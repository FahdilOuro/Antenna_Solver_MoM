"""
V5 MoM runs for the EurAAP Small Antenna Super-Gain Challenge (fiche V5).

    V5a   strip dipole 150 x 4 mm, PEC, 0.6-1.2 GHz:
          Zin(f), S11(f) (50 Ohm), D_phi in the xy plane at 0.9 GHz.
    V5b'  TUNED square loop (side 20 mm on the centre line, w in {1, 2, 4} mm):
          feed cut at the middle of the side y = -s/2, ideal capacitor on a second
          cut at the middle of y = +s/2 (w = 1 -> 0.48 pF, 2 -> 0.65 pF, 4 -> 1.08 pF,
          the CST values). At the series resonance: f_res, R_in, P_rad, P_cond,
          eta_rad = P_rad / (P_rad + P_cond) for PEC, copper zs_scale = 1, copper
          zs_scale = 0.5.
    V5b   (legacy, untuned single-port loop; not in any profile -- its CST
          counterpart did not resolve Re Zin, see the fiche)

Profiles (--profile, default quick):
    quick  V5a: 2 meshes, 21 frequencies.  V5b': w = 2 mm, n_width = 2, 4, 8
           (cosine edge grading).  Budget < 1 h of GPU.
    full   V5a: 3 meshes x 2 gradings, 121 frequencies.  V5b': w = 1, 2, 4 mm,
           3 meshes x 2 gradings.

V5b' method (the capacitor is NOT in the Z sweep)
-------------------------------------------------
The bare loop's two-port Zp(f) = Y(f)^{-1} varies slowly with frequency; the
capacitor creates the narrow resonance (Q ~ 330-440 in CST). So:
  1. Y(f) of the reduced 2-port model on 9 frequencies in [0.85, 0.93] GHz;
     PEC and copper share ONE Z per frequency (only Z_t = Z + alpha Zs G changes);
  2. Zp interpolated element-wise (cubic spline, Re and Im) on 2001 points;
  3. Zin(f) = Z11 - Z12 Z21 / (Z22 + 1/(j w C))  (the solve equations in
     impedance form), resonance = rising zero of Im Zin, by linear interpolation
     between the two bracketing grid points;
  4. validation: Z assembled EXACTLY at each f_res found, full reduced-model
     solve with the capacitor; the gap to the interpolated Zin is printed and
     saved, and P_rad, P_cond, eta_rad come from this exact point.

Ports are delta-gap cuts across the full strip width at x = 0
(backend.optim.ports.port_from_cut). CST uses 0.5 mm gaps: compare R_in and eta,
not f_res.

Outputs (one sub-folder per case under OUT_DIR): CSV files with units in the
column names, metadata.json (mesh N, parameters, timings, git version).

DO NOT run the full set casually (an estimate is printed and confirmation asked):

    python backend/antenna_simulation/euraap_challenge/v5_mom_runs.py --dry-run [--profile full]
    python backend/antenna_simulation/euraap_challenge/v5_mom_runs.py            # quick profile
    # smoke test (coarsest V5b' mesh only) into a scratch folder:
    python backend/antenna_simulation/euraap_challenge/v5_mom_runs.py --cases V5bp \
        --levels 0 --out <scratch> --yes

Completed cases (metadata.json with "complete": true) are skipped unless
--overwrite is given, so an interrupted run can be resumed.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.optim import ReducedMultiport, incidence_matrix                     # noqa: E402
from backend.optim.farfield import RadiationOperator                             # noqa: E402
from backend.optim.geometries import square_loop, strip_dipole                   # noqa: E402
from backend.optim.objectives import to_dbi                                      # noqa: E402
from backend.optim.reduced_model import terminated_input_impedance               # noqa: E402
from backend.optim.threads import limit_blas_threads, threads_for_size           # noqa: E402
from backend.rwg.rwg3 import calculate_z_matrice_sweep                           # noqa: E402
from backend.utils.impmet import rwg_gram_matrix, DEFAULT_FILL_MODE              # noqa: E402
from backend.utils.lossy_electric_conductor import calculate_normal_surface_impedance  # noqa: E402

# =============================================================================
# Parameters
# =============================================================================
OUT_DIR = Path(r"D:\PhD_Fahdilou_Leat\PUBLICATIONS\7-EurAAP Small Antenna Super-Gain Challenge"
               r"\fiches_cst\resultats_V5\mom")

Z0 = 50.0                    # Ohm, V5a reference impedance
SIGMA_CU = 5.8e7             # S/m
ZS_SCALES = (1.0, 0.5)       # surface-impedance factor alpha (open question, see reduced_model)
GRADINGS = ("uniform", "cosine")

V5A = dict(length=0.150, width=0.004, f_start=0.6e9, f_stop=1.2e9, n_freq=121, f_pattern=0.9e9,
           levels=((40, 2), (60, 4), (80, 8)))                      # (n_len, n_width)

#: V5b' loop geometry. CST's parameter "cote = 20" is the OUTER side (confirmed
#: 5/11): centre-line side = 20 - w (19 / 18 / 16 mm), ports at y = +-(10 - w/2).
#: "centerline" keeps the fiche's original reading (centre-line side 20 mm), still
#: useful for the Grover check.
LOOP_GEOMETRIES = ("cst", "centerline")
DEFAULT_LOOP_GEOMETRY = "cst"


def loop_centerline_side(w, geometry):
    if geometry == "cst":
        return V5BP["side"] - w            # outer side fixed at 20 mm
    if geometry == "centerline":
        return V5BP["side"]
    raise ValueError(f"geometry must be one of {LOOP_GEOMETRIES}, got {geometry!r}")


V5BP = dict(side=0.020, widths=(1e-3, 2e-3, 4e-3),
            cap_F={1.0: 0.48e-12, 2.0: 0.65e-12, 4.0: 1.08e-12},   # by w in mm (CST values)
            f_res_cst={1.0: 0.8912e9, 2.0: 0.8888e9, 4.0: 0.8845e9},  # CST copper resonances
            # Band: the instruction was 9 points in [0.85, 0.93] GHz, but with the CST
            # capacitors the MoM loop (delta gaps, no gap capacitance) resonates near
            # 0.78 GHz (smoke test 2026-11-05: Im Zin = +35..+77 Ohm over 0.85-0.93).
            # 13 points over 0.70-1.00 GHz cover both that and the CST f_res; the bare
            # loop's Zp is smooth there and the exact-point validation checks the spline.
            f_start=0.70e9, f_stop=1.00e9, n_freq=13, n_fine=2001,
            r_port=1.0,                                              # generator, as CST's 1 Ohm port
            levels=((16, 2), (16, 4), (16, 8)))                      # (n_side, n_width): width study

V5B_LEGACY = dict(side=0.020, widths=(1e-3, 2e-3, 4e-3), f_start=0.8e9, f_stop=1.0e9, n_freq=41,
                  levels=((12, 2), (16, 4), (24, 8)))

#: (label, zs_scale) -- PEC and both copper variants are derived from the SAME Z.
VARIANTS = (("PEC", None),) + tuple((f"Cu_zs{a:g}", a) for a in ZS_SCALES)

PROFILES = {
    "quick": dict(V5a=dict(levels=(0, 1), gradings=("uniform",), n_freq=21),
                  V5bp=dict(widths_mm=(2.0,), levels=(0, 1, 2), gradings=("cosine",))),
    "full": dict(V5a=dict(levels=(0, 1, 2), gradings=GRADINGS, n_freq=121),
                 V5bp=dict(widths_mm=(1.0, 2.0, 4.0), levels=(0, 1, 2), gradings=GRADINGS)),
}

PATTERN_PHI_DEG = np.arange(0.0, 360.0, 1.0)

# Z assembly in frequency blocks: bounds host memory and, above all, GPU memory --
# impedance_matrix_z_sweep_gpu's intermediates grow with the number of frequencies
# per call (41 frequencies at N = 280 exhausted a 4 GB T600 on 2026-11-04; 8 ran fine).
FREQ_BLOCK_MAX = 8
Z_BLOCK_BUDGET_BYTES = 0.6e9
USE_GPU = None               # None = auto (as calculate_z_matrice_sweep), True / False to force

# --- Timing model for the up-front estimate ----------------------------------
# Measured on the development laptop (NVIDIA T600 4 GB, OpenBLAS 16 threads), 2026-11-04:
#   GPU sweep block:  N=280: 5.4 s + 0.22 s/freq ; N=704: 12.6 s + 1.25 s/freq ;
#                     N=1840: 39.4 s + 4.9 s/freq
#   CPU sweep:        N=79: 0.7 s + 0.68 s/freq   (step 1: N=157, 1 freq: 5.3 s)
# Per-frequency part extrapolated in N^2 (conservative: measured exponent ~1.6 on
# GPU), fixed part in N.
GPU_REF = dict(N=704, fixed_s=12.6, per_freq_s=1.25)
CPU_REF = dict(N=79, fixed_s=0.7, per_freq_s=0.68)
LU_REF = dict(N=1000, s=0.1)          # one complex LU, multithreaded, extrapolated in N^3
FILL_COST_FACTOR = 1.8                # full_sym / mirror assembly time, measured 2026-11-05:
                                      # GPU 1.70 (N=880) / 1.77 (N=2208), CPU 1.86 (N=880)


# =============================================================================
# Case definitions
# =============================================================================
def _grading_arg(name):
    return None if name == "uniform" else name


def _with_frequency(f, f_needed):
    """Make sure f_needed is one of the sweep points."""
    if np.min(np.abs(f - f_needed)) > 1e-6 * f_needed:
        f = np.sort(np.append(f, f_needed))
    return f


def build_cases(profile="quick", cases=None, levels=None, gradings=None, widths_mm=None,
                n_freq=None, loop_geometry=DEFAULT_LOOP_GEOMETRY):
    """
    Cases of `profile`; any of cases / levels / gradings / widths_mm / n_freq given
    explicitly overrides the profile (n_freq applies to V5a and legacy V5b only: the
    V5b' spline needs its 9 points).
    """
    prof = PROFILES[profile]
    cases = tuple(cases) if cases else tuple(prof)
    out = []
    if "V5a" in cases:
        p = prof["V5a"]
        for lv in (levels or p["levels"]):
            n_len, n_w = V5A["levels"][lv]
            for g in (gradings or p["gradings"]):
                mesh = strip_dipole(V5A["length"], V5A["width"], n_len, n_w, _grading_arg(g))
                f = np.linspace(V5A["f_start"], V5A["f_stop"], n_freq or p["n_freq"])
                f = _with_frequency(f, V5A["f_pattern"])
                out.append(dict(id=f"V5a_dipole_nl{n_len}_nw{n_w}_{g}", kind="V5a", mesh=mesh,
                                frequencies=f, level=lv, grading=g,
                                params=dict(mesh.params, material="PEC")))
    if "V5bp" in cases:
        p = prof["V5bp"]
        for w_mm in (widths_mm or p["widths_mm"]):
            w = w_mm * 1e-3
            for lv in (levels or p["levels"]):
                n_side, n_w = V5BP["levels"][lv]
                for g in (gradings or p["gradings"]):
                    s_c = loop_centerline_side(w, loop_geometry)
                    mesh = square_loop(s_c, w, n_side, n_w, _grading_arg(g))
                    f = np.linspace(V5BP["f_start"], V5BP["f_stop"], V5BP["n_freq"])
                    cap = V5BP["cap_F"][float(w_mm)]
                    out.append(dict(id=f"V5bp_{loop_geometry}_w{w_mm:g}mm_ns{n_side}_nw{n_w}_{g}",
                                    kind="V5bp", mesh=mesh, frequencies=f, level=lv, grading=g,
                                    cap_F=cap,
                                    params=dict(mesh.params, sigma_S_per_m=SIGMA_CU,
                                                zs_scales=list(ZS_SCALES), capacitor_F=cap,
                                                r_port_ohm=V5BP["r_port"],
                                                n_fine=V5BP["n_fine"],
                                                loop_geometry=loop_geometry,
                                                outer_side_m=s_c + w)))
    if "V5b" in cases:
        for w_mm in (widths_mm or (1.0, 2.0, 4.0)):
            w = w_mm * 1e-3
            for lv in (levels or (0, 1, 2)):
                n_side, n_w = V5B_LEGACY["levels"][lv]
                for g in (gradings or GRADINGS):
                    mesh = square_loop(V5B_LEGACY["side"], w, n_side, n_w, _grading_arg(g))
                    f = np.linspace(V5B_LEGACY["f_start"], V5B_LEGACY["f_stop"],
                                    n_freq or V5B_LEGACY["n_freq"])
                    out.append(dict(id=f"V5b_loop_w{w_mm:g}mm_ns{n_side}_nw{n_w}_{g}",
                                    kind="V5b", mesh=mesh, frequencies=f, level=lv, grading=g,
                                    params=dict(mesh.params, sigma_S_per_m=SIGMA_CU,
                                                zs_scales=list(ZS_SCALES))))
    return out


# =============================================================================
# Estimate
# =============================================================================
def _gpu_available():
    try:
        from backend.utils.compute_backend import GPU_AVAILABLE
        return bool(GPU_AVAILABLE) and USE_GPU is not False
    except Exception:
        return False


def freq_block_size(n_edges):
    return int(max(1, min(FREQ_BLOCK_MAX, Z_BLOCK_BUDGET_BYTES // (16 * n_edges ** 2))))


def _z_seconds(N, nf, gpu):
    """Calibrated in fill_mode="mirror"; FILL_COST_FACTOR scales to the full_sym default."""
    if gpu:
        blocks = int(np.ceil(nf / freq_block_size(N)))
        t = (blocks * GPU_REF["fixed_s"] * N / GPU_REF["N"]
             + nf * GPU_REF["per_freq_s"] * (N / GPU_REF["N"]) ** 2)
    else:
        t = (CPU_REF["fixed_s"] + nf * CPU_REF["per_freq_s"]) * (N / CPU_REF["N"]) ** 2
    return t * (FILL_COST_FACTOR if DEFAULT_FILL_MODE != "mirror" else 1.0)


def estimate_seconds(case, gpu):
    N = case["mesh"].n_edges
    nf = len(case["frequencies"])
    lu = LU_REF["s"] * (N / LU_REF["N"]) ** 3
    if case["kind"] == "V5a":
        return _z_seconds(N, nf, gpu) + nf * lu
    n_var = len(VARIANTS)
    if case["kind"] == "V5bp":           # sweep + exact Z at <= 2 * len(VARIANTS) resonances
        n_ex = 2 * n_var
        return _z_seconds(N, nf, gpu) + _z_seconds(N, n_ex, gpu) + (nf * n_var + n_ex) * lu
    return _z_seconds(N, nf, gpu) + nf * n_var * lu


def _fmt_duration(s):
    s = int(round(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{sec:02d}s" if m else f"{sec}s")


# =============================================================================
# Z sweep in blocks (with GPU out-of-memory recovery)
# =============================================================================
def _free_gpu_pool():
    try:
        import cupy
        cupy.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


def z_sweep_blocks(mesh, frequencies, use_gpu=USE_GPU, log=print):
    """
    Yield (f_block, Z_block (nb, N, N)) over the sweep. On a GPU failure the block
    is retried with half as many frequencies, then on CPU for a single frequency.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    N = mesh.n_edges
    block = freq_block_size(N)
    i = 0
    while i < len(frequencies):
        f_blk = frequencies[i:i + block]
        try:
            *_, Z = calculate_z_matrice_sweep(mesh.points, mesh.triangles, mesh.edges, f_blk,
                                              use_gpu=use_gpu)
        except Exception as exc:                               # noqa: BLE001
            _free_gpu_pool()
            if use_gpu is False:
                raise
            if block > 1:
                block = max(1, block // 2)
                log(f"    Z block failed ({type(exc).__name__}); retrying with {block} "
                    f"frequencies per block.")
                continue
            log(f"    Z block failed on GPU with 1 frequency ({type(exc).__name__}): "
                f"assembling this frequency on CPU (slow).")
            *_, Z = calculate_z_matrice_sweep(mesh.points, mesh.triangles, mesh.edges, f_blk,
                                              use_gpu=False)
        _free_gpu_pool()
        yield f_blk, Z
        i += len(f_blk)


# =============================================================================
# V5a / legacy V5b post-processing (one row per frequency)
# =============================================================================
def _s11(z_in, z_ref=Z0):
    g = (z_in - z_ref) / (z_in + z_ref)
    return 20 * np.log10(abs(g)), np.degrees(np.angle(g))


def run_v5a_frequency(f, Z, A, mesh, pattern_out):
    model = ReducedMultiport.from_matrices(Z, A, f)
    sol = model.solve([1.0], [Z0])
    z_in = sol.z_in[0]
    s11_db, s11_ph = _s11(z_in)
    row = {"freq_GHz": f / 1e9, "Re_Zin_ohm": z_in.real, "Im_Zin_ohm": z_in.imag,
           "S11_mag_dB": s11_db, "S11_phase_deg": s11_ph, "P_rad_W_per_1V_emf": sol.p_rad}
    if abs(f - V5A["f_pattern"]) <= 1e-6 * f:
        op = RadiationOperator(mesh.points, mesh.triangles, mesh.edges, f)
        I = model.edge_currents(sol.v)
        p_sphere = op.radiated_power_sphere(I)
        theta = np.full(PATTERN_PHI_DEG.shape, np.pi / 2)
        u_t, u_p = op.intensity(I, theta, np.radians(PATTERN_PHI_DEG))
        pattern_out.update(
            phi_deg=PATTERN_PHI_DEG,
            D_phi_dBi=to_dbi(4 * np.pi * u_p / sol.p_rad),
            D_theta_dBi=to_dbi(4 * np.pi * u_t / sol.p_rad),
            D_total_dBi=to_dbi(4 * np.pi * (u_t + u_p) / sol.p_rad),
            p_rad_R=sol.p_rad, p_rad_sphere=p_sphere, freq_GHz=f / 1e9)
    return row


def run_v5b_frequency(f, Z, A, G):
    """Legacy single-port loop: PEC and both copper variants from the same Z."""
    pec = ReducedMultiport.from_matrices(Z, A, f, keep_matrices=False)
    s_pec = pec.solve([1.0], [Z0])
    z_pec = s_pec.z_in[0]
    row = {"freq_GHz": f / 1e9, "Re_Zin_PEC_ohm": z_pec.real, "Im_Zin_PEC_ohm": z_pec.imag}
    zs = calculate_normal_surface_impedance(f, SIGMA_CU)
    row["Re_Zs_ohm"] = zs.real
    for a in ZS_SCALES:
        cu = ReducedMultiport.from_matrices(Z, A, f, G=G, Zs=zs, zs_scale=a, keep_matrices=False)
        s = cu.solve([1.0], [Z0])
        z = s.z_in[0]
        i2 = abs(s.i[0]) ** 2
        tag = f"zs{a:g}"
        row[f"Re_Zin_Cu_{tag}_ohm"] = z.real
        row[f"Im_Zin_Cu_{tag}_ohm"] = z.imag
        row[f"eta_rad_{tag}"] = s.p_rad / (s.p_rad + s.p_cond)
        row[f"R_loss_{tag}_ohm"] = z.real - z_pec.real
        row[f"R_cond_{tag}_ohm"] = 2 * s.p_cond / i2
        row[f"R_rad_{tag}_ohm"] = 2 * s.p_rad / i2
    return row


# =============================================================================
# V5b' : tuned loop
# =============================================================================
def _variant_model(Z, A, f, G, alpha):
    """Reduced 2-port for one variant; PEC (alpha None) and copper share Z."""
    if alpha is None:
        return ReducedMultiport.from_matrices(Z, A, f, keep_matrices=False)
    zs = calculate_normal_surface_impedance(f, SIGMA_CU)
    return ReducedMultiport.from_matrices(Z, A, f, G=G, Zs=zs, zs_scale=alpha, keep_matrices=False)


def capacitor_impedance(f, C):
    return 1.0 / (1j * 2 * np.pi * np.asarray(f) * C)


def find_series_resonance(f, z_in):
    """
    Rising zero of X = Im Zin (series resonance) by linear interpolation between
    the bracketing points. Returns dict(f_res, R_in, dXdf, Q, index) or None.
    If several rising zeros exist, the one nearest the band centre is kept.
    """
    X = z_in.imag
    k = np.flatnonzero((X[:-1] <= 0) & (X[1:] > 0))
    if k.size == 0:
        return None
    k = int(k[np.argmin(np.abs(f[k] - 0.5 * (f[0] + f[-1])))])
    t = -X[k] / (X[k + 1] - X[k])
    f_res = f[k] + t * (f[k + 1] - f[k])
    R_in = z_in.real[k] + t * (z_in.real[k + 1] - z_in.real[k])
    dXdf = (X[k + 1] - X[k]) / (f[k + 1] - f[k])
    return dict(f_res=f_res, R_in=R_in, dXdf=dXdf, Q=f_res * dXdf / (2 * R_in), index=k)


def tuning_reactance(zp, x_guess):
    """
    Load reactance X (port 2) making Im Zin = 0 at port 1 for the 2-port zp:

        Zin = Z11 - p / (Z22 + jX),  p = Z12 Z21,  Z22 = r + j x22,  y = x22 + X
        Im Zin = 0  <=>  Im(Z11) (r^2 + y^2) - (Im(p) r - Re(p) y) = 0
                    <=>  a y^2 + Re(p) y + (a r^2 - Im(p) r) = 0,   a = Im Z11.

    Returns the root with X < 0 (a capacitor) closest to x_guess, or None.
    """
    p = zp[0, 1] * zp[1, 0]
    r, x22, a = zp[1, 1].real, zp[1, 1].imag, zp[0, 0].imag
    roots = np.roots([a, p.real, a * r * r - p.imag * r])
    X = [float(y.real - x22) for y in roots if abs(y.imag) < 1e-9 * (1 + abs(y.real))]
    X = [x for x in X if x < 0]
    return min(X, key=lambda x: abs(x - x_guess)) if X else None


def _spline_zp(f_s, zp_s, f_eval):
    out = np.empty((np.size(f_eval), 2, 2), dtype=complex)
    for i in range(2):
        for j in range(2):
            out[:, i, j] = (CubicSpline(f_s, zp_s[:, i, j].real)(f_eval)
                            + 1j * CubicSpline(f_s, zp_s[:, i, j].imag)(f_eval))
    return out


def run_v5bp_case(case, log=print):
    """
    Returns (rows_resonance, rows_fine, rows_samples, extra_meta). Two modes:
      'C_cst'   : the CST capacitor; the MoM loop resonates at its own f_res.
      'C_tuned' : capacitor re-tuned (per variant) so that the MoM resonates at
                  CST's f_res for this w -- R_in and eta then compare at the same
                  frequency (R_rad ~ f^4 makes the C_cst comparison biased).
    """
    mesh = case["mesh"]
    C_cst = case["cap_F"]
    w_mm = round(case["params"]["width"] * 1e3, 6)
    f_target = V5BP["f_res_cst"].get(w_mm)
    ports = [mesh.port("feed"), mesh.port("load")]
    A = incidence_matrix(ports, mesh.edges)
    t = time.perf_counter()
    G = rwg_gram_matrix(mesh.edges, mesh.triangles, mesh.points)
    timings = dict(t_gram_s=time.perf_counter() - t, t_z_sweep_s=0.0, t_reduce_s=0.0)

    # 1. Zp(f) = Y(f)^-1 on the coarse grid, three variants from one Z per frequency
    f_s = np.asarray(case["frequencies"], dtype=float)
    Zp = {v: [] for v, _ in VARIANTS}
    tz = time.perf_counter()
    for f_blk, Z_blk in z_sweep_blocks(mesh, f_s, log=log):
        timings["t_z_sweep_s"] += time.perf_counter() - tz
        tr = time.perf_counter()
        with limit_blas_threads(threads_for_size(mesh.n_edges)):
            for f, Z in zip(f_blk, Z_blk):
                for v, alpha in VARIANTS:
                    Zp[v].append(np.linalg.inv(_variant_model(Z, A, f, G, alpha).Y))
        timings["t_reduce_s"] += time.perf_counter() - tr
        tz = time.perf_counter()
    Zp = {v: np.array(z) for v, z in Zp.items()}                       # (nf, 2, 2)

    # 2-3. spline, capacitor, resonance -- for the CST capacitor and the re-tuned one
    f_fine = np.linspace(f_s[0], f_s[-1], V5BP["n_fine"])
    zin_fine, res, cap = {}, {}, {}
    for v, _ in VARIANTS:
        zp_f = _spline_zp(f_s, Zp[v], f_fine)
        caps = {"C_cst": C_cst}
        if f_target is not None and f_s[0] < f_target < f_s[-1]:
            zp_t = _spline_zp(f_s, Zp[v], [f_target])[0]
            X_t = tuning_reactance(zp_t, x_guess=-1.0 / (2 * np.pi * f_target * C_cst))
            if X_t is not None:
                caps["C_tuned"] = -1.0 / (2 * np.pi * f_target * X_t)
        for mode, C in caps.items():
            key = (mode, v)
            cap[key] = C
            zin_fine[key] = terminated_input_impedance(zp_f, capacitor_impedance(f_fine, C)[:, None])
            res[key] = find_series_resonance(f_fine, zin_fine[key])
            if res[key] is None:
                X = zin_fine[key].imag
                warnings.warn(f"{case['id']} {mode} {v}: no series resonance in "
                              f"[{f_s[0] / 1e9:.3f}, {f_s[-1] / 1e9:.3f}] GHz (Im Zin from "
                              f"{X.min():.1f} to {X.max():.1f} Ohm). Widen V5BP f_start/f_stop.")

    # 4. exact validation at each f_res found (one Z per distinct frequency; the
    #    re-tuned resonances all sit near f_target, so they share few assemblies)
    f_exact = sorted({r["f_res"] for r in res.values() if r is not None})
    exact = {}
    tz = time.perf_counter()
    for f_blk, Z_blk in z_sweep_blocks(mesh, f_exact, log=log):
        timings["t_z_sweep_s"] += time.perf_counter() - tz
        for f, Z in zip(f_blk, Z_blk):
            for (mode, v), r in res.items():
                if r is not None and r["f_res"] == f:
                    alpha = dict(VARIANTS)[v]
                    m = _variant_model(Z, A, f, G, alpha)
                    exact[(mode, v)] = m.solve([1.0, 0.0], [V5BP["r_port"],
                                                            capacitor_impedance(f, cap[(mode, v)])])
        tz = time.perf_counter()

    rows = []
    for (mode, v), r in res.items():
        alpha = dict(VARIANTS)[v]
        row = dict(mode=mode, variant=v, zs_scale="" if alpha is None else alpha,
                   C_pF=cap[(mode, v)] * 1e12)
        if r is None:
            rows.append(dict(row, f_res_GHz=np.nan))
            continue
        s = exact[(mode, v)]
        z_ex = complex(s.z_in[0])
        dz = z_ex - r["R_in"]                           # interpolated Zin at f_res is R_in + j0
        row.update(f_res_GHz=r["f_res"] / 1e9, R_in_interp_ohm=r["R_in"],
                   Q_interp=r["Q"], dXdf_ohm_per_MHz=r["dXdf"] * 1e6,
                   R_in_exact_ohm=z_ex.real, X_exact_ohm=z_ex.imag,
                   dZin_abs_ohm=abs(dz), dR_rel=(z_ex.real - r["R_in"]) / r["R_in"],
                   df_equiv_kHz=-z_ex.imag / r["dXdf"] / 1e3,
                   P_rad_W=s.p_rad, P_cond_W=s.p_cond,
                   eta_rad=s.p_rad / (s.p_rad + s.p_cond))
        rows.append(row)
        log(f"    {mode:7s} {v:9s} C = {row['C_pF']:.4f} pF  f_res = {r['f_res'] / 1e9:.5f} GHz  "
            f"R_in = {z_ex.real:.4f} Ohm (interp {r['R_in']:.4f}, |dZin| = {abs(dz):.2e} Ohm, "
            f"X_exact = {z_ex.imag:+.2e} Ohm <-> df = {row['df_equiv_kHz']:+.2f} kHz)  "
            f"Q = {r['Q']:.0f}  eta_rad = {row['eta_rad']:.4f}")

    fine = [dict(freq_GHz=f / 1e9, **{k: val for (mode, v) in zin_fine
                                       for k, val in ((f"Re_Zin_{mode}_{v}_ohm", zin_fine[(mode, v)][i].real),
                                                      (f"Im_Zin_{mode}_{v}_ohm", zin_fine[(mode, v)][i].imag))})
            for i, f in enumerate(f_fine)]
    samples = []
    for i, f in enumerate(f_s):
        row = dict(freq_GHz=f / 1e9)
        for v in Zp:
            for a in range(2):
                for b in range(2):
                    row[f"Re_Z{a + 1}{b + 1}_{v}_ohm"] = Zp[v][i, a, b].real
                    row[f"Im_Z{a + 1}{b + 1}_{v}_ohm"] = Zp[v][i, a, b].imag
        samples.append(row)
    return rows, fine, samples, dict(timings, f_exact_GHz=[f / 1e9 for f in f_exact],
                                     load_port_edges=ports[1].n_edges,
                                     f_target_cst_GHz=None if f_target is None else f_target / 1e9)


# =============================================================================
# Driver
# =============================================================================
def git_version():
    def _git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:
            return None
    return dict(commit=_git("rev-parse", "HEAD"), branch=_git("rev-parse", "--abbrev-ref", "HEAD"),
                dirty=bool(_git("status", "--porcelain")))


def _write_csv(path, rows):
    keys = list(rows[0].keys())
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in r.items()})


def run_case(case, out_dir, log=print):
    mesh = case["mesh"]
    N = mesh.n_edges
    case_dir = out_dir / case["id"]
    case_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(case=case["id"], kind=case["kind"], level=case["level"], grading=case["grading"],
                n_edges=N, n_triangles=int(mesh.triangles.total_of_triangles),
                n_points=int(mesh.points.total_of_points), feed_port_edges=mesh.port().n_edges,
                parameters=case["params"],
                frequencies_GHz=dict(start=float(case["frequencies"][0] / 1e9),
                                     stop=float(case["frequencies"][-1] / 1e9),
                                     n=len(case["frequencies"])),
                constants=dict(epsilon0=8.854e-12, mu0=1.257e-6, note="as rwg3.calculate_z_matrice"),
                port_model="delta-gap cuts across the full width at x = 0 (port_from_cut)",
                z_fill_mode=DEFAULT_FILL_MODE,
                git=git_version(), started=_dt.datetime.now().isoformat(timespec="seconds"),
                python=platform.python_version(), numpy=np.__version__,
                gpu=_gpu_available(), freq_block=freq_block_size(N), complete=False)
    t0 = time.perf_counter()

    if case["kind"] == "V5bp":
        rows, fine, samples, extra = run_v5bp_case(case, log=log)
        _write_csv(case_dir / "resonance.csv", rows)
        _write_csv(case_dir / "zin_fine.csv", fine)
        _write_csv(case_dir / "zp_samples.csv", samples)
        meta.update(extra)
    else:
        A = mesh.port().incidence_vector(mesh.edges)[:, None]
        G = rwg_gram_matrix(mesh.edges, mesh.triangles, mesh.points) if case["kind"] == "V5b" else None
        rows, pattern = [], {}
        t_z = t_post = 0.0
        tz0 = time.perf_counter()
        for f_blk, Z_blk in z_sweep_blocks(mesh, case["frequencies"], log=log):
            t_z += time.perf_counter() - tz0
            tp = time.perf_counter()
            with limit_blas_threads(threads_for_size(N)):
                for f, Z in zip(f_blk, Z_blk):
                    rows.append(run_v5a_frequency(f, Z, A, mesh, pattern) if case["kind"] == "V5a"
                                else run_v5b_frequency(f, Z, A, G))
            t_post += time.perf_counter() - tp
            log(f"    {len(rows)}/{len(case['frequencies'])} frequencies "
                f"(Z {t_z:.1f} s, post-processing {t_post:.1f} s)")
            tz0 = time.perf_counter()
        _write_csv(case_dir / "results.csv", rows)
        if pattern:
            prow = [{"phi_deg": float(p), "D_phi_dBi": float(a), "D_theta_dBi": float(b),
                     "D_total_dBi": float(c)}
                    for p, a, b, c in zip(pattern["phi_deg"], pattern["D_phi_dBi"],
                                          pattern["D_theta_dBi"], pattern["D_total_dBi"])]
            _write_csv(case_dir / "pattern_xy.csv", prow)
            meta["pattern"] = dict(freq_GHz=pattern["freq_GHz"], theta_deg=90.0,
                                   P_rad_R_W=pattern["p_rad_R"],
                                   P_rad_sphere_W=pattern["p_rad_sphere"],
                                   P_rad_rel_diff=(pattern["p_rad_sphere"] - pattern["p_rad_R"])
                                   / pattern["p_rad_R"])
        meta.update(t_z_assembly_s=t_z, t_postprocessing_s=t_post)

    meta.update(t_total_s=time.perf_counter() - t0, complete=True,
                finished=_dt.datetime.now().isoformat(timespec="seconds"))
    with open(case_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=float)
    return meta


def _is_complete(out_dir, case):
    p = out_dir / case["id"] / "metadata.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("complete", False)
    except Exception:
        return False


def _csv_list(s, cast=str):
    return None if s is None else tuple(cast(x.strip()) for x in s.split(","))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--profile", default="quick", choices=sorted(PROFILES))
    ap.add_argument("--cases", default=None, help="subset of V5a,V5bp (and legacy V5b)")
    ap.add_argument("--levels", default=None, help="density levels (0 = coarsest)")
    ap.add_argument("--gradings", default=None, help="uniform,cosine")
    ap.add_argument("--widths", default=None, help="loop strip widths in mm, e.g. 1,2,4")
    ap.add_argument("--n-freq", type=int, default=None, help="V5a / V5b number of frequencies")
    ap.add_argument("--loop-geometry", default=DEFAULT_LOOP_GEOMETRY, choices=LOOP_GEOMETRIES,
                    help="V5b': 'cst' = outer side 20 mm (CST's cote), 'centerline' = centre-line side 20 mm")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and estimate, then stop")
    ap.add_argument("--overwrite", action="store_true", help="re-run completed cases")
    args = ap.parse_args(argv)

    cases = build_cases(profile=args.profile, cases=_csv_list(args.cases),
                        levels=_csv_list(args.levels, int), gradings=_csv_list(args.gradings),
                        widths_mm=_csv_list(args.widths, float), n_freq=args.n_freq,
                        loop_geometry=args.loop_geometry)
    out_dir = Path(args.out)
    gpu = _gpu_available()

    todo = [c for c in cases if args.overwrite or not _is_complete(out_dir, c)]
    print(f"V5 MoM runs, profile '{args.profile}' -> {out_dir}")
    print(f"Backend for Z: {'GPU (auto)' if gpu else 'CPU'}; estimate model: "
          f"{'GPU' if gpu else 'CPU'} calibration of 2026-11-04 (see file header).")
    print(f"{'case':42s} {'N':>6s} {'n_f':>5s} {'block':>5s} {'estimate':>9s}")
    total = 0.0
    for c in cases:
        est = estimate_seconds(c, gpu)
        done = c not in todo
        total += 0 if done else est
        nf = len(c["frequencies"]) + (2 * len(VARIANTS) if c["kind"] == "V5bp" else 0)
        print(f"{c['id']:42s} {c['mesh'].n_edges:6d} {nf:5d} "
              f"{freq_block_size(c['mesh'].n_edges):5d} {_fmt_duration(est):>9s}"
              f"{'   (done, skipped)' if done else ''}")
    print(f"{len(todo)} case(s) to run out of {len(cases)}; estimated total {_fmt_duration(total)}.")
    if any(c["kind"] == "V5bp" for c in cases):
        print(f"(V5b' n_f = {V5BP['n_freq']} sweep points + up to {2 * len(VARIANTS)} exact Z "
              f"at the resonances found: CST capacitor and re-tuned capacitor, per variant.)")
    if args.dry_run or not todo:
        return 0
    if not args.yes:
        if input("Start? [y/N] ").strip().lower() not in ("y", "yes", "o", "oui"):
            print("Aborted.")
            return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    for k, c in enumerate(todo, 1):
        print(f"[{k}/{len(todo)}] {c['id']} (N = {c['mesh'].n_edges}, "
              f"estimate {_fmt_duration(estimate_seconds(c, gpu))})")
        meta = run_case(c, out_dir)
        print(f"    done in {_fmt_duration(meta['t_total_s'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
