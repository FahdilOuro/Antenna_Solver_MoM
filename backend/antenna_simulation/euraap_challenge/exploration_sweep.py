"""
Physical exploration in the challenge box (step 5) -- sweep A: two parallel strips.

Geometry (backend.optim.geometries.parallel_strip_array): two strips along x,
L = 180 mm, width w (4 mm, optionally 2 mm), z = 0, centre-to-centre spacing d
along y (15 ... 115 mm), one delta-gap port per strip at x = 0. Meshing rule of
the loss calibration (n_width = 4, cosine grading; backend.optim.loss_model),
cells of CELL_LENGTH along x.

For each (w, d, f0): ONE Z (fill_mode "full_sym", reused for everything below),
then for alpha in alpha_band(w) = alpha_cal * (0.85, 1, 1.15) a new LU of
Z + alpha Zs G, and two configurations:
    director  : el1 (-y, rear) fed, el2 (+y, front) loaded
    reflector : el2 (+y, front) fed, el1 (-y, rear) loaded
The load reactance maximises G (ideal matching) towards +y, phi polarisation
(backend.optim.inner.optimize_loads). Reported: G, G_r (50 Ohm), D, eta_rad, Zin,
front/back (+y/-y), G_max (geometry bound), G/G_max, X_opt and the equivalent L/C.

Sensitivity of the TOP best points (alpha_cal, best configuration): G at
f0 x (1 +- 0.5 %) and f0 x (1 +- 1 %) with the optimal load FROZEN as a constant L
or C (one extra Z sweep of 4 frequencies per point).

Mesh check (optional, on by default): one point (d = 35 mm, f0 = 0.9 GHz) again
with half the cell length along x -- the loss calibration only fixes the rule
ACROSS the width.

Outputs in OUT_DIR: sweepA_w<w>mm.csv (one row per d, f0, alpha, configuration),
sweepA_w<w>mm_sensitivity.csv, sweepA_w<w>mm_meta.json. Figures:
exploration_two_element.ipynb.

    python backend/antenna_simulation/euraap_challenge/exploration_sweep.py --dry-run
    python backend/antenna_simulation/euraap_challenge/exploration_sweep.py          # asks first
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import v5_mom_runs as v5                                                          # noqa: E402
from backend.optim import ReducedMultiport, incidence_matrix                     # noqa: E402
from backend.optim.farfield import RadiationOperator                             # noqa: E402
from backend.optim.geometries import parallel_strip_array, array_ports           # noqa: E402
from backend.optim.inner import optimize_loads                                   # noqa: E402
from backend.optim.loss_model import ALPHA_UNCERTAINTY, MESH_RULE, alpha_band, alpha_calibrated  # noqa: E402
from backend.optim.threads import limit_blas_threads, threads_for_size           # noqa: E402
from backend.utils.impmet import DEFAULT_FILL_MODE, rwg_gram_matrix              # noqa: E402
from backend.utils.lossy_electric_conductor import calculate_normal_surface_impedance  # noqa: E402

# =============================================================================
# Parameters
# =============================================================================
OUT_DIR = Path(r"D:\PhD_Fahdilou_Leat\PUBLICATIONS\7-EurAAP Small Antenna Super-Gain Challenge"
               r"\exploration")
LENGTH = 0.180
WIDTHS_MM = (4.0,)                                 # 2.0 optionally (--widths 4,2)
SPACINGS_MM = tuple(range(15, 116, 10))            # 15, 25, ..., 115
FREQS_GHZ = (0.80, 0.90, 1.00)
CELL_LENGTH = 4.5e-3                               # along x (not part of the calibrated rule)
SIGMA_CU = 5.8e7
Z_GEN = 50.0
DIRECTION = (np.pi / 2, np.pi / 2)                 # +y
BACK = (np.pi / 2, 3 * np.pi / 2)                  # -y
POLARIZATION = "phi"
N_STARTS = 32
X_BOUNDS = (-2000.0, 2000.0)
TOP_SENSITIVITY = 3
SENS_FACTORS = (0.99, 0.995, 1.005, 1.01)
CONFIGS = (("director", 0), ("reflector", 1))      # (name, fed element index; 0 = -y)
MESH_CHECK = dict(spacing_mm=35, f_GHz=0.90, cell_factor=0.5)

#: Measured inner-loop cost per optimize_loads call (32 starts, N ~ 870, bound
#: included), development laptop 2026-11-05 -- used by the estimate only.
T_INNER_S = 1.0
T_LU_S = 0.15


# =============================================================================
# Building blocks
# =============================================================================
def make_mesh(width, spacing, cell_length=CELL_LENGTH):
    return parallel_strip_array([LENGTH, LENGTH], width, [spacing], cell_length=cell_length,
                                **MESH_RULE)


def front_back_db(op, I):
    u_f = op.intensity(I, *DIRECTION)[1][0]
    u_b = op.intensity(I, *BACK)[1][0]
    return float(10 * np.log10(u_f / u_b)) if u_b > 0 else float("inf")


def db(x):
    return float(10 * np.log10(x)) if x and x > 0 else float("nan")


def evaluate_geometry(mesh, Zf, G, width_mm, n_starts=N_STARTS, seed=0, log=print):
    """All (f0, alpha, configuration) rows for one geometry; Zf = {f: Z}."""
    A = incidence_matrix([mesh.port("el1"), mesh.port("el2")], mesh.edges)
    rows = []
    for f, Z in Zf.items():
        op = RadiationOperator(mesh.points, mesh.triangles, mesh.edges, f)
        zs = calculate_normal_surface_impedance(f, SIGMA_CU)
        for factor, alpha in zip((1 - ALPHA_UNCERTAINTY, 1.0, 1 + ALPHA_UNCERTAINTY), alpha_band(width_mm)):
            with limit_blas_threads(threads_for_size(mesh.n_edges)):
                model = ReducedMultiport.from_matrices(Z, A, f, G=G, Zs=zs, zs_scale=alpha)
            group = []
            for config, fed in CONFIGS:
                res = optimize_loads(model, DIRECTION, POLARIZATION, "G", x_bounds=X_BOUNDS,
                                     n_starts=n_starts, seed=seed, operator=op, z_gen=Z_GEN,
                                     driven_port=fed)
                group.append(dict(
                    width_mm=width_mm, spacing_mm=round(mesh.params["spacings"][0] * 1e3, 6),
                    f_GHz=f / 1e9, alpha_factor=round(factor, 3), alpha=alpha, config=config,
                    fed=f"el{fed + 1}", loaded=f"el{2 - fed}",
                    X_opt_ohm=float(res.x_opt[0]), element=res.element_kind[0],
                    element_value=float(res.element_value[0]), element_label=res.element_label[0],
                    G=res.G, G_dBi=db(res.G), G_r_dBi=db(res.G_r), D_dBi=db(res.D),
                    eta_rad=res.eta_rad, Zin_re_ohm=res.z_in.real, Zin_im_ohm=res.z_in.imag,
                    FB_dB=front_back_db(op, res.edge_currents),
                    G_max_dBi=db(res.g_max), G_over_G_max=res.g_ratio,
                    N=mesh.n_edges, n_evaluations=res.n_evaluations))
            best = max(group, key=lambda r: r["G"])
            for r in group:
                r["best_config"] = r is best
            rows += group
        nominal = [r for r in rows if r["f_GHz"] == f / 1e9 and r["alpha_factor"] == 1.0]
        b = max(nominal, key=lambda r: r["G"])
        log(f"    d = {b['spacing_mm']:5.1f} mm  f0 = {f / 1e9:.2f} GHz  best {b['config']:9s} "
            f"G = {b['G_dBi']:6.2f} dBi  eta = {b['eta_rad']:.3f}  F/B = {b['FB_dB']:5.1f} dB  "
            f"X = {b['X_opt_ohm']:+8.1f} Ohm ({b['element_label']})  G/Gmax = {b['G_over_G_max']:.3f}")
    return rows


def load_reactance(kind, value, f):
    w = 2 * np.pi * f
    if kind == "C":
        return -1.0 / (w * value)
    if kind == "L":
        return w * value
    return 0.0


def sensitivity(width, row, log=print):
    """G at f0 * SENS_FACTORS with the load frozen as the L or C found at f0."""
    mesh = make_mesh(width, row["spacing_mm"] * 1e-3)
    A = incidence_matrix([mesh.port("el1"), mesh.port("el2")], mesh.edges)
    G = rwg_gram_matrix(mesh.edges, mesh.triangles, mesh.points)
    f0 = row["f_GHz"] * 1e9
    fed = 0 if row["fed"] == "el1" else 1
    out = []
    freqs = [f0 * s for s in SENS_FACTORS]
    for f_blk, Z_blk in v5.z_sweep_blocks(mesh, freqs, log=log):
        for f, Z in zip(f_blk, Z_blk):
            zs = calculate_normal_surface_impedance(f, SIGMA_CU)
            with limit_blas_threads(threads_for_size(mesh.n_edges)):
                model = ReducedMultiport.from_matrices(Z, A, f, G=G, Zs=zs, zs_scale=row["alpha"])
            op = RadiationOperator(mesh.points, mesh.triangles, mesh.edges, f)
            name = op.attach_functional(model, *DIRECTION, component=POLARIZATION)
            emf = np.zeros(2, complex)
            emf[fed] = 1.0
            zt = np.zeros(2, complex)
            zt[fed] = Z_GEN
            zt[1 - fed] = 1j * load_reactance(row["element"], row["element_value"], f)
            sol = model.solve(emf, zt)
            U = op.intensity_constant * abs(sol.functionals[name]) ** 2
            g, g_r = 4 * np.pi * U / sol.p_in_total, 4 * np.pi * U / sol.p_avail_total
            out.append(dict(width_mm=row["width_mm"], spacing_mm=row["spacing_mm"],
                            f0_GHz=row["f_GHz"], config=row["config"], alpha=row["alpha"],
                            element_label=row["element_label"], f_GHz=f / 1e9, f_over_f0=f / f0,
                            G_dBi=db(g), G_r_dBi=db(g_r), dG_dB=db(g) - row["G_dBi"],
                            eta_rad=sol.p_rad / sol.p_in_total))
    out.append(dict(width_mm=row["width_mm"], spacing_mm=row["spacing_mm"], f0_GHz=row["f_GHz"],
                    config=row["config"], alpha=row["alpha"], element_label=row["element_label"],
                    f_GHz=row["f_GHz"], f_over_f0=1.0, G_dBi=row["G_dBi"], G_r_dBi=row["G_r_dBi"],
                    dG_dB=0.0, eta_rad=row["eta_rad"]))
    return sorted(out, key=lambda r: r["f_GHz"])


# =============================================================================
# Driver
# =============================================================================
def write_csv(path, rows):
    keys = list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in r.items()})


def plan(widths_mm, spacings_mm, freqs_ghz, mesh_check, top):
    gpu = v5._gpu_available()
    items, total = [], 0.0
    for w_mm in widths_mm:
        for d in spacings_mm:
            m = make_mesh(w_mm * 1e-3, d * 1e-3)
            t = (v5._z_seconds(m.n_edges, len(freqs_ghz), gpu)
                 + len(freqs_ghz) * 3 * (T_LU_S + 2 * T_INNER_S))
            items.append((f"A w={w_mm:g} mm d={d} mm", m.n_edges, len(freqs_ghz), t))
            total += t
        n_ref = items[-1][1]
        t_s = top * (v5._z_seconds(n_ref, len(SENS_FACTORS), gpu) + len(SENS_FACTORS) * T_LU_S)
        items.append((f"sensitivity w={w_mm:g} mm, top {top}", n_ref, top * len(SENS_FACTORS), t_s))
        total += t_s
    if mesh_check:
        m = make_mesh(widths_mm[0] * 1e-3, mesh_check["spacing_mm"] * 1e-3,
                      CELL_LENGTH * mesh_check["cell_factor"])
        t = v5._z_seconds(m.n_edges, 1, gpu) + 3 * (T_LU_S + 2 * T_INNER_S)
        items.append((f"mesh check (cell along x / 2)", m.n_edges, 1, t))
        total += t
    return items, total, gpu


def main(argv=None):
    ap = argparse.ArgumentParser(description="Sweep A: two-element exploration in the box.")
    ap.add_argument("--widths", default=",".join(f"{w:g}" for w in WIDTHS_MM), help="mm, e.g. 4,2")
    ap.add_argument("--spacings", default=",".join(str(d) for d in SPACINGS_MM), help="mm")
    ap.add_argument("--freqs", default=",".join(f"{f:g}" for f in FREQS_GHZ), help="GHz")
    ap.add_argument("--n-starts", type=int, default=N_STARTS)
    ap.add_argument("--top", type=int, default=TOP_SENSITIVITY, help="points for the sensitivity study")
    ap.add_argument("--no-mesh-check", action="store_true")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args(argv)

    widths = [float(x) for x in args.widths.split(",")]
    for w in widths:
        alpha_calibrated(w)                                  # fails early outside [1, 4] mm
    spacings = [float(x) for x in args.spacings.split(",")]
    freqs = [float(x) for x in args.freqs.split(",")]
    mesh_check = None if args.no_mesh_check else MESH_CHECK
    out_dir = Path(args.out)

    items, total, gpu = plan(widths, spacings, freqs, mesh_check, args.top)
    print(f"Sweep A -> {out_dir}  (Z: {'GPU' if gpu else 'CPU'}, fill_mode {DEFAULT_FILL_MODE}; "
          f"alpha_cal: " + ", ".join(f"w={w:g} mm -> {alpha_calibrated(w):.3f}" for w in widths) + ")")
    print(f"{'item':40s} {'N':>6s} {'n_f':>4s} {'estimate':>9s}")
    for name, N, nf, t in items:
        print(f"{name:40s} {N:6d} {nf:4d} {v5._fmt_duration(t):>9s}")
    n_fill = sum(nf for _, _, nf, _ in items)
    print(f"{n_fill} Z fills (geometry x frequency); "
          f"{len(widths) * len(spacings) * len(freqs) * 3 * 2} load optimisations; "
          f"estimated total {v5._fmt_duration(total)}.")
    if args.dry_run:
        return 0
    if not args.yes and input("Start? [y/N] ").strip().lower() not in ("y", "yes", "o", "oui"):
        print("Aborted.")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    for w_mm in widths:
        w = w_mm * 1e-3
        tag = f"sweepA_w{w_mm:g}mm"
        rows, t0, timings = [], time.perf_counter(), []
        for d in spacings:
            mesh = make_mesh(w, d * 1e-3)
            print(f"[w = {w_mm:g} mm, d = {d:g} mm] N = {mesh.n_edges}")
            ts = time.perf_counter()
            G = rwg_gram_matrix(mesh.edges, mesh.triangles, mesh.points)
            Zf = {}
            for f_blk, Z_blk in v5.z_sweep_blocks(mesh, [f * 1e9 for f in freqs]):
                Zf.update(zip(f_blk, Z_blk))
            tz = time.perf_counter() - ts
            rows += evaluate_geometry(mesh, Zf, G, w_mm, n_starts=args.n_starts)
            timings.append(dict(spacing_mm=d, N=mesh.n_edges, t_z_s=tz,
                                t_total_s=time.perf_counter() - ts))
            write_csv(out_dir / f"{tag}.csv", rows)               # checkpoint after each geometry

        nominal_best = [r for r in rows if r["alpha_factor"] == 1.0 and r["best_config"]]
        top = sorted(nominal_best, key=lambda r: -r["G"])[:args.top]
        sens = []
        for r in top:
            print(f"[sensitivity] d = {r['spacing_mm']:g} mm, f0 = {r['f_GHz']:.2f} GHz, {r['config']}")
            sens += sensitivity(w, r)
        if sens:
            write_csv(out_dir / f"{tag}_sensitivity.csv", sens)
            for r in sens:
                print(f"    d = {r['spacing_mm']:5.1f} f0 = {r['f0_GHz']:.2f}  f/f0 = {r['f_over_f0']:.3f}  "
                      f"G = {r['G_dBi']:6.2f} dBi (dG = {r['dG_dB']:+.2f} dB)")

        check = None
        if mesh_check and w_mm == widths[0]:
            check = run_mesh_check(w, w_mm, rows)
        meta = dict(sweep="A", width_mm=w_mm, length_m=LENGTH, spacings_mm=spacings, freqs_GHz=freqs,
                    alpha_cal=alpha_calibrated(w_mm), alpha_band=list(alpha_band(w_mm)),
                    mesh_rule=MESH_RULE, cell_length_m=CELL_LENGTH, z_fill_mode=DEFAULT_FILL_MODE,
                    sigma_S_per_m=SIGMA_CU, z_gen_ohm=Z_GEN, target="+y, phi polarisation, G",
                    configs={c: f"el{fed + 1} fed" for c, fed in CONFIGS}, n_starts=args.n_starts,
                    top_sensitivity=[dict(spacing_mm=r["spacing_mm"], f_GHz=r["f_GHz"], config=r["config"],
                                          G_dBi=r["G_dBi"]) for r in top],
                    mesh_check=check, timings=timings, git=v5.git_version(),
                    finished=_dt.datetime.now().isoformat(timespec="seconds"),
                    t_total_s=time.perf_counter() - t0)
        (out_dir / f"{tag}_meta.json").write_text(json.dumps(meta, indent=2, default=float),
                                                   encoding="utf-8")
        print(f"wrote {out_dir / tag}.csv, _sensitivity.csv, _meta.json")
    return 0


def run_mesh_check(w, w_mm, rows):
    """Same point with half the cell length along x: change in G, eta, X_opt at alpha_cal."""
    d, f_ghz = MESH_CHECK["spacing_mm"], MESH_CHECK["f_GHz"]
    ref = [r for r in rows if r["spacing_mm"] == d and abs(r["f_GHz"] - f_ghz) < 1e-9
           and r["alpha_factor"] == 1.0]
    if not ref:
        return None
    mesh = make_mesh(w, d * 1e-3, CELL_LENGTH * MESH_CHECK["cell_factor"])
    print(f"[mesh check] d = {d} mm, f0 = {f_ghz} GHz, cell along x / 2: N = {mesh.n_edges}")
    G = rwg_gram_matrix(mesh.edges, mesh.triangles, mesh.points)
    Zf = {}
    for f_blk, Z_blk in v5.z_sweep_blocks(mesh, [f_ghz * 1e9]):
        Zf.update(zip(f_blk, Z_blk))
    fine = [r for r in evaluate_geometry(mesh, Zf, G, w_mm) if r["alpha_factor"] == 1.0]
    out = {}
    for r0 in ref:
        r1 = next(r for r in fine if r["config"] == r0["config"])
        out[r0["config"]] = {k: dict(coarse=r0[k], fine=r1[k]) for k in
                             ("N", "G_dBi", "eta_rad", "X_opt_ohm", "FB_dB", "Zin_re_ohm", "Zin_im_ohm")}
        print(f"    {r0['config']:9s} G {r0['G_dBi']:.3f} -> {r1['G_dBi']:.3f} dBi, eta {r0['eta_rad']:.4f} -> "
              f"{r1['eta_rad']:.4f}, X {r0['X_opt_ohm']:+.1f} -> {r1['X_opt_ohm']:+.1f} Ohm")
    return out


if __name__ == "__main__":
    sys.exit(main())
