"""
Shortening factor of the B3 strips on the RO4003C substrate (step 9, part B).

Why
---
CST V6b (B3 on the LEAT RO4003C, CST library material eps_r = 3.55, tan d = 0.0027,
h = 1.52 mm, copper 17.5 um) turns the main lobe to -y and puts the fed element near an
antiresonance: part of the field runs in the dielectric, which lengthens the strips
electrically. Not a new loss (eta_rad ~ 97.5 %): a detuning, corrected by shortening the
strips by the frequency-shift factor, then re-tuning the loads.

Method
------
Two CST Touchstone files of ONE strip 180 x 4 mm fed at its centre (gap 0.5 mm), without
and on the substrate (fiches_cst/resultats_V6/V6b_ruban_seul/):
    first resonance f_res = first rising zero of X = Im Z_in (negative -> positive),
    cubic-spline interpolation of X between the bracketing samples;
    k = f_res,substrate / f_res,air ;   eps_eff = 1 / k^2 ;
    X_new = 90 mm * k (half-length), rounded to 0.5 mm; box check (187 x 125 x 5 mm).
Constant rows at the ends of a CST export (extrapolation outside the simulated band, as
below 0.95 GHz in the V6b file) are dropped before the search.

Outputs
-------
fiches_cst/resultats_V6/V6_parametres_B3_1.00GHz_substrat.json : B3 with shortened strips
    (same y positions, initial loads 1 pF at -y and 0.5 pF at +y), substrate, the
    measured k, and the CST refinement sweep plan
fiches_cst/resultats_V6/V6b_ruban_seul/substrate_scaling.png : X(f) and R(f) of both files
If the MoM free-space Z_in of the same strip is available
(V6b_ruban_seul/mom_ruban_seul_180x4mm.csv, written by ruban_seul_resonance_mom.ipynb),
its first resonance is compared with CST without substrate (consistency check only).

    python solver/antenna_simulation/euraap_challenge/substrate_scaling.py [--air A.s1p] [--substrate B.s1p]
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from compare_v5 import read_touchstone_1port                           # noqa: E402

CHALLENGE = Path(r"D:\PhD_Fahdilou_Leat\PUBLICATIONS\7-EurAAP Small Antenna Super-Gain Challenge")
V6_DIR = CHALLENGE / "fiches_cst" / "resultats_V6"
STRIP_DIR = V6_DIR / "V6b_ruban_seul"
B3_JSON = V6_DIR / "V6_parametres_B3_1.00GHz.json"
OUT_JSON = V6_DIR / "V6_parametres_B3_1.00GHz_substrat.json"
MOM_CSV_NAME = "mom_ruban_seul_180x4mm.csv"

HALF_LENGTH_MM = 90.0                 # B3: strips of 180 mm, x in [-90, 90] mm
STRIP = dict(length_mm=180.0, width_mm=4.0, gap_mm=0.5, feed="centre (x = 0)")
#: challenge box (notes_comprehension/01_sujet_et_contraintes.md)
BOX_MM = dict(W_x=187.0, L_y=125.0, H_z=5.0)
SUBSTRATE = dict(name="Rogers RO4003C (LEAT)", model="CST library material, constant fit",
                 eps_r=3.55, tan_delta=0.0027, thickness_mm=1.52, copper_thickness_um=17.5,
                 board_mm=dict(x=187.0, y=125.0, note="CST V6b parameters X_max = 187, Y_max = 125"))
SWEEP_STEP_MM = 2.5
C_FRONT_PF = (0.3, 0.5, 0.8)          # +y element (director side), B3: 0.5 pF
C_BACK_PF = (0.8, 1.0, 1.2)           # -y element (reflector side), B3: 1 pF
ROUND_MM = 0.5


# =============================================================================
# Reading and resonance
# =============================================================================
def drop_constant_ends(f, z, min_run=3, rtol=1e-9):
    """
    Drop the constant runs (>= min_run identical Z) at the start and the end of a sweep:
    CST writes the edge value outside its simulated band. Returns (f, z, info).
    """
    f, z = np.asarray(f, float), np.asarray(z, complex)
    same = np.abs(np.diff(z)) <= rtol * np.maximum(np.abs(z[1:]), 1e-30)
    lo = 0
    while lo < len(same) and same[lo]:
        lo += 1
    hi = len(z) - 1
    while hi > 0 and same[hi - 1]:
        hi -= 1
    lo = lo if lo + 1 >= min_run else 0               # lo identical steps = lo + 1 equal points
    hi = hi if (len(z) - hi) >= min_run else len(z) - 1
    info = dict(n_dropped_start=int(lo), n_dropped_end=int(len(z) - 1 - hi),
                f_valid_GHz=[float(f[lo] / 1e9), float(f[hi] / 1e9)])
    return f[lo:hi + 1], z[lo:hi + 1], info


def read_strip_s1p(path, z_ref=None):
    f, z = read_touchstone_1port(path, r_ref=z_ref)
    order = np.argsort(f)
    return drop_constant_ends(f[order], z[order])


def first_resonance(f, z):
    """
    First rising zero of X = Im Z (series resonance): dict(f_res, R_in, dXdf, Q) or None.
    Cubic spline of X (>= 4 points) on the bracketing interval, linear otherwise.
    """
    f, z = np.asarray(f, float), np.asarray(z, complex)
    X = z.imag
    k = np.flatnonzero((X[:-1] < 0) & (X[1:] >= 0))
    if not k.size:
        return None
    k = int(k[0])
    if len(f) >= 4:
        sx, sr = CubicSpline(f, X), CubicSpline(f, z.real)
        f_res = brentq(lambda x: float(sx(x)), f[k], f[k + 1], xtol=1e-6)
        r_in, dxdf = float(sr(f_res)), float(sx(f_res, 1))
    else:
        t = -X[k] / (X[k + 1] - X[k])
        f_res = f[k] + t * (f[k + 1] - f[k])
        r_in = z.real[k] + t * (z.real[k + 1] - z.real[k])
        dxdf = (X[k + 1] - X[k]) / (f[k + 1] - f[k])
    return dict(f_res_Hz=float(f_res), R_in_ohm=r_in, dXdf_ohm_per_Hz=dxdf,
                Q=float(f_res * dxdf / (2 * r_in)), bracket_GHz=[float(f[k] / 1e9), float(f[k + 1] / 1e9)])


def scaling(f_air, f_sub, half_length_mm=HALF_LENGTH_MM, step_mm=ROUND_MM):
    k = f_sub / f_air
    x_exact = half_length_mm * k
    x_new = round(x_exact / step_mm) * step_mm
    return dict(k=k, eps_eff=1.0 / k ** 2, X_exact_mm=x_exact, X_new_mm=x_new,
                rounding_mm=step_mm, X_new_over_X=x_new / half_length_mm)


def box_check(x_half_mm, y_edges_mm, height_mm):
    dx = 2 * x_half_mm
    dy = max(e[1] for e in y_edges_mm) - min(e[0] for e in y_edges_mm)
    ok = dict(x=dx <= BOX_MM["W_x"] + 1e-9, y=dy <= BOX_MM["L_y"] + 1e-9, z=height_mm <= BOX_MM["H_z"] + 1e-9)
    return dict(extent_mm=dict(x=dx, y=dy, z=height_mm), box_mm=BOX_MM, ok=ok, all_ok=all(ok.values()))


# =============================================================================
# Sweep plan
# =============================================================================
def sweep_plan(x_new_mm, step_mm=SWEEP_STEP_MM, c_front=C_FRONT_PF, c_back=C_BACK_PF):
    xs = [x_new_mm - step_mm, x_new_mm, x_new_mm + step_mm]
    full = [dict(X_mm=x, C_front_pF=cf, C_back_pF=cb) for x in xs for cf in c_front for cb in c_back]
    stage1 = [dict(run=i + 1, X_mm=x, C_front_pF=0.5, C_back_pF=1.0) for i, x in enumerate(xs)]
    stage2 = []
    for cf in c_front:
        for cb in c_back:
            if (cf, cb) != (0.5, 1.0):
                stage2.append(dict(X_mm="X_best (stage 1)", C_front_pF=cf, C_back_pF=cb))
    cross = [r for r in stage2 if r["C_front_pF"] == 0.5 or r["C_back_pF"] == 1.0]
    rest = [r for r in stage2 if r not in cross]
    for i, r in enumerate(cross + rest):
        r["run"] = len(stage1) + i + 1
    return dict(
        grid=dict(X_mm=xs, C_front_pF=list(c_front), C_back_pF=list(c_back), n_cases=len(full),
                  note="C_front: element at +y (target side, B3 0.5 pF); C_back: element at -y (B3 1 pF)"),
        reduced_order=dict(
            stage1=dict(runs=stage1, choose="X_best: G(+y) highest with F/B(+y/-y) > 0 dB (main lobe back "
                                              "to +y) and Z_in of the fed strip closest to V6a "
                                              "(26.5 + j326 Ohm at 1 GHz)"),
            stage2a_cross=dict(runs=cross, note="vary one capacitor at a time around 0.5 / 1 pF at X_best"),
            stage2b_corners=dict(runs=rest, note="only if stage 2a does not give a clear optimum"),
            n_runs=dict(stage1=len(stage1), stage2a=len(cross), stage2b=len(rest),
                        total_max=len(stage1) + len(stage2))),
        then="match_from_cst.py on the s1p of the retained case (network recomputed on the CST impedance)")


# =============================================================================
# MoM comparison (optional file)
# =============================================================================
def read_mom_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    f = np.array([float(r["f_GHz"]) for r in rows]) * 1e9
    z = np.array([float(r["Re_Zin_ohm"]) + 1j * float(r["Im_Zin_ohm"]) for r in rows])
    return f, z


# =============================================================================
# Main
# =============================================================================
def find_inputs(folder):
    """(air, substrate) s1p in folder, from their names (air|sans|vide|free vs sub|ro4003|avec)."""
    files = sorted(Path(folder).glob("*.s1p"))
    air = [p for p in files if re.search(r"air|sans|vide|free|nosub", p.name, re.I)]
    sub = [p for p in files if re.search(r"substrat|subst|ro4003|avec|sub\b", p.name, re.I) and p not in air]
    if len(air) != 1 or len(sub) != 1:
        raise FileNotFoundError(
            f"{folder}: cannot identify the two strip files from their names "
            f"({[p.name for p in files]}); pass --air and --substrate.")
    return air[0], sub[0]


def _git_info():
    run = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    try:
        return dict(commit=run("rev-parse", "HEAD"), branch=run("rev-parse", "--abbrev-ref", "HEAD"),
                    dirty=bool(run("status", "--porcelain")))
    except Exception:
        return None


def substrate_design(b3, x_new_mm, sc, res_air, res_sub, plan, inputs, mom=None):
    """B3 parameters with shortened strips (same y), substrate and sweep plan."""
    d = copy.deepcopy(b3)
    d["campaign"] = "B3_1.00GHz_substrat"
    d["generated"] = dt.datetime.now().isoformat(timespec="seconds")
    d["git"] = _git_info()
    d["derived_from"] = B3_JSON.name
    d["material"] = dict(d.get("material", {}), thickness_um=SUBSTRATE["copper_thickness_um"],
                         note="copper as CST V6b (17.5 um)")
    d["substrate"] = SUBSTRATE
    for s in d["strips"]:
        s["x_extent_mm"] = dict(value=[-x_new_mm, x_new_mm], reference="EDGES (ends of the strip)")
        s["length_mm"] = 2 * x_new_mm
        if s.get("component"):
            s["component"]["note_step9"] = "initial value (B3); see sweep_plan_cst"
    d["shortening"] = dict(
        method="first rising zero of Im Z_in of one 180 x 4 mm strip fed at its centre (gap 0.5 mm), "
               "CST without / on the substrate; k = f_sub / f_air, eps_eff = 1/k^2, X_new = 90 mm k "
               f"rounded to {ROUND_MM} mm",
        strip=STRIP, inputs={k: str(v) for k, v in inputs.items()},
        f_res_air_GHz=res_air["f_res_Hz"] / 1e9, R_in_air_ohm=res_air["R_in_ohm"],
        f_res_substrate_GHz=res_sub["f_res_Hz"] / 1e9, R_in_substrate_ohm=res_sub["R_in_ohm"],
        **sc, mom_free_space_check=mom)
    ys = [s["y_edges_mm"]["value"] for s in d["strips"]]
    d["box_check"] = box_check(x_new_mm, ys, SUBSTRATE["thickness_mm"] + SUBSTRATE["copper_thickness_um"] / 1e3)
    d["sweep_plan_cst"] = plan
    d["expected"] = dict(note="no MoM prediction on the substrate (free-space solver): the CST sweep decides")
    for k in ("esr_esl_sensitivity", "mesh_check", "notes"):
        d.pop(k, None)
    return d


def run(air, substrate, out_json=OUT_JSON, b3_json=B3_JSON, z_ref=None, figure=True, mom_csv=None, log=print):
    f_a, z_a, info_a = read_strip_s1p(air, z_ref)
    f_s, z_s, info_s = read_strip_s1p(substrate, z_ref)
    res_a, res_s = first_resonance(f_a, z_a), first_resonance(f_s, z_s)
    for name, res, info in (("air", res_a, info_a), ("substrate", res_s, info_s)):
        if res is None:
            raise ValueError(f"{name}: no rising zero of Im Z_in in {info['f_valid_GHz']} GHz; extend the CST "
                             f"sweep (180 mm strip: first resonance ~0.75-0.8 GHz in air, lower on the substrate).")
        log(f"  {name:<9}: valid {info['f_valid_GHz'][0]:.4f} ... {info['f_valid_GHz'][1]:.4f} GHz "
            f"(dropped {info['n_dropped_start']} + {info['n_dropped_end']} constant rows), first resonance "
            f"{res['f_res_Hz'] / 1e9:.4f} GHz, R_in = {res['R_in_ohm']:.1f} Ohm, Q = {res['Q']:.2f}")
    sc = scaling(res_a["f_res_Hz"], res_s["f_res_Hz"])
    log(f"  k = f_sub / f_air = {sc['k']:.4f}, eps_eff = {sc['eps_eff']:.3f}; X_new = 90 x k = "
        f"{sc['X_exact_mm']:.2f} mm -> {sc['X_new_mm']:.1f} mm (strip {2 * sc['X_new_mm']:.1f} mm)")
    mom = None
    mom_csv = Path(mom_csv) if mom_csv else Path(air).parent / MOM_CSV_NAME
    if mom_csv.is_file():
        f_m, z_m = read_mom_csv(mom_csv)
        r_m = first_resonance(f_m, z_m)
        if r_m:
            mom = dict(file=mom_csv.name, f_res_GHz=r_m["f_res_Hz"] / 1e9, R_in_ohm=r_m["R_in_ohm"],
                       diff_vs_cst_air_pct=100 * (r_m["f_res_Hz"] / res_a["f_res_Hz"] - 1))
            log(f"  MoM free space ({mom_csv.name}): f_res = {mom['f_res_GHz']:.4f} GHz, R_in = {mom['R_in_ohm']:.1f}"
                f" Ohm ({mom['diff_vs_cst_air_pct']:+.2f} % vs CST air)")
    plan = sweep_plan(sc["X_new_mm"])
    b3 = json.loads(Path(b3_json).read_text(encoding="utf-8"))
    doc = substrate_design(b3, sc["X_new_mm"], sc, res_a, res_s, plan,
                           dict(air=Path(air), substrate=Path(substrate)), mom)
    bc = doc["box_check"]
    log(f"  box: {bc['extent_mm']['x']:.1f} x {bc['extent_mm']['y']:.1f} x {bc['extent_mm']['z']:.3f} mm "
        f"<= 187 x 125 x 5 mm: {'OK' if bc['all_ok'] else 'NOT OK ' + str(bc['ok'])}")
    print_plan(plan, log)
    if out_json is not None:
        Path(out_json).write_text(json.dumps(doc, indent=2), encoding="utf-8")
        log(f"  wrote {out_json}")
    if figure:
        _figure(Path(air).parent / "substrate_scaling.png", (f_a, z_a, res_a), (f_s, z_s, res_s), sc)
    return doc


def print_plan(plan, log=print):
    g = plan["grid"]
    log(f"  CST refinement grid: X in {g['X_mm']} mm x C_front {g['C_front_pF']} pF x C_back {g['C_back_pF']} pF "
        f"= {g['n_cases']} cases")
    ro = plan["reduced_order"]
    log("  reduced order:")
    for r in ro["stage1"]["runs"]:
        log(f"    run {r['run']:>2}  stage 1   X = {r['X_mm']:.1f} mm, C_front = {r['C_front_pF']} pF, "
            f"C_back = {r['C_back_pF']} pF")
    log(f"             -> {ro['stage1']['choose']}")
    for key, lab in (("stage2a_cross", "stage 2a"), ("stage2b_corners", "stage 2b")):
        for r in ro[key]["runs"]:
            log(f"    run {r['run']:>2}  {lab}  X = X_best, C_front = {r['C_front_pF']} pF, C_back = {r['C_back_pF']} pF")
    n = ro["n_runs"]
    log(f"    total: {n['stage1']} + {n['stage2a']} (+ {n['stage2b']} if needed) runs instead of {g['n_cases']}")


def _figure(path, air, sub, sc):
    import matplotlib
    if "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for (f, z, r), lab, c in ((air, "CST, air", "#2a78d6"), (sub, "CST, RO4003C 1.52 mm", "#eb6834")):
        axes[0].plot(f / 1e9, z.imag, color=c, label=lab)
        axes[1].plot(f / 1e9, z.real, color=c, label=lab)
        axes[0].axvline(r["f_res_Hz"] / 1e9, color=c, ls=":", lw=1)
    axes[0].axhline(0, color="#898781", lw=0.8)
    axes[0].set_ylabel("Im Z_in (Ohm)")
    axes[1].set_ylabel("Re Z_in (Ohm)")
    for ax in axes:
        ax.set_xlabel("f (GHz)")
        ax.grid(True, color="#e1e0d9")
    axes[0].legend(frameon=False)
    fig.suptitle(f"Strip 180 x 4 mm fed at the centre: k = {sc['k']:.4f}, eps_eff = {sc['eps_eff']:.3f}, "
                 f"X_new = {sc['X_new_mm']:.1f} mm", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--air", help="s1p of the strip without substrate")
    ap.add_argument("--substrate", help="s1p of the strip on the RO4003C")
    ap.add_argument("--dir", default=str(STRIP_DIR), help="folder searched when --air / --substrate are omitted")
    ap.add_argument("--z-ref", type=float, help="reference impedance overriding the files' R value")
    ap.add_argument("--out", default=str(OUT_JSON))
    ap.add_argument("--mom-csv", help=f"MoM free-space Z_in of the strip (default <dir>/{MOM_CSV_NAME})")
    ap.add_argument("--no-figure", action="store_true")
    a = ap.parse_args(argv)
    if a.air and a.substrate:
        air, sub = Path(a.air), Path(a.substrate)
    else:
        air, sub = find_inputs(a.dir)
    print(f"[substrate scaling] air: {air.name} | substrate: {sub.name}")
    run(air, sub, out_json=a.out, z_ref=a.z_ref, figure=not a.no_figure, mom_csv=a.mom_csv)


if __name__ == "__main__":
    main()
