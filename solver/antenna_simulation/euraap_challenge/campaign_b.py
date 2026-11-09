"""
Step 6 campaigns: 3 elements on a grid (B3) and 4 elements by Bayesian
optimisation (B4), real components, robust objective.

Common to both (solver.optim.design_eval):
  * K parallel copper strips, L = 180 mm (fixed: a centre load tunes a strip as a
    length change would, and a long strip has a larger radiation resistance, i.e.
    smaller relative loss -- to be re-checked on the final design), w = 4 mm,
    z = 0, along x, stacked along y; loss model alpha_cal(w) (loss_model);
  * one Z per geometry and frequency (fill_mode "full_sym"); EVERY choice of fed
    element is evaluated on it (only the role of the ports changes);
  * continuous loads -> Johanson S603DS kit (snap_to_kit) -> robust objective
    mean G over {alpha_cal x 0.85, 1, 1.15} x {tolerance -, 0, +} with a soft
    penalty when Re Zin < r_in_min (default 10 Ohm).

B3 : d1, d2 (centre to centre) = 15 ... 105 mm by 10 mm, edge-to-edge extent
     <= 120 mm, at f0 = 0.90 and 1.00 GHz (one Z sweep per geometry). Saved after
     every geometry; a restart skips the geometries already done.
B4 : 4 elements, BO (solver.optim.outer) over (s, u1, u2): extent s and simplex
     coordinates of the 3 gaps (d_i >= 15 mm), at the f0 that wins B3;
     n_init = 12, n_iter = 40, early stop if max EI < 0.01 dB for 8 iterations.
     The BO history is the resume point.

Post-checks on the TOP 3 of each campaign (and of each f0 for B3): G, Zin, ...
at f0 x (1 +- 0.5, 1, 1.5, 2 %) with the parts frozen (one Z sweep of 8
frequencies), input Q from the +-0.5 % points, and a mesh check (cell along x
halved). The best design of each campaign is exported for CST:
fiches_cst/resultats_V6/V6_parametres_<campaign>.json + expected MoM results
(CSV: Zin(f) over f0 +- 2 %, G, D, eta, F/B, G_r; cuts theta = 90 deg and
phi = 90 deg at f0).

    python solver/antenna_simulation/euraap_challenge/campaign_b.py b3 --dry-run
    python solver/antenna_simulation/euraap_challenge/campaign_b.py b4 --f0 0.9 --dry-run
    # overnight, unattended (B4 at the B3 winner):
    python solver/antenna_simulation/euraap_challenge/campaign_b.py b3 --yes && \
    python solver/antenna_simulation/euraap_challenge/campaign_b.py b4 --f0 auto --yes
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
from solver.optim.components import (                                           # noqa: E402
    IDEAL, JOHANSON_S603DS, IdealReactance, PROVISIONAL_ESL_GRID, PROVISIONAL_ESR_GRID,
    series_impedance,
)
from solver.optim.design_eval import (                                          # noqa: E402
    BACK, DIRECTION, POLARIZATION, Z_GEN, DesignContext, evaluate_design,
)
from solver.optim.geometries import parallel_strip_array                        # noqa: E402
from solver.optim.inner import optimize_loads                                   # noqa: E402
from solver.optim.loss_model import MESH_RULE, alpha_calibrated                 # noqa: E402
from solver.optim.outer import BayesianOptimizer, extent_bounds, spacings_from_params  # noqa: E402
from solver.utils.impmet import DEFAULT_FILL_MODE, rwg_gram_matrix              # noqa: E402

# =============================================================================
# Parameters
# =============================================================================
BASE = Path(r"D:\PhD_Fahdilou_Leat\PUBLICATIONS\7-EurAAP Small Antenna Super-Gain Challenge")
OUT_DIR = BASE / "exploration"
CST_DIR = BASE / "fiches_cst" / "resultats_V6"
LENGTH = 0.180
WIDTH = 4e-3
CELL_LENGTH = 4.5e-3
S_MAX = 0.120                      # edge-to-edge extent along y
D_MIN = 15e-3                      # minimum centre-to-centre gap
R_IN_MIN = 10.0
N_STARTS = 32
TOP = 3
CHECK_OFFSETS = (-0.02, -0.015, -0.01, -0.005, 0.005, 0.01, 0.015, 0.02)

B3_GRID_MM = tuple(range(15, 106, 10))
B3_FREQS_GHZ = (0.90, 1.00)
B4 = dict(n_init=12, n_iter=40, ei_stop=0.01, ei_patience=8, seed=0)

# Estimate model (development laptop, 2026-11-05): inner optimisation per loaded
# port and per fed position (32 starts), LU and bound of one geometry.
T_INNER_PER_LOAD_S = 0.75
T_SNAP_S = 0.3


# =============================================================================
# Geometry evaluation
# =============================================================================
def build_mesh(spacings, cell_length=None):
    return parallel_strip_array([LENGTH] * (len(spacings) + 1), WIDTH, list(spacings),
                                cell_length=CELL_LENGTH if cell_length is None else cell_length,
                                **MESH_RULE)


def assemble(mesh, freqs_hz, log=print):
    Zf = {}
    for f_blk, Z_blk in v5.z_sweep_blocks(mesh, list(freqs_hz), log=log):
        Zf.update(zip(f_blk, Z_blk))
    return Zf


def geometry_fields(mesh):
    p = mesh.params
    return dict(n_elements=len(p["y_centers"]), spacings_mm=[round(d * 1e3, 6) for d in p["spacings"]],
                extent_mm=p["extent_y"] * 1e3, y_centers_mm=[c * 1e3 for c in p["y_centers"]], N=mesh.n_edges)


def evaluate_geometry(spacings, freqs_hz, r_in_min, n_starts, log=print):
    mesh = build_mesh(spacings)
    t0 = time.perf_counter()
    gram = rwg_gram_matrix(mesh.edges, mesh.triangles, mesh.points)
    Zf = assemble(mesh, freqs_hz, log)
    t_z = time.perf_counter() - t0
    out = []
    for f, Z in Zf.items():
        ctx = DesignContext(mesh, Z, gram, f, WIDTH * 1e3)
        recs, best = evaluate_design(ctx, r_in_min=r_in_min, n_starts=n_starts)
        for k, r in enumerate(recs):
            r.update(geometry_fields(mesh), best_fed=(k == best), t_z_s=t_z)
        out += recs
        b = recs[best]
        log(f"    {geometry_fields(mesh)['spacings_mm']} mm  f0 = {f / 1e9:.2f} GHz  best fed {b['fed']}: "
            f"G kit {b['G_kit_dBi']:.2f} (cont {b['G_continuous_dBi']:.2f}), robust {b['G_robust_mean_dBi']:.2f} "
            f"/ worst {b['G_robust_worst_dBi']:.2f} dBi, eta {b['eta']:.3f}, Zin {b['Zin_re_ohm']:.1f}"
            f"{b['Zin_im_ohm']:+.1f}j, G/Gmax {b['G_over_G_max']:.3f}; "
            + ", ".join(f"{ld['port']} {ld['label']}" for ld in b["loads"])
            + (f"  [{'; '.join(b['notes'])}]" if b["notes"] else ""))
    return out


# =============================================================================
# Files
# =============================================================================
def append_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, default=float) + "\n")


def read_jsonl(path):
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


SUMMARY_KEYS = ("n_elements", "spacings_mm", "extent_mm", "f0_GHz", "fed", "best_fed", "objective_dB",
                "G_continuous_dBi", "G_kit_dBi", "G_robust_mean_dBi", "G_robust_worst_dBi", "G_r_dBi",
                "D_dBi", "eta", "Zin_re_ohm", "Zin_im_ohm", "FB_dB", "G_max_dBi", "G_over_G_max",
                "penalty", "esr_esl_G_min_dBi", "N")


def write_summary(path, records):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([*SUMMARY_KEYS, "loads", "notes"])
        for r in records:
            w.writerow([("/".join(f"{v:g}" for v in r[k]) if isinstance(r[k], list) else
                         (f"{r[k]:.6g}" if isinstance(r[k], float) else r[k])) for k in SUMMARY_KEYS]
                       + ["; ".join(f"{ld['port']}={ld['label']} (X*={ld['x_continuous_ohm']:.1f})"
                                    for ld in r["loads"]),
                          " | ".join(r["notes"])])


# =============================================================================
# Post-checks and CST export
# =============================================================================
_KIT = {c.label: c for c in JOHANSON_S603DS}


def frozen_parts(record):
    parts = []
    for ld in record["loads"]:
        if ld["kind"] == "C":
            parts.append(_KIT[ld["label"]])
        else:
            parts.append(IdealReactance(ld["x_ohm"], record["f0_GHz"] * 1e9, "kept from the optimisation"))
    return parts


def _solve_frozen(ctx, record, parts, f):
    z = np.array([series_impedance(p, f, IDEAL) for p in parts])
    return ctx.solve(record["fed_index"], z, 1.0)


def post_checks(record, log=print):
    """Frequency behaviour with frozen parts, input Q, mesh check; cuts at f0."""
    spacings = [d * 1e-3 for d in record["spacings_mm"]]
    f0 = record["f0_GHz"] * 1e9
    parts = frozen_parts(record)
    mesh = build_mesh(spacings)
    gram = rwg_gram_matrix(mesh.edges, mesh.triangles, mesh.points)
    freqs = sorted([f0] + [f0 * (1 + o) for o in CHECK_OFFSETS])
    Zf = assemble(mesh, freqs, log)
    rows, ctx0, res0 = [], None, None
    for f in freqs:
        ctx = DesignContext(mesh, Zf[f], gram, f, WIDTH * 1e3, alpha_factors=(1.0,), compute_bound=False)
        r = _solve_frozen(ctx, record, parts, f)
        rows.append(dict(f_GHz=f / 1e9, offset_pct=100 * (f / f0 - 1), Zin_re_ohm=r["zin"].real,
                         Zin_im_ohm=r["zin"].imag, G_dBi=10 * np.log10(r["G"]), G_r_dBi=10 * np.log10(r["G_r"]),
                         D_dBi=10 * np.log10(r["D"]), eta=r["eta"], FB_dB=ctx.front_back_db(r)))
        if abs(f - f0) < 1e-6 * f0:
            ctx0, res0 = ctx, r
    by = {round(r["offset_pct"], 3): r for r in rows}
    lo, hi = by[-0.5], by[0.5]
    dX_dw = (hi["Zin_im_ohm"] - lo["Zin_im_ohm"]) / (2 * np.pi * f0 * 0.01)
    q_in = 2 * np.pi * f0 / (2 * by[0.0]["Zin_re_ohm"]) * abs(dX_dw)

    # cuts at f0 (gain, phi and theta components)
    I = res0["model"].edge_currents(res0["sol"].v)
    p_in = res0["sol"].p_in_total
    phi = np.radians(np.arange(0, 360, 1.0))
    theta = np.radians(np.arange(0, 181, 1.0))
    ut, up = ctx0.op.intensity(I, np.full(phi.shape, np.pi / 2), phi)
    cut_xy = [dict(theta_deg=90.0, phi_deg=float(np.degrees(a)), G_phi_dBi=10 * np.log10(max(b, 1e-30) * 4 * np.pi / p_in),
                   G_theta_dBi=10 * np.log10(max(c, 1e-30) * 4 * np.pi / p_in)) for a, b, c in zip(phi, up, ut)]
    ut, up = ctx0.op.intensity(I, theta, np.full(theta.shape, np.pi / 2))
    cut_yz = [dict(theta_deg=float(np.degrees(a)), phi_deg=90.0, G_phi_dBi=10 * np.log10(max(b, 1e-30) * 4 * np.pi / p_in),
                   G_theta_dBi=10 * np.log10(max(c, 1e-30) * 4 * np.pi / p_in)) for a, b, c in zip(theta, up, ut)]

    # mesh check: same parts on a mesh with half the cell length along x
    fine = build_mesh(spacings, CELL_LENGTH / 2)
    gram_f = rwg_gram_matrix(fine.edges, fine.triangles, fine.points)
    Zfine = assemble(fine, [f0], log)[f0]
    ctx_f = DesignContext(fine, Zfine, gram_f, f0, WIDTH * 1e3, alpha_factors=(1.0,), compute_bound=False)
    r_f = _solve_frozen(ctx_f, record, parts, f0)
    cont_f = optimize_loads(ctx_f.models[1.0], DIRECTION, POLARIZATION, "G", n_starts=N_STARTS,
                            operator=ctx_f.op, z_gen=Z_GEN, driven_port=record["fed_index"], compute_bound=False)
    mesh_check = dict(N_coarse=mesh.n_edges, N_fine=fine.n_edges,
                      G_kit_dBi=[10 * np.log10(res0["G"]), 10 * np.log10(r_f["G"])],
                      Zin_ohm=[[res0["zin"].real, res0["zin"].imag], [r_f["zin"].real, r_f["zin"].imag]],
                      X_continuous_ohm=[[ld["x_continuous_ohm"] for ld in record["loads"]], cont_f.x_opt.tolist()])
    log(f"    checks {record['spacings_mm']} mm, f0 {record['f0_GHz']:.2f} GHz: Q_in = {q_in:.1f}; "
        + ", ".join(f"{r['offset_pct']:+.1f}%: {r['G_dBi']:.2f}" for r in rows)
        + f" dBi; mesh check G {mesh_check['G_kit_dBi'][0]:.3f} -> {mesh_check['G_kit_dBi'][1]:.3f} dBi")
    return dict(freq_rows=rows, Q_in=q_in, cut_xy=cut_xy, cut_yz=cut_yz, mesh_check=mesh_check)


def _write_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.8g}" if isinstance(v, float) else v) for k, v in r.items()})


def export_cst(record, checks, tag, log=print):
    """V6_parametres_<tag>.json + expected MoM results (see fiches_cst/V6_fiche_reseau.md)."""
    CST_DIR.mkdir(parents=True, exist_ok=True)
    strips = []
    loads = {ld["port"]: ld for ld in record["loads"]}
    for k, yc in enumerate(record["y_centers_mm"]):
        name = f"el{k + 1}"
        strips.append(dict(
            name=name, role="fed" if name == record["fed"] else "loaded",
            y_center_mm=dict(value=yc, reference="CENTRE of the strip"),
            y_edges_mm=dict(value=[yc - WIDTH * 1e3 / 2, yc + WIDTH * 1e3 / 2], reference="EDGES (bords)"),
            x_extent_mm=dict(value=[-LENGTH * 1e3 / 2, LENGTH * 1e3 / 2], reference="EDGES (ends of the strip)"),
            width_mm=WIDTH * 1e3, length_mm=LENGTH * 1e3,
            gap="at x = 0 (strip centre), MoM delta gap; CST: 0.5 mm gap",
            component=None if name == record["fed"] else dict(
                kit="Johanson S603DS" if loads[name]["kind"] == "C" else None,
                label=loads[name]["label"], value_F=loads[name].get("value_F"),
                tolerance_abs_F=loads[name].get("tol_abs_F"), tolerance_rel=loads[name].get("tol_rel"),
                x_continuous_ohm=loads[name]["x_continuous_ohm"],
                esr_ohm=0.0, esl_nH=0.0,
                parasitics_note="MoM nominal ESR = ESL = 0 (PROVISIONAL); sensitivity grid "
                                f"ESR {list(PROVISIONAL_ESR_GRID)} Ohm x ESL "
                                f"{[v * 1e9 for v in PROVISIONAL_ESL_GRID]} nH in the results")))
    doc = dict(
        campaign=tag, generated=_dt.datetime.now().isoformat(timespec="seconds"), git=v5.git_version(),
        frame="strips along x, stacked along y, plane z = 0; target +y (theta = 90, phi = 90), phi polarisation",
        f0_GHz=record["f0_GHz"], material=dict(sigma_S_per_m=5.8e7, thickness_um=35, model="sheet + alpha_cal Zs",
                                                alpha_cal=record["alpha_cal"]),
        spacings_mm=dict(value=record["spacings_mm"], reference="CENTRE to CENTRE"),
        extent_mm=dict(value=record["extent_mm"], reference="EDGE to EDGE along y"),
        fed_port=dict(strip=record["fed"], generator_ohm=Z_GEN), strips=strips,
        expected=dict(G_kit_dBi=record["G_kit_dBi"], G_robust_mean_dBi=record["G_robust_mean_dBi"],
                      G_robust_worst_dBi=record["G_robust_worst_dBi"], G_r_50ohm_dBi=record["G_r_dBi"],
                      D_dBi=record["D_dBi"], eta=record["eta"],
                      Zin_ohm=[record["Zin_re_ohm"], record["Zin_im_ohm"]], FB_dB=record["FB_dB"],
                      G_max_dBi=record["G_max_dBi"], Q_in=checks["Q_in"]),
        esr_esl_sensitivity=record["esr_esl"], notes=record["notes"], mesh_check=checks["mesh_check"])
    path = CST_DIR / f"V6_parametres_{tag}.json"
    path.write_text(json.dumps(doc, indent=2, default=float), encoding="utf-8")
    _write_rows(CST_DIR / f"V6_expected_{tag}_freq.csv", checks["freq_rows"])
    _write_rows(CST_DIR / f"V6_expected_{tag}_cut_theta90_xy.csv", checks["cut_xy"])
    _write_rows(CST_DIR / f"V6_expected_{tag}_cut_phi90_yz.csv", checks["cut_yz"])
    log(f"  exported {path} (+ freq / cut CSVs)")


def top_records(records, n=TOP, f_ghz=None):
    best = [r for r in records if r["best_fed"] and (f_ghz is None or abs(r["f0_GHz"] - f_ghz) < 1e-9)]
    seen, out = set(), []
    for r in sorted(best, key=lambda r: -r["objective"]):
        key = (tuple(r["spacings_mm"]), r["f0_GHz"])
        if key not in seen:
            seen.add(key)
            out.append(r)
        if len(out) == n:
            break
    return out


def run_checks(records, tag_prefix, f_list, out_dir, log=print):
    checks_all = []
    for f in f_list:
        tops = top_records(records, TOP, f)
        for rank, r in enumerate(tops):
            log(f"[checks] {tag_prefix} f0 = {f:.2f} GHz, rank {rank + 1}: {r['spacings_mm']} mm, fed {r['fed']}")
            c = post_checks(r, log)
            checks_all.append(dict(rank=rank + 1, f0_GHz=f, spacings_mm=r["spacings_mm"], fed=r["fed"],
                                   Q_in=c["Q_in"], freq_rows=c["freq_rows"], mesh_check=c["mesh_check"]))
            if rank == 0:
                tag = tag_prefix if len(f_list) == 1 else f"{tag_prefix}_{f:.2f}GHz"
                export_cst(r, c, tag, log)
    (out_dir / f"{tag_prefix}_checks.json").write_text(json.dumps(checks_all, indent=1, default=float),
                                                         encoding="utf-8")


# =============================================================================
# Plans (dry-run)
# =============================================================================
def b3_geometries():
    out = []
    for d1 in B3_GRID_MM:
        for d2 in B3_GRID_MM:
            if d1 + d2 + WIDTH * 1e3 <= S_MAX * 1e3 + 1e-9:
                out.append((d1, d2))
    return out


def _t_geometry(N, K, n_freq, gpu):
    lu = v5.LU_REF["s"] * (N / v5.LU_REF["N"]) ** 3
    per_f = 3 * lu + 4 * lu + K * ((K - 1) * T_INNER_PER_LOAD_S + T_SNAP_S)   # 3 LUs + bound (~4 LU)
    return v5._z_seconds(N, n_freq, gpu) + n_freq * per_f


def _t_checks(N, gpu):
    return v5._z_seconds(N, 1 + len(CHECK_OFFSETS), gpu) + v5._z_seconds(2 * N, 1, gpu) + 10


def print_plan(title, items, total):
    print(title)
    print(f"{'item':44s} {'N':>6s} {'fills':>6s} {'estimate':>9s}")
    for name, N, nf, t in items:
        print(f"{name:44s} {N:6d} {nf:6d} {v5._fmt_duration(t):>9s}")
    print(f"{sum(nf for _, _, nf, _ in items)} Z fills in total; estimated {v5._fmt_duration(total)} "
          f"(GPU={v5._gpu_available()}, fill_mode {DEFAULT_FILL_MODE}).")


# =============================================================================
# Campaigns
# =============================================================================
def run_b3(args):
    freqs = [float(f) for f in args.freqs.split(",")]
    geoms = b3_geometries()
    out_dir = Path(args.out)
    jl = out_dir / "B3_designs.jsonl"
    done = {(tuple(r["spacings_mm"]), round(r["f0_GHz"], 6)) for r in read_jsonl(jl)}
    todo = [g for g in geoms if not all((tuple(float(x) for x in g), round(f, 6)) in done for f in freqs)]
    gpu = v5._gpu_available()
    N = build_mesh([g * 1e-3 for g in geoms[0]]).n_edges
    t_geo = _t_geometry(N, 3, len(freqs), gpu)
    t_chk = _t_checks(N, gpu)
    items = [(f"B3 geometries to run ({len(todo)}/{len(geoms)})", N, len(todo) * len(freqs), len(todo) * t_geo),
             (f"checks: top {TOP} x {len(freqs)} f0 (8 freq + mesh x2)", N, TOP * len(freqs) * 10,
              TOP * len(freqs) * t_chk)]
    print_plan(f"B3 -> {out_dir}  (3 elements, L = {LENGTH * 1e3:g} mm, w = {WIDTH * 1e3:g} mm, f0 = {freqs} GHz, "
               f"alpha_cal = {alpha_calibrated(WIDTH * 1e3):.3f}, r_in_min = {args.r_in_min} Ohm)",
               items, sum(t for *_, t in items))
    if args.dry_run:
        return 0
    if not args.yes and input("Start B3? [y/N] ").strip().lower() not in ("y", "yes", "o", "oui"):
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, g in enumerate(todo, 1):
        print(f"[B3 {k}/{len(todo)}] d1 = {g[0]} mm, d2 = {g[1]} mm")
        recs = evaluate_geometry([g[0] * 1e-3, g[1] * 1e-3], [f * 1e9 for f in freqs], args.r_in_min, args.n_starts)
        append_jsonl(jl, recs)
        write_summary(out_dir / "B3_summary.csv", read_jsonl(jl))
    records = read_jsonl(jl)
    if not args.no_checks:
        run_checks(records, "B3", freqs, out_dir)
    return 0


def b3_winner_f0(out_dir):
    """f0 (GHz) of the best robust B3 design, or None without B3 results."""
    best = [r for r in read_jsonl(Path(out_dir) / "B3_designs.jsonl") if r["best_fed"]]
    return max(best, key=lambda r: r["objective"])["f0_GHz"] if best else None


def run_b4(args):
    out_dir = Path(args.out)
    if args.f0 == "auto":
        f_auto = b3_winner_f0(out_dir)
        if f_auto is None:
            if not args.dry_run:
                raise SystemExit("--f0 auto needs B3 results (B3_designs.jsonl): run b3 first or give --f0.")
            print("(--f0 auto: no B3 results yet; the estimate below uses 0.90 GHz)")
            f_auto = 0.90
        else:
            print(f"--f0 auto: B3 winner is {f_auto:.2f} GHz")
        args.f0 = f_auto
    f0 = float(args.f0) * 1e9
    lo, hi = extent_bounds(4, WIDTH, D_MIN, S_MAX)
    gpu = v5._gpu_available()
    N = build_mesh(spacings_from_params(0.5 * (lo + hi), [0.5, 0.5], 4, WIDTH, D_MIN)).n_edges
    n_total = B4["n_init"] + args.n_iter
    hist = out_dir / "B4_bo_history"
    n_done = len(json.loads(hist.with_suffix(".json").read_text())["records"]) if hist.with_suffix(".json").is_file() else 0
    t_eval = _t_geometry(N, 4, 1, gpu)
    items = [(f"B4 BO evaluations ({n_done} done, up to {n_total})", N, n_total - n_done, (n_total - n_done) * t_eval),
             (f"checks: top {TOP} (8 freq + mesh x2)", N, TOP * 10, TOP * _t_checks(N, gpu))]
    print_plan(f"B4 -> {out_dir}  (4 elements, BO over s in [{lo * 1e3:.0f}, {hi * 1e3:.0f}] mm + 2 simplex "
               f"coordinates, d_min = {D_MIN * 1e3:g} mm, f0 = {f0 / 1e9:.2f} GHz; early stop EI < "
               f"{B4['ei_stop']} dB x {B4['ei_patience']})", items, sum(t for *_, t in items))
    if args.dry_run:
        return 0
    if not args.yes and input("Start B4? [y/N] ").strip().lower() not in ("y", "yes", "o", "oui"):
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    jl = out_dir / "B4_designs.jsonl"

    def objective(x):
        s, u = float(x[0]), x[1:]
        d = spacings_from_params(s, u, 4, WIDTH, D_MIN)
        recs = evaluate_geometry(d, [f0], args.r_in_min, args.n_starts)
        append_jsonl(jl, recs)
        write_summary(out_dir / "B4_summary.csv", read_jsonl(jl))
        b = next(r for r in recs if r["best_fed"])
        return b["objective_dB"], dict(spacings_mm=b["spacings_mm"], fed=b["fed"], G_kit_dBi=b["G_kit_dBi"],
                                       G_robust_mean_dBi=b["G_robust_mean_dBi"], eta=b["eta"])

    bo = BayesianOptimizer(objective, [(lo, hi), (0, 1), (0, 1)], n_init=B4["n_init"], n_iter=args.n_iter,
                           seed=B4["seed"], history_path=str(hist), ei_stop=B4["ei_stop"],
                           ei_patience=B4["ei_patience"], names=["s_m", "u1", "u2"])
    best = bo.run()
    print(f"[B4] best objective {best['y']:.3f} dB after {best['n_evaluations']} evaluations: {best['record']['info']}")
    if not args.no_checks:
        run_checks(read_jsonl(jl), "B4", [f0 / 1e9], out_dir)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Step 6 campaigns B3 (grid) and B4 (Bayesian optimisation).")
    sub = ap.add_subparsers(dest="campaign", required=True)
    for name in ("b3", "b4"):
        p = sub.add_parser(name)
        p.add_argument("--out", default=str(OUT_DIR))
        p.add_argument("--r-in-min", type=float, default=R_IN_MIN)
        p.add_argument("--n-starts", type=int, default=N_STARTS)
        p.add_argument("--no-checks", action="store_true")
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--yes", action="store_true")
    sub.choices["b3"].add_argument("--freqs", default=",".join(f"{f:.2f}" for f in B3_FREQS_GHZ), help="GHz")
    sub.choices["b4"].add_argument("--f0", required=True,
                                   help="GHz, or 'auto' for the f0 of the best robust B3 design")
    sub.choices["b4"].add_argument("--n-iter", type=int, default=B4["n_iter"])
    args = ap.parse_args(argv)
    return run_b3(args) if args.campaign == "b3" else run_b4(args)


if __name__ == "__main__":
    sys.exit(main())
