"""
Matching network of the B3 design recomputed on the input impedance of a CST model (step 8).

Why
---
CST V6a validates B3 at 1 GHz with a ~1 % frequency offset (Z_in(CST, 1.00 GHz) =
26.5 + j326 Ohm vs 26.8 + j299 Ohm for the MoM) while the matched band is only
+-0.5 %: the network must be computed on the CST impedance, and recomputed for
every CST model version (V6a bare strips, V6b + substrate, V6c + feed).

Inputs
------
s1p          CST Touchstone 1-port at the gap of el2. Reference impedance from the
             option line ("# GHz S MA R 50"); --z-ref forces it (CST writes "R 0"
             when S is not renormalised; see compare_v5.read_touchstone_1port).
--gain-csv   optional CST gain G_A(f) towards +y (IEEE gain, phi pol.), columns
             f_GHz,G_dBi. Without it, G_A is the MoM one (V6_expected_B3_1.00GHz_freq.csv)
             SHIFTED IN FREQUENCY:  G_A,CST(f) = G_A,MoM(f (1 + s)),  where s is the
             offset between the MoM and CST reactances:  X_MoM(f_m) = X_CST(f_c)
             -> s = f_m / f_c - 1, sampled over the MoM range; the median is retained
             (printed with its spread). G_r(f) is only evaluated where G_A is known
             (no extrapolation); |S11|(f) uses the whole CST band.
             If CST far-field cuts *farfield_cut_theta=90_f=<f>GHz.txt sit next to the
             s1p, the shifted MoM G_A is checked against their phi = 90 deg Abs(Phi).

Models (solver/optim/matching.py, used as is): topologies a / b_sp / b_ps / c_sp / c_ps,
Johanson kits, Monte-Carlo over the tolerances x f0 (1 + [-0.5 ... +0.5] %), ESR.
Capacitor ESR / ESL and inductor Q / C_par are command-line parameters; when they are
not given, the step-6b values are used and flagged as a PROVISIONAL RANGE (not data):
ranking ESR_C = 0.1 Ohm, ESL_C = 0, Q_L = 40, C_par = 0; sensitivity ESR_C in
{0, 0.1, 0.3} Ohm, ESL_C in {0, 0.1, 0.5, 1.0} nH (0.1 nH: first jtisoft indication),
Q_L in {20, 40, 80}, C_par 0.1 pF.

Outputs (fiches_cst/resultats_V6/<version>/)
-------
adaptation_from_cst.json   CST version, solutions (top 10), sensitivity, ESL-aware
                           rankings, comparison with the step-6b network (MoM),
                           bench-tuning list (kit values moving the |S11| minimum by
                           -2, -1, +1, +2 %), spread of the minimum over the tolerances
adaptation_from_cst.png    |S11|(f) and G_r(f) of the 3 best distinct solutions (+ the 6b network)

    python solver/antenna_simulation/euraap_challenge/match_from_cst.py \
        "<...>/resultats_V6/V6a/Reseau end-fire.s1p" [--version V6a] [--z-ref 50] \
        [--gain-csv G.csv] [--esr-ohm 0.1] [--esl-nh 0.1] [--q-l 40] [--c-par-pf 0]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from compare_v5 import read_cst_farfield, read_touchstone_1port          # noqa: E402
from solver.optim import matching as mt                                 # noqa: E402
from solver.optim.components import (                                   # noqa: E402
    C_PAR_RANGE, CAPACITOR_KITS, INDUCTOR_KIT, JOHANSON_R14S_EXTRAPOLATED, JOHANSON_R14S_RF_1GHZ,
    PROVISIONAL_ESL_GRID, PROVISIONAL_ESR_GRID, Q_L_RANGE, ComponentModel, InductorModel, johanson_r14s_model,
)
from dataclasses import asdict                                          # noqa: E402

CHALLENGE = Path(r"D:\PhD_Fahdilou_Leat\PUBLICATIONS\7-EurAAP Small Antenna Super-Gain Challenge")
V6_DIR = CHALLENGE / "fiches_cst" / "resultats_V6"
MOM_FREQ_CSV = V6_DIR / "V6_expected_B3_1.00GHz_freq.csv"
REF_6B_JSON = V6_DIR / "V6_adaptation_B3_1.00GHz.json"

#: step-6b ranking values (PROVISIONAL RANGE, NOT DATA) used when not given on the command line
DEFAULT_ESR_OHM, DEFAULT_ESL_NH, DEFAULT_Q_L, DEFAULT_C_PAR_PF = 0.1, 0.0, 40.0, 0.0
#: first indication for the S-Series capacitors (jtisoft, step-8 instructions): added to the ESL
#: sensitivity cases, NOT used for the ranking unless passed with --esl-nh
FIRST_INDICATION_ESL_NH = 0.1
TUNING_SHIFTS_PCT = (-2.0, -1.0, 1.0, 2.0)

INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


# =============================================================================
# Inputs
# =============================================================================
def touchstone_reference(path):
    """Reference resistance written on the Touchstone option line (None if absent)."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.split("!")[0].strip()
            if s.startswith("#"):
                toks = s[1:].lower().split()
                if "r" in toks and toks.index("r") + 1 < len(toks):
                    return float(toks[toks.index("r") + 1])
                return None
    return None


def read_s1p(path, z_ref=None):
    """{'f_hz', 'z', 'z_ref_file', 'z_ref_used'}; z_ref overrides the file's R (needed for 'R 0')."""
    f, z = read_touchstone_1port(path, r_ref=z_ref)
    r_file = touchstone_reference(path)
    return dict(f_hz=f, z=z, z_ref_file=r_file, z_ref_used=float(z_ref if z_ref is not None else r_file))


def read_gain_csv(path):
    """(f_hz, G_lin) from a CSV with columns f_GHz, G_dBi."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    f = np.array([float(r["f_GHz"]) for r in rows]) * 1e9
    g = np.array([float(r["G_dBi"]) for r in rows])
    order = np.argsort(f)
    return f[order], 10 ** (g[order] / 10)


def load_mom_oneport(path, f0):
    with open(path, newline="", encoding="utf-8") as fh:
        return mt.AntennaOnePort.from_freq_rows(list(csv.DictReader(fh)), f0)


def estimate_frequency_shift(f_cst, z_cst, mom: mt.AntennaOnePort, f0, n=161):
    """
    Offset s between the MoM and CST reactances, X_MoM(f_m) = X_CST(f_c) -> s = f_m / f_c - 1,
    sampled over the MoM range (where the CST reactance covers it, on the monotonic branch of
    X_CST around f0). Retained value: the median. Also s at f0 (X_MoM(f0 (1 + s)) = X_CST(f0)).
    """
    xc = np.asarray(z_cst).imag
    k0 = int(np.argmin(np.abs(f_cst - f0)))
    dx = np.diff(xc)
    lo, hi = k0, k0
    while lo > 0 and dx[lo - 1] > 0:
        lo -= 1
    while hi < len(dx) and dx[hi] > 0:
        hi += 1
    if hi - lo < 2:
        raise ValueError("X_CST is not increasing around f0: give the CST gain with --gain-csv.")
    fc_seg, xc_seg = f_cst[lo:hi + 1], xc[lo:hi + 1]
    if len(fc_seg) >= 4:     # cubic spline of X_CST (as for the MoM data), resampled finely for the inversion
        spl = CubicSpline(fc_seg / f0 - 1, xc_seg)
        fc_seg = np.linspace(fc_seg[0], fc_seg[-1], 20001)
        xc_seg = spl(fc_seg / f0 - 1)
        if np.any(np.diff(xc_seg) <= 0):
            raise ValueError("spline of X_CST not monotonic around f0: give the CST gain with --gain-csv.")
    fm = np.linspace(mom.f_min, mom.f_max, n)
    xm = mom.zin(fm).imag
    inside = (xm >= xc_seg[0]) & (xm <= xc_seg[-1])
    if inside.sum() < 5:
        raise ValueError("MoM and CST reactances barely overlap: give the CST gain with --gain-csv.")
    s = fm[inside] / np.interp(xm[inside], xc_seg, fc_seg) - 1
    x_f0 = float(np.interp(f0, fc_seg, xc_seg))
    if not (np.all(np.diff(xm) > 0) and xm[0] <= x_f0 <= xm[-1]):
        s_f0 = None
    else:
        s_f0 = float(np.interp(x_f0, xm, fm) / f0 - 1)
    return dict(s=float(np.median(s)), s_at_f0=s_f0, s_min=float(s.min()), s_max=float(s.max()),
                n_samples=int(inside.sum()), f_mom_GHz=(fm[inside] / 1e9).tolist(), s_samples=s.tolist())


def build_antennas(f_cst, z_cst, f0, gain_csv=None, mom=None):
    """
    (ant, ant_z, info):
      ant   : AntennaOnePort with Z_CST and G_A, on the CST frequencies where G_A is known
      ant_z : AntennaOnePort with Z_CST over the whole CST band and G_A = 1 (|S11| work only)
    """
    order = np.argsort(f_cst)
    f_cst, z_cst = np.asarray(f_cst, float)[order], np.asarray(z_cst, complex)[order]
    ant_z = mt.AntennaOnePort(f_cst, z_cst, np.ones(len(f_cst)), f0)
    tol = 1e-9 * f0
    if gain_csv is not None:
        fg, gg = read_gain_csv(gain_csv)
        keep = (f_cst >= fg[0] - tol) & (f_cst <= fg[-1] + tol)
        g_db = CubicSpline(fg, 10 * np.log10(gg)) if len(fg) > 2 else (lambda x: np.interp(x, fg, 10 * np.log10(gg)))
        g = 10 ** (np.asarray(g_db(np.clip(f_cst[keep], fg[0], fg[-1]))) / 10)
        info = dict(source=f"CST gain table {Path(gain_csv).name}", shift=None,
                    f_range_GHz=[float(fg[0] / 1e9), float(fg[-1] / 1e9)])
    else:
        if mom is None:
            raise ValueError("No CST gain table: the MoM one-port is needed (shifted G_A).")
        sh = estimate_frequency_shift(f_cst, z_cst, mom, f0)
        s = sh["s"]
        lo, hi = mom.f_min / (1 + s), mom.f_max / (1 + s)
        keep = (f_cst >= lo - tol) & (f_cst <= hi + tol)
        g = mom.gain(np.clip(f_cst[keep] * (1 + s), mom.f_min, mom.f_max))
        info = dict(source="MoM G_A (V6_expected_B3_1.00GHz_freq.csv) shifted in frequency: "
                           "G_A,CST(f) = G_A,MoM(f (1 + s))",
                    shift=sh, f_range_GHz=[float(lo / 1e9), float(hi / 1e9)])
    if keep.sum() < 4:
        raise ValueError("Fewer than 4 CST frequencies where G_A is known.")
    ant = mt.AntennaOnePort(f_cst[keep], z_cst[keep], g, f0)
    if not (ant.f_min <= f0 * (1 + min(mt.F_OFFSETS_ROBUST)) + tol and ant.f_max >= f0 * (1 + max(mt.F_OFFSETS_ROBUST)) - tol):
        raise ValueError(f"G_A known on [{ant.f_min / 1e9:.4f}, {ant.f_max / 1e9:.4f}] GHz only: "
                         f"does not cover f0 +- 0.5 %.")
    info["f_range_used_GHz"] = [ant.f_min / 1e9, ant.f_max / 1e9]
    return ant, ant_z, info


def gain_csv_from_cuts(folder, out_csv, column="Abs(Phi)"):
    """
    CSV f_GHz,G_dBi of the CST gain towards +y (theta = 90, phi = 90 deg) read in the cuts
    *farfield_cut_theta=90_f=<f>GHz.txt of `folder`. G_dBi = `column` (default Abs(Phi):
    phi polarisation, the objective of the design); Abs(Gain) is kept as G_total_dBi.
    CST "Gain" is the IEEE gain (relative to the accepted power), i.e. G_A.
    """
    rows = []
    for p in sorted(Path(folder).glob("*farfield_cut_theta=90_f=*GHz.txt")):
        m = re.search(r"f=([0-9.]+)GHz", p.name)
        d = read_cst_farfield(p)
        k = int(np.argmin(np.abs(d["Phi"] - 90.0)))
        if not m or abs(d["Phi"][k] - 90.0) > 1e-6:
            continue
        rows.append((float(m.group(1)), float(d[column][k]), float(d["Abs(Gain)"][k]), p.name))
    if len(rows) < 2:
        raise ValueError(f"{folder}: fewer than 2 cuts theta = 90 deg with a phi = 90 deg sample.")
    rows.sort()
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["f_GHz", "G_dBi", "G_total_dBi", "source"])
        for r in rows:
            w.writerow([f"{r[0]:g}", f"{r[1]:.4f}", f"{r[2]:.4f}", r[3]])
    return out_csv, rows


def check_gain_against_cst_cuts(folder, ant):
    """Shifted / tabulated G_A vs CST cuts theta = 90 deg (phi = 90 deg, Abs(Phi)) found in folder."""
    rows = []
    for p in sorted(Path(folder).glob("*farfield_cut_theta=90_f=*GHz.txt")):
        m = re.search(r"f=([0-9.]+)GHz", p.name)
        if not m:
            continue
        f = float(m.group(1)) * 1e9
        d = read_cst_farfield(p)
        phi = d.get("Phi")
        col = d.get("Abs(Phi)", d.get("Abs(Gain)"))
        if phi is None or col is None:
            continue
        k = int(np.argmin(np.abs(phi - 90.0)))
        if abs(phi[k] - 90.0) > 1e-6 or not (ant.f_min <= f <= ant.f_max):
            continue
        g_used = float(10 * np.log10(ant.gain(f)))
        rows.append(dict(file=p.name, f_GHz=f / 1e9, G_cst_phi_dBi=float(col[k]), G_A_used_dBi=g_used,
                         diff_dB=g_used - float(col[k])))
    return rows


# =============================================================================
# |S11| minimum
# =============================================================================
def s11_minimum(positions, is_ind, values, ant_z, cap_model, ind_model, f_grid):
    """
    Frequency and depth of the |S11| minimum for value arrays (..., n_el) on f_grid (Hz),
    parabolic refinement of |S11|^2 around the grid minimum. f_min = NaN if at a grid edge.
    """
    values = np.asarray(values, float)
    r = mt.respond(positions, is_ind, values[..., None, :], f_grid, ant_z, cap_model, ind_model)
    p = np.abs(r["s11"]) ** 2
    k = np.argmin(p, axis=-1)
    n = len(f_grid)
    kk = np.clip(k, 1, n - 2)
    pm = np.take_along_axis(p, (kk - 1)[..., None], -1)[..., 0]
    p0 = np.take_along_axis(p, kk[..., None], -1)[..., 0]
    pp = np.take_along_axis(p, (kk + 1)[..., None], -1)[..., 0]
    den = pm - 2 * p0 + pp
    d = np.where(den > 0, 0.5 * (pm - pp) / np.where(den > 0, den, 1), 0.0)
    d = np.clip(d, -1, 1)
    df = f_grid[1] - f_grid[0]
    f_min = f_grid[kk] + d * df
    p_min = np.maximum(p0 - 0.25 * (pm - pp) * d, 1e-12)
    edge = (k == 0) | (k == n - 1)
    return np.where(edge, np.nan, f_min), np.where(edge, 10 * np.log10(p.min(axis=-1)), 10 * np.log10(p_min))


def net_arrays(net: mt.Network):
    return (np.array([p.is_inductor for p in net.parts]), np.array([p.value for p in net.parts]),
            np.array([p.tol_abs for p in net.parts]))


def network_minimum(net, ant_z, cap_model, ind_model, n=4001):
    f = np.linspace(ant_z.f_min, ant_z.f_max, n)
    is_ind, val, _ = net_arrays(net)
    fm, sm = s11_minimum(net.positions, is_ind, val[None, :], ant_z, cap_model, ind_model, f)
    return (None if np.isnan(fm[0]) else float(fm[0])), float(sm[0])


def minimum_spread(net, ant_z, cap_model, ind_model, u, n=2001):
    """Spread of the |S11| minimum frequency over the part tolerances (draws u)."""
    f = np.linspace(ant_z.f_min, ant_z.f_max, n)
    is_ind, val, tol = net_arrays(net)
    vals = val[None, :] + u[:, :len(val)] * tol[None, :]
    fm, sm = s11_minimum(net.positions, is_ind, vals, ant_z, cap_model, ind_model, f)
    ok = ~np.isnan(fm)
    q = np.percentile(fm[ok], [5, 50, 95]) if ok.any() else [np.nan] * 3
    return dict(n_draws=int(len(u)), n_inside_band=int(ok.sum()),
                f_min_p5_GHz=float(q[0] / 1e9), f_min_median_GHz=float(q[1] / 1e9),
                f_min_p95_GHz=float(q[2] / 1e9), f_min_std_pct=float(100 * np.std(fm[ok]) / ant_z.f0),
                s11_min_median_dB=float(np.median(sm[ok])))


# =============================================================================
# Bench tuning list
# =============================================================================
def bench_tuning(best: mt.Network, ant, ant_z, cap_model, ind_model, shifts_pct=TUNING_SHIFTS_PCT,
                 f_tol_pct=0.25, depth_max_db=-15.0, n_show=3, chunk=4000):
    """
    Every kit-value combination in the topology of `best` (same part types per position):
    frequency of the |S11| minimum. For each target shift d (relative to the minimum of `best`),
    networks whose minimum lies within f_tol_pct of f_best (1 + d) and is at least
    depth_max_db deep, ranked by number of changed parts, then distance to target, then depth.
    """
    eparts = mt.electrical_parts(list(CAPACITOR_KITS) + list(INDUCTOR_KIT))
    caps = [p for p in eparts if not p.is_inductor]
    inds = [p for p in eparts if p.is_inductor]
    pools = [inds if p.is_inductor else caps for p in best.parts]
    idx = np.stack([m.ravel() for m in np.meshgrid(*[np.arange(len(pl)) for pl in pools], indexing="ij")], axis=1)
    vals = np.array([[pools[j][i].value for j, i in enumerate(row)] for row in idx])
    is_ind = np.array([p.is_inductor for p in best.parts])
    f_best, s_best = network_minimum(best, ant_z, cap_model, ind_model)
    f = np.linspace(ant_z.f_min, ant_z.f_max, 1001)
    fm, sm = np.empty(len(idx)), np.empty(len(idx))
    for s in range(0, len(idx), chunk):
        fm[s:s + chunk], sm[s:s + chunk] = s11_minimum(best.positions, is_ind, vals[s:s + chunk], ant_z,
                                                       cap_model, ind_model, f)
    out = []
    for d in shifts_pct:
        f_t = f_best * (1 + d / 100)
        err = 100 * (fm / f_t - 1)
        ok = np.flatnonzero((np.abs(err) <= f_tol_pct) & (sm <= depth_max_db))
        cands, seen = [], set()
        for i in ok:
            net = mt.Network(best.topology, tuple(pools[j][k] for j, k in enumerate(idx[i])))
            changed = part_changes(best, net)
            key = tuple((c["position"], c["from"], c["to"]) for c in changed)   # same labels = same swap
            if key in seen:
                continue
            seen.add(key)
            cands.append((len(changed), abs(err[i]), sm[i], net, changed, i))
        cands.sort(key=lambda c: (c[0], round(c[1] / 0.05), c[2]))
        rows = []
        for n_changed, e, depth, net, changed, i in cands[:n_show]:
            f_i = float(fm[i])
            g_r = (float(10 * np.log10(mt.network_response(net, f_i, ant, cap_model, ind_model)["G_r"]))
                   if ant.f_min <= f_i <= ant.f_max else None)
            band = band_around(net, ant_z, cap_model, ind_model, f_i)
            rows.append(dict(n_changed=int(n_changed), changes=changed,
                             description=net.describe(), elements_port_to_gap=mt.part_records(net),
                             f_min_GHz=f_i / 1e9, shift_achieved_pct=float(100 * (f_i / f_best - 1)),
                             s11_min_dB=float(depth), G_r_at_f_min_dBi=g_r, band_10dB_MHz=band["bw_MHz"]))
        out.append(dict(target_shift_pct=d, f_target_GHz=f_t / 1e9, n_candidates=len(cands), options=rows))

    # ladder of single-part swaps (|S11| min <= -10 dB, within +-3 %), sorted by shift
    ladder, seen = [], set()
    for i in np.flatnonzero(~np.isnan(fm) & (sm <= -10.0)):
        net = mt.Network(best.topology, tuple(pools[j][k] for j, k in enumerate(idx[i])))
        changed = part_changes(best, net)
        shift = 100 * (fm[i] / f_best - 1)
        key = tuple((c["position"], c["from"], c["to"]) for c in changed)
        if len(changed) != 1 or key in seen or abs(shift) > 3.0:
            continue
        seen.add(key)
        ladder.append(dict(change=changed[0], f_min_GHz=float(fm[i] / 1e9), shift_pct=float(shift),
                           s11_min_dB=float(sm[i]),
                           band_10dB_MHz=band_around(net, ant_z, cap_model, ind_model, float(fm[i]))["bw_MHz"]))
    ladder.sort(key=lambda r: r["shift_pct"])
    return dict(reference=dict(description=best.describe(), f_min_GHz=f_best / 1e9, s11_min_dB=s_best),
                criteria=dict(f_tol_pct=f_tol_pct, depth_max_dB=depth_max_db,
                              ranking="fewest changed parts, then distance to target (0.05 % bins), then depth",
                              shift_reference="relative to the |S11| minimum of the reference network, "
                                              "antenna = this CST model"),
                n_combinations=int(len(idx)), targets=out, single_swap_ladder=ladder)


def band_around(net, ant_z, cap_model, ind_model, f_c, n=4001):
    """|S11| <= -10 dB band around f_c (instead of f0): mt.band_10db on the one-port re-centred at f_c."""
    recentred = mt.AntennaOnePort(ant_z.f_hz, ant_z.zin_data, ant_z.gain_data, f_c)
    return mt.band_10db(net, recentred, cap_model, ind_model, n=n)


def _part_label(item):
    is_ind, value, _ = item
    return f"{value * 1e9:g} nH" if is_ind else f"{value * 1e12:g} pF"


def part_changes(ref: mt.Network, net: mt.Network):
    """
    Parts to swap on the board to go from `ref` to `net` (same topology), as
    [{'position', 'from', 'to'}]; adjacent series parts commute, so a group of
    series parts is compared as a multiset.
    """
    from collections import Counter
    out = []
    for (pos, items_r), (_, items_n) in zip(ref.key(), net.key()):
        removed = sorted((Counter(items_r) - Counter(items_n)).elements())
        added = sorted((Counter(items_n) - Counter(items_r)).elements())
        for a, b in zip(removed, added):
            out.append({"position": pos, "from": _part_label(a), "to": _part_label(b)})
    return out


# =============================================================================
# 6b network
# =============================================================================
def network_from_record(rec):
    """Rebuild a Network from a solution record of a matching JSON (step 6b format)."""
    eparts = mt.electrical_parts(list(CAPACITOR_KITS) + list(INDUCTOR_KIT))
    parts = []
    for el in rec["elements_port_to_gap"]:
        is_ind = el["type"] == "inductor"
        hit = [p for p in eparts if p.is_inductor == is_ind and abs(p.value - el["value"]) < 1e-18
               and abs(p.tol_abs - el["tolerance_abs"]) < 1e-21]
        if not hit:
            raise ValueError(f"part {el['label']} not found in the kits")
        parts.append(hit[0])
    return mt.Network(rec["topology"], tuple(parts))


# =============================================================================
# Main
# =============================================================================
def _git_info():
    run = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    try:
        return dict(commit=run("rev-parse", "HEAD"), branch=run("rev-parse", "--abbrev-ref", "HEAD"),
                    dirty=bool(run("status", "--porcelain")))
    except Exception:
        return None


def _sol_record(e, net, rank):
    return dict(rank=rank, topology=net.topology, topology_text=mt.TOPOLOGY_TEXT[net.topology],
                description=net.describe(), elements_port_to_gap=mt.part_records(net), has_inductor=net.has_inductor,
                expected={k: e[k] for k in ("G_r_nominal_dBi", "G_r_mean_dBi", "G_r_p10_dBi", "S11_dB",
                                            "mismatch_dB", "network_loss_dB", "eta_net", "z_port_ohm", "band")})


def infer_version(s1p):
    m = re.fullmatch(r"V\d+[a-z]?", Path(s1p).parent.name)
    return m.group(0) if m else None


def run(s1p, version, out_dir=None, z_ref=None, gain_csv=None, esr_ohm=None, esl_nh=None, q_l=None,
        c_par_pf=None, f0=1e9, mom_csv=MOM_FREQ_CSV, ref_6b=REF_6B_JSON, n_mc=500, seed=0,
        esl_rankings=True, figure=True, show=False, cap_data=None, out_name=None, previous=None, tune_rank=1,
        log=print):
    t_start = time.perf_counter()
    out_dir = Path(out_dir) if out_dir is not None else V6_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- models (explicit; step-6b values flagged as provisional when not given)
    given = dict(esr=esr_ohm is not None, esl=esl_nh is not None, q=q_l is not None, cpar=c_par_pf is not None)
    esr = DEFAULT_ESR_OHM if esr_ohm is None else float(esr_ohm)
    esl = (DEFAULT_ESL_NH if esl_nh is None else float(esl_nh)) * 1e-9
    q = DEFAULT_Q_L if q_l is None else float(q_l)
    cpar = (DEFAULT_C_PAR_PF if c_par_pf is None else float(c_par_pf)) * 1e-12
    ind = lambda qq=q, cc=cpar: InductorModel(q=qq, c_par=cc)
    st = lambda g: "user-supplied (command line)" if g else "PROVISIONAL, NOT DATA (step-6b value)"
    tab = johanson_r14s_model()
    if cap_data == "johanson":
        if given["esr"] or given["esl"]:
            raise ValueError("--cap-data johanson sets ESR / ESL per value: do not pass --esr-ohm / --esl-nh.")
        cap = tab
        esr = JOHANSON_R14S_EXTRAPOLATED["esr_ohm"]           # scalar base of the sensitivity grids
        esl = JOHANSON_R14S_EXTRAPOLATED["esl_nh"] * 1e-9
        st_cap = "MANUFACTURER DATA (jtisoft, 1 GHz) for 0.5/1.0/1.2/2.7 pF; " + JOHANSON_R14S_EXTRAPOLATED["status"]
        cap_text = "Johanson R14S data (jtisoft)"
    elif cap_data is None:
        cap = ComponentModel(esr=esr, esl=esl)
        st_cap = None
        cap_text = f"ESR_C = {esr:g} Ohm, ESL_C = {esl * 1e9:g} nH"
    else:
        raise ValueError(f"unknown cap_data {cap_data!r} (only 'johanson')")
    out_name = out_name or ("adaptation_from_cst" + ("_johanson" if cap_data == "johanson" else ""))

    # ---- antenna
    data = read_s1p(s1p, z_ref)
    mom = load_mom_oneport(mom_csv, f0) if Path(mom_csv).is_file() else None
    ant, ant_z, ginfo = build_antennas(data["f_hz"], data["z"], f0, gain_csv, mom)
    za = complex(ant_z.zin(f0))
    ga_db = float(10 * np.log10(ant.gain(f0)))
    gam = (za - mt.Z0) / (za + mt.Z0)
    w_dxdw = (ant_z.zin(f0 * 1.0005).imag - ant_z.zin(f0 * 0.9995).imag) / 0.001
    q_in = float(abs(w_dxdw) / (2 * za.real))                          # Q_in = (w / 2R) |dX/dw|, as step 6b
    cuts = check_gain_against_cst_cuts(Path(s1p).parent, ant) if gain_csv is None else []
    log(f"[{version}] {Path(s1p).name}: {len(data['f_hz'])} points, {data['f_hz'][0] / 1e9:.4f} ... "
        f"{data['f_hz'][-1] / 1e9:.4f} GHz, reference R = {data['z_ref_used']:g} Ohm "
        f"(file: {data['z_ref_file']})")
    log(f"  Z_CST(f0) = {za.real:.2f} {za.imag:+.2f}j Ohm, |S11| without network = {20 * np.log10(abs(gam)):.2f} dB, "
        f"Q_in ~ {q_in:.1f}")
    if mom is not None:
        zm = complex(mom.zin(f0))
        log(f"  Z_MoM(f0) = {zm.real:.2f} {zm.imag:+.2f}j Ohm")
    if ginfo["shift"]:
        sh = ginfo["shift"]
        log(f"  G_A: MoM shifted, s = {100 * sh['s']:+.3f} % retained (median over {sh['n_samples']} points; "
            f"range {100 * sh['s_min']:+.3f} ... {100 * sh['s_max']:+.3f} %, at f0 "
            f"{'n/a' if sh['s_at_f0'] is None else f'{100 * sh['s_at_f0']:+.3f} %'}): "
            f"G_A,CST(f) = G_A,MoM(f (1 {sh['s']:+.5f}))")
    else:
        log(f"  G_A: {ginfo['source']}")
    log(f"  G_A(f0) = {ga_db:.2f} dBi, known on {ant.f_min / 1e9:.4f} ... {ant.f_max / 1e9:.4f} GHz")
    for c in cuts:
        log(f"  check vs CST cut {c['f_GHz']:.3f} GHz: G_A used {c['G_A_used_dBi']:.2f} dBi, CST "
            f"{c['G_cst_phi_dBi']:.2f} dBi ({c['diff_dB']:+.2f} dB)")
    if st_cap:
        log(f"  models: capacitors {tab.label} [{st_cap}]")
        for r in JOHANSON_R14S_RF_1GHZ:
            log(f"    {r.value_pf:g} pF {r.part_number}: ESR {1e3 * r.esr_ohm:.1f} mOhm, ESL {r.esl_nh:g} nH, "
                f"SRF {r.srf_ghz:g} GHz")
        log(f"  Q_L = {q:g} [{st(given['q'])}], C_par = {cpar * 1e12:g} pF [{st(given['cpar'])}]")
    else:
        log(f"  models: ESR_C = {esr:g} Ohm [{st(given['esr'])}], ESL_C = {esl * 1e9:g} nH [{st(given['esl'])}], "
            f"Q_L = {q:g} [{st(given['q'])}], C_par = {cpar * 1e12:g} pF [{st(given['cpar'])}]")

    # ---- search and top 10
    u = mt.tolerance_draws(n_mc, 3, seed)
    res = mt.search(ant, cap, ind(), n_mc=n_mc, seed=seed, n_keep=None)
    log("  combinations with |S11(f0)| <= -10 dB: " + ", ".join(f"{t} {res.counts[t]}/{res.counts_total[t]}"
                                                               for t in mt.TOPOLOGIES))
    top = []
    for net, _, _ in res.ranked[:10]:
        e = mt.evaluate_solution(net, ant, cap, ind(), u)
        e["band"] = mt.band_10db(net, ant_z, cap, ind(), n=2001)
        e["net"] = net
        e["f_min"], e["s11_min"] = network_minimum(net, ant_z, cap, ind())
        top.append(e)
    log(f"  {'#':>2} {'topo':>5} {'G_r':>6} {'mean':>6} {'P10':>6} {'S11':>6} {'BW':>5} {'f_min':>7}  elements (port -> gap)")
    for i, e in enumerate(top, 1):
        fmin = f"{e['f_min'] / 1e9:.4f}" if e["f_min"] else "  n/a "
        log(f"  {i:>2} {e['topology']:>5} {e['G_r_nominal_dBi']:6.2f} {e['G_r_mean_dBi']:6.2f} {e['G_r_p10_dBi']:6.2f} "
            f"{e['S11_dB']:6.1f} {e['band']['bw_MHz']:5.1f} {fmin}  {e['description']}")
    best = top[0]["net"]

    # ---- 3 best distinct (by response) for the figure / sensitivity
    shown, sigs = [], set()
    for e in top:
        sig = (round(e["S11_dB"], 2), round(e["G_r_nominal_dBi"], 3))
        if sig not in sigs:
            sigs.add(sig)
            shown.append(e)
        if len(shown) == 3:
            break

    # ---- sensitivity
    esr_set = sorted(set(PROVISIONAL_ESR_GRID) | {esr})
    esl_set = sorted(set(PROVISIONAL_ESL_GRID) | {esl, FIRST_INDICATION_ESL_NH * 1e-9})
    cases = ([(f"ESR_C = {r:g} Ohm", ComponentModel(r, esl), ind()) for r in esr_set]
             + [(f"ESL_C = {l * 1e9:g} nH", ComponentModel(esr, l), ind()) for l in esl_set]
             + [(f"Q_L = {qq:g}", cap, ind(qq)) for qq in Q_L_RANGE]
             + [(f"C_par = {c * 1e12:g} pF", cap, ind(cc=c)) for c in C_PAR_RANGE if c > 0]
             + [("Johanson R14S data (jtisoft)", tab, ind())])
    sensitivity = []
    for e in shown:
        rows = []
        for name, cm, lm in cases:
            s = mt.evaluate_solution(e["net"], ant, cm, lm, u)
            fmin, smin = network_minimum(e["net"], ant_z, cm, lm)
            rows.append(dict(case=name, G_r_nominal_dBi=s["G_r_nominal_dBi"], G_r_mean_dBi=s["G_r_mean_dBi"],
                             G_r_p10_dBi=s["G_r_p10_dBi"], S11_dB=s["S11_dB"], network_loss_dB=s["network_loss_dB"],
                             f_min_GHz=None if fmin is None else fmin / 1e9, s11_min_dB=smin))
        sensitivity.append(dict(network=e["description"], rows=rows))

    # ---- ESL-aware rankings
    esl_rank = {}
    if esl_rankings:
        for l in [x for x in esl_set if x != esl]:
            cm = ComponentModel(esr, l)
            r = mt.search(ant, cm, ind(), n_mc=n_mc, seed=seed, n_keep=5)
            rows = []
            for i, (n, _, _) in enumerate(r.ranked, 1):
                s = mt.evaluate_solution(n, ant, cm, ind(), u)
                rows.append(dict(rank=i, description=n.describe(), elements_port_to_gap=mt.part_records(n),
                                 **{k: s[k] for k in ("G_r_nominal_dBi", "G_r_mean_dBi", "G_r_p10_dBi", "S11_dB",
                                                      "network_loss_dB")}))
            esl_rank[f"{l * 1e9:g} nH"] = rows
            log(f"  ESL_C = {l * 1e9:g} nH: #1 {rows[0]['description']} (mean {rows[0]['G_r_mean_dBi']:.2f} dBi)")

    # ---- comparison with step 6b (network computed on the MoM)
    comp = None
    if ref_6b is not None and Path(ref_6b).is_file():
        doc6 = json.loads(Path(ref_6b).read_text(encoding="utf-8"))
        rec6 = next(s for s in doc6["solutions"] if s["rank"] == doc6.get("recommended_rank", 1))
        net6 = network_from_record(rec6)
        e6 = mt.evaluate_solution(net6, ant, cap, ind(), u)
        f6, s6 = network_minimum(net6, ant_z, cap, ind())
        f6m, s6m = (network_minimum(net6, mt.AntennaOnePort(mom.f_hz, mom.zin_data, np.ones(len(mom.f_hz)), f0),
                                    cap, ind()) if mom is not None else (None, None))
        same = net6.key() == best.key()
        comp = dict(network_6b=net6.describe(), same_components_as_best=bool(same),
                    on_mom=dict(f_min_GHz=None if f6m is None else f6m / 1e9, s11_min_dB=s6m,
                                G_r_mean_dBi_6b_record=rec6["expected"]["G_r_mean_dBi"]),
                    on_cst=dict(f_min_GHz=None if f6 is None else f6 / 1e9, s11_min_dB=s6,
                                S11_f0_dB=e6["S11_dB"], G_r_nominal_dBi=e6["G_r_nominal_dBi"],
                                G_r_mean_dBi=e6["G_r_mean_dBi"], G_r_p10_dBi=e6["G_r_p10_dBi"],
                                band=mt.band_10db(net6, ant_z, cap, ind(), n=2001)),
                    best_on_cst=dict(f_min_GHz=None if top[0]["f_min"] is None else top[0]["f_min"] / 1e9,
                                     s11_min_dB=top[0]["s11_min"]))
        log(f"  6b network ({net6.describe()}): on MoM min |S11| at "
            f"{'n/a' if f6m is None else f'{f6m / 1e9:.4f} GHz'} ({s6m:.1f} dB); on {version} at "
            f"{'n/a' if f6 is None else f'{f6 / 1e9:.4f} GHz'} ({s6:.1f} dB), |S11(f0)| = {e6['S11_dB']:.1f} dB, "
            f"G_r mean {e6['G_r_mean_dBi']:.2f} dBi; same components as #1: {same}")
        comp["net"] = net6

    # ---- solutions of an earlier run (e.g. step 8, provisional ESR/ESL) re-evaluated with the current model
    prev = None
    if previous is not None and Path(previous).is_file() and Path(previous).resolve() != (out_dir / f"{out_name}.json").resolve():
        docp = json.loads(Path(previous).read_text(encoding="utf-8"))
        rows = []
        mc = docp["models"]["capacitor"]
        prev_model = mc.get("label") or f"ESR {mc['esr_ohm']:g} Ohm, ESL {mc['esl_nH']:g} nH"
        log(f"  solutions of {Path(previous).name} ({prev_model}) re-evaluated with the current model:")
        for s in docp["solutions"][:3]:
            n = network_from_record(s)
            e = mt.evaluate_solution(n, ant, cap, ind(), u)
            fmin, smin = network_minimum(n, ant_z, cap, ind())
            band = mt.band_10db(n, ant_z, cap, ind(), n=2001)
            rank_now = next((i + 1 for i, t in enumerate(top) if t["net"].key() == n.key()), None)
            rows.append(dict(rank_previous=s["rank"], description=n.describe(),
                             previous=dict(G_r_mean_dBi=s["expected"]["G_r_mean_dBi"],
                                           G_r_p10_dBi=s["expected"]["G_r_p10_dBi"], f_min_GHz=s.get("f_min_GHz")),
                             now=dict(G_r_nominal_dBi=e["G_r_nominal_dBi"], G_r_mean_dBi=e["G_r_mean_dBi"],
                                      G_r_p10_dBi=e["G_r_p10_dBi"], S11_f0_dB=e["S11_dB"],
                                      f_min_GHz=None if fmin is None else fmin / 1e9, s11_min_dB=smin,
                                      bw_MHz=band["bw_MHz"], rank_now=rank_now)))
            r = rows[-1]
            log(f"   prev #{s['rank']} {n.describe()}: mean {r['previous']['G_r_mean_dBi']:.2f} -> "
                f"{e['G_r_mean_dBi']:.2f} dBi, P10 {r['previous']['G_r_p10_dBi']:.2f} -> {e['G_r_p10_dBi']:.2f}, "
                f"min |S11| {r['previous']['f_min_GHz'] or float('nan'):.4f} -> "
                f"{float('nan') if fmin is None else fmin / 1e9:.4f} GHz ({smin:.1f} dB), |S11(f0)| {e['S11_dB']:.1f} dB, "
                f"BW {band['bw_MHz']:.1f} MHz, rank now {rank_now}")
        prev = dict(file=str(Path(previous)), solutions=rows)

    # ---- bench tuning and spread of the minimum
    tune_net = top[tune_rank - 1]["net"]
    tuning = bench_tuning(tune_net, ant, ant_z, cap, ind())
    tuning["reference_rank"] = tune_rank
    spread = minimum_spread(tune_net, ant_z, cap, ind(), u)
    log(f"  bench tuning around #{tune_rank} (minimum at {tuning['reference']['f_min_GHz']:.4f} GHz, "
        f"{tuning['reference']['s11_min_dB']:.1f} dB); tolerance spread of the minimum: "
        f"std {spread['f_min_std_pct']:.2f} %, P5-P95 {spread['f_min_p5_GHz']:.4f} ... {spread['f_min_p95_GHz']:.4f} GHz")
    for t in tuning["targets"]:
        log(f"   {t['target_shift_pct']:+.0f} % (-> {t['f_target_GHz']:.4f} GHz): {t['n_candidates']} candidates")
        for o in t["options"]:
            ch = ", ".join(f"{c['position']} {c['from']} -> {c['to']}" for c in o["changes"]) or "none"
            g = "n/a" if o["G_r_at_f_min_dBi"] is None else f"{o['G_r_at_f_min_dBi']:.2f}"
            log(f"     [{o['n_changed']}] {ch}: min at {o['f_min_GHz']:.4f} GHz ({o['shift_achieved_pct']:+.2f} %), "
                f"{o['s11_min_dB']:.1f} dB, BW {o['band_10dB_MHz']:.1f} MHz, G_r {g} dBi")
    log("   single-part swaps (|S11| min <= -10 dB, within +-3 %):")
    for r in tuning["single_swap_ladder"]:
        c = r["change"]
        log(f"     {r['shift_pct']:+6.2f} %  {r['f_min_GHz']:.4f} GHz  {r['s11_min_dB']:6.1f} dB  "
            f"BW {r['band_10dB_MHz']:4.1f} MHz   {c['position']} {c['from']} -> {c['to']}")

    # ---- figure
    fig_path = out_dir / f"{out_name}.png"
    curves = []
    f_all = np.linspace(max(ant_z.f_min, f0 * 0.97), min(ant_z.f_max, f0 * 1.03), 1201)
    f_g = np.linspace(ant.f_min, ant.f_max, 801)
    for e in shown:
        rs = mt.network_response(e["net"], f_all, ant_z, cap, ind())
        rg = mt.network_response(e["net"], f_g, ant, cap, ind())
        curves.append(dict(rank=top.index(e) + 1, description=e["description"],
                           f_GHz=(f_all / 1e9).tolist(), S11_dB=(20 * np.log10(np.abs(rs["s11"]))).tolist(),
                           f_G_GHz=(f_g / 1e9).tolist(), G_r_dBi=(10 * np.log10(rg["G_r"])).tolist()))
    if figure:
        _figure(fig_path, version, f0, f_all, f_g, shown, top, comp, ant, ant_z, cap, ind(), za, ga_db, gam,
                cap_text, q)
        if show:
            import matplotlib.pyplot as plt
            plt.show()

    # ---- JSON
    doc = dict(
        campaign=f"adaptation_from_cst_{version}", cst_version=version,
        generated=dt.datetime.now().isoformat(timespec="seconds"), git=_git_info(),
        cst_data=dict(s1p=str(Path(s1p)), z_ref_file_ohm=data["z_ref_file"], z_ref_used_ohm=data["z_ref_used"],
                      f_range_GHz=[float(data["f_hz"].min() / 1e9), float(data["f_hz"].max() / 1e9)],
                      n_freq=int(len(data["f_hz"]))),
        design=dict(f0_GHz=f0 / 1e9, fed_port="el2 (gap at x = 0)", Z_A_f0_ohm=[za.real, za.imag],
                    Z_A_f0_mom_ohm=None if mom is None else [complex(mom.zin(f0)).real, complex(mom.zin(f0)).imag],
                    G_A_f0_dBi=ga_db, Q_in_approx=q_in, S11_without_network_dB=float(20 * np.log10(abs(gam))),
                    G_r_without_network_dBi=float(ga_db + 10 * np.log10(1 - abs(gam) ** 2))),
        gain=dict(ginfo, check_vs_cst_cuts=cuts),
        reference_plane="50-Ohm port at the gap of el2 (as the CST port); elements listed port -> gap",
        models=dict(
            capacitor=(dict(model="R_ESR + j w L_ESL + 1/(j w C)", esr_ohm=esr, esr_status=st(given["esr"]),
                            esl_nH=esl * 1e9, esl_status=st(given["esl"]),
                            esr_sensitivity_ohm=esr_set, esl_sensitivity_nH=[x * 1e9 for x in esl_set])
                       if cap_data is None else
                       dict(model="R_ESR(C) + j w L_ESL(C) + 1/(j w C), nearest kit nominal", label=tab.label,
                            status=st_cap, manufacturer_data=[asdict(r) for r in JOHANSON_R14S_RF_1GHZ],
                            extrapolated=JOHANSON_R14S_EXTRAPOLATED,
                            esr_ohm=esr, esl_nH=esl * 1e9, scalar_base_note="esr_ohm / esl_nH: extrapolated "
                            "values, used as the base of the scalar sensitivity grids",
                            esr_sensitivity_ohm=esr_set, esl_sensitivity_nH=[x * 1e9 for x in esl_set])),
            inductor=dict(model="(w L/Q + j w L) parallel C_par", q=q, q_status=st(given["q"]),
                          c_par_pF=cpar * 1e12, c_par_status=st(given["cpar"]),
                          q_sensitivity=list(Q_L_RANGE), c_par_sensitivity_pF=[c * 1e12 for c in C_PAR_RANGE if c > 0])),
        robust=dict(distribution="uniform within +- tolerance, independent", n_mc=n_mc, seed=seed,
                    f_offsets_pct=[100 * o for o in mt.F_OFFSETS_ROBUST],
                    statistic="mean (linear) and 10th percentile of G_r"),
        combinations_s11_le_m10dB=res.counts, combinations_total=res.counts_total,
        solutions=[dict(_sol_record(e, e["net"], i + 1), f_min_GHz=None if e["f_min"] is None else e["f_min"] / 1e9,
                        s11_min_dB=e["s11_min"]) for i, e in enumerate(top)],
        recommended_rank=tune_rank, sensitivity=sensitivity, esl_aware_rankings=esl_rank,
        comparison_6b=None if comp is None else {k: v for k, v in comp.items() if k != "net"},
        previous_solutions_reevaluated=prev,
        bench_tuning=tuning, minimum_spread_tolerances=dict(spread, rank=tune_rank),
        curves_top3_distinct=curves, figure=fig_path.name if figure else None,
        run_time_s=time.perf_counter() - t_start,
    )
    json_path = out_dir / f"{out_name}.json"
    json_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    log(f"  wrote {json_path}" + (f" and {fig_path.name}" if figure else "") + f" ({doc['run_time_s']:.1f} s)")
    doc["_top"] = top
    return doc


def _figure(path, version, f0, f_all, f_g, shown, top, comp, ant, ant_z, cap, ind, za, ga_db, gam, cap_text, q):
    import matplotlib
    if "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "lines.linewidth": 1.6, "legend.frameon": False,
                         "font.size": 9, "axes.titlesize": 10, "figure.dpi": 110})
    off, off_g = 100 * (f_all / f0 - 1), 100 * (f_g / f0 - 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for i, e in enumerate(shown):
        rs = mt.network_response(e["net"], f_all, ant_z, cap, ind)
        rg = mt.network_response(e["net"], f_g, ant, cap, ind)
        lab = f"#{top.index(e) + 1} {e['topology']}: {e['description']}"
        axes[0].plot(off, 20 * np.log10(np.abs(rs["s11"])), color=SERIES[i], label=lab)
        axes[1].plot(off_g, 10 * np.log10(rg["G_r"]), color=SERIES[i], label=lab)
    if comp is not None:
        n6 = comp["net"]
        rs = mt.network_response(n6, f_all, ant_z, cap, ind)
        rg = mt.network_response(n6, f_g, ant, cap, ind)
        lab = f"step 6b (MoM) network on {version}: {n6.describe()}"
        axes[0].plot(off, 20 * np.log10(np.abs(rs["s11"])), color=MUTED, ls="--", lw=1.3, label=lab)
        axes[1].plot(off_g, 10 * np.log10(rg["G_r"]), color=MUTED, ls="--", lw=1.3, label=lab)
    axes[1].plot(off_g, 10 * np.log10(ant.gain(f_g)), color=INK, ls=":", label="G_A: lossless conjugate match (upper bound)")
    axes[0].axhline(-10, color=MUTED, ls="--", lw=1)
    for ax in axes:
        ax.axvspan(-0.5, 0.5, color=GRID, alpha=0.6, lw=0)
        ax.set_xlabel("(f - f0) / f0 (%)")
    for ax in axes:
        ax.set_xlim(off[0], off[-1])
    axes[0].set_ylabel("|S11| (dB)")
    axes[1].set_ylabel("G_r (dBi), +y, phi")
    g0 = ga_db + 10 * np.log10(1 - abs(gam) ** 2)
    fig.suptitle(f"Matching network on the CST impedance, {version}, f0 = {f0 / 1e9:.2f} GHz "
                 f"(Z_CST(f0) = {za.real:.1f} {za.imag:+.1f}j Ohm; {cap_text}, Q_L = {q:g})\nshaded: f0 +- 0.5 %; without network G_r = {g0:.2f} dBi", fontsize=10, color=INK)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=1)
    fig.tight_layout(rect=(0, 0.27, 1, 1))
    fig.savefig(path, dpi=150)
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("s1p", help="CST Touchstone 1-port (port at the gap of el2)")
    ap.add_argument("--version", help="CST model version (V6a, V6b, V6c); default: the s1p folder name")
    ap.add_argument("--z-ref", type=float, help="reference impedance (Ohm) overriding the file's 'R' value")
    ap.add_argument("--gain-csv", help="CST G_A(f) towards +y: CSV f_GHz,G_dBi (default: shifted MoM G_A)")
    ap.add_argument("--gain-from-cuts", action="store_true",
                    help="build the gain CSV from the CST cuts theta = 90 deg next to the s1p (value at phi = 90 deg, "
                         "Abs(Phi)), write it to <out-dir>/gain_plus_y_from_cuts.csv and use it (no MoM shift)")
    ap.add_argument("--esr-ohm", type=float, help=f"capacitor ESR (default {DEFAULT_ESR_OHM}, step 6b, provisional)")
    ap.add_argument("--esl-nh", type=float, help=f"capacitor ESL in nH (default {DEFAULT_ESL_NH}, step 6b, provisional)")
    ap.add_argument("--cap-data", choices=["johanson"],
                    help="capacitor parasitics from manufacturer data (johanson: jtisoft R14S at 1 GHz for "
                         "0.5/1.0/1.2/2.7 pF, ESR 0.145 Ohm / ESL 0.3 nH extrapolated for the other values)")
    ap.add_argument("--out-name", help="output base name (default adaptation_from_cst[_johanson])")
    ap.add_argument("--previous", help="earlier adaptation JSON whose top-3 solutions are re-evaluated "
                                       "(default with --cap-data: <out-dir>/adaptation_from_cst.json)")
    ap.add_argument("--q-l", type=float, help=f"inductor Q at f0 (default {DEFAULT_Q_L}, middle of the 6b range, not data)")
    ap.add_argument("--c-par-pf", type=float, help=f"inductor parallel capacitance in pF (default {DEFAULT_C_PAR_PF})")
    ap.add_argument("--f0-ghz", type=float, default=1.0)
    ap.add_argument("--mom-csv", default=str(MOM_FREQ_CSV), help="MoM frequency table of the design")
    ap.add_argument("--ref-6b", default=str(REF_6B_JSON), help="step-6b matching JSON (comparison)")
    ap.add_argument("--out-dir", help="output folder (default: resultats_V6/<version>)")
    ap.add_argument("--n-mc", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-esl-rankings", action="store_true")
    ap.add_argument("--tune-rank", type=int, default=1, help="solution (rank) around which the bench-tuning "
                                                             "list and the tolerance spread are computed")
    ap.add_argument("--show", action="store_true", help="also display the figure")
    a = ap.parse_args(argv)
    version = a.version or infer_version(a.s1p)
    if not version:
        ap.error("cannot infer the CST version from the folder name: pass --version (V6a, V6b, V6c)")
    gain_csv = a.gain_csv
    if a.gain_from_cuts:
        if gain_csv:
            ap.error("--gain-from-cuts and --gain-csv are exclusive")
        out_dir = Path(a.out_dir) if a.out_dir else V6_DIR / version
        out_dir.mkdir(parents=True, exist_ok=True)
        gain_csv, rows = gain_csv_from_cuts(Path(a.s1p).parent, out_dir / "gain_plus_y_from_cuts.csv")
        print(f"gain CSV from {len(rows)} CST cuts: " + ", ".join(f"{r[0]:g} GHz {r[1]:.2f} dBi" for r in rows)
              + f" -> {gain_csv}")
    previous = a.previous
    if previous is None and a.cap_data:
        previous = Path(a.out_dir) if a.out_dir else V6_DIR / version
        previous = previous / "adaptation_from_cst.json"
    run(a.s1p, version, out_dir=a.out_dir, z_ref=a.z_ref, gain_csv=gain_csv, esr_ohm=a.esr_ohm,
        esl_nh=a.esl_nh, q_l=a.q_l, c_par_pf=a.c_par_pf, f0=a.f0_ghz * 1e9, mom_csv=a.mom_csv,
        ref_6b=a.ref_6b, n_mc=a.n_mc, seed=a.seed, esl_rankings=not a.skip_esl_rankings, show=a.show,
        cap_data=a.cap_data, out_name=a.out_name, previous=previous, tune_rank=a.tune_rank)


if __name__ == "__main__":
    main()
