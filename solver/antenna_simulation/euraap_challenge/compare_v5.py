"""
Compare the V5 MoM results (v5_mom_runs.py) with CST exports, as PNG figures.

Inputs
------
MoM : <mom_root>/<case>/metadata.json with results.csv (+ pattern_xy.csv) for V5a,
      resonance.csv + zin_fine.csv for V5b'
CST : <cst_root>, searched recursively and tolerant to naming:
      V5a*/          *.s1p (complex Zin via S11), Zin.txt (|Zin| or Re/Im),
                     *theta*90*0.9*.txt (far-field cut in the xy plane)
      V5b'/<mat>_w=<w>/  (tuned loop) *.s1p read with a 1 Ohm reference (the files say
                     "R 0": not renormalised, S relative to CST's 1 Ohm port), and
                     Rad_efficacity.txt / Tot_efficacity.txt (dB, 0.8/0.9/1.0 GHz),
                     interpolated in dB to CST's f_res. CST's loop has an OUTER side
                     of 20 mm ("cote"): only MoM cases with loop_geometry = "cst"
                     are compared when they exist.
      V5b/<mat>_w=<w>/   (untuned loop) loaded but NEVER compared: Re Zin is not
                     resolved there (|S11| ~ 0.9994), see the fiche
      cst_efficiency.csv (optional, hand-copied from the CST farfield results):
                     columns case (V5bp), w_mm, material, f_GHz, rad_eff [, tot_eff]
      Default cst_root: resultats_V5/cst if it exists, else resultats_V5 itself.

The ASCII readers skip every non-numeric line and take the frequency unit from
the header ("Frequency / GHz", "[MHz]", Touchstone "# GHz S MA R 50", ...). The
Touchstone reader accepts a user-imposed reference impedance (r_ref).

V5b' is compared on R_in and eta_rad at resonance, with the MoM capacitor
RE-TUNED to CST's f_res ('C_tuned' rows): the gaps and ports differ, so f_res is
not compared, and R_rad ~ f^4 forbids comparing R_in at different frequencies.

Figures (into <fig_dir>)
------------------------
    V5a_zin.png                Re/Im Zin(f), every MoM mesh vs CST
    V5a_pattern_xy.png         D_phi(phi) at 0.9 GHz, theta = 90 deg
    V5a_mesh_convergence.png   Re/Im Zin at 0.9 GHz vs N
    V5bp_rin_eta_vs_w.png      R_in and eta_rad at resonance vs w: MoM zs_scale 1 / 0.5 vs CST
    V5bp_mesh_convergence.png  R_in at resonance vs n_width, one panel per w
    V5bp_zin_w<w>mm.png        Re/Im Zin around CST's f_res: MoM (re-tuned) vs CST
    V5b_*.png                  legacy untuned-loop MoM plots (no CST overlay), if such runs exist

    python solver/antenna_simulation/euraap_challenge/compare_v5.py [--cst DIR] [--mom DIR] [--fig DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

BASE = Path(r"D:\PhD_Fahdilou_Leat\PUBLICATIONS\7-EurAAP Small Antenna Super-Gain Challenge"
            r"\fiches_cst\resultats_V5")
F_REF_GHZ = 0.9

#: CST V5b' (tuned loop, copper, 1 Ohm discrete face port): w_mm -> (f_res Hz, R_in Ohm)
CST_V5BP_REF = {1.0: (0.8912e9, 0.789), 2.0: (0.8888e9, 0.511), 4.0: (0.8845e9, 0.291)}
CST_V5BP_PORT_OHM = 1.0

_UNIT = {"thz": 1e12, "ghz": 1e9, "mhz": 1e6, "khz": 1e3, "hz": 1.0}
_NUM = re.compile(r"[\s,;]+")

# ---- Chart styling (reference palette of the dataviz method, light mode) ----
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # fixed categorical order
DENSITY = ["#86b6ef", "#3987e5", "#184f95"]                     # sequential blue, coarse -> fine
GRADING_STYLE = {"uniform": "-", "cosine": "--"}


# =============================================================================
# Tolerant readers
# =============================================================================
def read_numeric_table(path):
    """
    (header_lines, data) of an ASCII table: every line whose tokens all parse as
    floats is data (keeping the most common column count), anything else header.
    """
    header, rows = [], []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            toks = [t for t in _NUM.split(s) if t]
            try:
                rows.append([float(t) for t in toks])
            except ValueError:
                header.append(s)
    if not rows:
        raise ValueError(f"{path}: no numeric data.")
    width = Counter(len(r) for r in rows).most_common(1)[0][0]
    return header, np.array([r for r in rows if len(r) == width])


def frequency_scale(header_lines, default=1e9):
    """Multiplier to Hz from the first frequency unit found in the header."""
    text = " ".join(header_lines)
    m = re.search(r"freq[^\n]*?\b(THz|GHz|MHz|kHz|Hz)\b", text, re.I) \
        or re.search(r"\b(THz|GHz|MHz|kHz|Hz)\b", text, re.I)
    return _UNIT[m.group(1).lower()] if m else default


def read_touchstone_1port(path, r_ref=None):
    """
    (f_Hz, Zin) from a 1-port Touchstone file (S/Z/Y; MA/DB/RI; any unit).

    r_ref : reference impedance to use instead of the file's "R" value. Needed
            for CST files written with "R 0" ("S-Parameters are not renormalized"):
            S is then relative to the port impedance, which the file does not say.
    """
    unit, param, fmt, r_file = "ghz", "s", "ma", 50.0
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.split("!")[0].strip()
            if not s:
                continue
            if s.startswith("#"):
                toks = s[1:].lower().split()
                for i, t in enumerate(toks):
                    if t in _UNIT:
                        unit = t
                    elif t in ("s", "y", "z"):
                        param = t
                    elif t in ("ma", "db", "ri"):
                        fmt = t
                    elif t == "r" and i + 1 < len(toks):
                        r_file = float(toks[i + 1])
                continue
            try:
                vals = [float(t) for t in s.split()]
            except ValueError:
                continue
            if len(vals) >= 3:
                rows.append(vals[:3])
    if r_ref is None:
        if r_file <= 0:
            raise ValueError(f"{path}: the file's reference impedance is R {r_file:g} (CST writes "
                             f"'R 0' when S is not renormalised). Pass r_ref = the port "
                             f"impedance used in CST (1 Ohm for V5b').")
        r_ref = r_file
    a = np.array(rows)
    f = a[:, 0] * _UNIT[unit]
    if fmt == "ri":
        x = a[:, 1] + 1j * a[:, 2]
    else:
        mag = 10 ** (a[:, 1] / 20) if fmt == "db" else a[:, 1]
        x = mag * np.exp(1j * np.radians(a[:, 2]))
    if param == "s":
        z = r_ref * (1 + x) / (1 - x)
    elif param == "z":
        z = r_ref * x
    else:
        z = r_ref / x
    return f, z


def read_cst_zin(path):
    """(f_Hz, Zin) where Zin is complex if Re/Im columns exist, else |Zin| (real)."""
    header, d = read_numeric_table(path)
    f = d[:, 0] * frequency_scale(header)
    h = " ".join(header).lower()
    has_re = re.search(r"\bre\b|real", h) is not None
    has_im = re.search(r"\bim\b|imag", h) is not None
    if d.shape[1] >= 3 and has_re and has_im:
        return f, d[:, 1] + 1j * d[:, 2]
    return f, d[:, 1]


def read_cst_farfield(path):
    """{column name without spaces: values} for a CST far-field ASCII export."""
    header, d = read_numeric_table(path)
    names = None
    for line in header:
        found = re.findall(r"([A-Za-z.]+\s*(?:\([^)]*\))?)\s*\[[^\]]*\]", line)
        if len(found) >= d.shape[1]:
            names = [re.sub(r"\s+", "", n) for n in found[:d.shape[1]]]
            break
    if names is None:
        names = [f"col{i}" for i in range(d.shape[1])]
    return {n: d[:, i] for i, n in enumerate(names)}


def _find(root, patterns):
    for pat in patterns:
        hits = sorted(p for p in root.rglob(pat) if p.is_file())
        if hits:
            return hits[0]
    return None


def _loop_family(dirname):
    """'V5bp' for the tuned-loop folder (V5b', V5b′, V5bp, V5b_prime), 'V5b' for V5b, else None."""
    n = dirname.lower().replace("′", "'").replace("’", "'")
    if n in ("v5b'", "v5bp", "v5b_prime", "v5bprime"):
        return "V5bp"
    if n == "v5b":
        return "V5b"
    return None


def load_cst(cst_root, r_ref_v5bp=CST_V5BP_PORT_OHM):
    """
    {'V5a': {'zin', 'zin_mag', 'cut'},
     'V5b':  {(material, w_mm): {'zin', 'zin_mag'}}   (legacy, not compared),
     'V5bp': {(material, w_mm): {'zin', 'zin_mag'}}   (s1p read with r_ref_v5bp),
     'eff': rows of cst_efficiency.csv}
    """
    cst_root = Path(cst_root)
    out = {"V5a": {}, "V5b": {}, "V5bp": {}, "eff": []}
    v5a = next((p for p in sorted(cst_root.rglob("*")) if p.is_dir()
                and p.name.lower().startswith("v5a")), None)
    if v5a is not None:
        out["V5a"] = _load_cst_dir(v5a)
        cut = _find(v5a, ["*theta*90*0.9*.txt", "*cut*0.9*.txt"])
        out["V5a"]["cut"] = read_cst_farfield(cut) if cut else None
    for d in sorted(p for p in cst_root.rglob("*") if p.is_dir()):
        m = re.match(r"(?i)(pec|copper|cu)[_\- ]*w\s*=?\s*(\d+(?:[.,]\d+)?)", d.name)
        family = _loop_family(d.parent.name)
        if m and family:
            mat = "pec" if m.group(1).lower() == "pec" else "cu"
            key = (mat, float(m.group(2).replace(",", ".")))
            out[family][key] = _load_cst_dir(d, r_ref=r_ref_v5bp if family == "V5bp" else None)
    eff = cst_root / "cst_efficiency.csv"
    if eff.is_file():
        with open(eff, newline="", encoding="utf-8") as fh:
            out["eff"] = list(csv.DictReader(fh))
    return out


def read_cst_efficiency(path):
    """(f_Hz, efficiency as a linear ratio) from a CST efficiency export (dB or linear)."""
    header, d = read_numeric_table(path)
    f = d[:, 0] * frequency_scale(header)
    v = d[:, 1]
    is_db = re.search(r"\bdB\b", " ".join(header)) is not None
    return f, (10 ** (v / 10) if is_db else v)


def _load_cst_dir(d, r_ref=None):
    res = {"zin": None, "zin_mag": None, "rad_eff": None, "tot_eff": None}
    for key, pats in (("rad_eff", ["*Rad*effic*.txt"]), ("tot_eff", ["*Tot*effic*.txt"])):
        p = _find(d, pats)
        if p:
            res[key] = read_cst_efficiency(p)
    s1p = _find(d, ["*.s1p"])
    if s1p:
        res["zin"] = read_touchstone_1port(s1p, r_ref=r_ref)
    zt = _find(d, ["Zin*.txt", "*Z1,1*.txt", "*zin*.txt"])
    if zt:
        f, z = read_cst_zin(zt)
        if np.iscomplexobj(z) and res["zin"] is None:
            res["zin"] = (f, z)
        else:
            res["zin_mag"] = (f, np.abs(z))
    return res


def load_mom(mom_root):
    """
    List of cases: dict(meta, data={column: array} or None, pattern or None,
    resonance=list of row dicts or None, fine={column: array} or None).
    """
    cases = []
    for meta_path in sorted(Path(mom_root).glob("*/metadata.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not meta.get("complete", False):
            continue
        d = meta_path.parent
        opt = lambda name, fn: fn(d / name) if (d / name).is_file() else None   # noqa: E731
        cases.append(dict(meta=meta, data=opt("results.csv", _read_csv_columns),
                          pattern=opt("pattern_xy.csv", _read_csv_columns),
                          resonance=opt("resonance.csv", _read_csv_rows),
                          fine=opt("zin_fine.csv", _read_csv_columns)))
    return cases


def _to_num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return s


def _read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return [{k: _to_num(v) for k, v in r.items()} for r in csv.DictReader(fh)]


def _read_csv_columns(path):
    rows = _read_csv_rows(path)
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


# =============================================================================
# Figures
# =============================================================================
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.6, "legend.frameon": False, "legend.fontsize": 8,
        "axes.titlesize": 10, "axes.labelsize": 9, "font.family": "sans-serif",
    })
    return plt


def _save(fig, path, legend_ax=None, ncol=2):
    """
    Save; with legend_ax, one figure-level legend below the plots (never over data),
    gathering the entries of every panel (legend_ax first), each label once.
    """
    if legend_ax is not None:
        handles, labels = [], []
        for ax in [legend_ax] + [a for a in fig.axes if a is not legend_ax]:
            for h, lab in zip(*ax.get_legend_handles_labels()):
                if lab not in labels:
                    handles.append(h)
                    labels.append(lab)
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        rows = int(np.ceil(len(labels) / ncol))
        frac = min(0.45, (0.035 * rows + 0.02) * 6.0 / fig.get_size_inches()[1])
        fig.legend(handles, labels, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=(0, frac, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def _at(data, col, f_ghz=F_REF_GHZ):
    i = int(np.argmin(np.abs(data["freq_GHz"] - f_ghz)))
    return data[col][i]


def _label(meta):
    p = meta["parameters"]
    nl = p.get("n_len", p.get("n_side"))
    return f"MoM n_w={p['n_width']}, n_l={nl}, {meta['grading']} (N={meta['n_edges']})"


def _density_color(meta):
    return DENSITY[min(int(meta.get("level", 0)), len(DENSITY) - 1)]


def _finest(cases):
    top = max(c["meta"]["level"] for c in cases)
    return [c for c in cases if c["meta"]["level"] == top]


def plot_v5a(mom, cst, fig_dir):
    plt = _plt()
    out = []
    cases = [c for c in mom if c["meta"]["kind"] == "V5a"]
    c5 = cst.get("V5a", {})
    if not cases and not c5:
        return out

    has_mag = c5.get("zin") is None and c5.get("zin_mag") is not None
    fig, axes = plt.subplots(3 if has_mag else 2, 1, figsize=(7, 7 if has_mag else 5.5),
                             sharex=True)
    for c in cases:
        d, m = c["data"], c["meta"]
        kw = dict(color=_density_color(m), ls=GRADING_STYLE.get(m["grading"], "-"), label=_label(m))
        axes[0].plot(d["freq_GHz"], d["Re_Zin_ohm"], **kw)
        axes[1].plot(d["freq_GHz"], d["Im_Zin_ohm"], **kw)
        if has_mag:
            axes[2].plot(d["freq_GHz"], np.hypot(d["Re_Zin_ohm"], d["Im_Zin_ohm"]), **kw)
    if c5.get("zin") is not None:
        f, z = c5["zin"]
        axes[0].plot(f / 1e9, z.real, color=INK, lw=1.2, label="CST (from S11)")
        axes[1].plot(f / 1e9, z.imag, color=INK, lw=1.2, label="CST (from S11)")
    if has_mag:
        f, zm = c5["zin_mag"]
        axes[2].plot(f / 1e9, zm, color=INK, lw=1.2, label="CST |Zin|")
        axes[2].set_ylabel("|Zin| (Ohm)")
    axes[0].set_ylabel("Re Zin (Ohm)")
    axes[1].set_ylabel("Im Zin (Ohm)")
    axes[-1].set_xlabel("Frequency (GHz)")
    axes[0].set_title("V5a strip dipole 150 x 4 mm, PEC: input impedance")
    out.append(_save(fig, fig_dir / "V5a_zin.png", legend_ax=axes[0]))

    pats = [c for c in cases if c["pattern"] is not None]
    if pats or (c5.get("cut") is not None):
        fig, ax = plt.subplots(figsize=(7, 3.8))
        peak = []
        for c in _finest(pats) if pats else []:
            p, m = c["pattern"], c["meta"]
            ax.plot(p["phi_deg"], p["D_phi_dBi"], color=_density_color(m),
                    ls=GRADING_STYLE.get(m["grading"], "-"), label=_label(m))
            peak.append(np.max(p["D_phi_dBi"]))
        cut = c5.get("cut")
        if cut is not None:
            phi = cut.get("Phi", cut.get("col1"))
            col = cut.get("Abs(Phi)", cut.get("Abs(Dir.)", cut.get("col2")))
            ax.plot(phi, col, color=INK, lw=1.2, label="CST")
            peak.append(np.max(col))
        top = max(peak) if peak else 0
        ax.set_ylim(top - 40, top + 3)
        ax.set_xlim(0, 360)
        ax.set_xticks(range(0, 361, 45))
        ax.set_xlabel("phi (deg), theta = 90 deg")
        ax.set_ylabel("D_phi (dBi)")
        ax.set_title(f"V5a: directivity D_phi in the xy plane at {F_REF_GHZ} GHz")
        out.append(_save(fig, fig_dir / "V5a_pattern_xy.png", legend_ax=ax))

    if cases:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
        for g in sorted({c["meta"]["grading"] for c in cases}):
            sub = sorted((c for c in cases if c["meta"]["grading"] == g), key=lambda c: c["meta"]["n_edges"])
            N = [c["meta"]["n_edges"] for c in sub]
            for ax, col in zip(axes, ("Re_Zin_ohm", "Im_Zin_ohm")):
                ax.plot(N, [_at(c["data"], col) for c in sub], marker="o", ms=5,
                        color=SERIES[0], ls=GRADING_STYLE.get(g, "-"), label=f"MoM, {g}")
        if c5.get("zin") is not None:
            f, z = c5["zin"]
            zr = z[int(np.argmin(np.abs(f / 1e9 - F_REF_GHZ)))]
            for ax, v in zip(axes, (zr.real, zr.imag)):
                ax.axhline(v, color=INK, lw=1.0, label="CST")
        axes[0].set_ylabel(f"Re Zin at {F_REF_GHZ} GHz (Ohm)")
        axes[1].set_ylabel(f"Im Zin at {F_REF_GHZ} GHz (Ohm)")
        for ax in axes:
            ax.set_xlabel("RWG edges N")
        fig.suptitle("V5a: mesh convergence", fontsize=10, color=INK)
        out.append(_save(fig, fig_dir / "V5a_mesh_convergence.png", legend_ax=axes[0], ncol=3))
    return out


def _w_mm(meta):
    return round(meta["parameters"]["width"] * 1e3, 6)


def _cst_eta(cst, w, f_ghz=F_REF_GHZ):
    """CST eta_rad at width w: hand-copied value if present, else Re Z_PEC / Re Z_Cu."""
    for r in cst.get("eff", []):
        try:
            if (abs(float(r["w_mm"]) - w) < 1e-9 and r["material"].lower() in ("cu", "copper")
                    and abs(float(r["f_GHz"]) - f_ghz) < 1e-6):
                return float(r["rad_eff"]), "CST (logged)"
        except (KeyError, ValueError):
            continue
    pec, cu = cst["V5b"].get(("pec", w)), cst["V5b"].get(("cu", w))
    if pec and cu and pec["zin"] is not None and cu["zin"] is not None:
        zp = _interp(pec["zin"], f_ghz)
        zc = _interp(cu["zin"], f_ghz)
        return zp.real / zc.real, "CST (Re Z_PEC / Re Z_Cu)"
    return None, None


def _interp(fz, f_ghz):
    f, z = fz
    return np.interp(f_ghz * 1e9, f, z.real) + 1j * np.interp(f_ghz * 1e9, f, z.imag)


def plot_v5b(mom, cst, fig_dir):
    plt = _plt()
    out = []
    cases = [c for c in mom if c["meta"]["kind"] == "V5b"]
    if not cases:
        return out
    # The untuned-loop CST exports do not resolve Re Zin (|S11| ~ 0.9994): never overlay them.
    cst = dict(cst, V5b={}, eff=[])
    widths = sorted({_w_mm(c["meta"]) for c in cases})
    if not widths:
        return out
    zs_tags = sorted({k.split("_")[2] for c in cases for k in c["data"] if k.startswith("eta_rad_")},
                     reverse=True)                                    # e.g. ['zs1', 'zs0.5']

    for w in widths:
        fig, axes = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)
        same_w = [c for c in cases if _w_mm(c["meta"]) == w]
        for c in (_finest(same_w) if same_w else []):
            d, m = c["data"], c["meta"]
            ls = GRADING_STYLE.get(m["grading"], "-")
            axes[0].plot(d["freq_GHz"], d["Re_Zin_PEC_ohm"], color=MUTED, ls=ls,
                         label=f"MoM PEC, {m['grading']} (N={m['n_edges']})")
            axes[1].plot(d["freq_GHz"], d["Im_Zin_PEC_ohm"], color=MUTED, ls=ls,
                         label=f"MoM PEC, {m['grading']}")
            for k, tag in enumerate(zs_tags):
                kw = dict(color=SERIES[k], ls=ls, label=f"MoM Cu {tag.replace('zs', 'zs_scale=')}, {m['grading']}")
                axes[0].plot(d["freq_GHz"], d[f"Re_Zin_Cu_{tag}_ohm"], **kw)
                axes[1].plot(d["freq_GHz"], d[f"Im_Zin_Cu_{tag}_ohm"], **kw)
        for mat, mk in (("pec", ":"), ("cu", "-")):
            entry = cst["V5b"].get((mat, w))
            if entry and entry["zin"] is not None:
                f, z = entry["zin"]
                lab = f"CST {'PEC' if mat == 'pec' else 'Cu'}"
                axes[0].plot(f / 1e9, z.real, color=INK, ls=mk, lw=1.2, label=lab)
                axes[1].plot(f / 1e9, z.imag, color=INK, ls=mk, lw=1.2, label=lab)
        axes[0].set_ylabel("Re Zin (Ohm)")
        axes[1].set_ylabel("Im Zin (Ohm)")
        axes[1].set_xlabel("Frequency (GHz)")
        axes[0].set_title(f"V5b square loop, w = {w:g} mm: input impedance (finest MoM mesh)")
        out.append(_save(fig, fig_dir / f"V5b_zin_w{w:g}mm.png", legend_ax=axes[0]))

    # eta_rad vs w
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for g in sorted({c["meta"]["grading"] for c in cases}):
        for k, tag in enumerate(zs_tags):
            pts = []
            for w in widths:
                sub = [c for c in cases if _w_mm(c["meta"]) == w and c["meta"]["grading"] == g]
                if sub:
                    c = max(sub, key=lambda c: c["meta"]["level"])
                    pts.append((w, _at(c["data"], f"eta_rad_{tag}")))
            if pts:
                x, y = zip(*pts)
                ax.plot(x, y, marker="o", ms=6, color=SERIES[k], ls=GRADING_STYLE.get(g, "-"),
                        label=f"MoM {tag.replace('zs', 'zs_scale=')}, {g}")
    cpts = [(w, *_cst_eta(cst, w)) for w in widths]
    cpts = [(w, e, lab) for w, e, lab in cpts if e is not None]
    if cpts:
        ax.plot([p[0] for p in cpts], [p[1] for p in cpts], marker="s", ms=6, color=INK, lw=1.2,
                label=cpts[0][2])
    ax.set_xlabel("strip width w (mm)")
    ax.set_ylabel(f"eta_rad at {F_REF_GHZ} GHz")
    ax.set_ylim(0, 1)
    ax.set_title("V5b: radiation efficiency of the copper loop vs strip width")
    out.append(_save(fig, fig_dir / "V5b_eta_vs_w.png", legend_ax=ax))

    # mesh convergence
    if cases and zs_tags:
        tag = zs_tags[0]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
        for k, w in enumerate(sorted({_w_mm(c["meta"]) for c in cases})):
            for g in sorted({c["meta"]["grading"] for c in cases}):
                sub = sorted((c for c in cases if _w_mm(c["meta"]) == w and c["meta"]["grading"] == g),
                             key=lambda c: c["meta"]["n_edges"])
                if not sub:
                    continue
                N = [c["meta"]["n_edges"] for c in sub]
                kw = dict(marker="o", ms=5, color=SERIES[k % len(SERIES)], ls=GRADING_STYLE.get(g, "-"),
                          label=f"w = {w:g} mm, {g}")
                axes[0].plot(N, [_at(c["data"], f"R_loss_{tag}_ohm") for c in sub], **kw)
                axes[1].plot(N, [_at(c["data"], f"eta_rad_{tag}") for c in sub], **kw)
        axes[0].set_ylabel(f"R_loss at {F_REF_GHZ} GHz (Ohm), {tag.replace('zs', 'zs_scale=')}")
        axes[1].set_ylabel(f"eta_rad at {F_REF_GHZ} GHz, {tag.replace('zs', 'zs_scale=')}")
        for ax in axes:
            ax.set_xscale("log")
            ax.set_xlabel("RWG edges N")
        fig.suptitle("V5b: mesh convergence (edge grading should converge faster)", fontsize=10, color=INK)
        out.append(_save(fig, fig_dir / "V5b_mesh_convergence.png", legend_ax=axes[1], ncol=3))
    return out


def cst_resonance(fz):
    """Rising zero of Im Zin (series resonance) from CST data: (f_res Hz, R_in Ohm) or None."""
    f, z = fz
    X = z.imag
    k = np.flatnonzero((X[:-1] <= 0) & (X[1:] > 0))
    if k.size == 0:
        return None
    k = int(k[np.argmax(np.abs(X[k + 1] - X[k]))])          # steepest crossing = the resonance
    t = -X[k] / (X[k + 1] - X[k])
    return f[k] + t * (f[k + 1] - f[k]), z.real[k] + t * (z.real[k + 1] - z.real[k])


def _bp_row(case, mode, variant):
    for r in case["resonance"] or []:
        if r.get("mode") == mode and r.get("variant") == variant:
            return r
    return None


def cst_rad_efficiency_v5bp(cst, w, f_hz):
    """CST copper radiation efficiency at f_hz (interpolated in dB between the exported
    frequencies), from Rad_efficacity.txt; None if absent."""
    d = cst.get("V5bp", {}).get(("cu", w))
    if not d or d.get("rad_eff") is None:
        return None
    f, eta = d["rad_eff"]
    return float(10 ** (np.interp(f_hz, f, 10 * np.log10(eta)) / 10))


def _cst_eff_v5bp(cst, w):
    """Hand-copied cst_efficiency.csv value (legacy route), else None."""
    for r in cst.get("eff", []):
        try:
            if (str(r.get("case", "V5bp")).lower() in ("v5bp", "v5b'", "v5b′")
                    and abs(float(r["w_mm"]) - w) < 1e-9 and r["material"].lower() in ("cu", "copper")):
                return float(r["rad_eff"])
        except (KeyError, ValueError):
            continue
    return None


def cst_v5bp_resonances(cst):
    """{(material, w): (f_res, R_in)} from the CST V5b' s1p files (1 Ohm reference)."""
    out = {}
    for key, d in cst.get("V5bp", {}).items():
        if d.get("zin") is not None:
            r = cst_resonance(d["zin"])
            if r is not None:
                out[key] = r
    return out


def cst_eta_estimate(rec, w):
    """
    CST eta_rad at w from the PEC and copper resonances alone:
        eta ~ R_rad / R_in,Cu,  R_rad = R_in,PEC * (f_Cu / f_PEC)^4
    (small-loop R_rad ~ f^4 brings the PEC value to the copper resonance).
    """
    cu, pec = rec.get(("cu", w)), rec.get(("pec", w))
    if not (cu and pec):
        return None
    return pec[1] * (cu[0] / pec[0]) ** 4 / cu[1]


def plot_v5bp(mom, cst, fig_dir, log=print):
    plt = _plt()
    out = []
    cases = [c for c in mom if c["meta"]["kind"] == "V5bp" and c["resonance"]]
    # CST's loop is "cote = 20 mm" OUTER side: compare with the MoM cases built that way.
    cst_geom = [c for c in cases if c["meta"]["parameters"].get("loop_geometry") == "cst"]
    if cst_geom:
        cases = cst_geom
    elif cases:
        log("WARNING: no V5b' MoM case with loop_geometry='cst' (outer side 20 mm, as CST): "
            "comparing other geometries.")
    widths = sorted({_w_mm(c["meta"]) for c in cases} | {w for (_, w) in cst.get("V5bp", {})})
    if not widths:
        return out
    variants = [("Cu_zs1", "MoM Cu, zs_scale=1", SERIES[0]),
                ("Cu_zs0.5", "MoM Cu, zs_scale=0.5", SERIES[1]),
                ("PEC", "MoM PEC", MUTED)]
    gradings = sorted({c["meta"]["grading"] for c in cases})
    rec = cst_v5bp_resonances(cst)

    # --- table: CST (reference + recomputed from s1p) vs MoM finest, re-tuned capacitor
    log("V5b' at resonance (MoM capacitor re-tuned to CST f_res):")
    log(f"  {'w':>4s} {'CST f_res':>10s} {'CST R_Cu':>9s} {'s1p R_Cu':>9s} {'s1p R_PEC':>9s} "
        f"{'CST eta':>8s}  " + "  ".join(f"{lab.replace('MoM ', ''):>22s}" for _, lab, _ in variants))
    for w in widths:
        ref = CST_V5BP_REF.get(w, (np.nan, np.nan))
        cu, pec = rec.get(("cu", w)), rec.get(("pec", w))
        eta_c = cst_rad_efficiency_v5bp(cst, w, ref[0]) if np.isfinite(ref[0]) else None
        if eta_c is None:
            eta_c = cst_eta_estimate(rec, w)
        sub = [c for c in cases if _w_mm(c["meta"]) == w]
        fin = max(sub, key=lambda c: (c["meta"]["level"], c["meta"]["grading"] == "cosine")) if sub else None
        cells = []
        for v, _, _ in variants:
            r = _bp_row(fin, "C_tuned", v) if fin else None
            cells.append(f"{r['R_in_exact_ohm']:.4f} Ohm, eta {r['eta_rad']:.3f}" if r else "-")
        log(f"  {w:4g} {ref[0] / 1e9:10.4f} {ref[1]:9.3f} {(cu[1] if cu else np.nan):9.4f} "
            f"{(pec[1] if pec else np.nan):9.4f} {(eta_c if eta_c else np.nan):8.4f}  "
            + "  ".join(f"{c:>22s}" for c in cells))

    # --- R_in and eta vs w (finest mesh per grading)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for v, lab, col in variants:
        for g in gradings:
            pts = []
            for w in widths:
                sub = [c for c in cases if _w_mm(c["meta"]) == w and c["meta"]["grading"] == g]
                if sub:
                    r = _bp_row(max(sub, key=lambda c: c["meta"]["level"]), "C_tuned", v)
                    if r and np.isfinite(r.get("f_res_GHz", np.nan)):
                        pts.append((w, r["R_in_exact_ohm"], r["eta_rad"]))
            if pts:
                x, rin, eta = zip(*pts)
                kw = dict(marker="o", ms=6, color=col, ls=GRADING_STYLE.get(g, "-"), label=f"{lab}, {g}")
                axes[0].plot(x, rin, **kw)
                if v != "PEC":
                    axes[1].plot(x, eta, **kw)
    cw = [w for w in widths if w in CST_V5BP_REF]
    if cw:
        axes[0].plot(cw, [CST_V5BP_REF[w][1] for w in cw], marker="s", ms=6, color=INK, lw=1.2,
                     label="CST copper (reference values)")
    cp = [(w, rec[("pec", w)][1]) for w in widths if ("pec", w) in rec]
    if cp:
        axes[0].plot(*zip(*cp), marker="s", ms=6, color=INK, ls=":", lw=1.2, mfc=SURFACE,
                     label="CST PEC (from s1p)")
    ce = [(w, _cst_eff_v5bp(cst, w)) for w in widths]
    ce = [(w, e) for w, e in ce if e is not None]
    if ce:
        axes[1].plot(*zip(*ce), marker="s", ms=6, color=INK, lw=1.2, label="CST copper (logged)")
    ef = [(w, cst_rad_efficiency_v5bp(cst, w, CST_V5BP_REF[w][0])) for w in widths if w in CST_V5BP_REF]
    ef = [(w, e) for w, e in ef if e is not None]
    if ef:
        axes[1].plot(*zip(*ef), marker="s", ms=6, color=INK, lw=1.2,
                     label="CST copper, Rad. efficiency at f_res")
    else:
        ee = [(w, cst_eta_estimate(rec, w)) for w in widths]
        ee = [(w, e) for w, e in ee if e is not None]
        if ee:
            axes[1].plot(*zip(*ee), marker="s", ms=6, color=INK, ls=":", lw=1.2, mfc=SURFACE,
                         label="CST est. R_PEC (f_Cu/f_PEC)^4 / R_Cu")
    axes[0].set_ylabel("R_in at resonance (Ohm)")
    axes[1].set_ylabel("eta_rad at resonance")
    axes[1].set_ylim(0, 1)
    for ax in axes:
        ax.set_xlabel("strip width w (mm)")
    fig.suptitle("V5b' tuned loop: MoM capacitor re-tuned to CST f_res (same frequency)",
                 fontsize=10, color=INK)
    out.append(_save(fig, fig_dir / "V5bp_rin_eta_vs_w.png", legend_ax=axes[0], ncol=3))

    # --- mesh convergence: one panel per width
    wm = sorted({_w_mm(c["meta"]) for c in cases})
    if wm:
        fig, axes = plt.subplots(1, len(wm), figsize=(max(7.0, 4.2 * len(wm)), 3.8), squeeze=False)
        n_ticks = sorted({c["meta"]["parameters"]["n_width"] for c in cases})
        for ax, w in zip(axes[0], wm):
            for v, lab, col in variants[:2]:
                for g in gradings:
                    sub = sorted((c for c in cases if _w_mm(c["meta"]) == w and c["meta"]["grading"] == g),
                                 key=lambda c: c["meta"]["parameters"]["n_width"])
                    pts = [(c["meta"]["parameters"]["n_width"], _bp_row(c, "C_tuned", v)) for c in sub]
                    pts = [(n, r["R_in_exact_ohm"]) for n, r in pts if r and np.isfinite(r.get("f_res_GHz", np.nan))]
                    if pts:
                        ax.plot(*zip(*pts), marker="o", ms=6, color=col, ls=GRADING_STYLE.get(g, "-"),
                                label=f"{lab}, {g}")
            if w in CST_V5BP_REF:
                ax.axhline(CST_V5BP_REF[w][1], color=INK, lw=1.0, label="CST copper")
            ax.set_xscale("log", base=2)
            ax.set_xticks(n_ticks)
            ax.set_xticklabels([str(n) for n in n_ticks])
            ax.minorticks_off()
            ax.set_xlabel("cells across the width, n_width")
            ax.set_title(f"w = {w:g} mm", fontsize=9)
        axes[0][0].set_ylabel("R_in at resonance (Ohm)")
        fig.suptitle("V5b': convergence of R_in with the width discretisation", fontsize=10, color=INK)
        out.append(_save(fig, fig_dir / "V5bp_mesh_convergence.png", legend_ax=axes[0][0], ncol=2))

    # --- Zin around CST's f_res
    for w in widths:
        sub = [c for c in cases if _w_mm(c["meta"]) == w and c["fine"] is not None]
        s1p = cst.get("V5bp", {}).get(("cu", w))
        if not sub and not s1p:
            continue
        f0 = CST_V5BP_REF.get(w, (F_REF_GHZ * 1e9,))[0] / 1e9
        fig, axes = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)
        if sub:
            fin = max(sub, key=lambda c: (c["meta"]["level"], c["meta"]["grading"] == "cosine"))
            d = fin["fine"]
            sel = np.abs(d["freq_GHz"] - f0) <= 0.02
            for v, lab, col in variants[:2]:
                key = f"Re_Zin_C_tuned_{v}_ohm"
                if key in d:
                    axes[0].plot(d["freq_GHz"][sel], d[key][sel], color=col, label=f"{lab} (N={fin['meta']['n_edges']})")
                    axes[1].plot(d["freq_GHz"][sel], d[f"Im_Zin_C_tuned_{v}_ohm"][sel], color=col,
                                 label=f"{lab} (N={fin['meta']['n_edges']})")
        if s1p and s1p["zin"] is not None:
            f, z = s1p["zin"]
            sel = np.abs(f / 1e9 - f0) <= 0.02
            axes[0].plot(f[sel] / 1e9, z.real[sel], color=INK, lw=1.2, label="CST copper (1 Ohm port)")
            axes[1].plot(f[sel] / 1e9, z.imag[sel], color=INK, lw=1.2, label="CST copper (1 Ohm port)")
        axes[0].set_ylabel("Re Zin (Ohm)")
        axes[1].set_ylabel("Im Zin (Ohm)")
        axes[1].set_xlabel("Frequency (GHz)")
        axes[0].set_title(f"V5b' tuned loop, w = {w:g} mm: input impedance near CST f_res")
        out.append(_save(fig, fig_dir / f"V5bp_zin_w{w:g}mm.png", legend_ax=axes[0]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Overlay V5 MoM results and CST exports.")
    default_cst = BASE / "cst" if (BASE / "cst").is_dir() else BASE
    ap.add_argument("--cst", default=str(default_cst))
    ap.add_argument("--mom", default=str(BASE / "mom"))
    ap.add_argument("--fig", default=str(BASE / "figures"))
    args = ap.parse_args(argv)

    fig_dir = Path(args.fig)
    fig_dir.mkdir(parents=True, exist_ok=True)
    cst = load_cst(args.cst)
    mom = load_mom(args.mom) if Path(args.mom).is_dir() else []
    print(f"MoM cases: {len(mom)}; CST: V5a {'found' if cst['V5a'] else 'missing'}, "
          f"V5b' entries {sorted(cst['V5bp'])} (legacy V5b loaded, not compared)")
    figs = plot_v5a(mom, cst, fig_dir) + plot_v5bp(mom, cst, fig_dir) + plot_v5b(mom, cst, fig_dir)
    for f in figs:
        print(f"  wrote {f}")
    return figs


if __name__ == "__main__":
    main()
