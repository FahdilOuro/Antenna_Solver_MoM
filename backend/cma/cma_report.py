"""
Markdown data sheets for Characteristic Mode Analysis runs.

A CMA run leaves behind a .mat holding eigenvalues and eigenvectors. That is
enough to re-plot, but not enough to READ months later: it carries no record of
which mesh produced it, at what electrical size, with which solver settings, or
whether the impedance matrix behind it was numerically sound.

These reports write all of that down, together with the per-mode quantities
derived from the eigenvalues (modal significance, characteristic angle), in a
form a human or a later analysis can work from.

They deliberately stop at recording data and defining the quantities. They do
NOT decide where the antenna resonates, how good it is, or which mode to feed:
those readings depend on context a script does not have, and the obvious
automations are traps -- a sign change in lambda, for instance, is a resonance
when lambda passes through zero but a POLE when it runs off through infinity,
and the two look identical to a naive scan while meaning opposite things about
whether the mode can be excited.
"""

import os
from datetime import datetime

import numpy as np

from backend.cma.modal_significance import modal_significance
from backend.cma.characteristic_angle import characteristic_angle
from backend.cma.z_matrix_diagnostics import (
    check_reciprocity, check_r_positive_semidefinite, RECIPROCITY_TOL,
)

# Speed of light, used only for the wavelength / electrical-size rows.
_C0 = 299792458.0


def _as_posix(p):
    """Forward slashes, so a Windows-generated report reads the same everywhere."""
    return str(p).replace(os.sep, "/")


def _fmt_freq(f_hz):
    """Frequency with a unit that keeps the number readable."""
    if f_hz >= 1e9:
        return f"{f_hz / 1e9:.4f} GHz"
    if f_hz >= 1e6:
        return f"{f_hz / 1e6:.3f} MHz"
    if f_hz >= 1e3:
        return f"{f_hz / 1e3:.3f} kHz"
    return f"{f_hz:.3f} Hz"


def _geometry_rows(points, triangles, edges):
    """Mesh facts that bound what the results can mean."""
    p = points.points
    spans = p.max(axis=1) - p.min(axis=1)
    edge_lengths = np.asarray(edges.edges_length, dtype=float)
    return {
        'spans': spans,
        'max_dim': float(np.max(spans)),
        'sphere_radius': 0.5 * float(np.sqrt(np.sum(spans ** 2))),
        'mean_edge': float(np.mean(edge_lengths)),
        'min_edge': float(np.min(edge_lengths)),
        'max_edge': float(np.max(edge_lengths)),
        'rows': [
            "| Quantity | Value |",
            "|---|---|",
            f"| Points | {points.total_of_points} |",
            f"| Triangles | {triangles.total_of_triangles} |",
            f"| RWG edges (unknowns) | {edges.total_number_of_edges} |",
            f"| Bounding box | {spans[0]:.4f} x {spans[1]:.4f} x {spans[2]:.4f} m |",
            f"| Edge length min / mean / max | {np.min(edge_lengths) * 1e3:.2f} / "
            f"{np.mean(edge_lengths) * 1e3:.2f} / {np.max(edge_lengths) * 1e3:.2f} mm |",
        ],
    }


def _electrical_size_rows(geom, frequency, label=None):
    """Electrical size and mesh density, both stated as plain numbers."""
    wavelength = _C0 / frequency
    heading = f"Frequency ({label})" if label else "Frequency"
    return [
        f"| {heading} | {_fmt_freq(frequency)} |",
        f"| Wavelength | {wavelength:.4f} m |",
        f"| Largest dimension / wavelength | {geom['max_dim'] / wavelength:.3f} |",
        f"| Electrical size ka (enclosing sphere) | {2.0 * np.pi * geom['sphere_radius'] / wavelength:.3f} |",
        f"| RWG edges per wavelength (mean) | {wavelength / geom['mean_edge']:.1f} |",
    ]


def _health_rows(Z):
    """
    Measured properties of the impedance matrix the modes came from, with the
    reference values next to them so they can be judged without hunting through
    the source.
    """
    reciprocity = check_reciprocity(Z)
    psd = check_r_positive_semidefinite(Z)
    eig = psd['eigenvalues']
    rank = int(np.sum(eig > 1e-8 * eig[-1]))
    return [
        "| Measurement | Value | Reference |",
        "|---|---|---|",
        f"| Reciprocity, max\\|Z-Z^T\\|/max\\|Z\\| | {reciprocity['relative_asymmetry']:.3e} | "
        f"expected ~1e-4; above {RECIPROCITY_TOL:.0e} is suspect |",
        f"| min eig(R) | {eig[0]:.4e} Ohm | should be >= 0 (R is the radiation operator) |",
        f"| max eig(R) | {eig[-1]:.4e} Ohm | - |",
        f"| min eig(R) / max eig(R) | {eig[0] / eig[-1]:.3e} | negative only at roundoff level |",
        f"| Numerical rank of R (> 1e-8 x max) | {rank} of {len(eig)} | "
        f"expected << N: only a few current distributions radiate |",
        "",
        "Reciprocity reads exactly `0.000e+00` here because the assembler computes",
        "only the `m <= n` half of Z and mirrors it, so `Z = Z^T` holds by",
        "construction rather than by agreement. It is therefore a check that the",
        "mirroring ran, not a check that the kernel is right -- a wrong kernel would",
        "read `0` too. The independent version (both halves computed separately,",
        "`impedance_matrix_z(..., full_fill=True)`) is what still detects a broken",
        "integration scheme; on a correct assembly it lands near 1e-4.",
    ]


_LEGEND = """
## Definitions

Characteristic Mode Analysis decomposes what a structure can support on its
own, independently of any feed. Each mode is a current pattern; an excitation
only decides how much of each one is stirred up.

Three quantities describe a mode, all carrying the same information:

| Symbol | Definition | Reference values |
|---|---|---|
| `lambda_n` | characteristic value, from `X J = lambda R J` | `0` at resonance; `> 0` inductive (stores magnetic energy); `< 0` capacitive (stores electric energy); unbounded |
| `MS_n` | modal significance, `1 / sqrt(1 + lambda^2)` | `1` at resonance, `0` for an unexcitable mode; `>= 0.707` (i.e. `\\|lambda\\| <= 1`) is the usual "significant" threshold |
| `delta_n` | characteristic angle, `180 deg - atan(lambda)` | `180 deg` at resonance; `-> 90 deg` strongly inductive; `-> 270 deg` strongly capacitive |

Other reference values used above:

- **RWG edges per wavelength**: below roughly 10, the discretisation rather
  than the geometry starts to set the higher-order results.
- **ka**: electrical size of the enclosing sphere. Well below 1 the structure
  is electrically small, and every `|lambda|` is large.
"""

_SWEEP_CAVEAT = """
## Caveat on mode numbering

At each frequency the modes are sorted independently by `|lambda|`; no attempt
is made to follow a given physical mode across the sweep. Where two eigenvalue
curves cross, the labels "mode 1" and "mode 2" can therefore swap between
adjacent rows of the tables below.

A related trap when reading `lambda(f)`: a sign change is a resonance when
`lambda` passes through **zero**, but a **pole** when it runs off through
infinity, and the second means the mode is at its least excitable, not its
most. `MS` distinguishes them at a glance -- it goes to 1 at a resonance and to
0 at a pole.
"""


_TRACKED_CAVEAT = """
## Caveat on mode numbering

The modes in the tables below have been tracked across the sweep: at each step
the eigenvectors are correlated with the previous frequency in the `R`-weighted
metric and matched one-to-one by the Hungarian algorithm, so a column follows
one physical current rather than the solver's per-frequency `|lambda|` rank. A
`swap` row in the tracking events is therefore a crossing that was followed
through, not an error.

The tracking section above lists what could NOT be decided that way. Inside a
quasi-degenerate pair the eigenvectors are set by roundoff and no correlation
metric can separate them, so a `degeneracy` or `avoided_crossing` row marks a
band where the mode labels may still be interchanged.

A separate trap when reading `lambda(f)`: a sign change is a resonance when
`lambda` passes through **zero**, but a **pole** when it runs off through
infinity, and the second means the mode is at its least excitable, not its most.
`MS` distinguishes them at a glance -- it goes to 1 at a resonance and to 0 at a
pole.
"""


def _tracking_rows(tracking, freqs):
    """Record what cross-frequency tracking did and what it could not decide."""
    n_steps = tracking.correlation.shape[0]
    n_modes = tracking.n_modes
    applied = tracking.is_applied

    lines = [
        "## Mode tracking",
        "",
        "| Setting | Value |",
        "|---|---|",
        f"| Correlation metric | `{tracking.metric}` "
        + ("(R-weighted, `C_mn = \\|J_m^H R J_n\\| / sqrt(J_m^H R J_m . J_n^H R J_n)`) |"
           if tracking.metric == 'R' else "(plain Euclidean overlap) |"),
        "| Assignment | Hungarian (linear sum assignment), maximising the total `C` |",
        f"| Correction stage | `{tracking.correction}` |",
        f"| Applied to the tables below | {'yes' if applied else 'no -- raw solver order'} |",
    ]
    if n_steps:
        lines += [
            f"| Accepted correlation, min / median | {tracking.correlation.min():.4f} / "
            f"{np.median(tracking.correlation):.4f} |",
            f"| Assignment margin, min | {tracking.margin.min():.4f} |",
        ]
    lines.append("")

    if applied:
        lines += [
            "The saved `.mat` keeps the raw per-frequency order -- the permutation is",
            "stored beside it as `track_order`, so both views are recoverable. The tables",
            "in this report are the tracked one.",
            "",
        ]
    else:
        lines += [
            "The permutation is recorded but NOT applied here: the tables below are in",
            "the solver's per-frequency order. Pass `sweep_data.tracked()` to this report",
            "(and to the plotters) for the tracked view.",
            "",
        ]

    counts = {kind: len(tracking.events_of(kind))
              for kind in ('swap', 'degeneracy', 'avoided_crossing', 'low_correlation')}
    lines += [
        "| Event | Count | Meaning |",
        "|---|---|---|",
        f"| `swap` | {counts['swap']} | two curves crossed and the solver's labels "
        "inverted; tracking followed the modes through |",
        f"| `degeneracy` | {counts['degeneracy']} | `lambda_m ~ lambda_n`, so the "
        "eigenvector pair is set by roundoff and correlation cannot separate them |",
        f"| `avoided_crossing` | {counts['avoided_crossing']} | quasi-degenerate with no "
        "label crossing and weak correlation: the currents exchanged character |",
        f"| `low_correlation` | {counts['low_correlation']} | the best match fell below "
        "the threshold, usually a sweep step too coarse for that band |",
        "",
    ]

    if tracking.events:
        lines += ["### Events", "",
                  "| Frequency | Kind | Mode(s) | Measurement |", "|---|---|---|---|"]
        for e in tracking.events:
            modes = ", ".join(str(m + 1) for m in e.modes)
            # A pipe inside the measurement would end the table cell.
            detail = e.detail.replace("|", "\\|")
            lines.append(f"| {_fmt_freq(e.frequency)} | `{e.kind}` | {modes} | {detail} |")
        lines.append("")
    else:
        lines += ["No swap, degeneracy or low-confidence step was flagged.", ""]

    if n_steps:
        lines += [
            "### Accepted correlation per step",
            "",
            "`C` between each mode and its match at the previous frequency. Values near",
            "1 mean the current barely changed over the step; a dip is where the",
            "identity of that mode is least certain.",
            "",
            "| Frequency | " + " | ".join(f"mode {n + 1}" for n in range(n_modes)) + " |",
            "|---" * (n_modes + 1) + "|",
        ]
        for i in range(n_steps):
            lines.append(f"| {_fmt_freq(freqs[i + 1])} | "
                         + " | ".join(f"{tracking.correlation[i, n]:.4f}"
                                      for n in range(n_modes)) + " |")
        lines.append("")
    return lines


def _write(out_path, lines):
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f"CMA report written to: {out_path}")
    return out_path


def write_single_frequency_report(path, modal_data, points, triangles, edges, Z=None,
                                  params=None):
    """
    Write the Markdown data sheet for a single-frequency CMA run.

    Parameters
    ----------
    path        : SimpleNamespace from setup_save_file_paths()
    modal_data  : ModalData from CharacteristicModeSolver.solve()
    points, triangles, edges : the mesh objects the run used
    Z           : the impedance matrix, if available -- adds the health section
    params      : CMAParameters, if available -- records the solver settings

    Returns
    -------
    str : the file written
    """
    cma_dir = 'data/antennas_cma'
    os.makedirs(cma_dir, exist_ok=True)
    out = os.path.join(cma_dir, f"{path.name}_cma_report.md")

    freq = float(modal_data.frequency)
    geom = _geometry_rows(points, triangles, edges)
    lam = np.asarray(modal_data.eigenvalues, dtype=float)
    ms = modal_significance(lam)
    delta = characteristic_angle(lam)

    lines = [
        f"# Characteristic Mode Analysis — {path.name}",
        "",
        f"Single frequency: **{_fmt_freq(freq)}**  ",
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Structure and discretisation",
        "",
    ] + geom['rows'] + _electrical_size_rows(geom, freq) + [""]

    if params is not None:
        lines += [
            "## Solver settings",
            "",
            "| Setting | Value |",
            "|---|---|",
            f"| Modes extracted | {modal_data.n_modes} |",
            f"| Mode ordering | `{params.sort_by}` "
            f"({'closest to resonance first' if params.sort_by == 'magnitude' else 'algebraic'}) |",
            f"| Regularization epsilon | {params.regularization_epsilon:.1e} |",
            "",
        ]

    lines += ["## Modes", "",
              "| Mode | lambda | MS | delta (deg) |",
              "|---|---|---|---|"]
    for n in range(len(lam)):
        lines.append(f"| {n + 1} | {lam[n]:+.6g} | {ms[n]:.4f} | {delta[n]:.2f} |")
    lines.append("")

    if Z is not None:
        lines += ["## Impedance matrix", ""] + _health_rows(Z) + [""]

    lines += [_LEGEND.strip(), "",
              "## Source data", "",
              f"- Eigenvalues and eigenvectors: `{_as_posix(getattr(path, 'mat_cma', 'n/a'))}`",
              "- Per-mode surface currents and far fields: `cma_algorithm(..., save_image=True)` "
              "and `compute_modal_farfield`.",
              ""]
    return _write(out, lines)


def write_sweep_report(path, sweep_data, points, triangles, edges, Z_all=None, params=None):
    """
    Write the Markdown data sheet for a CMA frequency sweep.

    Parameters
    ----------
    path       : SimpleNamespace from setup_save_file_paths()
    sweep_data : CMASweepData from cma_frequency_sweep()
    points, triangles, edges : the mesh objects the run used
    Z_all      : (n_freq, E, E) impedance matrices, if available -- the health
                 section is then reported at the mid-band frequency
    params     : CMAParameters, if available -- records the solver settings

    Returns
    -------
    str : the file written
    """
    cma_dir = 'data/antennas_cma'
    os.makedirs(cma_dir, exist_ok=True)
    out = os.path.join(cma_dir, f"{path.name}_cma_sweep_report.md")

    freqs = np.asarray(sweep_data.frequencies, dtype=float)
    lam_all = np.asarray(sweep_data.eigenvalues, dtype=float)     # (n_freq, n_modes)
    n_modes = lam_all.shape[1]
    geom = _geometry_rows(points, triangles, edges)

    lines = [
        f"# Characteristic Mode Analysis — {path.name}",
        "",
        f"Frequency sweep: **{_fmt_freq(freqs[0])} to {_fmt_freq(freqs[-1])}**, "
        f"{len(freqs)} points, {n_modes} modes  ",
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Structure and discretisation",
        "",
    ] + geom['rows']
    # Both ends of the band: electrical size is smallest at the bottom, and the
    # mesh density constraint is tightest at the top.
    lines += _electrical_size_rows(geom, freqs[0], "band start")
    lines += _electrical_size_rows(geom, freqs[-1], "band stop")
    lines += [""]

    if params is not None:
        lines += [
            "## Solver settings",
            "",
            "| Setting | Value |",
            "|---|---|",
            f"| Modes tracked | {n_modes} |",
            f"| Mode ordering | `{params.sort_by}` "
            f"({'closest to resonance first' if params.sort_by == 'magnitude' else 'algebraic'}) |",
            f"| Regularization epsilon | {params.regularization_epsilon:.1e} |",
            "",
        ]

    tracking = getattr(sweep_data, 'tracking', None)
    if tracking is not None:
        lines += _tracking_rows(tracking, freqs)

    ms_all = modal_significance(lam_all)
    delta_all = characteristic_angle(lam_all)

    for title, note, data, fmt in (
        ("Characteristic values lambda", "0 marks a resonance; the sign gives the reactive character.",
         lam_all, "{:+.6g}"),
        ("Modal significance MS", "1 at resonance, 0 for an unexcitable mode.", ms_all, "{:.4f}"),
        ("Characteristic angle delta (deg)", "180 deg marks a resonance.", delta_all, "{:.2f}"),
    ):
        lines += ["", f"## {title}", "", note, "",
                  "| Frequency | " + " | ".join(f"mode {n + 1}" for n in range(n_modes)) + " |",
                  "|---" * (n_modes + 1) + "|"]
        for i, f in enumerate(freqs):
            lines.append(f"| {_fmt_freq(f)} | "
                         + " | ".join(fmt.format(data[i, n]) for n in range(n_modes)) + " |")

    if Z_all is not None:
        mid = len(freqs) // 2
        lines += ["", f"## Impedance matrix (at {_fmt_freq(freqs[mid])}, mid-band)", ""]
        lines += _health_rows(np.asarray(Z_all)[mid])

    caveat = (_TRACKED_CAVEAT if tracking is not None and tracking.is_applied
              else _SWEEP_CAVEAT)
    lines += ["", caveat.strip(), "", _LEGEND.strip(), "",
              "## Source data", "",
              f"- Eigenvalues and eigenvectors: `{_as_posix(getattr(path, 'mat_cma_sweep', 'n/a'))}`",
              "- Plots: `ModalVisualizer.plot_characteristic_angle_vs_frequency`, "
              "`plot_eigenvalues_vs_frequency`, `plot_modal_significance`.",
              ""]
    return _write(out, lines)
