"""
Calibrated conductor-loss factor alpha for zero-thickness strips (EMPIRICAL).

Why a calibration
-----------------
The MoM models a copper strip as a zero-thickness sheet with surface impedance
alpha * Zs (solver.optim.reduced_model). On such a sheet the current is
singular at the strip edges, |J|^2 ~ 1/d, so the conductor loss grows like
ln(1 / h_edge) as the edge cells shrink: refining the mesh never converges.
In the real strip the thickness (t = 35 um) cuts the singularity off. The sheet
model therefore has no mesh-independent alpha. Decision (step 5, 5/11):

  * freeze ONE meshing rule for every strip of the antenna:
        n_width = 4 cells across the width, cosine edge grading
    (MESH_RULE; the rule of the V5b' study);
  * calibrate alpha on CST for that rule:
        alpha_cal(w) = R_loss^CST(w) / R_loss^MoM(w, alpha = 1, rule above),
    R_loss = Re Zin(copper) - Re Zin(PEC) of the tuned loop V5b' at resonance
    (CST geometry: outer side 20 mm, copper 35 um, ~0.89 GHz; MoM fill_mode
    "full_sym", capacitor re-tuned to CST's f_res).

Calibration data (fiches_cst/resultats_V5, MoM runs of 2026-11-05):

    w (mm)   R_loss CST (Ohm)   R_loss MoM, alpha=1 (Ohm)   alpha_cal
      1          0.441               0.620                    0.71
      2          0.246               0.303                    0.81
      4          0.152               0.140                    1.09

Validity domain: copper 35 um, ~0.9 GHz, strip widths 1 to 4 mm, strips meshed
with MESH_RULE. alpha_cal is linearly interpolated in w and NOT extrapolated
(ValueError outside [1, 4] mm). Uncertainty: +-15 % (ALPHA_UNCERTAINTY); results
should always be reported for alpha_cal * (0.85, 1, 1.15) -- alpha_band(). Changing
alpha only needs a new LU of Z + alpha Zs G: Z is reused.
"""

import numpy as np

#: Meshing rule the calibration is valid for (pass to the planar geometry builders).
MESH_RULE = dict(n_width=4, edge_grading="cosine")

#: (width mm, R_loss CST Ohm, R_loss MoM alpha=1 Ohm, alpha_cal) -- see module docstring.
CALIBRATION = (
    (1.0, 0.441, 0.620, 0.71),
    (2.0, 0.246, 0.303, 0.81),
    (4.0, 0.152, 0.140, 1.09),
)
ALPHA_UNCERTAINTY = 0.15
VALIDITY = dict(conductor="copper, 35 um", frequency_GHz=0.9, width_mm=(1.0, 4.0),
                mesh_rule=MESH_RULE, status="empirical calibration on CST V5b'")


def alpha_calibrated(width_mm: float) -> float:
    """alpha_cal at strip width `width_mm` (mm), linear interpolation of CALIBRATION.

    Raises ValueError outside [1, 4] mm: no extrapolation of an empirical fit.
    """
    w = np.asarray([c[0] for c in CALIBRATION])
    a = np.asarray([c[3] for c in CALIBRATION])
    width_mm = float(width_mm)
    if not (w[0] - 1e-9 <= width_mm <= w[-1] + 1e-9):
        raise ValueError(f"alpha_calibrated: width {width_mm:g} mm is outside the calibrated "
                         f"domain [{w[0]:g}, {w[-1]:g}] mm (copper 35 um, ~0.9 GHz, n_width = 4 "
                         f"cosine). Choose a width in the domain or recalibrate on CST.")
    return float(np.interp(width_mm, w, a))


def alpha_band(width_mm: float):
    """(alpha_cal * 0.85, alpha_cal, alpha_cal * 1.15): the three values to report."""
    a = alpha_calibrated(width_mm)
    return (a * (1 - ALPHA_UNCERTAINTY), a, a * (1 + ALPHA_UNCERTAINTY))
