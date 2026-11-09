"""
Antenna figures of merit from radiation intensity and port powers.

    D    = 4 pi U / P_rad,      P_rad = 1/2 Re(I^H R I),  R = Re(Z)  (lossless MoM)
    G    = 4 pi U / P_in        (conductor and load losses included)
    G_r  = 4 pi U / P_avail     (mismatch included: realized gain)

With a single driven port and a real generator impedance,
P_in = P_avail (1 - |Gamma|^2) and eta_rad = P_rad / P_in, so

    G_r = (1 - |Gamma|^2) * eta_rad * D       (mismatch x loss x directivity)

U is any intensity in W/sr (one polarisation component or the total), for the
same peak-phasor excitation as the powers.
"""

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError


def directivity_from_R(U, I, R):
    """D = 4 pi U / (1/2 I^H R I), with R = Re(Z) of the lossless (PEC) MoM matrix."""
    I = np.asarray(I, dtype=np.complex128).reshape(-1)
    p_rad = 0.5 * float(np.real(I.conj() @ (np.asarray(R) @ I)))
    return 4.0 * np.pi * np.asarray(U) / p_rad


def realized_gain(sol, U):
    """G_r = 4 pi U / P_avail (sum of available powers of the driven ports)."""
    return 4.0 * np.pi * np.asarray(U) / sol.p_avail_total


def gain(sol, U):
    """
    G = 4 pi U / P_in (sum of powers accepted at the driven ports).

    This is also the gain with an IDEAL (lossless, conjugate) matching network at
    the port: such a network makes P_in = P_avail, so G_r -> G. G says what the
    structure can do; realized_gain what it does on a bare 50 Ohm generator.
    With lossless (reactive) loads it does not depend on the generator impedance.
    """
    return 4.0 * np.pi * np.asarray(U) / sol.p_in_total


matched_gain = gain


def loss_matrix(model):
    """
    M = 1/2 (R + Re(alpha Zs) G), R = Re(Z) symmetrised, so that for ANY edge
    currents I (reactive loads, any port excitation):
        P_rad + P_cond = I^H M I.
    Needs a model built with keep_matrices=True.
    """
    if model.Z is None:
        raise ValueError("The model was built with keep_matrices=False: Z is not available.")
    R = 0.5 * (model.Z.real + model.Z.real.T)
    loss = 0.0 if model.zs is None else float(np.real(model.zs_scale * model.zs))
    M = 0.5 * R
    if loss != 0.0:
        M = M + 0.5 * loss * model.G
    return M, loss


def gain_from_currents(I, operator, M, theta, phi, component="phi"):
    """G(I) = 4 pi U_component(theta, phi) / (I^H M I), for arbitrary edge currents I."""
    I = np.asarray(I, dtype=np.complex128).reshape(-1)
    u_t, u_p = operator.intensity(I, np.atleast_1d(theta)[:1], np.atleast_1d(phi)[:1])
    U = float({"theta": u_t, "phi": u_p}[component][0])
    return 4.0 * np.pi * U / float(np.real(I.conj() @ (M @ I)))


@dataclass(frozen=True)
class GainBound:
    """
    g_max  : upper bound of G for this geometry (linear)
    cond   : 2-norm condition number of M (lambda_max / lambda_min)
    I_opt  : edge currents attaining it, M^{-1} a (any complex multiple works)
    a      : conj(f), f the far-field vector of the chosen component
    """
    g_max: float
    cond: float
    I_opt: np.ndarray
    a: np.ndarray


def geometry_gain_bound(model, operator, theta, phi, component="phi",
                        verbose=True) -> Optional[GainBound]:
    """
    Largest gain G = 4 pi U / P_in in direction (theta, phi), component
    `component`, over ALL possible edge currents on this geometry -- whatever
    feeds or loads produce them. Derivation:

      U   = k^2 eta0 / (32 pi^2) |N|^2,  N = f^T I = a^H I,  a = conj(f)
            (farfield.RadiationOperator.vectors)
      P_in = P_rad + P_cond = 1/2 I^H R I + 1/2 Re(alpha Zs) I^H G I = I^H M I,
            M = 1/2 (R + Re(alpha Zs) G)   (reactive loads dissipate nothing)
      G(I) = 4 pi U / P_in = (k^2 eta0 / (8 pi)) |a^H I|^2 / (I^H M I)

    M is Hermitian (real symmetric) positive definite once there are losses, so
    by Cauchy-Schwarz in the M inner product,
      |a^H I|^2 = |<M^{-1} a, I>_M|^2 <= (a^H M^{-1} a)(I^H M I),
    with equality iff I is proportional to M^{-1} a. Hence

      G_max = (k^2 eta0 / (8 pi)) a^H M^{-1} a,   attained for I = M^{-1} a.

    (4 pi * k^2 eta0 / (32 pi^2) = k^2 eta0 / (8 pi): the prefactor checks.)

    Without conductor loss M = R/2 is only positive SEMI-definite and nearly
    singular for a small antenna (few radiating modes): the "bound" is then
    unbounded in practice. A warning is issued and None returned.
    The condition number of M is computed (eigvalsh) and printed if verbose.
    """
    M, loss = loss_matrix(model)
    if loss <= 0.0:
        warnings.warn("geometry_gain_bound: no conductor loss (zs is None or Re(alpha Zs) = 0). "
                      "M = R/2 is nearly singular for a small antenna and the lossless "
                      "supergain bound is not meaningful: no bound returned.")
        return None
    lam = np.linalg.eigvalsh(M)
    cond = float(lam[-1] / lam[0]) if lam[0] > 0 else np.inf
    if verbose:
        print(f"[geometry_gain_bound] cond(M) = {cond:.3e} "
              f"(lambda_min = {lam[0]:.3e}, lambda_max = {lam[-1]:.3e})")
    try:
        c = cho_factor(M, lower=True)
    except LinAlgError:
        warnings.warn("geometry_gain_bound: M is not numerically positive definite "
                      f"(cond = {cond:.3e}); no bound returned.")
        return None
    f_t, f_p = operator.vectors(np.atleast_1d(theta)[:1], np.atleast_1d(phi)[:1])
    f = {"theta": f_t, "phi": f_p}[component][0]
    a = np.conj(f)
    x = cho_solve(c, a)
    g_max = operator.k ** 2 * operator.eta / (8 * np.pi) * float(np.real(a.conj() @ x))
    return GainBound(g_max=g_max, cond=cond, I_opt=x, a=a)


def intensity_from_functional(sol, name, operator):
    """U = k^2 eta0/(32 pi^2) |N|^2 for a far-field functional registered with
    RadiationOperator.attach_functional (W/sr)."""
    return operator.intensity_constant * abs(sol.functionals[name]) ** 2


def to_dbi(x):
    """10 log10(x), with -inf for x <= 0 instead of a warning."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(np.where(x > 0, x, 0.0))
