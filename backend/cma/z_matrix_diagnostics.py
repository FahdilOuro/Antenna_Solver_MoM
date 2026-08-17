"""
Numerical sanity checks on an EFIE/MoM impedance matrix Z, independent of
CMA itself -- useful whenever GeneralizedEigenSolver needs unusually large
regularization (see eigen_solver.py), to tell apart two very different
root causes:

  1. Z is physically/numerically healthy, and R = Re(Z) is just close to
     singular because the structure is electrically small at that
     frequency (most RWG basis functions barely radiate there) -- expected,
     not a bug.
  2. Z itself is wrong (assembly bug, mesh defect, wrong quadrature order,
     ...), which would typically also show up as broken reciprocity and/or
     R eigenvalues that are negative by much more than floating-point noise.
"""

import numpy as np


def check_reciprocity(Z):
    """
    Check Z's reciprocity (symmetry). The EXACT Galerkin MoM impedance matrix
    is symmetric: testing and basis functions are the same set, and the
    free-space Green's function is symmetric in its two arguments, so nothing
    in the physics breaks Z = Z^T.

    The assembled matrix is not symmetric to machine precision, though, and
    that is expected rather than a defect: the SELF branch of impmet uses
    Gauss-Dunavant on the TEST triangle and Duffy on the SOURCE triangle, so
    swapping the (m,n) roles applies a different quadrature rule to the same
    integral. The residual is pure Duffy truncation error and converges only
    slowly with duffy_order (measured ~1.5e-7 at order 7, ~1.6e-8 at order 20).

    That residual is harmless in practice: on the reference pixel-grid loop it
    sits 5-6 orders of magnitude BELOW the symmetric part of the quadrature
    error (Re(Z) ~4.5e-3, Im(Z) ~3.2e-1 relative to AToM), and symmetrising Z
    changes the error vs that reference by less than 7 significant digits. The
    consumers that care already symmetrise anyway (GeneralizedEigenSolver
    forms R_sym/X_sym before solving). So this check is a coarse tripwire for
    a genuinely broken assembly, not a precision measurement.

    Parameters
    ----------
    Z : complex (N, N) ndarray

    Returns
    -------
    dict with:
        max_abs_diff        : max|Z - Z^T|
        max_abs_z            : max|Z|
        relative_asymmetry  : max_abs_diff / max_abs_z (expect ~1e-7 with the
                              default duffy_order; orders of magnitude above
                              that means something is actually wrong)
    """
    Z = np.asarray(Z)
    diff = Z - Z.T
    max_abs_diff = float(np.max(np.abs(diff)))
    max_abs_z = float(np.max(np.abs(Z)))
    relative_asymmetry = max_abs_diff / max_abs_z if max_abs_z > 0 else 0.0
    return {
        'max_abs_diff': max_abs_diff,
        'max_abs_z': max_abs_z,
        'relative_asymmetry': relative_asymmetry,
    }


def check_r_positive_semidefinite(Z):
    """
    Check that R = Re(Z) is positive semi-definite, as required for a
    lossless PEC: radiated power P_rad = 0.5 * I^H R I >= 0 for ANY current
    I. A raw eigenvalue in Ohms is not meaningful on its own -- radiation
    resistance can legitimately be tiny for weakly-radiating modes/low
    electrical size (see eigen_solver.py's regularization comment) -- so
    this also reports the smallest eigenvalue relative to R's own scale
    (||R||_F / N), the same normalization eigen_solver.py's regularization
    floor uses, to judge whether a negative eigenvalue is ordinary
    floating-point noise or something larger worth investigating.

    Parameters
    ----------
    Z : complex (N, N) ndarray

    Returns
    -------
    dict with:
        eigenvalues       : full eigenvalue spectrum of R_sym, ascending
        min_eigenvalue    : smallest eigenvalue (Ohm)
        frobenius_norm    : ||R_sym||_F
        relative_min_eig  : min_eigenvalue / (frobenius_norm / N)
    """
    Z = np.asarray(Z)
    R = np.real(Z)
    R_sym = (R + R.T) * 0.5
    N = R_sym.shape[0]

    eigenvalues = np.linalg.eigvalsh(R_sym)
    frob = float(np.linalg.norm(R_sym, 'fro'))
    scale = frob / N if N > 0 else 1.0

    return {
        'eigenvalues': eigenvalues,
        'min_eigenvalue': float(eigenvalues[0]),
        'frobenius_norm': frob,
        'relative_min_eig': eigenvalues[0] / scale if scale > 0 else 0.0,
    }


def diagnose_z_matrix(Z, frequency=None, label="", reciprocity_tol=1e-5, relative_min_eig_tol=-1e-6):
    """
    Print a combined reciprocity + R-positive-semidefinite report for an
    impedance matrix, and return the underlying numbers.

    Parameters
    ----------
    Z                     : complex (N, N) impedance matrix
    frequency             : float or None, Hz -- only used in the printed header
    label                 : str, prefix for the printed header (e.g. antenna name)
    reciprocity_tol       : relative_asymmetry above this prints a WARNING
                            (default 1e-5: comfortably above the ~1e-7 that the
                            Gauss-outer/Duffy-inner SELF branch legitimately
                            produces -- see check_reciprocity -- and still low
                            enough to catch a genuinely broken assembly)
    relative_min_eig_tol  : relative_min_eig below this prints a WARNING
                            (default -1e-6: comfortably above floating-point
                            noise, comfortably below "clearly wrong")

    Returns
    -------
    dict with keys 'reciprocity' and 'positive_semidefinite' -- see
    check_reciprocity() / check_r_positive_semidefinite()
    """
    header = "Z-matrix diagnostics"
    if label:
        header += f" -- {label}"
    if frequency is not None:
        header += f" @ {frequency / 1e6:.2f} MHz"
    print(f"\n{header}")
    print("=" * len(header))

    reciprocity = check_reciprocity(Z)
    ok_reciprocity = reciprocity['relative_asymmetry'] < reciprocity_tol
    print(f"Reciprocity   : max|Z-Z^T| = {reciprocity['max_abs_diff']:.3e}  "
          f"max|Z| = {reciprocity['max_abs_z']:.3e}  "
          f"-> relative asymmetry = {reciprocity['relative_asymmetry']:.3e}  "
          f"[{'OK' if ok_reciprocity else 'WARNING: not reciprocal'}]")

    psd = check_r_positive_semidefinite(Z)
    ok_psd = psd['relative_min_eig'] > relative_min_eig_tol
    print(f"R = Re(Z) PSD : min eig(R) = {psd['min_eigenvalue']:.3e} Ohm  "
          f"(relative to ||R||_F/N = {psd['relative_min_eig']:.3e})  "
          f"[{'OK, ~0 within noise' if ok_psd else 'WARNING: significantly negative'}]")

    return {'reciprocity': reciprocity, 'positive_semidefinite': psd}
