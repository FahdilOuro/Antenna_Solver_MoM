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

# Relative asymmetry above which Z stops looking like a quadrature artefact and
# starts looking like a broken assembly.
#
# This value is NOT to be relaxed to accommodate a measurement. It was raised to
# 5e-3 once, to make a measured 1.23e-3 on a CubeSat mesh read as acceptable;
# that hid a real residual instead of fixing it. If a mesh cannot meet it, the
# assembly is what needs changing.
#
# And that is what happened: the residual turned out to be plain truncation
# error in the outer Gauss rule of the EDGE/VERTEX branch, whose order had been
# pinned at the largest rule the quadrature table offered. Raising it (impmet's
# NEAR_ORDER_DEFAULT, 11 -> 16) took the same mesh to 5.2e-4, under this
# tolerance, at no measurable cost. See
# backend/utils/quadrature/outer_order_study.py.
RECIPROCITY_TOL = 1e-3


def check_reciprocity(Z):
    """
    Check Z's reciprocity (symmetry). The EXACT Galerkin MoM impedance matrix
    is symmetric: testing and basis functions are the same set, and the
    free-space Green's function is symmetric in its two arguments, so nothing
    in the physics breaks Z = Z^T.

    Since the assembler fills only m <= n and mirrors (impmet's
    _mirror_upper_triangle), production matrices satisfy Z = Z^T EXACTLY, and
    this function reads 0. That is worth being explicit about: a zero here
    confirms the mirroring ran, and nothing else. It cannot fail, so it cannot
    warn -- a completely wrong kernel would also read 0.

    What used to make this measurement informative was that the two halves were
    computed independently, by schemes that treat the pair's two triangles
    differently: SELF uses Gauss-Dunavant on the TEST triangle and Duffy on the
    SOURCE one, EDGE/VERTEX integrates the 1/R half of the kernel analytically
    over the SOURCE only. Their disagreement was therefore a direct readout of
    the quadrature's own error -- 5e-5 to 5e-4 when healthy, and 0.994 the one
    time a closed-form 1/R integral silently assumed the observation point was
    coplanar with the source triangle. That is what caught the bug.

    To recover that signal, assemble with impedance_matrix_z(..., full_fill=True)
    and pass the result here: at twice the fill cost, the two halves are
    independent again and the number means something. See
    test/test_impedance_non_coplanar.py, which keeps a standing 3D case.

    Two limits are worth knowing before reading a small value as "Z is accurate",
    because it does not mean that. This bounds the EDGE/VERTEX branch and, in
    practice, nothing else: a SELF pair puts both integrals on the SAME triangle,
    so swapping (m,n) leaves the geometry unchanged and its error very nearly
    cancels -- measured 8e-7 here against a true SELF error around 3e-2 on the
    same mesh. And on a small, locally symmetric mesh a vertex-permutation-
    invariant Gauss rule can make the two halves' errors cancel outright, so this
    reads roundoff while the branch is 1e-3 wrong. On a real mesh it does track
    the true EDGE error closely (1.27e-3 against 1.14e-3 on the CubeSat mesh) --
    it is a lower bound on the assembly's error, never a measure of it.

    Parameters
    ----------
    Z : complex (N, N) ndarray

    Returns
    -------
    dict with:
        max_abs_diff        : max|Z - Z^T|
        max_abs_z            : max|Z|
        relative_asymmetry  : max_abs_diff / max_abs_z (expect 5e-5 to 5e-4 with
                              the default settings, depending on mesh size;
                              orders of magnitude above that means something is
                              actually wrong)
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


def diagnose_z_matrix(Z, frequency=None, label="", reciprocity_tol=RECIPROCITY_TOL,
                      relative_min_eig_tol=-1e-6):
    """
    Print a combined reciprocity + R-positive-semidefinite report for an
    impedance matrix, and return the underlying numbers.

    Parameters
    ----------
    Z                     : complex (N, N) impedance matrix
    frequency             : float or None, Hz -- only used in the printed header
    label                 : str, prefix for the printed header (e.g. antenna name)
    reciprocity_tol       : relative_asymmetry above this prints a WARNING
                            (defaults to RECIPROCITY_TOL -- see its derivation
                            at the top of this module)
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
