"""
Acceptance criteria checks for the impedance matrix Z.

Call check_z_properties(Z) to get a dict of diagnostics.
These are the same metrics used before and after the quadrature refactor
so results are directly comparable.

Metrics
-------
sym_error_R   : ||R - R.T|| / ||R||_F          (should be ~0)
sym_error_X   : ||X - X.T|| / ||X||_F          (should be ~0)
sym_error_Z   : ||Z - Z.T|| / ||Z||_F          (overall symmetry)
min_eig_R     : minimum eigenvalue of (R+R.T)/2 (should be >= 0 for PEC)
cond_R        : condition number of (R+R.T)/2   (was ~2.68e15 with legacy code)
rank_R        : numerical rank of R at tol=1e-10*||R||_2
"""

import numpy as np


def check_z_properties(Z: np.ndarray, label: str = "") -> dict:
    """
    Parameters
    ----------
    Z     : complex (E, E) impedance matrix
    label : optional string printed with the report

    Returns
    -------
    dict with keys described in the module docstring.
    """
    R = np.real(Z)
    X = np.imag(Z)

    frob_R = np.linalg.norm(R, 'fro')
    frob_X = np.linalg.norm(X, 'fro')
    frob_Z = np.linalg.norm(Z, 'fro')

    sym_error_R = np.linalg.norm(R - R.T, 'fro') / frob_R if frob_R > 0 else 0.0
    sym_error_X = np.linalg.norm(X - X.T, 'fro') / frob_X if frob_X > 0 else 0.0
    sym_error_Z = np.linalg.norm(Z - Z.T, 'fro') / frob_Z if frob_Z > 0 else 0.0

    R_sym = (R + R.T) * 0.5
    eigs_R = np.linalg.eigvalsh(R_sym)   # all real, sorted ascending
    min_eig_R = float(eigs_R[0])
    max_eig_R = float(eigs_R[-1])
    cond_R = max_eig_R / max(abs(min_eig_R), 1e-300)

    # Numerical rank: count eigenvalues above tol * max_eig
    tol_rank = 1e-10 * max_eig_R
    rank_R = int(np.sum(eigs_R > tol_rank))

    results = {
        "label":        label,
        "size":         Z.shape[0],
        "sym_error_R":  sym_error_R,
        "sym_error_X":  sym_error_X,
        "sym_error_Z":  sym_error_Z,
        "min_eig_R":    min_eig_R,
        "max_eig_R":    max_eig_R,
        "cond_R":       cond_R,
        "rank_R":       rank_R,
    }

    _print_report(results)
    return results


def _print_report(r: dict):
    tag = f"[{r['label']}] " if r['label'] else ""
    print(f"\n{tag}Z-matrix properties  (E = {r['size']})")
    print(f"  Symmetry  ||R-R.T||/||R||  = {r['sym_error_R']:.3e}")
    print(f"  Symmetry  ||X-X.T||/||X||  = {r['sym_error_X']:.3e}")
    print(f"  Symmetry  ||Z-Z.T||/||Z||  = {r['sym_error_Z']:.3e}")
    print(f"  min eig(R_sym)             = {r['min_eig_R']:.4e}")
    print(f"  max eig(R_sym)             = {r['max_eig_R']:.4e}")
    print(f"  cond(R_sym)                = {r['cond_R']:.4e}")
    print(f"  rank(R_sym) at 1e-10*lmax = {r['rank_R']}")


def compare_z(Z_ref: np.ndarray, Z_new: np.ndarray) -> dict:
    """
    Element-wise comparison of two Z matrices.

    Returns relative Frobenius error and max element-wise relative error,
    both overall and split by pair category if category array is provided.
    """
    diff = Z_new - Z_ref
    rel_frob = np.linalg.norm(diff, 'fro') / np.linalg.norm(Z_ref, 'fro')

    with np.errstate(divide='ignore', invalid='ignore'):
        rel_elem = np.abs(diff) / (np.abs(Z_ref) + 1e-300)

    max_rel = float(np.max(rel_elem))
    median_rel = float(np.median(rel_elem))

    results = {
        "rel_frob":   rel_frob,
        "max_rel":    max_rel,
        "median_rel": median_rel,
    }
    print(f"\nZ comparison:")
    print(f"  ||Z_new - Z_ref||_F / ||Z_ref||_F = {rel_frob:.3e}")
    print(f"  max element-wise relative error    = {max_rel:.3e}")
    print(f"  median element-wise relative error = {median_rel:.3e}")
    return results
