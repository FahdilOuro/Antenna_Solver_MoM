"""
Per-mode physical diagnostics for Characteristic Mode Analysis.

A CMA run always returns modes. Whether those modes correspond to currents the
structure can actually support is a separate question, and lambda / MS alone
cannot answer it: a mode living in the null space of R = Re(Z) radiates nothing,
yet the regularized eigensolver hands it an ordinary-looking lambda, and
MS = 1/sqrt(1+lambda^2) will happily report 0.99 for it.

Two independent quantities separate the two cases.

Radiated power per unit current amplitude. For a real current vector J,

    P_rad = (J^T R J) / (J^T J)      (R is the radiation operator, in Ohms)

is >= 0 for a lossless PEC and zero for a non-radiating current. The division
matters: eigensolvers return J normalised so that J^T R J = 1 exactly, for every
mode, so the raw numerator reports 1.0 even for a mode lying entirely in R's null
space (it just comes back with an enormous ||J||). Compared against the strongest
mode in the same set, a physical mode lands within a few orders of magnitude; an
artefact of adding eps*I to R lands far below.

Null-space leakage. Writing R = U diag(mu) U^T and collecting into `null` the
columns whose mu <= 1e-6 * mu_max, the fraction

    leak = ||null^T J|| / ||J||

is the part of the mode that lies in directions R cannot see. A physical mode
is built out of radiating directions and leaks little; a spurious one is almost
entirely null-space and leaks close to 1.

The two criteria are not redundant: leakage catches a mode that is mostly
null-space but picked up enough radiating content to keep P_rad off the floor.

One caveat worth keeping in view. Modes produced by
CMAParameters(solver='rank_truncated') are built inside the range of R, so their
leakage is 0 BY CONSTRUCTION and cannot flag anything -- exactly as max|Z-Z^T|
reads 0 once the assembler mirrors its upper triangle. For those modes the
leakage column confirms that the truncation ran; the discriminating quantity is
P_rad, which is still measured independently.
"""

import numpy as np

# A mode radiating less than this fraction of the strongest mode in the set is
# not radiating in any physical sense.
SPURIOUS_P_REL_TOL = 1e-6

# Above this fraction of the mode living in R's null space, the mode is an
# artefact of regularization rather than a current the structure supports.
SPURIOUS_LEAK_TOL = 0.5

# Eigenvalues of R at or below this fraction of the largest are null space.
NULL_SPACE_TOL = 1e-6


def mode_report(J, R, X=None, eigenvalues=None,
                p_rel_tol=SPURIOUS_P_REL_TOL, leak_tol=SPURIOUS_LEAK_TOL,
                null_tol=NULL_SPACE_TOL):
    """
    Physical diagnostics for one or more characteristic modes.

    Parameters
    ----------
    J           : (N,) or (N, K) real eigenvector(s), one mode per column
    R           : (N, N) Re(Z), the radiation operator, in Ohms
    X           : (N, N) Im(Z), optional -- enables the eigenvalue residual
    eigenvalues : (K,) optional lambda per mode; required for the residual
    p_rel_tol   : P_rad relative to the strongest mode below which -> SPURIOUS
    leak_tol    : null-space leakage above which -> SPURIOUS
    null_tol    : eigenvalues of R at or below null_tol * max are null space

    Returns
    -------
    dict of (K,) arrays plus scalars:
        p_rad       (J^T R J)/(J^T J) per mode, in Ohms -- see module docstring
        p_rel       p_rad / p_rad.max() -- the STEP D criterion
        leak        ||null^T J|| / ||J||, in [0, 1]
        residual    ||X J - lambda R J|| / ||X J||, NaN if X/eigenvalues absent
        spurious    bool, p_rel < p_rel_tol OR leak > leak_tol
        verdict     'PHYSICAL' or 'SPURIOUS' per mode
        reason      why each mode was flagged ('' when PHYSICAL)
        rank_R      numerical rank of R at null_tol
        n_null      size of the null space
    """
    J = np.asarray(J, dtype=np.float64)
    if J.ndim == 1:
        J = J[:, None]
    R = np.asarray(R, dtype=np.float64)
    R_sym = (R + R.T) * 0.5

    mu, U = np.linalg.eigh(R_sym)                     # ascending
    is_null = mu <= null_tol * mu[-1]
    null = U[:, is_null]                              # (N, n_null)

    RJ = R_sym @ J
    j_sq = np.einsum('ik,ik->k', J, J)                # J^T J
    j_sq_safe = np.where(j_sq > 0, j_sq, 1.0)

    # Radiated power PER UNIT CURRENT AMPLITUDE -- the Rayleigh quotient of R,
    # not the raw J^T R J.
    #
    # This distinction is the whole measurement. scipy.linalg.eigh(X, R) returns
    # eigenvectors normalised so that J^T R J = 1 EXACTLY, for every mode,
    # including ones that live almost entirely in R's null space: such a mode
    # simply comes back with a huge ||J||, since a near-null direction has to be
    # scaled up enormously to reach unit R-norm. Taking J^T R J at face value
    # therefore reports 1.0 for every mode and can never flag anything -- it
    # measures the normalisation convention, not the physics. Dividing by J^T J
    # removes the convention and leaves how much this current shape radiates for
    # the amount of current it carries, which is what "does this mode radiate"
    # was asking.
    p_rad = np.einsum('ik,ik->k', J, RJ) / j_sq_safe
    p_max = p_rad.max()
    p_rel = p_rad / p_max if p_max > 0 else np.zeros_like(p_rad)

    leak = (np.linalg.norm(null.T @ J, axis=0) / np.sqrt(j_sq_safe)
            if null.shape[1] else np.zeros(J.shape[1]))

    # Residual of X J = lambda R J, measured ON THE RADIATING SUBSPACE ONLY.
    #
    # Projecting is not a way of being lenient, it is the only measurement that
    # means anything here. On R's null space the equation reads X J = lambda * 0,
    # which no finite lambda satisfies, so the full-space residual of a
    # rank-truncated mode is O(1) by construction -- measured ~0.8 on the
    # synthetic fixture in test_cma_rank_truncation.py, while the projected
    # residual of the same modes is ~1e-14. A truncated solve produces exact
    # eigenvectors of the problem restricted to where R is invertible, which is
    # the only region where the problem is posed at all. Regularized modes that
    # are physical satisfy this equally well; the ones that are not are already
    # caught by P_rel and leak.
    residual = np.full(J.shape[1], np.nan)
    if X is not None and eigenvalues is not None:
        X_sym = (np.asarray(X, dtype=np.float64) + np.asarray(X, dtype=np.float64).T) * 0.5
        range_R = U[:, ~is_null]                      # (N, rank)
        XJ = X_sym @ J
        lhs = XJ - np.asarray(eigenvalues, dtype=np.float64)[None, :] * RJ
        lhs_p = range_R.T @ lhs
        denom = np.linalg.norm(range_R.T @ XJ, axis=0)
        residual = np.linalg.norm(lhs_p, axis=0) / np.where(denom > 0, denom, 1.0)

    low_power = p_rel < p_rel_tol
    high_leak = leak > leak_tol
    spurious = low_power | high_leak

    reason = []
    for lp, hl in zip(low_power, high_leak):
        parts = []
        if lp:
            parts.append(f"P_rel < {p_rel_tol:.0e}")
        if hl:
            parts.append(f"leak > {leak_tol:g}")
        reason.append(" and ".join(parts))

    return {
        'p_rad': p_rad,
        'p_rel': p_rel,
        'leak': leak,
        'residual': residual,
        'spurious': spurious,
        'verdict': np.where(spurious, 'SPURIOUS', 'PHYSICAL'),
        'reason': np.array(reason, dtype=object),
        'rank_R': int(np.count_nonzero(~is_null)),
        'n_null': int(np.count_nonzero(is_null)),
        'lambda_max_R': float(mu[-1]),
    }


def format_mode_report(report, eigenvalues=None, title=None):
    """Render `mode_report`'s output as a Markdown table (also readable on a terminal)."""
    K = len(report['p_rel'])
    lines = []
    if title:
        lines += [title, ""]
    lines.append(f"rank(R) = {report['rank_R']} "
                 f"(null space {report['n_null']}, "
                 f"lambda_max(R) = {report['lambda_max_R']:.4e} Ohm)")
    lines.append("")

    has_lam = eigenvalues is not None
    has_res = not np.all(np.isnan(report['residual']))

    head = ["Mode"]
    if has_lam:
        head += ["lambda", "MS"]
    head += ["P_rad (Ohm)", "P_rel", "leak", "VERDICT"]
    if has_res:
        head += ["residual"]
    head += ["reason"]
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "---|" * len(head))

    for n in range(K):
        row = [f"{n + 1}"]
        if has_lam:
            lam = float(eigenvalues[n])
            row += [f"{lam:.5g}", f"{1.0 / np.sqrt(1.0 + lam ** 2):.4f}"]
        row += [f"{report['p_rad'][n]:.4e}", f"{report['p_rel'][n]:.3e}",
                f"{report['leak'][n]:.4f}", str(report['verdict'][n])]
        if has_res:
            row += [f"{report['residual'][n]:.2e}"]
        row += [str(report['reason'][n]) or "-"]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
