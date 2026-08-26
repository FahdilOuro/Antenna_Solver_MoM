from dataclasses import dataclass


@dataclass
class CMAParameters:
    """Configuration for the Characteristic Mode Analysis solver."""

    n_modes: int = 10
    # Relative perturbation applied to R before eigh: eps_adapt = epsilon * ||R||_F / N
    regularization_epsilon: float = 1e-10
    # Apply sign convention (largest-magnitude component positive)
    normalize: bool = True
    # Ordering of extracted modes: 'magnitude' sorts by |λ| (resonant modes first)
    sort_by: str = 'magnitude'   # 'magnitude' | 'value'

    # How to handle R = Re(Z) being rank-deficient.
    #
    # 'regularized'    solve X J = λ (R + eps I) J on the full N-dimensional
    #                  space. Every basis function yields a mode, including the
    #                  many that radiate nothing -- and for those, eps I *is*
    #                  the operator being inverted, so their λ is set by the
    #                  regularization rather than by the structure. They come
    #                  back looking like ordinary modes, sometimes with a modal
    #                  significance near 1, and they move from run to run.
    # 'rank_truncated' project onto the subspace R actually spans before
    #                  solving, so non-radiating directions are removed rather
    #                  than assigned fictitious eigenvalues. Returns at most
    #                  rank(R) modes -- fewer than n_modes when the structure is
    #                  electrically small.
    solver: str = 'regularized'   # 'regularized' | 'rank_truncated'

    # Eigenvalues of R below rank_tol * λ_max(R) count as null space for the
    # 'rank_truncated' solver. 1e-6 sits well below the smallest genuinely
    # radiating eigenvalue measured on real meshes and well above the float64
    # noise floor of forming R (~1e-14 relative).
    rank_tol: float = 1e-6
