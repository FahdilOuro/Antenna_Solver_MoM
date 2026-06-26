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
