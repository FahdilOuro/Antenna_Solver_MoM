import numpy as np


def modal_significance(eigenvalues):
    """
    Modal Significance:  MS_n = |1 / (1 + j*lambda_n)| = 1 / sqrt(1 + lambda_n^2)

    A normalized (0, 1] alternative to the raw characteristic value lambda_n
    for judging how strongly a mode can be excited by an external source:
    MS_n = 1 exactly at resonance (lambda_n = 0) and decays as |lambda_n|
    grows, regardless of sign -- unlike lambda_n itself, which is unbounded
    and does not saturate away from resonance.

    Parameters
    ----------
    eigenvalues : array-like, characteristic values lambda_n (real, from
        GeneralizedEigenSolver.solve or a CMAFrequencySweepResult)

    Returns
    -------
    ndarray, same shape as `eigenvalues`, modal significance MS_n in (0, 1]
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    return 1.0 / np.sqrt(1.0 + eigenvalues ** 2)
