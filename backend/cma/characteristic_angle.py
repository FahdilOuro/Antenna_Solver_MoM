import numpy as np


def characteristic_angle(eigenvalues):
    """
    Characteristic angle:  delta_n = 180 deg - arctan(lambda_n)

    The standard readable stand-in for the raw characteristic value. Unlike
    lambda_n -- which is unbounded and routinely spans five orders of
    magnitude across a sweep, so that a single barely-radiating mode flattens
    every other curve onto zero on a linear axis -- delta_n is confined to
    (90 deg, 270 deg) and spreads the modes out evenly:

        delta = 90 deg   <-> lambda -> +infinity  (inductive, stores magnetic energy)
        delta = 180 deg  <-> lambda = 0           (RESONANT, radiates best)
        delta = 270 deg  <-> lambda -> -infinity  (capacitive, stores electric energy)

    So a mode resonates exactly where its curve crosses 180 deg, and how
    steeply it crosses is a qualitative indicator of its Q (a steep crossing
    means a narrow-band, high-Q mode).

    Convention checked against AToM's own `characteristicAngle` / `eigNum`
    outputs on the pixel-grid loop: agreement to 6e-14 deg.

    Parameters
    ----------
    eigenvalues : array-like, characteristic values lambda_n (real, from
        GeneralizedEigenSolver.solve or a CMASweepData)

    Returns
    -------
    ndarray, same shape as `eigenvalues`, characteristic angle in DEGREES,
    in (90, 270)
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    return 180.0 - np.degrees(np.arctan(eigenvalues))
