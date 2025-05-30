import numpy as np


def calculate_nPoints(fLow, fHigh, fC, min_points=5):
    step = (fHigh - fLow) / (min_points - 1)
    if (fC - fLow) % step == 0:
        return min_points
    else:
        # Trouve le plus petit nPoints qui inclut fC
        nPoints = min_points
        while True:
            frequencies = np.linspace(fLow, fHigh, nPoints)
            if fC in frequencies:
                return nPoints
            nPoints += 1

def calc_frequencies(fC, delta_f, nPoints):
    """
    Calcule fLow, fHigh et la liste des fréquences centrées sur fC,
    avec un écart delta_f et nPoints échantillons.
    """
    fLow = fC - delta_f * (nPoints // 2)
    fHigh = fC + delta_f * ((nPoints - 1) // 2)
    frequencies = np.linspace(fLow, fHigh, nPoints)
    return fLow, fHigh, frequencies