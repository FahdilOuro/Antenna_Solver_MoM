import numpy as np


def calc_frequencies(fC, delta_f, nPoints):
    """
        Calculates fLow, fHigh and the list of frequencies centered on fC,
        with a spacing delta_f and nPoints samples.
    """
    fLow = fC - delta_f * (nPoints // 2)
    fHigh = fC + delta_f * ((nPoints - 1) // 2)
    frequencies = np.linspace(fLow, fHigh, nPoints)
    return fLow, fHigh, frequencies

def calculate_nPoints(fLow, fHigh, fC, min_points=27):
    step = (fHigh - fLow) / (min_points - 1)
    if (fC - fLow) % step == 0:
        return min_points
    else:
        # Find the smallest nPoints that includes fC
        nPoints = min_points
        while True:
            frequencies = np.linspace(fLow, fHigh, nPoints)
            if fC in frequencies:
                return nPoints
            nPoints += 1

def generate_frequencies(fLow, fHigh, fC, step):
    """
        Generates a list of frequencies between fLow and fHigh with a given step.
        Checks that fC is included in the list and returns its index and the number of points.

        Parameters:
            - fLow: lower frequency (in Hz)
            - fHigh: upper frequency (in Hz)
            - fC: central frequency to include (in Hz)
            - step: frequency step (in Hz)

        Returns:
            - frequencies: list of frequencies
            - fC_included: boolean indicating if fC is in the list
            - fC_index: index of fC in the list (or None if not included)
            - nPoints: total number of generated frequencies
    """
    if (fC - fLow) % step != 0:
        raise ValueError("fC does not fall on a frequency step. Adjust fLow, fC, or the step.")

    nPoints = int((fHigh - fLow) // step) + 1
    frequencies = [fLow + i * step for i in range(nPoints)]
    fC_included = fC in frequencies
    fC_index = frequencies.index(fC) if fC_included else None

    return frequencies, fC_index, nPoints
