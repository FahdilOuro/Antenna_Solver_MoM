import numpy as np


def calc_frequencies(fC, delta_f, nPoints):
    """
    Calcule fLow, fHigh et la liste des fréquences centrées sur fC,
    avec un écart delta_f et nPoints échantillons.
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
        # Trouve le plus petit nPoints qui inclut fC
        nPoints = min_points
        while True:
            frequencies = np.linspace(fLow, fHigh, nPoints)
            if fC in frequencies:
                return nPoints
            nPoints += 1

def generate_frequencies(fLow, fHigh, fC, step):
    """
    Génère une liste de fréquences entre fLow et fHigh avec un pas donné.
    Vérifie que fC est inclus dans la liste et retourne son index et le nombre de points.
    
    Paramètres:
    - fLow : fréquence basse (en Hz)
    - fHigh : fréquence haute (en Hz)
    - fC : fréquence centrale à inclure (en Hz)
    - step : pas entre les fréquences (en Hz)
    
    Retourne:
    - frequencies : liste des fréquences
    - fC_included : booléen indiquant si fC est dans la liste
    - fC_index : index de fC dans la liste (ou None si non inclus)
    - nPoints : nombre total de fréquences générées
    """
    if (fC - fLow) % step != 0:
        raise ValueError("fC ne tombe pas sur un pas de fréquence. Ajuste fLow, fC ou le step.")

    nPoints = int((fHigh - fLow) // step) + 1
    frequencies = [fLow + i * step for i in range(nPoints)]
    fC_included = fC in frequencies
    fC_index = frequencies.index(fC) if fC_included else None

    return frequencies, fC_index, nPoints
