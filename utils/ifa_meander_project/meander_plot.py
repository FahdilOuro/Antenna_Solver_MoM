from matplotlib import pyplot as plt
import numpy as np


def plot_s11_curve(fLow, fHigh, nPoints, s11_db, fC=None):

    frequencies = np.linspace(fLow, fHigh, nPoints)
    frequencies_mhz = np.array(frequencies) / 1e6
    s11_db = np.array(s11_db)

    # Trouver le minimum de S11
    min_index = np.argmin(s11_db)
    f_resonance = frequencies[min_index] / 1e6
    s11_min = s11_db[min_index]

    # Tracé
    fig_size = 8
    Fibonacci = (1 + np.sqrt(5)) / 2
    plt.figure(figsize=(fig_size, fig_size / Fibonacci))
    plt.plot(frequencies_mhz, s11_db, label="S11 (dB)", color='blue')
    plt.plot(f_resonance, s11_min, 'ro', label=f"Résonance: {f_resonance:.2f} MHz")
    
    if fC:
        plt.axvline(fC / 1e6, color='green', linestyle='--', label=f"fC = {fC/1e6:.2f} MHz")

    plt.xlabel("Fréquence (MHz)")
    plt.ylabel("S11 (dB)")
    plt.title("Courbe de S11 vs Fréquence")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_impedance(fLow, fHigh, nPoints, impedances, s11_db, fC=None):
    # Vecteur de fréquences
    frequencies = np.linspace(fLow, fHigh, nPoints)
    frequencies_mhz = frequencies / 1e6

    # Partie réelle de l’impédance
    impedances_real = np.real(impedances)

    # Trouver la fréquence de résonance (minimum de S11)
    min_index = np.argmin(s11_db)
    f_resonance = frequencies[min_index] / 1e6
    impedance_resonance = impedances_real[min_index]

    # Préparer le tracé
    fig_size = 8
    golden_ratio = (1 + np.sqrt(5)) / 2
    plt.figure(figsize=(fig_size, fig_size / golden_ratio))

    # Courbe d'impédance réelle
    plt.plot(frequencies_mhz, impedances_real, label="Re(Z)", color='red')

    # Marqueur de résonance
    plt.plot(f_resonance, impedance_resonance, 'bo', label=f"Résonance: {f_resonance:.2f} MHz")

    # Si une fréquence cible est donnée
    if fC is not None:
        fC_mhz = fC / 1e6
        index_fC = np.argmin(np.abs(frequencies - fC))
        impedance_fC = impedances_real[index_fC]
        plt.plot(fC_mhz, impedance_fC, 'go', label=f"fC: {fC_mhz:.2f} MHz")
        plt.axvline(fC_mhz, color='green', linestyle='--')

    # Mise en forme
    plt.xlabel("Fréquence (MHz)")
    plt.ylabel("Impédance réelle (Ohm)")
    plt.title("Impédance réelle vs Fréquence")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()