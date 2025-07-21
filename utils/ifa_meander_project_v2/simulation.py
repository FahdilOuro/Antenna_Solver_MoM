from src.radiation_algorithm.radiation_algorithm import radiation_algorithm
from efield.efield4 import *
import numpy as np

from utils.frequency_sweep import load_cst_data, plot_smith_chart_CST_MoM

def analysis(frequencies, ifa_meander_mat, feed_point):
    Z0=50
    s11_db = []
    impedances = []
    nPoints = len(frequencies)
    for idx, frequency in enumerate(frequencies):
        visualiser = False
        impedance, _ = radiation_algorithm(ifa_meander_mat, frequency, feed_point, 0.5, show=visualiser)
        impedances.append(impedance)
        s11 = (impedance - Z0) / (impedance + Z0)
        s11_db.append(20 * np.log10(abs(s11)))
        print(f"Simulation {idx+1}/{nPoints} | f = {frequency/1e6:.2f} MHz | S11 = {s11_db[-1]:.2f} dB")
    # Résultats
    min_index = np.argmin(s11_db)
    f_resonance = frequencies[min_index]
    Z_at_res = impedances[min_index]
    R_res = Z_at_res.real
    X_res = Z_at_res.imag

    print(f"\n📡 Résultats de simulation :")
    print(f"→ Fréquence de résonance = {f_resonance / 1e6:.2f} MHz")
    print(f"→ Impédance à f_res      = {Z_at_res:.2f} Ω")

    return f_resonance, s11_db, R_res

def simulate(frequencies, ifa_meander_mat, fC, feed_point):
    Z0=50
    s11_db = []
    impedances = []
    nPoints = len(frequencies)
    for idx, frequency in enumerate(frequencies):
        visualiser = (frequency == fC)
        impedance, _ = radiation_algorithm(ifa_meander_mat, frequency, feed_point, voltage_amplitude=0.5, show=visualiser, save_image=True)
        impedances.append(impedance)
        s11 = (impedance - Z0) / (impedance + Z0)
        s11_db.append(20 * np.log10(abs(s11)))
        print(f"Simulation {idx+1}/{nPoints} | f = {frequency/1e6:.2f} MHz | S11 = {s11_db[-1]:.2f} dB")
    # Résultats
    min_index = np.argmin(s11_db)
    f_resonance = frequencies[min_index]
    Z_at_res = impedances[min_index]
    R_res = Z_at_res.real
    X_res = Z_at_res.imag
    plot_smith_chart(impedances, frequencies, fC, save_image=True)

    """ # filepath = 'data/plot_file/plot_smith_chart_cst_Antenne_Ta_92_Tb_55_a_27.txt'
    filepath = 'data/plot_file/plot_smith_chart_cst_Antenne_Ta_89_Tb_44_a_29.txt'
    frequencies_cst, mag_cst, phase_cst = load_cst_data(filepath)
    
    plot_smith_chart_CST_MoM(frequencies, impedances, frequencies_cst, mag_cst, phase_cst, fC) """
    

    print(f"\n📡 Résultats de simulation :")
    print(f"→ Fréquence de résonance = {f_resonance / 1e6:.2f} MHz")
    print(f"→ Impédance à f_res      = {Z_at_res:.2f} Ω")

    return f_resonance, s11_db, R_res, X_res