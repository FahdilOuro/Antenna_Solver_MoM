import numpy as np
import pandas as pd

from src.radiation_algorithm.radiation_algorithm import radiation_algorithm

def simulate_frequency_sweep(frequencies, fC, ifa_meander_mat, feed_point, voltage_amplitude, Z0=50):
    s11_db = []
    impedances = []
    nPoints = len(frequencies)
    for idx, frequency in enumerate(frequencies):
        visualiser = (frequency == fC)
        impedance, _ = radiation_algorithm(ifa_meander_mat, frequency, feed_point, voltage_amplitude, show=visualiser)
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

    return f_resonance, R_res, X_res, s11_db

def generate_antenna_database(
        fLow, fHigh, nPoints, 
        distance_short, widths, nombre_meanders,
        fC, ifa_meander_mat, feed_point,
        output_file = "data/antenna_database.csv"
        ):
    frequencies = np.linspace(fLow, fHigh, nPoints)
    voltage_amplitude = 0.5
    data = []

    for dis_short in distance_short:
        for wid in widths:
            for n in nombre_meanders:
                f_res, Zr, Zi, S11 = simulate_frequency_sweep(
                    frequencies, fC, ifa_meander_mat, feed_point, voltage_amplitude, Z0=50
                    )

                data.append({
                    "position_sc": dis_short,
                    "width": wid,
                    "n_meanders": n,
                    "f_res": f_res,
                    "Zin_real": Zr,
                    "Zin_imag": Zi,
                    "S11_min": S11
                })

    # Création de la base de données
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)

    print(f"Base de données enregistrée avec succès dans {output_file}.")
