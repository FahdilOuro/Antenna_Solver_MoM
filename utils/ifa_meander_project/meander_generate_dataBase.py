import os
import numpy as np
import pandas as pd

from src.radiation_algorithm.radiation_algorithm import radiation_algorithm
from utils.gmsh_function import *
from utils.ifa_meander_project.ifa_creation_functions import ifa_creation_optimisation, trace_meander_new
from utils.ifa_meander_project.ifa_meander_gmsh import antenna_ifa_meander

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
    s11_db_min = s11_db[min_index]
    f_resonance = frequencies[min_index]
    Z_at_res = impedances[min_index]
    R_res = Z_at_res.real
    X_res = Z_at_res.imag

    print(f"\n📡 Résultats de simulation :")
    print(f"→ Fréquence de résonance = {f_resonance / 1e6:.2f} MHz")
    print(f"→ Impédance à f_res      = {Z_at_res:.2f} Ω")

    return f_resonance, R_res, X_res, s11_db_min

def generate_antenna_database(
        fLow, fHigh, nPoints, fC,
        distance_shorts, L_shorts, widths, nombre_meanders,
        ifa_meander_mat, ifa_meander_msh,
        feed_point,
        a, b, Lenght_antenna, feed, feed_wid,
        save_mesh_folder, mesh_name,
        x_t, y_t,
        output_file
        ):
    # verifier l'existance du dossier output_file
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    frequencies = np.linspace(fLow, fHigh, nPoints)
    voltage_amplitude = 0.5
    data = []

    for L_short in L_shorts:
        for dis_short in distance_shorts:
            for wid in widths:
                for n in nombre_meanders:

                    # -------------

                    x, y = ifa_creation_optimisation(Lenght_antenna, a, b, wid, n, L_short)
                    x_m, y_m = trace_meander_new(x, y, wid)
                    feed_x = np.array([0, L_short-wid/2, L_short-wid/2, 0])
                    feed_y = np.array([feed + feed_wid/2, feed + feed_wid/2, feed - feed_wid/2, feed - feed_wid/2])
                    antenna_ifa_meander(x_m, y_m, x_t, y_t, feed_x, feed_y, save_mesh_folder, mesh_name, 2.25/1000)

                    extract_msh_to_mat(ifa_meander_msh, ifa_meander_mat)

                    f_res, Zr, Zi, S11 = simulate_frequency_sweep(
                        frequencies, fC, ifa_meander_mat, feed_point, voltage_amplitude, Z0=50
                        )

                    # -------------

                    data.append({
                        "distance_shorts": dis_short,
                        "Lshort": L_short,
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
