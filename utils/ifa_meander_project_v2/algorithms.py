from utils.gmsh_function import extract_msh_to_mat
from utils.ifa_meander_project_v2.geometry import *
from utils.ifa_meander_project_v2.meshing import *
from utils.ifa_meander_project_v2.simulation import *
from utils.ifa_meander_project_v2.meshing import *

import time
from matplotlib import pyplot as plt
import os

def adapt_with_ratio_square(distance_meandre, largeur_piste, ratio):
    return distance_meandre / (ratio**2), largeur_piste / (ratio**2)

def adapt_with_ratio_cube(distance_meandre, largeur_piste, ratio):
    return distance_meandre / (ratio**3), largeur_piste / (ratio**3)

def creation_ifa(msh_file, mat_file, largeur, hauteur, width, dist_meandre, feed, x_t, y_t, save_mesh_folder, mesh_name, mesh_size):
    x, y, N, distance_meandre = ifa_creation(largeur, hauteur, width, dist_meandre)
    x_m, y_m = trace_meander(x, y, width)
    feed_wid = width # La largeur de la piste est la meme partout
    feed_x = np.array([0, distance_meandre, distance_meandre, 0])
    feed_y = np.array([feed + feed_wid/2, feed + feed_wid/2, feed -feed_wid/2, feed - feed_wid/2])
    antenna_ifa_meander(x_m, y_m, x_t, y_t, feed_x, feed_y, save_mesh_folder, mesh_name, mesh_size)
    extract_msh_to_mat(msh_file, mat_file)
    return N, distance_meandre

def plot_s11_curve(fLow, fHigh, nPoints, s11_db, fC=None):
    plt.style.use('fivethirtyeight')
    plt.rcParams['font.family'] = 'JetBrains Mono'
    frequencies = np.linspace(fLow, fHigh, nPoints)
    frequencies_mhz = np.array(frequencies) / 1e6
    s11_db = np.array(s11_db)

    # Trouver le minimum de S11
    min_index = np.argmin(s11_db)
    f_resonance = frequencies[min_index] / 1e6
    s11_min = s11_db[min_index]

    # Tracé
    fig_size = 12
    Fibonacci = (1 + np.sqrt(5)) / 2
    plt.figure(figsize=(fig_size, fig_size / Fibonacci))
    plt.plot(frequencies_mhz, s11_db, label="S11 (dB)", color='blue')
    plt.plot(f_resonance, s11_min, 'ro', 
            label=f"Résonance: {f_resonance:.2f} MHz (S11={s11_min:.2f} dB)")
    
    if fC:
        fC_mhz = fC / 1e6
        idx_fc = np.argmin(np.abs(frequencies - fC))
        s11_fc = s11_db[idx_fc]
        plt.axvline(fC_mhz, color='green', linestyle='--', 
                   label=f"fC = {fC_mhz:.2f} MHz (S11={s11_fc:.2f} dB)")

    plt.xlabel("Fréquence (MHz)")
    plt.ylabel("S11 (dB)")
    plt.title("Courbe de S11 vs Fréquence")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def optimize_ifa(msh_file, mat_file,
                 frequencies, frequence_Coupure, fLow, fHigh, nPoints, fc_index,
                 a, b, largeur_piste, distance_meandre, feed, feed_point, x_t, y_t, save_mesh_folder, mesh_name, mesh_size):
    start_time = time.time()
    iteration = 1
    max_iter = 20
    Accuracy = 0.01
    converged = False
    while iteration <= max_iter and not converged:
        try:
            print(f"\n------------------------------------------------------Iteration N°{iteration}------------------------------------------------------\n")
            print(f"distance meandres = {distance_meandre * 1000:.3f} mm\n")
            
            N_list_elem, new_distance_meandre_elem = creation_ifa(msh_file, mat_file, a, b, largeur_piste, distance_meandre, feed, x_t, y_t, save_mesh_folder, mesh_name, mesh_size)
            print(f"Number of meanders = {N_list_elem}\n")

            print(f"New distance meandres = {new_distance_meandre_elem * 1000:.3f} mm")
            print(f"Largeur de piste ifa = {largeur_piste * 1000:.3f} mm")
            print(f"position feed = {feed * 1000:.3f} mm\n")

            frequence_resonance, s11_db, R_res, X_res = simulate(frequencies, mat_file, frequence_Coupure, feed_point)
            # Q = calculate_Q(frequencies, s11_db, frequence_resonance)
            plot_s11_curve(fLow, fHigh, nPoints, s11_db, frequence_Coupure)

            ratio = frequence_resonance / frequence_Coupure
            print(f"\nRatio = {ratio}\n")
            # distance_meandre = distance_meandre / ratio

            if (abs((frequence_Coupure - frequence_resonance)/frequence_Coupure) < Accuracy):
                min_index = np.argmin(s11_db)
                min_s11 = s11_db[min_index]
                s11_fc = s11_db[fc_index]
                if s11_fc < -10 or min_s11 < -20:
                    converged = True
                    print("\nRequired Accuracy is met!")
                    break
                else:
                    print("\nOn cherche le matching !!!")

                    if ratio == 1:
                        print("Ratio == 1 on modifie le feed parce quon a pas une bonne adaptation")
                        ratio_adapt_feed = math.sqrt(R_res / 50)
                        feed = max(min(feed / ratio_adapt_feed, b - 3 * largeur_piste / 2 - 2.5 / 1000), largeur_piste / 2)
                    else:
                        adapt_with_ratio_cube(distance_meandre, largeur_piste, ratio)

                        feed = max(min(feed * ratio**2, b - 3 * largeur_piste / 2 - 2.5 / 1000), largeur_piste / 2)

                        print(f"\nresultat feed = {feed * 1000:.3f} mm")

                    if feed >= b - 3 * largeur_piste / 2 - 2.5 / 1000 or feed <= largeur_piste / 2:
                        distance_meandre, largeur_piste = adapt_with_ratio_square(distance_meandre, largeur_piste, ratio)

                    feed_point       = [0, feed, 0]

            elif abs(frequence_resonance - frequence_Coupure) < 0.07 * frequence_Coupure:
                print(f"\nWe are within 2% of fc!\n")
                feed = max(min(feed * ratio**2, b - 3 * largeur_piste / 2 - 2.5 / 1000), largeur_piste / 2)

                if feed >= b - 3 * largeur_piste / 2 - 2.5 / 1000 or feed <= largeur_piste / 2:
                    print("\nBord extreme atteint\n")
                    distance_meandre, largeur_piste = adapt_with_ratio_square(distance_meandre, largeur_piste, ratio)
                
                feed_point       = [0, feed, 0]
            else:
                print(f"\nWe are FAR of fc!\n")
                distance_meandre, largeur_piste = adapt_with_ratio_square(distance_meandre, largeur_piste, ratio)
            
            iteration += 1
        except ValueError as e:
            print(f"Error: {e}")

    end_time = time.time()
    simulation_time = end_time - start_time
    simulation_time_minutes = simulation_time / 60
    simulation_time_seconds = simulation_time % 60
    print(f"Temps de simulation : {simulation_time_minutes:.0f} minutes et {simulation_time_seconds:.2f} secondes")
    if converged:
        print(f"Convergence atteinte à l'itération {iteration}")
    else:
        print(f"Convergence non atteinte après {max_iter} itérations")
