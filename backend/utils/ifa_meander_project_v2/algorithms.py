from utils.gmsh_function import extract_msh_to_mat
from utils.ifa_meander_project_v2.geometry import *
from utils.ifa_meander_project_v2.meshing import *
from utils.ifa_meander_project_v2.simulation import *
from utils.ifa_meander_project_v2.meshing import *

import time
from matplotlib import pyplot as plt
import json

def adapt_with_ratio_square(distance_meandre, largeur_piste, ratio):
    return distance_meandre / (ratio**2), largeur_piste / (ratio**2)

def adapt_with_ratio_cube(distance_meandre, largeur_piste, ratio):
    return distance_meandre / (ratio**3), largeur_piste / (ratio**3)

def creation_ifa(msh_file, mat_file, largeur, hauteur, width, dist_meandre, feed, x_t, y_t, save_mesh_folder, mesh_name, mesh_size):
    x, y, N, distance_meandre = ifa_creation(largeur, hauteur, width, dist_meandre)
    x_m, y_m = trace_meander(x, y, width)
    feed_wid = width                # The width of the track is the same everywhere.
    feed_x = np.array([0, distance_meandre, distance_meandre, 0])
    feed_y = np.array([feed + feed_wid/2, feed + feed_wid/2, feed -feed_wid/2, feed - feed_wid/2])
    antenna_ifa_meander(x_m, y_m, x_t, y_t, feed_x, feed_y, save_mesh_folder, mesh_name, mesh_size)
    extract_msh_to_mat(msh_file, mat_file)
    return N, distance_meandre

def plot_s11_curve(fLow, fHigh, nPoints, s11_db, fC=None):
    plt.style.use('seaborn-v0_8-talk')
    plt.rcParams['font.family'] = 'Lucida Console'
    plt.rcParams['font.size'] = 11
    frequencies = np.linspace(fLow, fHigh, nPoints)
    frequencies_mhz = np.array(frequencies) / 1e6
    s11_db = np.array(s11_db)

    # Find the minimum of S11
    min_index = np.argmin(s11_db)
    f_resonance = frequencies[min_index] / 1e6
    s11_min = s11_db[min_index]

    # Plot
    fig_size = 12
    Fibonacci = (1 + np.sqrt(5)) / 2
    plt.figure(figsize=(fig_size, fig_size / Fibonacci))
    plt.plot(frequencies_mhz, s11_db, label="S11 (dB)", color='blue')
    plt.plot(f_resonance, s11_min, 'ro', 
            label=f"Resonance: {f_resonance:.2f} MHz (S11={s11_min:.2f} dB)")
    
    if fC:
        fC_mhz = fC / 1e6
        idx_fc = np.argmin(np.abs(frequencies - fC))
        s11_fc = s11_db[idx_fc]
        plt.axvline(fC_mhz, color='green', linestyle='--', 
                   label=f"fC = {fC_mhz:.2f} MHz (S11={s11_fc:.2f} dB)")

    plt.xlabel("Frequency (MHz)")
    plt.ylabel("S11 (dB)")
    plt.title("S11 curve vs Frequency")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def optimize_ifa(msh_file, mat_file,
                 frequencies, frequence_Coupure, fLow, fHigh, nPoints, fc_index,
                 a, b, largeur_piste, distance_meandre, feed, feed_point, x_t, y_t, save_mesh_folder, mesh_name, mesh_size):
    start_time = time.time()
    iteration = 1
    Z0 = 50
    max_iter = 20
    Accuracy = 0.01
    converged = False
    while iteration <= max_iter and not converged:
        try:
            print(f"\n------------------------------------------------------Iteration N°{iteration}------------------------------------------------------\n")
            print(f"distance meanders = {distance_meandre * 1000:.3f} mm\n")
            
            N_list_elem, new_distance_meandre_elem = creation_ifa(msh_file, mat_file, a, b, largeur_piste, distance_meandre, feed, x_t, y_t, save_mesh_folder, mesh_name, mesh_size)
            print(f"Number of meanders = {N_list_elem}\n")

            print(f"New distance meanders = {new_distance_meandre_elem * 1000:.3f} mm")
            print(f"IFA track width = {largeur_piste * 1000:.3f} mm")
            print(f"feed position = {feed * 1000:.3f} mm\n")

            frequence_resonance, s11_db, R_res, X_res = simulate(frequencies, mat_file, frequence_Coupure, feed_point)
            # Q = calculate_Q(frequencies, s11_db, frequence_resonance)
            plot_s11_curve(fLow, fHigh, nPoints, s11_db, frequence_Coupure)

            ratio = frequence_resonance / frequence_Coupure
            print(f"\nRatio = {ratio}\n")

            if (abs((frequence_Coupure - frequence_resonance)/frequence_Coupure) < Accuracy):
                min_index = np.argmin(s11_db)
                min_s11 = s11_db[min_index]
                s11_fc = s11_db[fc_index]
                if s11_fc < -10 or min_s11 < -20:
                    converged = True
                    print("\nRequired accuracy is met!")
                    break
                else:
                    print("\nLooking for matching!!!")

                    if ratio == 1:
                        print("Ratio == 1, modifying the feed because matching is not good")
                        ratio_adapt_feed = math.sqrt(R_res / Z0)
                        feed = max(min(feed / ratio_adapt_feed, b - 3 * largeur_piste / 2 - 2.5 / 1000), largeur_piste / 2)
                    else:
                        adapt_with_ratio_cube(distance_meandre, largeur_piste, ratio)

                        feed = max(min(feed * ratio**2, b - 3 * largeur_piste / 2 - 2.5 / 1000), largeur_piste / 2)

                        print(f"\nfeed result = {feed * 1000:.3f} mm")

                    if feed >= b - 3 * largeur_piste / 2 - 2.5 / 1000 or feed <= largeur_piste / 2:
                        distance_meandre, largeur_piste = adapt_with_ratio_square(distance_meandre, largeur_piste, ratio)

                    feed_point       = np.array([0, feed, 0])

            elif abs(frequence_resonance - frequence_Coupure) < 0.07 * frequence_Coupure:
                print(f"\nWe are within 2% of fc!\n")
                feed = max(min(feed * ratio**2, b - 3 * largeur_piste / 2 - 2.5 / 1000), largeur_piste / 2)

                if feed >= b - 3 * largeur_piste / 2 - 2.5 / 1000 or feed <= largeur_piste / 2:
                    print("\nExtreme border reached\n")
                    distance_meandre, largeur_piste = adapt_with_ratio_square(distance_meandre, largeur_piste, ratio)
                
                feed_point       = np.array([0, feed, 0])
            else:
                print(f"\nWe are FAR from fc!\n")
                distance_meandre, largeur_piste = adapt_with_ratio_square(distance_meandre, largeur_piste, ratio)
            
            iteration += 1
        except ValueError as e:
            print(f"Error: {e}")

    end_time = time.time()
    simulation_time = end_time - start_time
    simulation_time_minutes = simulation_time / 60
    simulation_time_seconds = simulation_time % 60
    print(f"Simulation time: {simulation_time_minutes:.0f} minutes and {simulation_time_seconds:.2f} seconds")
    if converged:
        print(f"Convergence reached at iteration {iteration}")
    else:
        print(f"Convergence not reached after {max_iter} iterations")

def optimize_ifa_input_impedance(msh_file, mat_file,
                 frequencies, frequence_Coupure, fLow, fHigh, nPoints, fc_index,
                 a, b, largeur_piste, distance_meandre, feed, feed_point, x_t, y_t, save_mesh_folder, mesh_name, mesh_size, Z0=25, filename="data/optimization_parameters/ifa_optimization_params.json"):
    start_time = time.time()
    iteration = 1
    max_iter = 30
    Accuracy = 0.01
    converged = False
    while iteration <= max_iter and not converged:
        try:
            print(f"\n------------------------------------------------------Iteration N°{iteration}------------------------------------------------------\n")
            print(f"distance meanders = {distance_meandre * 1000:.3f} mm\n")
            
            N_list_elem, new_distance_meandre_elem = creation_ifa(msh_file, mat_file, a, b, largeur_piste, distance_meandre, feed, x_t, y_t, save_mesh_folder, mesh_name, mesh_size)
            print(f"Number of meanders = {N_list_elem}\n")

            print(f"New distance meanders = {new_distance_meandre_elem * 1000:.3f} mm")
            print(f"IFA track width = {largeur_piste * 1000:.3f} mm")
            print(f"feed position = {feed * 1000:.3f} mm\n")

            frequence_resonance, s11_db, R_res, X_res, impedances = simulate_433_project(frequencies, mat_file, frequence_Coupure, feed_point, Z0=Z0)

            print("--- Input Impedance at resonance ---")
            print(f"R_res = {R_res:.2f} Ohm")
            print(f"X_res = {X_res:.2f} Ohm")
            print("--- Input Impedance at resonance ---")

            # Q = calculate_Q(frequencies, s11_db, frequence_resonance)
            plot_s11_curve(fLow, fHigh, nPoints, s11_db, frequence_Coupure)

            ratio = frequence_resonance / frequence_Coupure
            print(f"\nRatio = {ratio}\n")

            if (abs((frequence_Coupure - frequence_resonance)/frequence_Coupure) < Accuracy):
                min_index = np.argmin(s11_db)
                min_s11 = s11_db[min_index]
                s11_fc = s11_db[fc_index]
                if (s11_fc <= -10):   # Ici je veux savoir combien mettre comme valeur de s11 a fc pour valider la convergance en fonction de Z_adapt
                    converged = True
                    print("\nRequired accuracy is met!")
                    break
                else:
                    print("\nLooking for matching!!!")

                    if ratio == 1:
                        print("Ratio == 1, modifying the feed because matching is not good")
                        ratio_adapt_feed = math.sqrt(R_res / Z0)
                        feed = max(min(feed / (ratio_adapt_feed * 1.2), b - 3 * largeur_piste / 2 - 2.5 / 1000), largeur_piste / 2)
                    else:
                        adapt_with_ratio_cube(distance_meandre, largeur_piste, ratio)

                        feed = max(min(feed * ratio**2, b - 3 * largeur_piste / 2 - 2.5 / 1000), largeur_piste / 2)

                        print(f"\nfeed result = {feed * 1000:.3f} mm")

                    if feed >= b - 3 * largeur_piste / 2 - 2.5 / 1000 or feed <= largeur_piste / 2:
                        distance_meandre, largeur_piste = adapt_with_ratio_square(distance_meandre, largeur_piste, ratio)

                    feed_point       = np.array([0, feed, 0])

            elif (abs((frequence_Coupure - frequence_resonance)/frequence_Coupure) < 0.07):
                print(f"\nWe are within 2% of fc!\n")
                feed = max(min(feed * ratio**2, b - 3 * largeur_piste / 2 - 2.5 / 1000), largeur_piste / 2)

                if feed >= b - 3 * largeur_piste / 2 - 2.5 / 1000 or feed <= largeur_piste / 2:
                    print("\nExtreme border reached\n")
                    distance_meandre, largeur_piste = adapt_with_ratio_square(distance_meandre, largeur_piste, ratio)
                
                feed_point       = np.array([0, feed, 0])
            else:
                print(f"\nWe are FAR from fc!\n")
                distance_meandre, largeur_piste = adapt_with_ratio_square(distance_meandre, largeur_piste, ratio)
            
            iteration += 1
        except ValueError as e:
            print(f"Error: {e}")

    end_time = time.time()
    simulation_time = end_time - start_time
    simulation_time_minutes = simulation_time / 60
    simulation_time_seconds = simulation_time % 60
    print(f"Simulation time: {simulation_time_minutes:.0f} minutes and {simulation_time_seconds:.2f} seconds")
    if converged:
        print(f"Convergence reached at iteration {iteration}")
    else:
        print(f"Convergence not reached after {max_iter} iterations")

    # Save only a, largeur_piste, distance_meandre, and feed in JSON format
    data = {
        "a": a,
        "largeur_piste": largeur_piste,
        "distance_meandre": distance_meandre,
        "feed": feed,
        "mesh_size": mesh_size
    }

    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Optimization parameters saved to {filename}")

    return impedances

def impedance_to_s11(frequencies, impedances, filename="data/reflexion_coef_files/new.s1p", Z0=50):
    """
    Convert impedance data to S11 parameters and save as .s1p file
    
    Parameters:
    -----------
    frequencies : list or array
        List of frequencies in Hz
    impedances : list or array
        List of complex impedances in Ohms (same length as frequencies)
    filename : str
        Output filename (default: "output.s1p")
    Z0 : float
        Reference impedance (default: 50 ohms)
    """
    
    # Check that both lists have the same length
    if len(frequencies) != len(impedances):
        raise ValueError(f"Frequencies and impedances must have the same length. "
                        f"Got {len(frequencies)} frequencies and {len(impedances)} impedances.")
    
    # Create directory if it doesn't exist
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # Open file for writing
    with open(filename, 'w') as f:
        # Write header
        f.write("! S1P file generated from impedance data\n")
        f.write(f"! Reference impedance Z0 = {Z0} Ohms\n")
        f.write(f"# Hz S RI R {Z0}\n")
        f.write("!\n")
        
        # Process each frequency point
        for freq, Z in zip(frequencies, impedances):
            # Convert impedance to S11 using reflection coefficient formula
            # S11 = (Z - Z0) / (Z + Z0)
            s11 = (Z - Z0) / (Z + Z0)
            
            # Extract real and imaginary parts
            s11_real = np.real(s11)
            s11_imag = np.imag(s11)
            
            # Write frequency and S11 (real, imaginary) to file
            f.write(f"{freq:.6e} {s11_real:.6e} {s11_imag:.6e}\n")
    
    print(f"S1P file '{filename}' created successfully with {len(frequencies)} frequency points")


"""def optimize_ifa_input_impedance(msh_file, mat_file,
                 frequencies, frequence_Coupure, fLow, fHigh, nPoints, fc_index,
                 a, b, largeur_piste, distance_meandre, feed, feed_point, x_t, y_t, save_mesh_folder, mesh_name, mesh_size, Z0=25, filename="data/optimization_parameters/ifa_optimization_params.json"):
    start_time = time.time()
    iteration = 1
    max_iter = 30
    Accuracy = 0.01
    converged = False
    while iteration <= max_iter and not converged:
        try:
            print(f"\n------------------------------------------------------Iteration N°{iteration}------------------------------------------------------\n")
            print(f"distance meanders = {distance_meandre * 1000:.3f} mm\n")
            
            N_list_elem, new_distance_meandre_elem = creation_ifa(msh_file, mat_file, a, b, largeur_piste, distance_meandre, feed, x_t, y_t, save_mesh_folder, mesh_name, mesh_size)
            print(f"Number of meanders = {N_list_elem}\n")

            print(f"New distance meanders = {new_distance_meandre_elem * 1000:.3f} mm")
            print(f"IFA track width = {largeur_piste * 1000:.3f} mm")
            print(f"feed position = {feed * 1000:.3f} mm\n")

            frequence_resonance, s11_db, R_res, X_res, impedances = simulate_433_project(frequencies, mat_file, frequence_Coupure, feed_point, Z0=Z0)

            print("--- Input Impedance at resonance ---")
            print(f"R_res = {R_res:.2f} Ohm")
            print(f"X_res = {X_res:.2f} Ohm")
            print("--- Input Impedance at resonance ---")

            # Q = calculate_Q(frequencies, s11_db, frequence_resonance)
            plot_s11_curve(fLow, fHigh, nPoints, s11_db, frequence_Coupure)

            ratio = frequence_resonance / frequence_Coupure
            print(f"\nRatio = {ratio}\n")

            if (abs((frequence_Coupure - frequence_resonance)/frequence_Coupure) < Accuracy):
                min_index = np.argmin(s11_db)
                min_s11 = s11_db[min_index]
                s11_fc = s11_db[fc_index]
                if (s11_fc <= -6):   # Ici je veux savoir combien mettre comme valeur de s11 a fc pour valider la convergance en fonction de Z_adapt
                    converged = True
                    print("\nRequired accuracy is met!")
                    break
                else:
                    print("\nLooking for matching!!!")

                    if ratio == 1:
                        print("Ratio == 1, modifying the feed because matching is not good")
                        ratio_adapt_feed = math.sqrt(R_res / Z0)
                        feed = max(min(feed / (ratio_adapt_feed * 1.2), b - 3 * largeur_piste / 2 - 2.5 / 1000), largeur_piste / 2)
                    else:
                        adapt_with_ratio_cube(distance_meandre, largeur_piste, ratio)

                        feed = max(min(feed * ratio**2, b - 3 * largeur_piste / 2 - 2.5 / 1000), largeur_piste / 2)

                        print(f"\nfeed result = {feed * 1000:.3f} mm")

                    if feed >= b - 3 * largeur_piste / 2 - 2.5 / 1000 or feed <= largeur_piste / 2:
                        distance_meandre, largeur_piste = adapt_with_ratio_square(distance_meandre, largeur_piste, ratio)

                    feed_point       = np.array([0, feed, 0])

            elif (abs((frequence_Coupure - frequence_resonance)/frequence_Coupure) < 0.07):
                print(f"\nWe are within 2% of fc!\n")
                feed = max(min(feed * ratio**2, b - 3 * largeur_piste / 2 - 2.5 / 1000), largeur_piste / 2)

                if feed >= b - 3 * largeur_piste / 2 - 2.5 / 1000 or feed <= largeur_piste / 2:
                    print("\nExtreme border reached\n")
                    distance_meandre, largeur_piste = adapt_with_ratio_square(distance_meandre, largeur_piste, ratio)
                
                feed_point       = np.array([0, feed, 0])
            else:
                print(f"\nWe are FAR from fc!\n")
                distance_meandre, largeur_piste = adapt_with_ratio_square(distance_meandre, largeur_piste, ratio)
            
            iteration += 1
        except ValueError as e:
            print(f"Error: {e}")

    end_time = time.time()
    simulation_time = end_time - start_time
    simulation_time_minutes = simulation_time / 60
    simulation_time_seconds = simulation_time % 60
    print(f"Simulation time: {simulation_time_minutes:.0f} minutes and {simulation_time_seconds:.2f} seconds")
    if converged:
        print(f"Convergence reached at iteration {iteration}")
    else:
        print(f"Convergence not reached after {max_iter} iterations")

    # Save only a, largeur_piste, distance_meandre, and feed in JSON format
    data = {
        "a": a,
        "largeur_piste": largeur_piste,
        "distance_meandre": distance_meandre,
        "feed": feed
    }

    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Optimization parameters saved to {filename}")

    return impedances"""