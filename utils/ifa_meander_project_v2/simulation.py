from src.radiation_algorithm.radiation_algorithm import radiation_algorithm
from efield.efield2 import radiation_intensity_distribution_over_sphere_surface, load_gain_power_data
from efield.efield3 import antenna_directivity_pattern
from efield.efield4 import *
import matplotlib.pyplot as plt
import numpy as np

from efield.efield4 import load_cst_data, plot_smith_chart_CST_MoM

def analysis(frequencies, ifa_meander_mat, feed_point):
    Z0 = 50
    s11_db = []
    impedances = []
    nPoints = len(frequencies)
    for idx, frequency in enumerate(frequencies):
        visualiser = False
        impedance, *_ = radiation_algorithm(ifa_meander_mat, frequency, feed_point, 0.5, show=visualiser)
        impedances.append(impedance)
        s11 = (impedance - Z0) / (impedance + Z0)
        s11_db.append(20 * np.log10(abs(s11)))
        print(f"Simulation {idx+1}/{nPoints} | f = {frequency/1e6:.2f} MHz | S11 = {s11_db[-1]:.2f} dB")
    
    # Results
    min_index = np.argmin(s11_db)
    f_resonance = frequencies[min_index]
    Z_at_res = impedances[min_index]
    R_res = Z_at_res.real
    X_res = Z_at_res.imag

    print(f"\n📡 Simulation results:")
    print(f"→ Resonance frequency = {f_resonance / 1e6:.2f} MHz")
    print(f"→ Impedance at f_res  = {Z_at_res:.2f} Ω")

    return f_resonance, s11_db, R_res

def simulate(frequencies, ifa_meander_mat, fC, feed_point, Z0=50):
    print(f"Z0 = {Z0} Ohms")
    s11_db = []
    impedances = []
    nPoints = len(frequencies)
    for idx, frequency in enumerate(frequencies):
        visualiser = (frequency == fC)
        impedance, *_ = radiation_algorithm(ifa_meander_mat, frequency, feed_point, voltage_amplitude=0.5, show=visualiser, save_image=False)
        impedances.append(impedance)
        s11 = (impedance - Z0) / (impedance + Z0)
        s11_db.append(20 * np.log10(abs(s11)))
        print(f"Simulation {idx+1}/{nPoints} | f = {frequency/1e6:.2f} MHz | S11 = {s11_db[-1]:.2f} dB")
    
    # Results
    min_index = np.argmin(s11_db)
    f_resonance = frequencies[min_index]
    Z_at_res = impedances[min_index]
    R_res = Z_at_res.real
    X_res = Z_at_res.imag
    plot_smith_chart(impedances, frequencies, fC, save_image=False, Z0=Z0)

    """ # filepath = 'data/plot_file/plot_smith_chart_cst_Antenne_Ta_92_Tb_55_a_27.txt'
    filepath = 'data/plot_file/plot_smith_chart_cst_Antenne_Ta_89_Tb_44_a_29.txt'
    frequencies_cst, mag_cst, phase_cst = load_cst_data(filepath)
    
    plot_smith_chart_CST_MoM(frequencies, impedances, frequencies_cst, mag_cst, phase_cst, fC) """
    
    print(f"\n📡 Simulation results:")
    print(f"→ Resonance frequency = {f_resonance / 1e6:.2f} MHz")
    print(f"→ Impedance at f_res  = {Z_at_res:.2f} Ω")

    return f_resonance, s11_db, R_res, X_res

def simulate_efficiency(frequencies, ifa_meander_mat, fC, feed_point, save_image=False):
    plt.style.use('seaborn-v0_8-talk')
    plt.rcParams['font.family'] = 'Lucida Console'
    plt.rcParams['font.size'] = 11
    Z0 = 50
    s11_db = []
    impedances = []
    nPoints = len(frequencies)
    efficiency_tot_log_table = []
    voltage = 0.5  # voltage amplitude in volts
    for idx, frequency in enumerate(frequencies):
        visualiser = (frequency == fC)
        impedance, *_ = radiation_algorithm(ifa_meander_mat, frequency, feed_point, voltage_amplitude=voltage, show=visualiser)
        impedances.append(impedance)
        s11 = (impedance - Z0) / (impedance + Z0)
        s11_db.append(20 * np.log10(abs(s11)))
        print(f"Simulation {idx+1}/{nPoints} | f = {frequency/1e6:.2f} MHz | S11 = {s11_db[-1]:.2f} dB")

        ifa_meander_mesh2 = 'data/antennas_mesh2/sim_optimize_ifa_mesh2.mat'
        ifa_meander_current = 'data/antennas_current/sim_optimize_ifa_current.mat'
        ifa_meander_gain_power = 'data/antennas_gain_power/sim_optimize_ifa_gain_power.mat'
        filename_sphere_dense = '../../data/sphere_mesh/sphere_dense.mat'

        # Step 2: Distribution of radiation intensity over a sphere
        print("Calculating radiation intensity distribution over sphere surface...")
        radiation_intensity_distribution_over_sphere_surface(ifa_meander_mesh2, ifa_meander_current, filename_sphere_dense, radiation=True, voltage_amplitude=voltage, show=False)

        # Step 3: Generating the directivity diagram
        print("Generating antenna directivity pattern...")
        antenna_directivity_pattern(ifa_meander_mesh2, ifa_meander_current, ifa_meander_gain_power, radiation=True, show=False)

        efficiency_total = load_gain_power_data(ifa_meander_gain_power)[-1]
        efficiency_tot_log_table.append(efficiency_total)

        print("")

    # Save efficiency_tot_table and frequencies in the same .npy file
    np.save('data/antennas_gain_power/efficiency_tot_and_freq_table.npy', 
            {'efficiency_tot_table': np.array(efficiency_tot_log_table), 'frequencies': np.array(frequencies)})

    # Plot efficiency_tot_log_table as a function of frequencies (in MHz) as a percentage
    plt.figure()
    efficiency_percent = np.array(efficiency_tot_log_table) * 100
    plt.plot(np.array(frequencies) / 1e6, efficiency_percent, linestyle='--', color='blue', linewidth=2.5)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Total Efficiency (%)')
    plt.title('Total Efficiency (%) vs Frequency')
    plt.grid(True)

    if save_image:
        # Create the directory if it does not exist
        output_dir_fig_image = "data/fig_image/"
        if not os.path.exists(output_dir_fig_image):
            os.makedirs(output_dir_fig_image)

        # saving the figure
        pdf_path = os.path.join(output_dir_fig_image, 'Efficiency_vs_Frequency' + ".pdf")
        plt.tight_layout()
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        plt.close()  # Close the figure after saving to avoid a blank image

def simulate_433_project(frequencies, ifa_meander_mat, fC, feed_point, Z0=50):
    s11_db = []
    impedances = []
    nPoints = len(frequencies)
    for idx, frequency in enumerate(frequencies):
        visualiser = (frequency == fC)
        impedance, *_ = radiation_algorithm(ifa_meander_mat, frequency, feed_point, voltage_amplitude=0.5, show=visualiser, save_image=False)
        impedances.append(impedance)
        s11 = (impedance - Z0) / (impedance + Z0)
        s11_db.append(20 * np.log10(abs(s11)))
        print(f"Simulation {idx+1}/{nPoints} | f = {frequency/1e6:.2f} MHz | S11 = {s11_db[-1]:.2f} dB")
    
    # Results
    min_index = np.argmin(s11_db)
    f_resonance = frequencies[min_index]
    Z_at_res = impedances[min_index]
    R_res = Z_at_res.real
    X_res = Z_at_res.imag
    plot_smith_chart(impedances, frequencies, fC, save_image=False, Z0=Z0)
    
    print(f"\n📡 Simulation results:")
    print(f"→ Resonance frequency = {f_resonance / 1e6:.2f} MHz")
    print(f"→ Impedance at f_res  = {Z_at_res:.2f} Ω")
    print(f"frequency resonance / S11 min = {f_resonance / 1e6:.2f} MHz / {s11_db[min_index]:.2f} dB")
    print(f"s")

    return f_resonance, s11_db, R_res, X_res, impedances