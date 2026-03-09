from matplotlib import pyplot as plt
import numpy as np
from scipy.io import savemat, loadmat
from scipy.interpolate import make_interp_spline

import os

from backend.src.radiation_algorithm.radiation_algorithm import radiation_algorithm


def generate_freq_step(fLow, fHigh, step=2e6):
    nPoints = int((fHigh - fLow) // step) + 1
    frequencies = [fLow + i * step for i in range(nPoints)]
    return frequencies

'''def frequency_sweep(path, freq_start, freq_stop,feed_point, step=2e6, voltage_amplitude=1,
                    excitation_unit_vector=None, gap_width=0.05, voltage_phase=None, load_lumped_elements=False,
                    LoadPoint=None, LoadValue=None, LoadDir=None, Z0=50, show=False):
    s11_db = []
    impedances = []
    currents = []
    gap_currents = []
    feed_powers = []

    frequencies = generate_freq_step(freq_start, freq_stop, step)
    nPoints = len(frequencies)
    print(f"Starting frequency sweep from {freq_start/1e6:.2f} MHz to {freq_stop/1e6:.2f} MHz with {nPoints} points.\n")


    for idx, frequency in enumerate(frequencies):
        _, _, current, _, gap_current, _, impedance, feed_power = radiation_algorithm(path, frequency,
                                                                                    feed_point,
                                                                                    voltage_amplitude=voltage_amplitude,
                                                                                    excitation_unit_vector=excitation_unit_vector,
                                                                                    gap_width=gap_width,
                                                                                    voltage_phase=voltage_phase,
                                                                                    show=show,
                                                                                    load_lumped_elements=load_lumped_elements,
                                                                                    LoadPoint=LoadPoint, LoadValue=LoadValue,
                                                                                    LoadDir=LoadDir)
        
        impedances.append(impedance)
        currents.append(current)
        gap_currents.append(gap_current)
        feed_powers.append(feed_power)
        s11 = (impedance - Z0) / (impedance + Z0)
        s11_db.append(20 * np.log10(abs(s11)))
        # print(f"Simulation {idx+1}/{nPoints} | f = {frequency/1e6:.2f} MHz | S11 = {s11_db[-1]:.2f} dB")

    # Convert lists to arrays where appropriate
    data_to_save = {
        'frequencies': np.array(frequencies),
        'impedances': np.array(impedances),
        's11_db': np.array(s11_db),
        'currents': np.array(currents),
        'gap_currents': np.array(gap_currents),
        'feed_powers': np.array(feed_powers)
    }
    savemat(path.mat_freq_sweep, data_to_save)'''

def frequency_sweep(path, freq_start, freq_stop, feed_point, step=2e6, voltage_amplitude=1,
                    excitation_unit_vector=None, gap_width=0.05, voltage_phase=None, load_lumped_elements=False,
                    LoadPoint=None, LoadValue=None, LoadDir=None, Z0=50, show=False):
    """
    Performs a frequency sweep and calculates multi-port parameters (Zin, Snn, Power).
    
    This function analyzes the feed points, iterates through frequencies, solves 
    the MoM system, and saves structured data for all ports to a MATLAB file.

    Parameters:
        Z0 : float, reference impedance for S-parameter calculation (default 50 Ohm).
        (Other parameters same as radiation_algorithm)

    Returns:
        None (Saves data to path.mat_freq_sweep)
    """
    # 1. Analyze feed points to identify ports and locations
    feed_points_2d = np.atleast_2d(feed_point)
    num_ports = feed_points_2d.shape[0]
    
    # Initialize storage lists
    all_s11_db = []
    all_impedances = []
    all_current_vectors = []
    all_gap_currents = []
    all_feed_powers = []

    # Frequency setup
    frequencies = generate_freq_step(freq_start, freq_stop, step)
    nPoints = len(frequencies)

    # --- Phase distribution logic for display ---
    # Convert voltage_phase into a consistent array of size num_ports
    if voltage_phase is None:
        display_phases = np.zeros(num_ports)
    elif np.isscalar(voltage_phase):
        display_phases = np.full(num_ports, voltage_phase)
    else:
        display_phases = np.atleast_1d(voltage_phase)

    # Header display
    print("="*75)
    print(f"MULTIPLE PORT FREQUENCY SWEEP CONFIGURATION")
    print(f"Number of Ports detected: {num_ports}")
    
    for i in range(num_ports):
        # Extract location and phase for each specific port
        loc = feed_points_2d[i]
        ph = display_phases[i] * 180 / np.pi
        print(f"  > Port {i} | Location: {loc} | Phase Offset: {ph:6.2f}°")
        
    print(f"Range: {freq_start/1e6:.2f} MHz to {freq_stop/1e6:.2f} MHz ({nPoints} points)")
    print("="*75 + "\n")

    # 2. Main frequency loop
    for idx, freq in enumerate(frequencies):
        # Solve for the current frequency
        # Expected return order: Z_mat, V_vec, I_vec, J_surf, I_gaps, V_gaps, Z_gaps, P_gaps
        results = radiation_algorithm(
            path, freq, feed_point,
            voltage_amplitude=voltage_amplitude,
            excitation_unit_vector=excitation_unit_vector,
            gap_width=gap_width,
            voltage_phase=voltage_phase,
            show=show,
            load_lumped_elements=load_lumped_elements,
            LoadPoint=LoadPoint, LoadValue=LoadValue,
            LoadDir=LoadDir
        )
        
        # Unpacking based on your radiation_algorithm output
        current_vec = results[2]
        gap_I = results[4]
        gap_Z = results[6]
        gap_P = results[7]

        # 3. Calculate S-parameters (Reflection Coefficient) for all ports
        # Formula: S11 = (Zin - Z0) / (Zin + Z0)
        # We ensure gap_Z is a numpy array for vectorized calculation
        gap_Z_arr = np.atleast_1d(gap_Z)
        s11 = (gap_Z_arr - Z0) / (gap_Z_arr + Z0)
        s11_db = 20 * np.log10(np.abs(s11))

        # 4. Store step results
        all_s11_db.append(s11_db)
        all_impedances.append(gap_Z_arr)
        all_gap_currents.append(np.atleast_1d(gap_I))
        all_feed_powers.append(np.atleast_1d(gap_P))
        all_current_vectors.append(current_vec)

        # Dynamic progress print for each frequency
        # Handles both scalar and array results for display
        port_s11_str = " | ".join([f"P{i}: {s11_db[i]:.2f}dB" for i in range(len(s11_db))])
        print(f"Step {idx+1}/{nPoints} | {freq/1e6:7.2f} MHz | {port_s11_str}")

    # 5. Save structured data to MATLAB
    data_to_save = {
        'frequencies': np.array(frequencies),
        'port_locations': feed_points_2d,
        'num_ports': num_ports,
        'impedances': np.array(all_impedances),
        's11_db': np.array(all_s11_db),
        'gap_currents': np.array(all_gap_currents),
        'feed_powers': np.array(all_feed_powers),
        'current_vectors': np.array(all_current_vectors, dtype=object)
    }

    savemat(path.mat_freq_sweep, data_to_save)
    print(f"\nSweep complete. Results for {num_ports} ports saved to: {path.mat_freq_sweep}")

def load_freq_sweep_data(filepath):
    data = loadmat(filepath)
    frequencies = data.get('frequencies', np.array([])).squeeze()
    impedances = data.get('impedances', np.array([])).squeeze()
    s11_db = data.get('s11_db', np.array([])).squeeze()
    currents = data.get('currents', np.array([])).squeeze()
    gap_currents = data.get('gap_currents', np.array([])).squeeze()
    feed_powers = data.get('feed_powers', np.array([])).squeeze()

    return frequencies, impedances, s11_db, currents, gap_currents, feed_powers

def plot_s11(path, freq_unit='MHz', interpolation_threshold=50,
                     save_pdf=False, save_folder=None):
    """
    Plots the S11 parameter with Adelle font, spline interpolation, 
    and optional PDF export capability.

    Args:
        path: Object containing path.mat_freq_sweep.
        freq_unit: Frequency unit ('Hz', 'kHz', 'MHz', 'GHz').
        interpolation_threshold: Min points to trigger cubic spline smoothing.
        save_pdf: Boolean, if True saves the figure as a PDF file.
        save_folder: Directory path where the PDF will be stored.
    """
    # Load simulation data
    frequencies, _, s11_db, _, _, _ = load_freq_sweep_data(path.mat_freq_sweep)
    
    # Unit conversion
    unit_factors = {'Hz': 1, 'kHz': 1e3, 'MHz': 1e6, 'GHz': 1e9}
    freq_divisor = unit_factors.get(freq_unit, 1e6)
    freq_converted = frequencies / freq_divisor

    # --- Font Configuration (Adelle) ---
    # Attempt to use Adelle if installed, fallback to serif
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Adelle', 'DejaVu Serif', 'Times New Roman']
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    
    # --- Interpolation Logic ---
    if len(freq_converted) > interpolation_threshold:
        print("Interpolation Applied")
        # Create a smooth curve using 500 points for high-resolution rendering
        freq_smooth = np.linspace(freq_converted.min(), freq_converted.max(), 500)
        spline = make_interp_spline(freq_converted, s11_db, k=3)
        s11_smooth = spline(freq_smooth)
        
        ax.plot(freq_smooth, s11_smooth, color="#c90f0f", linewidth=2.5, 
                label=r'$S_{11}$ Interpolated', zorder=2)
        # Add discrete points with low opacity to show raw data
        # ax.scatter(freq_converted, s11_db, color='#2c3e50', s=15, alpha=0.3, zorder=3)
    else:
        ax.plot(freq_converted, s11_db, marker='o', markersize=5, color='#2c3e50', 
                linewidth=1.5, markerfacecolor='white', label=r'$S_{11}$ Data')

    # --- Aesthetic Styling ---
    # ax.set_title(r'$S_{11}$ Parameter vs Frequency', fontsize=16, pad=20, fontweight='bold')
    ax.set_xlabel(f'Frequency ({freq_unit})', fontsize=13, labelpad=10)
    ax.set_ylabel(r'$S_{11}$ (dB)', fontsize=13, labelpad=10)
    
    # Threshold line at -10 dB
    ax.axhline(-10, color="#2825e7", linestyle='--', linewidth=1.2, 
               label='Return Loss Limit (-10 dB)', alpha=0.9)
    
    # Subtle grid and spine management
    ax.grid(True, which='both', linestyle=':', alpha=0.5, color='#bdc3c7')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)

    ax.legend(frameon=False, fontsize=10)
    
    plt.tight_layout()

    # --- PDF Export Logic ---
    if save_pdf:
        # Determine the destination folder: default to the .mat file directory if not provided
        if save_folder is None:
            save_folder = os.path.dirname(path.mat_freq_sweep)
        
        # Ensure the directory exists
        os.makedirs(save_folder, exist_ok=True)
        
        # Construct the filename using the path name attribute
        # We use os.path.join to safely merge the folder path and the filename
        pdf_filename = f"S11_Sweep_Result_{path.name}.pdf"
        full_save_path = os.path.join(save_folder, pdf_filename)
        
        # Save with high quality and vector format
        plt.savefig(full_save_path, format='pdf', bbox_inches='tight')
        print(f"Graph successfully saved to: {full_save_path}")

    plt.show()

def plot_s_parameters(path, port_idx=None, freq_unit='MHz', interpolation_threshold=50,
                      save_pdf=False, save_folder=None):
    """
    Plots S_ii parameters for one or all ports with safety checks and spline interpolation.

    Args:
        path: Object containing path.mat_freq_sweep.
        port_idx: Integer for a specific port (0 to num_ports-1) or None to plot all.
        freq_unit: Frequency unit ('Hz', 'kHz', 'MHz', 'GHz').
        interpolation_threshold: Min points to trigger cubic spline smoothing.
        save_pdf: Boolean, if True saves the figure as a PDF file.
        save_folder: Directory path where the PDF will be stored.
    """
    # 1. Load simulation data
    # Expected: frequencies (N,), s11_matrix (N, num_ports)
    frequencies, _, s11_matrix, _, _, _ = load_freq_sweep_data(path.mat_freq_sweep)
    
    # Ensure s11_matrix is 2D for consistent indexing
    if s11_matrix.ndim == 1:
        s11_matrix = s11_matrix[:, np.newaxis]
    
    num_ports = s11_matrix.shape[1]

    # 2. Port index validation and user feedback
    if port_idx is not None:
        if port_idx < 0 or port_idx >= num_ports:
            print(f"Error: Port index {port_idx} is out of range. "
                  f"Available ports: 0 to {num_ports - 1}.")
            return
    elif num_ports > 1:
        print(f"Plotting all {num_ports} ports. "
              f"Note: To view a specific port, use 'port_idx=n'.")

    # Unit conversion setup
    unit_factors = {'Hz': 1, 'kHz': 1e3, 'MHz': 1e6, 'GHz': 1e9}
    freq_divisor = unit_factors.get(freq_unit, 1e6)
    freq_converted = frequencies / freq_divisor

    # --- Font Configuration ---
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Adelle', 'DejaVu Serif', 'Times New Roman']
    
    fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
    
    # Color palette for multiple ports
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Define which ports to iterate over
    ports_to_plot = range(num_ports) if port_idx is None else [port_idx]

    # 3. Plotting Loop
    for i in ports_to_plot:
        data_y = s11_matrix[:, i]
        port_label = rf'$S_{{{i+1}{i+1}}}$'
        # Distinct color for multi-port, standard red for individual focus
        color = colors[i % 10] if port_idx is None else "#c90f0f"

        if len(freq_converted) > interpolation_threshold:
            # High-resolution cubic spline for smooth visualization
            freq_smooth = np.linspace(freq_converted.min(), freq_converted.max(), 500)
            spline = make_interp_spline(freq_converted, data_y, k=3)
            s_smooth = spline(freq_smooth)
            
            ax.plot(freq_smooth, s_smooth, color=color, linewidth=2, 
                    label=f'{port_label} (Interp.)', zorder=2)
        else:
            ax.plot(freq_converted, data_y, marker='o', markersize=4, color=color, 
                    linewidth=1.5, markerfacecolor='white', label=f'{port_label} Data')

    # --- Aesthetic Styling ---
    ax.set_xlabel(f'Frequency ({freq_unit})', fontsize=13, labelpad=10)
    ax.set_ylabel('Magnitude (dB)', fontsize=13, labelpad=10)
    
    # Reference line for antenna bandwidth (-10 dB rule of thumb)
    ax.axhline(-10, color="#2c3e50", linestyle='--', linewidth=1.2, 
               label='Standard Limit (-10 dB)', alpha=0.6)
    
    ax.grid(True, which='both', linestyle=':', alpha=0.5, color='#bdc3c7')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)

    # Place legend based on number of traces to avoid clutter
    if len(ports_to_plot) > 3:
        ax.legend(frameon=False, fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
    else:
        ax.legend(frameon=False, fontsize=10)
    
    plt.tight_layout()

    # --- PDF Export ---
    if save_pdf:
        if save_folder is None:
            save_folder = os.path.dirname(path.mat_freq_sweep)
        os.makedirs(save_folder, exist_ok=True)
        
        suffix = "AllPorts" if port_idx is None else f"Port{port_idx+1}"
        pdf_filename = f"S_Parameters_{suffix}_{path.name}.pdf"
        full_save_path = os.path.join(save_folder, pdf_filename)
        
        plt.savefig(full_save_path, format='pdf', bbox_inches='tight')
        print(f"Figure exported to: {full_save_path}")

    plt.show()