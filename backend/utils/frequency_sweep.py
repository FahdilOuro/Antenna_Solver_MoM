from matplotlib import pyplot as plt
import numpy as np
from scipy.io import savemat, loadmat
from scipy.interpolate import make_interp_spline

import os
import time

from backend.rwg.rwg1 import Points, Triangles, load_mesh_file, filter_complexes_jonctions, DataManager_rwg1
from backend.rwg.rwg2 import Vecteurs_Rho, DataManager_rwg2
from backend.rwg.rwg3 import calculate_z_matrice_sweep, DataManager_rwg3
from backend.rwg.rwg4 import DataManager_rwg4, compute_port_results
from backend.utils.gap_source import multiple_gap_sources
from backend.utils.gmsh_function import extract_msh_to_mat
from backend.utils.impmet import rwg_gram_matrix
from backend.utils.lossy_electric_conductor import calculate_normal_surface_impedance
from backend.src.radiation_algorithm.radiation_algorithm import radiation_algorithm
from backend.efield.efield2 import load_gain_power_data, radiation_intensity_distribution_over_sphere_surface


def generate_freq_points(freq_start, freq_stop, n_points):
    """
    Generate `n_points` frequencies evenly spaced between freq_start and
    freq_stop, both endpoints included (VNA-style sweep specified by point
    count, like "401 points from 45 MHz to 120 MHz" rather than by step size).
    The step size is derived from n_points, not the other way around, so the
    sweep always lands exactly on freq_stop.

    Parameters
    ----------
    freq_start, freq_stop : float  sweep bounds in Hz (freq_stop > freq_start > 0)
    n_points : int  number of frequency points, >= 2 (both endpoints are
               always included)

    Returns
    -------
    frequencies : ndarray (n_points,)

    Raises
    ------
    ValueError : if freq_start/freq_stop are not finite, strictly positive
                 numbers with freq_stop > freq_start, or if n_points is not
                 an integer >= 2.
    """
    if not (np.isfinite(freq_start) and np.isfinite(freq_stop)):
        raise ValueError(
            f"freq_start/freq_stop must be finite numbers, got "
            f"freq_start={freq_start}, freq_stop={freq_stop}."
        )
    if freq_start <= 0 or freq_stop <= 0:
        raise ValueError(
            f"freq_start/freq_stop must be strictly positive (Hz), got "
            f"freq_start={freq_start}, freq_stop={freq_stop}."
        )
    if freq_stop <= freq_start:
        raise ValueError(
            f"freq_stop ({freq_stop:.6g} Hz) must be strictly greater than "
            f"freq_start ({freq_start:.6g} Hz)."
        )
    if not isinstance(n_points, (int, np.integer)) or isinstance(n_points, bool) or n_points < 2:
        raise ValueError(
            f"n_points must be an integer >= 2 (at least the start and stop "
            f"frequencies), got {n_points!r}."
        )

    return np.linspace(freq_start, freq_stop, int(n_points))


def frequency_sweep(path, freq_start, freq_stop, feed_point, n_points=41, voltage_amplitude=1,
                    excitation_unit_vector=None, gap_width=0.05, voltage_phase=None,
                    compute_radiation_intensity=False,
                    load_lumped_elements=False, LoadPoint=None, LoadValue=None, LoadDir=None,
                    lossy_conductor=False, conductivity_conductor=None,
                    Z0=50, show=False):
    """
    Performs a frequency sweep and calculates multi-port parameters (Zin, Snn, Power).

    This function analyzes the feed points, iterates through `n_points`
    frequencies evenly spaced between freq_start and freq_stop (inclusive),
    solves the MoM system, and saves structured data for all ports to a
    MATLAB file.

    Unlike calling radiation_algorithm() once per frequency (which reloads and
    recomputes the mesh geometry, RWG rho vectors, and the whole Z(f) assembly
    from scratch at every point), this function computes every frequency-
    INDEPENDENT quantity exactly ONCE for the whole sweep:
      - mesh geometry (points/triangles/edges, areas, centers) and RWG rho
        vectors (rwg1/rwg2 preprocessing)
      - the gap-source excitation vector V (`multiple_gap_sources` has no
        frequency dependence: it depends only on mesh geometry and the fixed
        excitation amplitude/phase/gap_width)
      - the Gram matrix for the lossy-conductor SIBC term (pure geometry)
      - inside the Z(f) assembly itself, every triangle-pair distance and
        Duffy quadrature node (see impmet.impedance_matrix_z_sweep)

    Only what is genuinely frequency-dependent is repeated per point: the
    Green's-function kernel inside Z(f), the surface impedance Zs(f) for
    lossy conductors, and the E x E linear solve. See workflow.md,
    "frequency_sweep one-time precompute" for the profiled speedup.

    NOTE: lumped elements are not yet supported by this fast path (they use
    a separate, non-frequency-vectorised impedance-matrix build). If
    `load_lumped_elements=True`, this function falls back to the previous
    behaviour of calling radiation_algorithm() once per frequency.

    Parameters:
        n_points : int, number of frequency points between freq_start and
                   freq_stop, inclusive (default 41). The step size is
                   derived from this, not requested directly.
        Z0 : float, reference impedance for S-parameter calculation (default 50 Ohm).
        (Other parameters same as radiation_algorithm)

    Raises:
        ValueError: if freq_start/freq_stop/n_points are invalid (see
            generate_freq_points), or if lossy_conductor=True without
            conductivity_conductor.

    Returns:
        None (Saves data to path.mat_freq_sweep)
    """
    # Fail fast on bad inputs, before any mesh generation/assembly work.
    frequencies = generate_freq_points(freq_start, freq_stop, n_points)
    nPoints = len(frequencies)

    if lossy_conductor and conductivity_conductor is None:
        raise ValueError("conductivity_conductor must be provided when lossy_conductor=True.")

    if load_lumped_elements:
        _frequency_sweep_per_point_fallback(
            path, frequencies, feed_point, voltage_amplitude,
            excitation_unit_vector, gap_width, voltage_phase,
            compute_radiation_intensity, load_lumped_elements,
            LoadPoint, LoadValue, LoadDir, Z0, show,
        )
        return

    # 1. Analyze feed points to identify ports and locations
    feed_points_2d = np.atleast_2d(feed_point)
    num_ports = feed_points_2d.shape[0]

    # --- Phase distribution logic for display ---
    if voltage_phase is None:
        display_phases = np.zeros(num_ports)
    elif np.isscalar(voltage_phase):
        display_phases = np.full(num_ports, voltage_phase)
    else:
        display_phases = np.atleast_1d(voltage_phase)

    print("="*75)
    print(f"MULTIPLE PORT FREQUENCY SWEEP CONFIGURATION")
    print(f"Number of Ports detected: {num_ports}")
    for i in range(num_ports):
        loc = feed_points_2d[i]
        ph = display_phases[i] * 180 / np.pi
        print(f"  > Port {i} | Location: {loc} | Phase Offset: {ph:6.2f}°")
    print(f"Range: {freq_start/1e6:.2f} MHz to {freq_stop/1e6:.2f} MHz ({nPoints} points)")
    print("="*75 + "\n")

    # 2. Mesh geometry + RWG preprocessing -- ONCE for the whole sweep
    #    (previously redone at every frequency inside radiation_algorithm())
    extract_msh_to_mat(path.msh, path.mat)
    p, t = load_mesh_file(path.mat)
    points = Points(p)
    triangles = Triangles(t)
    triangles.calculate_triangles_area_and_center(points)
    edges = triangles.get_edges()
    filter_complexes_jonctions(points, triangles, edges)
    edges.compute_edges_length(points)
    DataManager_rwg1.save_data(path, points, triangles, edges)

    vecteurs_rho = Vecteurs_Rho()
    vecteurs_rho.calculate_vecteurs_rho(points, triangles, edges)
    DataManager_rwg2.save_data(path, vecteurs_rho)

    E = edges.total_number_of_edges
    print(f"Mesh ready: {points.total_of_points} points, {triangles.total_of_triangles} "
          f"triangles, {E} edges\n")

    # 3. Gap-source excitation vector -- frequency-independent, computed ONCE
    voltage, all_feeding_edges = multiple_gap_sources(
        triangles, edges, vecteurs_rho, voltage_amplitude, feed_point,
        excitation_unit_vector, gap_width, voltage_phase
    )

    # 4. Gram matrix for the lossy-conductor SIBC term -- pure geometry, ONCE
    #    (conductivity_conductor was already validated above, before mesh gen)
    Gram_rwg = rwg_gram_matrix(edges, triangles, points) if lossy_conductor else None

    # 5. Assemble Z(f) for every sweep frequency in a single batched pass,
    #    reusing all triangle-pair geometry across the whole sweep instead of
    #    recomputing it from scratch at each frequency.
    print(f"Assembling Z(f) for {nPoints} frequencies ({E}x{E} edges) -- "
          f"geometry reused across the whole sweep...")
    t0 = time.time()
    omega_arr, mu, epsilon, light_speed_c, eta, Z_all = calculate_z_matrice_sweep(
        points, triangles, edges, frequencies)
    dt = time.time() - t0
    print(f"Z(f) assembly done in {dt:.1f} s ({dt/nPoints:.3f} s/frequency amortised)\n")

    # Initialize storage lists
    all_s11_db           = []
    all_impedances       = []
    all_current_vectors  = []
    all_gap_currents     = []
    all_feed_powers      = []
    all_efficiency_total = []
    last_port_results     = None

    # 6. Cheap per-frequency loop: only Zs(f) (if lossy), the linear solve,
    #    and the port-metric formulas are repeated per point.
    for idx, freq in enumerate(frequencies):
        Z = Z_all[idx]

        if lossy_conductor:
            impedance_zs = calculate_normal_surface_impedance(freq, conductivity_conductor)
            Z_solve = Z + impedance_zs * Gram_rwg
        else:
            Z_solve = Z

        current_vec = np.linalg.solve(Z_solve, voltage)
        port_results = compute_port_results(voltage, current_vec, all_feeding_edges,
                                             voltage_amplitude, feed_point, verbose=False)
        last_port_results = port_results

        gap_Z_arr = np.array([res['impedance'] for res in port_results])
        gap_I_arr = np.array([res['gap_current'] for res in port_results])
        gap_P_arr = np.array([res['power'] for res in port_results])

        s11 = (gap_Z_arr - Z0) / (gap_Z_arr + Z0)
        s11_db = 20 * np.log10(np.abs(s11))

        efficiency_total = np.nan
        if compute_radiation_intensity:
            # This feature reads path.mat_impedance/path.mat_current from disk
            # (backend.efield), so those files must reflect THIS frequency --
            # pay the per-frequency disk write only when actually requested.
            gap_V_arr = np.array([res['source_voltage'] for res in port_results])
            DataManager_rwg3.save_data(path, freq, omega_arr[idx], mu, epsilon,
                                        light_speed_c, eta, Z)
            DataManager_rwg4.save_data_for_radiation(
                path, freq, omega_arr[idx], mu, epsilon, light_speed_c, eta,
                voltage, current_vec, gap_I_arr, gap_V_arr, gap_Z_arr, gap_P_arr)
            print("Calculating radiation intensity distribution over sphere surface...")
            radiation_intensity_distribution_over_sphere_surface(path, show=False)
            efficiency_total = load_gain_power_data(path.mat_gain_power)[-1]

        all_s11_db.append(s11_db)
        all_impedances.append(gap_Z_arr)
        all_gap_currents.append(gap_I_arr)
        all_feed_powers.append(gap_P_arr)
        all_current_vectors.append(current_vec)
        all_efficiency_total.append(efficiency_total)

        port_s11_str = " | ".join([f"P{i}: {s11_db[i]:.2f}dB" for i in range(len(s11_db))])
        print(f"Step {idx+1}/{nPoints} | {freq/1e6:7.2f} MHz | {port_s11_str}")

    # 7. Persist the last frequency's impedance/current to disk, if not
    #    already done above (some downstream tools expect path.mat_impedance
    #    / path.mat_current to reflect the most recently solved frequency).
    if not compute_radiation_intensity:
        gap_V_last = np.array([res['source_voltage'] for res in last_port_results])
        gap_Z_last = np.array([res['impedance'] for res in last_port_results])
        gap_I_last = np.array([res['gap_current'] for res in last_port_results])
        gap_P_last = np.array([res['power'] for res in last_port_results])
        DataManager_rwg3.save_data(path, frequencies[-1], omega_arr[-1], mu, epsilon,
                                    light_speed_c, eta, Z_all[-1])
        DataManager_rwg4.save_data_for_radiation(
            path, frequencies[-1], omega_arr[-1], mu, epsilon, light_speed_c, eta,
            voltage, all_current_vectors[-1], gap_I_last, gap_V_last, gap_Z_last, gap_P_last)

    # 8. Save structured data to MATLAB
    data_to_save = {
        'frequencies': np.array(frequencies),
        'port_locations': feed_points_2d,
        'num_ports': num_ports,
        'impedances': np.array(all_impedances),
        's11_db': np.array(all_s11_db),
        'gap_currents': np.array(all_gap_currents),
        'feed_powers': np.array(all_feed_powers),
        'current_vectors': np.array(all_current_vectors, dtype=object),
        'efficiencies_total': np.array(all_efficiency_total)
    }

    savemat(path.mat_freq_sweep, data_to_save)
    print(f"\nSweep complete. Results for {num_ports} ports saved to: {path.mat_freq_sweep}")


def _frequency_sweep_per_point_fallback(path, frequencies, feed_point, voltage_amplitude,
                                        excitation_unit_vector, gap_width, voltage_phase,
                                        compute_radiation_intensity, load_lumped_elements,
                                        LoadPoint, LoadValue, LoadDir, Z0, show):
    """
    Original per-frequency sweep: calls radiation_algorithm() once per point,
    which redoes mesh/RWG preprocessing and the full Z(f) assembly from
    scratch every time. Kept only as the fallback for load_lumped_elements=True
    (calculate_z_matrice_lumped_elements has not been vectorised over
    frequency -- see impmet.impedance_matrix_z_sweep for the fast path used
    otherwise).

    `frequencies` is the array already generated (and validated) by
    generate_freq_points() in the caller.
    """
    feed_points_2d = np.atleast_2d(feed_point)
    num_ports = feed_points_2d.shape[0]

    all_s11_db, all_impedances, all_current_vectors = [], [], []
    all_gap_currents, all_feed_powers, all_efficiency_total = [], [], []

    nPoints = len(frequencies)

    print("="*75)
    print(f"MULTIPLE PORT FREQUENCY SWEEP CONFIGURATION (per-frequency fallback: lumped elements)")
    print(f"Number of Ports detected: {num_ports}")
    print(f"Range: {frequencies[0]/1e6:.2f} MHz to {frequencies[-1]/1e6:.2f} MHz ({nPoints} points)")
    print("="*75 + "\n")

    for idx, freq in enumerate(frequencies):
        results = radiation_algorithm(
            path, freq, feed_point,
            voltage_amplitude=voltage_amplitude,
            excitation_unit_vector=excitation_unit_vector,
            gap_width=gap_width,
            voltage_phase=voltage_phase,
            show=show,
            load_lumped_elements=load_lumped_elements,
            LoadPoint=LoadPoint, LoadValue=LoadValue,
            LoadDir=LoadDir,
        )

        current_vec = results[2]
        gap_I = results[4]
        gap_Z = results[6]
        gap_P = results[7]

        gap_Z_arr = np.atleast_1d(gap_Z)
        s11 = (gap_Z_arr - Z0) / (gap_Z_arr + Z0)
        s11_db = 20 * np.log10(np.abs(s11))

        efficiency_total = np.nan
        if compute_radiation_intensity:
            print("Calculating radiation intensity distribution over sphere surface...")
            radiation_intensity_distribution_over_sphere_surface(path, show=False)
            efficiency_total = load_gain_power_data(path.mat_gain_power)[-1]

        all_s11_db.append(s11_db)
        all_impedances.append(gap_Z_arr)
        all_gap_currents.append(np.atleast_1d(gap_I))
        all_feed_powers.append(np.atleast_1d(gap_P))
        all_current_vectors.append(current_vec)
        all_efficiency_total.append(efficiency_total)

        port_s11_str = " | ".join([f"P{i}: {s11_db[i]:.2f}dB" for i in range(len(s11_db))])
        print(f"Step {idx+1}/{nPoints} | {freq/1e6:7.2f} MHz | {port_s11_str}")

    data_to_save = {
        'frequencies': np.array(frequencies),
        'port_locations': feed_points_2d,
        'num_ports': num_ports,
        'impedances': np.array(all_impedances),
        's11_db': np.array(all_s11_db),
        'gap_currents': np.array(all_gap_currents),
        'feed_powers': np.array(all_feed_powers),
        'current_vectors': np.array(all_current_vectors, dtype=object),
        'efficiencies_total': np.array(all_efficiency_total)
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

def plot_s_parameters(path, port_idx=None, freq_unit='MHz', interpolation_threshold=101,
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

def plot_radiation_efficiency(path, freq_unit='MHz', interpolation_threshold=50, 
                    save_pdf=False, save_folder=None):
    """
    Plots the total efficiency over frequency with a professional aesthetic style.

    Args:
        path: Object containing path.mat_freq_sweep.
        freq_unit: Frequency unit ('Hz', 'kHz', 'MHz', 'GHz').
        interpolation_threshold: Min points to trigger cubic spline smoothing.
        save_pdf: Boolean, if True saves the figure as a PDF file.
        save_folder: Directory path where the PDF will be stored.
    """
    # 1. Load simulation data from MATLAB file
    try:
        data = loadmat(path.mat_freq_sweep)
        frequencies = data['frequencies'].flatten()
        # Ensure the key matches your data_to_save dictionary
        efficiency_total = data['efficiencies_total'].flatten()
    except KeyError as e:
        print(f"Error: Key {e} not found in {path.mat_freq_sweep}. Check your save dictionary.")
        return

    # 2. Frequency unit conversion
    unit_factors = {'Hz': 1, 'kHz': 1e3, 'MHz': 1e6, 'GHz': 1e9}
    freq_divisor = unit_factors.get(freq_unit, 1e6)
    freq_converted = frequencies / freq_divisor

    # Convert efficiency to percentage if it's in decimal format (0-1)
    if np.max(efficiency_total) <= 1.01:
        efficiency_total = efficiency_total * 100

    # --- Font & Style Configuration ---
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Adelle', 'DejaVu Serif', 'Times New Roman']
    
    fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
    
    # Using a professional deep blue for efficiency plots
    main_color = "#2980b9" 

    # 3. Plotting Logic (Spline vs Linear)
    if len(freq_converted) > interpolation_threshold:
        # High-resolution cubic spline for smooth visualization
        freq_smooth = np.linspace(freq_converted.min(), freq_converted.max(), 500)
        spline = make_interp_spline(freq_converted, efficiency_total, k=3)
        eff_smooth = spline(freq_smooth)
        
        # Clip values to ensure they don't exceed 100% due to spline overshoot
        eff_smooth = np.clip(eff_smooth, 0, 100)
        
        ax.plot(freq_smooth, eff_smooth, color=main_color, linewidth=2.5, 
                label='Total Efficiency (Interp.)', zorder=2)
    else:
        ax.plot(freq_converted, efficiency_total, marker='s', markersize=5, 
                color=main_color, linewidth=1.5, markerfacecolor='white', 
                label='Total Efficiency Data')

    # --- Aesthetic Styling ---
    ax.set_xlabel(f'Frequency ({freq_unit})', fontsize=13, labelpad=10)
    ax.set_ylabel('Total Efficiency (%)', fontsize=13, labelpad=10)
    
    # Set Y-axis limits for percentage
    ax.set_ylim(0, 105) 
    
    # Reference lines (e.g., 50% and 90% benchmarks)
    ax.axhline(90, color="#27ae60", linestyle='--', linewidth=1, alpha=0.4, label='High Efficiency (90%)')
    ax.axhline(50, color="#e67e22", linestyle='--', linewidth=1, alpha=0.4)
    
    # Grid and Spines styling
    ax.grid(True, which='both', linestyle=':', alpha=0.5, color='#bdc3c7')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)

    ax.legend(frameon=False, fontsize=10, loc='lower right')
    
    plt.tight_layout()

    # --- PDF Export ---
    if save_pdf:
        if save_folder is None:
            save_folder = os.path.dirname(path.mat_freq_sweep)
        os.makedirs(save_folder, exist_ok=True)
        
        pdf_filename = f"Efficiency_Total_{path.name}.pdf"
        full_save_path = os.path.join(save_folder, pdf_filename)
        
        plt.savefig(full_save_path, format='pdf', bbox_inches='tight')
        print(f"Efficiency plot exported to: {full_save_path}")

    plt.show()