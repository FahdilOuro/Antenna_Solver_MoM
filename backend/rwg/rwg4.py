import os

import numpy as np
from scipy.io import savemat, loadmat

from backend.rwg.rwg2 import DataManager_rwg2
from backend.rwg.rwg3 import DataManager_rwg3
from backend.utils.gap_source import *


# Definition of the incident field
# Example: wave_incident_direction = [0, 0, -1] means the incident field comes from the "-z" direction
# Example: polarization = [1, 0, 0] means the incident field is polarized in the "x" direction

def calculate_current_scattering(filename_mesh_2, filename_impedance, wave_incident_direction, polarization):
    """
        Computes the current and voltage vector resulting from the scattering of an incident wave on a structure.

        This function uses meshed data and calculated impedance data to solve the method of moments (MoM) equations,
        modeling the electromagnetic response of a structure.

        Parameters:
            * filename_mesh_2 : str, path to the file containing the meshed data (_mesh2 file).
            * filename_impedance : str, path to the file containing the impedance data (_impedance file).
            * wave_incident_direction : n-d-array (3,), direction of propagation of the incident wave (unit vector).
            * polarization : n-d-array (3,), vector describing the polarization of the incident electric field
              (e.g., 'x' or 'y' direction).

        Returns:
            * frequency : float, frequency used in the electromagnetic calculation (Hz).
            * omega : float, associated angular frequency (rad/s).
            * mu : float, magnetic permeability of free space (H/m).
            * epsilon : float, permittivity of free space (F/m).
            * light_speed_c : float, speed of light in vacuum (m/s).
            * eta : float, characteristic impedance of free space (Ω).
            * voltage : n-d-array, voltage vector resulting from the MoM equations (Z * I = V).
            * current : n-d-array, current vector solving the MoM equations.

        Behavior:
            1. Loads the meshed data and impedance data from the specified files.
            2. Computes the wave vector `kv` from the incident wave direction and the wave number 'k'.
            3. Initializes a 'voltage' vector (MoM RHS) from edge contributions and scalar products related to associated triangles.
            4. Solves the linear system of MoM equations to obtain the 'current' vector.
            5. Displays computation time for solving the linear system.

        Notes:
            * The method relies on the accuracy of the provided meshed and impedance data.
            * The incident wave direction ('wave_incident_direction') and polarization must be properly
              normalized to ensure consistent results.
    """
    # Load meshed data (points, triangles, edges, barycenters, rho vectors)
    _, triangles, edges, _, vecteurs_rho = DataManager_rwg2.load_data(filename_mesh_2)
    # Load impedance data (frequency, EM parameters, Z matrix)
    frequency, omega, mu, epsilon, light_speed_c, eta, matrice_z = DataManager_rwg3.load_data(filename_impedance)

    # Compute physical constants
    k = omega / light_speed_c               # Wave number
    kv = k * wave_incident_direction        # Incident wave vector

    # Initialize voltage vector (MoM RHS)
    voltage = np.zeros(edges.total_number_of_edges, dtype=complex)
    
    # === Prepare triangle centers associated with edges ===
    centers_plus = triangles.triangles_center[:, triangles.triangles_plus]    # (3, N_edges)
    centers_minus = triangles.triangles_center[:, triangles.triangles_minus]  # (3, N_edges)

    # === Compute scalar products kv . r_plus and kv . r_minus ===
    scalar_product_plus = np.einsum('i,ij->j', kv, centers_plus)   # (N_edges,)
    scalar_product_minus = np.einsum('i,ij->j', kv, centers_minus) # (N_edges,)

    # === Compute complex wave factors (em_plus and em_minus) ===
    # Broadcasting : (3, 1) * (1, N_edges) -> (3, N_edges)
    em_plus = polarization[:, None] * np.exp(-1j * scalar_product_plus)[None, :]
    em_minus = polarization[:, None] * np.exp(-1j * scalar_product_minus)[None, :]

    # === Scalar products with rho vectors ===
    # em_plus and vecteur_rho_plus are both (3, N_edges)
    scalar_plus = np.einsum('ij,ij->j', em_plus, vecteurs_rho.vecteur_rho_plus)   # (N_edges,)
    scalar_minus = np.einsum('ij,ij->j', em_minus, vecteurs_rho.vecteur_rho_minus) # (N_edges,)

    # === Assemble the final "voltage" vector ===
    voltage = edges.edges_length * 0.5 * (scalar_plus + scalar_minus)  # (N_edges,)

    # Solve the linear system (Z * I = V) to obtain the current vector
    current = np.linalg.solve(matrice_z, voltage)

    return frequency, omega, mu, epsilon, light_speed_c, eta, voltage, current

def calculate_current_radiation(path, feed_point, voltage_amplitude, excitation_unit_vector=None, gap_width=0.05, voltage_phase=None):
    """
    Calculates the currents, input impedance, and radiated power for one or multiple antenna ports.

    This function solves the Method of Moments (MoM) system, rectifies the current flow 
    based on physical consistency (Re(Z) > 0), and returns detailed metrics for each feed.

    Parameters:
        path : object, contains attributes path.mat_mesh2 and path.mat_impedance.
        feed_point : n-d-array, coordinates of the feed point(s). Can be (3,) or (N, 3).
        voltage_amplitude : float or list, amplitude of the excitation signal.
        excitation_unit_vector : str or n-d-array, direction of the gap excitation.
        gap_width : float, physical width of the gap source.
        voltage_phase : float or list, phase of the excitation in radians.

    Returns:
        frequency : float, operating frequency (Hz).
        omega : float, angular frequency (rad/s).
        mu, epsilon : float, vacuum constants.
        light_speed_c : float, speed of light (m/s).
        eta : float, free space impedance (Ohm).
        voltage_vector : n-d-array, the global excitation vector.
        current_vector : n-d-array, solved currents for all edges.
        port_results : list, list of dicts containing 'current', 'impedance', 'power', 
                       and 'source_voltage' for each individual port.
    """
    # Load meshed and impedance data
    _, triangles, edges, _, vecteurs_rho = DataManager_rwg2.load_data(path.mat_mesh2)
    frequency, _, _, _, _, _, matrice_z = DataManager_rwg3.load_data(path.mat_impedance)

    # 1. Initialize the global voltage vector and retrieve feeding indices per port
    voltage, all_feeding_indices = multiple_gap_sources(
        triangles, edges, vecteurs_rho, voltage_amplitude, feed_point, 
        excitation_unit_vector, gap_width, voltage_phase
    )

    # 2. Solve the linear system (Z * I = V)
    current = np.linalg.solve(matrice_z, voltage)

    # 3. Process each port to calculate specific metrics
    feed_points_2d = np.atleast_2d(feed_point)
    num_ports = len(all_feeding_indices)
    port_results = []

    for i in range(num_ports):
        indices = all_feeding_indices[i]

        if indices.size == 0:
            print(f"Warning: Port {i} at {feed_points_2d[i]} has no feeding edges.")
            continue

        # Extract local current coefficients and edge lengths for the gap
        local_coeffs = current[indices]
        edge_lengths = edges.edges_length[indices]
        is_values = local_coeffs * edge_lengths

        # --- RECTIFICATION OF GAP CURRENT ---
        # Calculate mean magnitude to avoid cancellation from arbitrary RWG edge orientations
        gap_current_mag = np.mean(np.abs(is_values))
        
        # Determine the initial complex phase from the sum of currents
        is_sum = np.sum(is_values)
        dominant_phase = np.exp(1j * np.angle(is_sum))
        
        # Estimate the complex gap current
        temp_gap_current = gap_current_mag * dominant_phase
        
        # Determine specific source voltage for this port
        amp = voltage_amplitude[i] if not np.isscalar(voltage_amplitude) else voltage_amplitude
        phi = 0 if voltage_phase is None else (voltage_phase[i] if not np.isscalar(voltage_phase) else voltage_phase)
        source_voltage = amp * np.exp(1j * phi)

        # --- PHYSICAL CONSISTENCY CHECK ---
        # Ensure Re(Z) > 0 for passive antennas by flipping phase if necessary
        if temp_gap_current != 0:
            z_test = source_voltage / temp_gap_current
            final_gap_current = -temp_gap_current if z_test.real < 0 else temp_gap_current
        else:
            final_gap_current = 0j

        # 4. Calculate final port metrics
        impedance = source_voltage / final_gap_current if final_gap_current != 0 else np.inf
        active_power = 0.5 * np.real(source_voltage * np.conj(final_gap_current))

        # Store detailed results for this port
        port_results.append({
            'port_index': i,
            'location': feed_points_2d[i],
            'gap_current': final_gap_current,
            'impedance': impedance,
            'power': active_power,
            'source_voltage': source_voltage
        })

        # print(f"Port_result : {port_results}")

    # Returns the environmental constants, global vectors, and the list of port-specific data
    return frequency, voltage, current, port_results

class DataManager_rwg4:
    """
        A class to manage saving and loading data related to electromagnetic wave problems,
        such as scattering or radiation, using MATLAB files.

        Methods:
            * save_data_for_scattering : Save data related to wave scattering.
            * save_data_for_radiation : Save data related to radiation.
            * load_data : Load data from a MATLAB file.
    """
    @staticmethod
    def save_data_for_scattering(filename_mesh2, save_folder_name, frequency,
                                 omega, mu, epsilon, light_speed_c, eta,
                                 wave_incident_direction, polarization, voltage, current):
        """
            Saves data related to electromagnetic wave scattering into a MATLAB file.

            Parameters:
                * filename_mesh2 (str) : Name of the mesh file used for simulation.
                * save_folder_name (str) : Directory where data will be saved.
                * frequency (float) : Wave frequency.
                * omega (float) : Angular frequency.
                * mu (float) : Magnetic permeability of the medium.
                * epsilon (float) : Electric permittivity of the medium.
                * light_speed_c (float) : Speed of light in the medium.
                * eta (float) : Impedance of the medium.
                * wave_incident_direction (np.n-d-array) : Direction of the incident wave.
                * polarization (np.n-d-array) : Polarization of the incident wave.
                * voltage (np.n-d-array) : Simulated voltages.
                * current (np.n-d-array) : Simulated currents.

            Returns:
            save_file_name (str) : Name of the generated save file.
        """
        # Construct file name
        base_name = os.path.splitext(os.path.basename(filename_mesh2))[0]
        base_name = base_name.replace('_mesh2', '')  # Remove '_mesh2' part
        save_file_name = base_name + '_current.mat'  # Add '_current' suffix
        full_save_path = os.path.join(save_folder_name, save_file_name)

        # Check and create directory if needed
        if not os.path.exists(save_folder_name):
            os.makedirs(save_folder_name)

        # Save data including wave incident direction and polarization
        data = {
            'frequency': frequency,
            'omega': omega,
            'mu': mu,
            'epsilon': epsilon,
            'light_speed_c': light_speed_c,
            'eta': eta,
            'wave_incident_direction': wave_incident_direction,
            'polarization': polarization,
            'voltage': voltage,
            'current': current
        }

        # Save the data
        savemat(full_save_path, data)

        return save_file_name

    @staticmethod
    def save_data_for_radiation(path, frequency, omega, mu, epsilon, light_speed_c, eta,
                                voltage, current, gap_currents, gap_voltages, impedances, feed_powers):
        """
        Saves electromagnetic radiation data for one or multiple ports into a MATLAB file.

        Parameters:
            path : object, contains path.mat_current attribute.
            frequency, omega, mu, epsilon, light_speed_c, eta : float, physical constants and frequency.
            voltage : np.ndarray, global excitation vector (Z-matrix RHS).
            current : np.ndarray, global solved current vector.
            gap_currents : np.ndarray, array of complex currents for each port.
            gap_voltages : np.ndarray, array of complex voltages for each port.
            impedances : np.ndarray, array of complex input impedances for each port.
            feed_powers : np.ndarray, array of active power values for each port.

        Returns:
            None
        """
        data = {
            'frequency': frequency,
            'omega': omega,
            'mu': mu,
            'epsilon': epsilon,
            'light_speed_c': light_speed_c,
            'eta': eta,
            'voltage': voltage,
            'current': current,
            'gap_currents': gap_currents,
            'gap_voltages': gap_voltages,
            'impedances': impedances,
            'feed_powers': feed_powers
        }

        savemat(path.mat_current, data)

    @staticmethod
    def load_data(filename, radiation=False, scattering=False):
        """
        Loads data from a MATLAB file, handling either radiation or scattering results.

        Parameters:
            filename (str) : Full path to the file to load.
            radiation (bool) : If True, expects and returns antenna feed port data.
            scattering (bool) : If True, expects and returns incident wave data.

        Returns:
            tuple : Extracted data in the order defined by the saving process.
        """
        try:
            if not os.path.isfile(filename):
                raise FileNotFoundError(f"File '{filename}' does not exist.")

            # Load the raw .mat data
            data = loadmat(filename)
            
            # 1. Common global parameters extraction
            frequency = data['frequency'].squeeze()
            omega = data['omega'].squeeze()
            mu = data['mu'].squeeze()
            epsilon = data['epsilon'].squeeze()
            light_speed_c = data['light_speed_c'].squeeze()
            eta = data['eta'].squeeze()
            voltage = data['voltage'].squeeze()
            current = data['current'].squeeze()

            # 2. Scattering-specific logic
            if scattering:
                if 'wave_incident_direction' in data and 'polarization' in data:
                    wave_incident_direction = data['wave_incident_direction'].squeeze()
                    polarization = data['polarization'].squeeze()
                    return (frequency, omega, mu, epsilon, light_speed_c, eta, 
                            wave_incident_direction, polarization, voltage, current)
                else:
                    raise KeyError("Missing scattering keys in file.")

            # 3. Radiation-specific logic (Updated for multi-port support)
            if radiation:
                # Check for the updated plural keys
                required_keys = ['gap_currents', 'gap_voltages', 'impedances', 'feed_powers']
                if all(k in data for k in required_keys):
                    gap_currents = data['gap_currents'].squeeze()
                    gap_voltages = data['gap_voltages'].squeeze()
                    impedances = data['impedances'].squeeze()
                    feed_powers = data['feed_powers'].squeeze()
                    
                    # Return in the exact order as specified in the calculation return
                    return (frequency, omega, mu, epsilon, light_speed_c, eta, 
                            voltage, current, gap_currents, gap_voltages, impedances, feed_powers)
                else:
                    raise KeyError("Missing radiation port keys in file.")

            if not scattering and not radiation:
                raise ValueError("You must specify either 'radiation=True' or 'scattering=True'.")

        except FileNotFoundError as e:
            print(f"Error: {e}")
        except KeyError as e:
            print(f"Key Error: {e}")
        except ValueError as e:
            print(f"Value Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during loading: {e}")