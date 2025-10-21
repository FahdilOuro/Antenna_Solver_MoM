import os

import numpy as np
from scipy.io import savemat, loadmat

from rwg.rwg2 import DataManager_rwg2
from rwg.rwg3 import DataManager_rwg3


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

def find_feed_edges(points, edges, feed_point, monopole=False):
    # --- Normalize feed_point / Feed ---
    Feed = np.asarray(feed_point.T)
    if Feed.ndim == 1:
        Feed = Feed.reshape(3, 1)
    elif Feed.ndim == 2 and Feed.shape[0] == 3:
        pass
    else:
        raise ValueError("feed_point must have shape (3,) or (3, N)")
    
    # --- Vectorized computation of edge geometric centers ---
    centers = 0.5 * (points.points[:, edges.first_points] + points.points[:, edges.second_points])  # (3, E)

    # --- Compute distances between each feed point and each edge ---
    diff = centers[:, :, np.newaxis] - Feed[:, np.newaxis, :]  # (3, E, N)
    dist_squared = np.sum(diff ** 2, axis=0)                   # (E, N)

    # --- Select feeding edges ---
    if monopole:
        index_feeding_edges = np.argsort(dist_squared, axis=0)[:2, :]  # (2, N)
        index_feeding_edges = index_feeding_edges.flatten(order='F')   # (2*N,)
    else:
        index_feeding_edges = np.argmin(dist_squared, axis=0)          # (N,)
    
    return index_feeding_edges

def calculate_current_radiation(filename_mesh_2, filename_impedance, feed_point, voltage_amplitude, monopole=False, simulate_array_antenna=False):
    """
        Calculates the currents, input impedance, and radiated power of an antenna.

        This function uses meshed data and impedance data to solve the Method of Moments (MoM) equations.
        It simulates the effect of a feed point on the antenna and deduces its operating parameters.

        Parameters:
            * filename_mesh_2 : str, path to the file containing meshed data (_mesh2).
            * filename_impedance : str, path to the file containing impedance data (_impedance).
            * feed_point : n-d-array (3,), coordinates of the feed point on the antenna.
            * voltage_amplitude : float, amplitude of the signal applied at the feed point.

        Returns:
            * frequency : float, operating frequency (Hz).
            * omega : float, associated angular frequency (rad/s).
            * mu : float, vacuum magnetic permeability (H/m).
            * epsilon : float, vacuum permittivity (F/m).
            * light_speed_c : float, speed of light in vacuum (m/s).
            * eta : float, characteristic impedance of free space (Ω).
            * voltage : n-d-array, voltage vector applied to edges.
            * current : n-d-array, current vector resulting from solving the MoM equations.
            * impedance : complex, input impedance at the feed point (Ω).
            * feed_power : float, active power delivered to the antenna (W).

        Behavior:
            1. Loads the necessary meshed and impedance data for the calculation.
            2. Identifies the edge closest to the feed point (feed_point).
            3. Sets the voltage vector with excitation applied to the fed edge.
            4. Solves the MoM equations to obtain currents flowing in the network.
            5. Calculates the electrical parameters of the antenna, including:
               * Input impedance at the feed point.
               * Active power delivered to the antenna.

        Notes:
            * The feed point (feed_point) should be located near one of the mesh edges.
            * Impedance and mesh data must correspond to ensure consistent calculations.
            * The linear system solution relies on a correctly formed impedance matrix.
    """
    # Load meshed and impedance data
    points, _, edges, *_ = DataManager_rwg2.load_data(filename_mesh_2)
    frequency, omega, mu, epsilon, light_speed_c, eta, matrice_z = DataManager_rwg3.load_data(filename_impedance)

    # Initialize the voltage vector
    voltage = np.zeros(edges.total_number_of_edges, dtype=complex)

    index_feeding_edges = find_feed_edges(points, edges, feed_point, monopole)

    # --- Apply voltage with progressive phase if simulating array ---
    if simulate_array_antenna:
        N = len(index_feeding_edges)
        phase = 0       # e.g., 0 (broadside), -2 * np.pi / 3 (end-fire), etc.
        phase_shift = np.exp(1j * phase * np.arange(N))
        voltage[index_feeding_edges] = voltage_amplitude * edges.edges_length[index_feeding_edges] * phase_shift
    else:
        voltage[index_feeding_edges] = voltage_amplitude * edges.edges_length[index_feeding_edges]

    # --- Solve the linear system (Z * I = V) ---
    current = np.linalg.solve(matrice_z, voltage)

    # --- Impedance / power ---
    if simulate_array_antenna:
        edge_lengths = edges.edges_length[index_feeding_edges]
        current_vals = current[index_feeding_edges]
        voltage_vals = voltage[index_feeding_edges]

        gap_current = current_vals * edge_lengths
        gap_voltage = voltage_vals / edge_lengths
        impedance = gap_voltage / gap_current
        feed_power = 0.5 * np.real(gap_current * np.conj(gap_voltage))
    else:
        gap_current = np.sum(current[index_feeding_edges] * edges.edges_length[index_feeding_edges])
        gap_voltage = np.mean(voltage[index_feeding_edges] / edges.edges_length[index_feeding_edges])
        impedance = gap_voltage / gap_current
        feed_power = 0.5 * np.real(gap_current * np.conj(gap_voltage))

    return frequency, omega, mu, epsilon, light_speed_c, eta, voltage, current, gap_current, gap_voltage, impedance, feed_power, index_feeding_edges


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
    def save_data_for_radiation(filename_mesh2, save_folder_name, frequency, omega,
                                mu, epsilon, light_speed_c, eta,
                                voltage, current, gap_current, gap_voltage, impedance, feed_power):
        """
            Saves data related to electromagnetic wave radiation into a MATLAB file.

            Parameters:
                (Same as 'save_data_for_scattering', with additionally:)
                * impedance (np.n-d-array) : Measured impedance.
                * feed_power (np.n-d-array) : Feed power.

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

        # Save data including currents, impedance, and feed power
        data = {
            'frequency': frequency,
            'omega': omega,
            'mu': mu,
            'epsilon': epsilon,
            'light_speed_c': light_speed_c,
            'eta': eta,
            'voltage': voltage,
            'current': current,
            'gap_current': gap_current,
            'gap_voltage': gap_voltage,
            'impedance': impedance,
            'feed_power': feed_power
        }

        # Save the data
        savemat(full_save_path, data)

        return save_file_name

    @staticmethod
    def load_data(filename, radiation=False, scattering=False):
        """
            Loads data from a MATLAB file.

            Parameters:
            filename (str) : Full path to the file to load.

            Returns:
            tuple : Contents of the loaded data, depending on the keys present in the file.

            Handled exceptions:
                * FileNotFoundError : If the specified file does not exist.
                * KeyError : If expected keys are missing from the file.
                * ValueError : If the data is malformed.
        """
        try:
            # Check if the file exists
            if not os.path.isfile(filename):
                raise FileNotFoundError(f"File '{filename}' does not exist.")

            # Extract main data
            data = loadmat(filename)
            frequency = data['frequency'].squeeze()
            omega = data['omega'].squeeze()
            mu = data['mu'].squeeze()
            epsilon = data['epsilon'].squeeze()
            light_speed_c = data['light_speed_c'].squeeze()
            eta = data['eta'].squeeze()
            voltage = data['voltage'].squeeze()
            current = data['current'].squeeze()

            # Extract specific fields
            if 'wave_incident_direction' in data and 'polarization' in data and scattering:
                wave_incident_direction = data['wave_incident_direction'].squeeze()
                polarization = data['polarization'].squeeze()
                return frequency, omega, mu, epsilon, light_speed_c, eta, wave_incident_direction, polarization, voltage, current

            if 'feed_power' in data and 'impedance' in data and 'gap_voltage' in data and 'gap_current' in data and radiation:
                impedance = data['voltage'].squeeze()
                feed_power = data['current'].squeeze()
                gap_voltage = data['gap_voltage'].squeeze()
                gap_current = data['gap_current'].squeeze()
                return frequency, omega, mu, epsilon, light_speed_c, eta, voltage, current, gap_voltage, gap_current, impedance, feed_power

            if not scattering and not radiation:
                raise ValueError("Error: 'scattering' and 'radiation' cannot both be False. Please specify one.")

        except FileNotFoundError as e:
            print(f"Error: {e}")
        except KeyError as e:
            print(f"Key Error: {e}")
        except ValueError as e:
            print(f"Value Error (likely malformed data): {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")