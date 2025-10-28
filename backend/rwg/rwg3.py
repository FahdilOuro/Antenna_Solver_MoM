import os
from scipy.io import savemat, loadmat
from backend.utils.impmet import *

def calculate_z_matrice(triangles, edges, barycentric_triangles, vecteurs_rho, frequency):
    """
        Calculates the electromagnetic impedance matrix for a given system.

        Parameters:
            * triangles : Object representing the mesh triangles.
            * edges : Object representing the mesh edges.
            * barycentric_triangles : Object containing the barycentric data of triangles.
            * vecteurs_rho : Object containing the Rho vectors associated with the triangles.
            * frequency : Electromagnetic signal frequency (in Hz).

        Returns:
            * omega : Angular frequency (rad/s).
            * mu : Magnetic permeability of free space (H/m).
            * epsilon : Permittivity of free space (F/m).
            * light_speed_c : Speed of light in vacuum (m/s).
            * eta : Characteristic impedance of free space (Ω).
            * matrice_z : Calculated impedance matrix.
    """

    # Electromagnetic parameters
    epsilon = 8.854e-12  # Permittivity of free space (F/m)
    mu = 1.257e-6        # Magnetic permeability of free space (H/m)

    # Calculate electromagnetic constants
    light_speed_c = 1 / np.sqrt(epsilon * mu)  # Speed of light in vacuum (m/s)
    eta = np.sqrt(mu / epsilon)                # Characteristic impedance of free space (Ω)
    omega = 2 * np.pi * frequency              # Angular frequency (rad/s)
    k = omega / light_speed_c                  # Wave number (rad/m)
    complexe_k = 1j * k                        # Complex wave number for calculations

    # Constants to optimize computations
    constant_1 = mu / (4 * np.pi)                        # Constant for magnetic field calculation
    constant_2 = 1 / (1j * 4 * np.pi * omega * epsilon)  # Constant for electric field calculation
    factor = 1 / 9                                       # Weight factor for integrated calculations

    # Edge-specific factors
    factor_a = factor * (1j * omega * edges.edges_length / 4) * constant_1  # Factor for vector potential
    factor_fi = factor * edges.edges_length * constant_2                     # Factor for scalar potential

    # Impedance matrix calculation
    matrice_z = impedance_matrice_z(edges, triangles, barycentric_triangles, vecteurs_rho, complexe_k, factor_a, factor_fi)

    return omega, mu, epsilon, light_speed_c, eta, matrice_z

def compute_lumped_impedance(load_values, omega):
    """
        Computes Z = jωL + 1/(jωC) + R from load_values (3, N),
        with special handling:
        - C = None ⇒ replaced by a very large value (C = 1e64)
        - C = 0    ⇒ raises an explicit error
        - L, R : None or 0 ⇒ treated as 0 (no inductance or resistance)
    """
    N = load_values.shape[1]

    # Initialization
    L_vals = np.zeros(N, dtype=np.complex128)
    C_vals = np.empty(N, dtype=np.complex128)
    R_vals = np.zeros(N, dtype=np.complex128)

    for i in range(N):
        L = load_values[0, i]
        C = load_values[1, i]
        R = load_values[2, i]

        # Inductor L
        if L not in (None, 0):
            L_vals[i] = L

        # Capacitor C
        if C is None:
            C_vals[i] = 1e64  # Simulates a very large C ⇒ Z_C ≈ 0
        elif C == 0:
            raise ValueError(f"Invalid C value (0) at index {i}. Use `None` to ignore C.")
        else:
            C_vals[i] = C

        # Resistor R
        if R not in (None, 0):
            R_vals[i] = R

    Z_L = 1j * omega * L_vals
    with np.errstate(divide='ignore', invalid='ignore'):
        Z_C = 1 / (1j * omega * C_vals)
    Z_R = R_vals

    DeltaZ = Z_L + Z_C + Z_R

    return DeltaZ

def calculate_z_matrice_lumped_elements(points, triangles, edges, barycentric_triangles, vecteurs_rho, frequency, LoadPoint, LoadValue, LoadDir):
    """
        Computes the electromagnetic impedance matrix for a given system with lumped elements.

        Parameters:
            * triangles : Object representing the mesh triangles.
            * edges : Object representing the mesh edges.
            * barycentric_triangles : Object containing barycentric data of the triangles.
            * vecteurs_rho : Object containing Rho vectors associated with the triangles.
            * frequency : Electromagnetic signal frequency (Hz).
            * LoadPoint : Locations of lumped elements.
            * LoadValue : Values of L, C, R for lumped elements (3, LNumber).
            * LoadDir : Directions of the lumped elements (3, LNumber).

        Returns:
            * omega : Angular frequency (rad/s).
            * mu : Magnetic permeability of free space (H/m).
            * epsilon : Permittivity of free space (F/m).
            * light_speed_c : Speed of light in vacuum (m/s).
            * eta : Characteristic impedance of free space (Ω).
            * matrice_z : Calculated impedance matrix.
            * ImpArray : Indices of edges where lumped elements are applied.
    """

    # Electromagnetic parameters
    epsilon = 8.854e-12  # Permittivity of free space (F/m)
    mu = 1.257e-6        # Magnetic permeability of free space (H/m)

    # Compute electromagnetic constants
    light_speed_c = 1 / np.sqrt(epsilon * mu)  # Speed of light (m/s)
    eta = np.sqrt(mu / epsilon)                # Characteristic impedance of free space (Ω)
    omega = 2 * np.pi * frequency              # Angular frequency (rad/s)
    k = omega / light_speed_c                  # Wavenumber (rad/m)
    complexe_k = 1j * k                        # Complex wavenumber for calculations

    # Factors to optimize calculations
    constant_1 = mu / (4 * np.pi)                        # Factor for magnetic field calculation
    constant_2 = 1 / (1j * 4 * np.pi * omega * epsilon)  # Factor for electric field calculation
    factor = 1 / 9                                       # Weight factor for integrated calculations

    # Edge-specific factors
    factor_a = factor * (1j * omega * edges.edges_length / 4) * constant_1  # Vector potential factor
    factor_fi = factor * edges.edges_length * constant_2                     # Scalar potential factor

    # Compute impedance matrix
    matrice_z = impedance_matrice_z(edges, triangles, barycentric_triangles, vecteurs_rho, complexe_k, factor_a, factor_fi)

    # Lumped impedance implementation
    LoadPoint = LoadPoint.T  # (3, LNumber)
    LoadValue = LoadValue.T  # (3, LNumber)
    LoadDir = LoadDir.T      # (3, LNumber)

    LNumber = LoadPoint.shape[1]

    DeltaZ = compute_lumped_impedance(LoadValue, omega)

    ImpArray = []
    tol = 1e-3  # Tolerance for orientation

    for k in range(LNumber):
        EdgeCenters = 0.5 * (points.points[:, edges.first_points] + points.points[:, edges.second_points])  # (3, EdgesTotal)
        EdgeVectors = (points.points[:, edges.first_points] - points.points[:, edges.second_points]) / edges.edges_length[np.newaxis, :]

        diff = EdgeCenters - LoadPoint[:, k][:, np.newaxis]
        Dist = np.linalg.norm(diff, axis=0)
        Orien = np.abs(np.einsum('ij,i->j', EdgeVectors, LoadDir[:, k]))

        index = np.argsort(Dist)

        for idx in index:
            if Orien[idx] < tol:
                ImpArray.append(idx)
                matrice_z[idx, idx] += edges.edges_length[idx]**2 * DeltaZ[k]
                break

    ImpArray = np.array(ImpArray)  # Convert to numpy array

    return omega, mu, epsilon, light_speed_c, eta, matrice_z, ImpArray


class DataManager_rwg3:
    """
        A class to manage saving and loading electromagnetic data related to the impedance matrix.

        This class provides two main methods:
            * save_data : to save computed data into a .mat file.
            * load_data : to load saved data from a .mat file.
    """

    @staticmethod
    def save_data(filename_mesh2, save_folder_name, frequency, omega, mu, epsilon, light_speed_c, eta, matrice_z):
        """
            Saves the computed data into a .mat file.

            Parameters:
                * filename_mesh2 : str, name of the source file containing the base meshed data.
                * save_folder_name : str, folder where the data should be saved.
                * frequency : float, frequency used for electromagnetic computation (Hz).
                * omega : float, angular frequency (rad/s).
                * mu : float, magnetic permeability of free space (H/m).
                * epsilon : float, permittivity of free space (F/m).
                * light_speed_c : float, speed of light in vacuum (m/s).
                * eta : float, characteristic impedance of free space (Ω).
                * matrice_z : n-d-array, computed impedance matrix.

            Returns:
                save_file_name : str, name of the saved file.

            Behavior:
                * Adds a '_impedance' suffix to the base name of the source file.
                * Saves the data in a .mat file at the specified path.
                * Creates the specified folder if it does not exist yet.
        """
        # Prepare data for saving
        data = {
            'frequency': frequency,
            'omega': omega,
            'mu': mu,
            'epsilon': epsilon,
            'light_speed_c': light_speed_c,
            'eta': eta,
            'matrice_z': matrice_z,
        }

        # Build the save file name
        base_name = os.path.splitext(os.path.basename(filename_mesh2))[0]
        base_name = base_name.replace('_mesh2', '')         # Remove previous '_mesh2' suffix
        save_file_name = base_name + '_impedance.mat'       # Add new '_impedance' suffix
        full_save_path = os.path.join(save_folder_name, save_file_name)

        # Check and create folder if needed
        if not os.path.exists(save_folder_name):
            os.makedirs(save_folder_name)

        # Save data to .mat file
        savemat(full_save_path, data)

        return save_file_name

    @staticmethod
    def load_data(filename):
        """
            Loads data from a .mat file.

            Parameters:
                filename : str, path to the .mat file to load.

            Returns:
                tuple containing the following data:
                  frequency (float), omega (float), mu (float), epsilon (float),
                  light_speed_c (float), eta (float), matrice_z (n-d-array).

            Behavior:
                * Checks if the file exists before loading.
                * Loads the data while ensuring array dimensions are correct.
                * Handles common exceptions, such as file not found or malformed data.
        """
        try:
            # Check if the file exists
            if not os.path.isfile(filename):
                raise FileNotFoundError(f"File '{filename}' does not exist.")

            # Load the data
            data = loadmat(filename)
            frequency = data['frequency'].squeeze()
            omega = data['omega'].squeeze()
            mu = data['mu'].squeeze()
            epsilon = data['epsilon'].squeeze()
            light_speed_c = data['light_speed_c'].squeeze()
            eta = data['eta'].squeeze()
            matrice_z = data['matrice_z'].squeeze()
            return frequency, omega, mu, epsilon, light_speed_c, eta, matrice_z
        
        except FileNotFoundError as e:
            print(f"Error: {e}")
        except KeyError as e:
            print(f"Key Error: {e}")
        except ValueError as e:
            print(f"Value Error (likely malformed data): {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")