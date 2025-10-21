import os

import numpy as np
from matplotlib import pyplot as plt

from efield.efield2 import load_gain_power_data
from rwg.rwg2 import DataManager_rwg2
from rwg.rwg4 import DataManager_rwg4
from utils.dipole_parameters import compute_dipole_center_moment, compute_e_h_field

def compute_circle_points(radius, num_points, plane="yz"):
    """
    Computes observation points arranged in a circle on a specified plane.

    Parameters:
        radius : Radius of the circle
        num_points : Number of points
        plane : Plane of the circle ("yz", "xy", "xz")

    Returns:
        np.ndarray of shape (num_points+1, 3)
    """
    angles = np.linspace(0, 2 * np.pi, num_points)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)

    if plane == "yz":
        return np.column_stack((np.zeros_like(x), x, y))  # (0, y, z)
    elif plane == "xy":
        return np.column_stack((x, y, np.zeros_like(x)))  # (x, y, 0)
    elif plane == "xz":
        return np.column_stack((x, np.zeros_like(x), y))  # (x, 0, z)
    else:
        raise ValueError("Plane must be 'yz', 'xy', or 'xz'")


def compute_polar(observation_point_list_phi, numbers_of_points, eta, complex_k, dipole_moment, dipole_center, total_power):
    """
    Computes the distribution of the field intensity (in dB) on a given polar plane.

    Parameters:
        * observation_point_list_phi : List of observation points (Nx3 n-d-array).
        * numbers_of_points : Total number of observation points (int).
        * eta : Medium impedance (float).
        * complex_k : Complex wave number (1j * k) (complex).
        * dipole_moment : Dipole moments (complex n-d-array).
        * dipole_center : Dipole centers (n-d-array).
        * total_power : Total power radiated by the antenna (float).

    Returns:
        np.n-d-array : Polar plot of the normalized intensity in dB (1D array).
    """
    e_field_total = np.zeros((3, numbers_of_points), dtype=complex)  # Electric field
    h_field_total = np.zeros((3, numbers_of_points), dtype=complex)  # Magnetic field
    poynting_vector = np.zeros((3, numbers_of_points))  # Poynting vector
    w = np.zeros(numbers_of_points)  # Energy density
    u = np.zeros(numbers_of_points)  # Power density

    index_point = 0
    for angular_phi in observation_point_list_phi:
        observation_point = angular_phi
        (e_field_total[:, index_point],
         h_field_total[:, index_point],
         poynting_vector[:, index_point],
         w[index_point], u[index_point],
         norm_observation_point) = compute_e_h_field(observation_point,
                                                     eta,
                                                     complex_k,
                                                     dipole_moment,
                                                     dipole_center)
        index_point += 1

    polar = 10 * np.log10(4 * np.pi * u / total_power)  # Conversion to dB
    return polar

def antenna_directivity_pattern(filename_mesh2_to_load, filename_current_to_load, filename_gain_power_to_load, scattering=False, radiation=False, show=True, save_image=False):
    """
        Generates the antenna directivity pattern in the Phi = 0° and Phi = 90° planes.
        This function loads the necessary data (mesh, currents, radiated power),
        computes polar intensity plots, and displays the results.
        Parameters:
            * filename_mesh2_to_load : Path to the file containing the antenna mesh.
            * filename_current_to_load : Path to the file containing the currents on the antenna.
            * filename_gain_power_to_load : Path to the file containing gain and power data.
    """
    # Extract and modify the base file name
    base_name = os.path.splitext(os.path.basename(filename_mesh2_to_load))[0]
    base_name = base_name.replace('_mesh2', '')
    
    # Load the necessary data
    _, triangles, edges, *_ = DataManager_rwg2.load_data(filename_mesh2_to_load)
    if scattering:
        _, omega, _, _, light_speed_c, eta, _, _, _, current = DataManager_rwg4.load_data(filename_current_to_load, scattering=scattering)
    elif radiation:
        _, omega, _, _, light_speed_c, eta, _, current, *_ = DataManager_rwg4.load_data(filename_current_to_load, radiation=radiation)
    elif (radiation is False and scattering is False) or (radiation is True and scattering is True):
        raise ValueError("Either radiation or scattering must be True, but not both or neither.")
    
    total_power, *_ = load_gain_power_data(filename_gain_power_to_load)
    
    # Compute fundamental parameters
    k = omega / light_speed_c    # Wave number (rad/m)
    complex_k = 1j * k           # Complex wave number component
    dipole_center, dipole_moment = compute_dipole_center_moment(triangles, edges, current)  # Dipole moments
    
    numbers_of_points = 100    # Number of points per plane
    radius = 100               # Radius of the observation sphere
    
    # Compute observation points for Phi = 0° and Phi = 90°
    theta = np.linspace(0, 2 * np.pi, numbers_of_points)    # Theta angles (0 to 360°)
    points_yz = compute_circle_points(radius, numbers_of_points, plane="yz")
    points_xy = compute_circle_points(radius, numbers_of_points, plane="xy")
    
    # Compute polar intensity plots
    polar_0 = compute_polar(points_yz, numbers_of_points, eta, complex_k, dipole_moment, dipole_center, total_power)
    polar_90 = compute_polar(points_xy, numbers_of_points, eta, complex_k, dipole_moment, dipole_center, total_power)
    
    if show or save_image:
        # Visualize the polar plot
        ax = plt.subplot(projection='polar')
        ax.plot(theta, polar_0, color='red', label='Phi = 0°')
        ax.plot(theta, polar_90, color='blue', label='Phi = 90°')
        
        # Configure axes and legends
        # ax.set_theta_zero_location("N")    # 0° at north
        # ax.set_theta_direction(-1)         # Clockwise direction for angles
        ax.set_rlabel_position(-30)      # Radial label position
        ax.text(0, max(polar_0) + 5, "z", ha='center', va='bottom', fontsize=10, color='red')
        ax.legend()
        ax.grid(True)
        ax.set_title(base_name + " E-field pattern in Phi = 0° and 90° plane", va='bottom')
        
        # IMPORTANT: Save BEFORE showing
        if save_image:
            # Create directory if it does not exist
            output_dir_fig_image = "data/fig_image/"
            if not os.path.exists(output_dir_fig_image):
                os.makedirs(output_dir_fig_image)
            
            # Save the figure
            pdf_path = os.path.join(output_dir_fig_image, 'antenna_directivity_pattern' + ".pdf")
            plt.tight_layout()
            plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
            print(f"The figure has been saved in {pdf_path}")
        
        # Show AFTER saving
        if show:
            plt.show()
        else:
            plt.close()  # Close the figure if not showing to free memory