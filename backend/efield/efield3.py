import os

import numpy as np
from matplotlib import pyplot as plt

from backend.efield.efield2 import load_gain_power_data
from backend.rwg.rwg2 import DataManager_rwg2
from backend.rwg.rwg4 import DataManager_rwg4
from backend.utils.dipole_parameters import compute_dipole_center_moment, compute_e_h_field

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

def antenna_directivity_pattern(path, mode='radiation', show=True, save_image=False):
    """
    Generates a modern and professional polar plot of the antenna directivity pattern.
    
    This function processes MoM simulation data to visualize the radiation intensity 
    in the Phi = 0° and Phi = 90° planes with enhanced aesthetics. The 90° angle 
    is positioned at the top of the plot for specific visualization requirements.

    Parameters:
        path (Namespace): Object containing file paths (mesh, current, gain_power).
        mode (str): Simulation mode, either 'radiation' or 'scattering'.
        show (bool): If True, displays the plot.
        save_image (bool): If True, saves the plot as a high-resolution PDF.
    """
    # Load data
    _, triangles, edges, *_ = DataManager_rwg2.load_data(path.mat_mesh2)
    
    radiation = (mode == 'radiation')
    scattering = (mode == 'scattering')
    
    if scattering:
        _, omega, _, _, light_speed_c, eta, _, _, _, current = DataManager_rwg4.load_data(path.mat_current, scattering=scattering)
    elif radiation:
        _, omega, _, _, light_speed_c, eta, _, current, *_ = DataManager_rwg4.load_data(path.mat_current, radiation=radiation)
    else:
        raise ValueError("Mode must be 'radiation' or 'scattering'.")
    
    total_power, *_ = load_gain_power_data(path.mat_gain_power)
    
    # Physics computation
    k = omega / light_speed_c
    complex_k = 1j * k
    dipole_center, dipole_moment = compute_dipole_center_moment(triangles, edges, current)
    
    points_count = 200 # Higher resolution
    radius = 100
    
    theta = np.linspace(0, 2 * np.pi, points_count)
    points_yz = compute_circle_points(radius, points_count, plane="yz")
    points_xy = compute_circle_points(radius, points_count, plane="xy")
    
    polar_0 = compute_polar(points_yz, points_count, eta, complex_k, dipole_moment, dipole_center, total_power)
    polar_90 = compute_polar(points_xy, points_count, eta, complex_k, dipole_moment, dipole_center, total_power)
    
    if show or save_image:
        # Style configuration
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({'font.size': 11, 'font.family': 'sans-serif'})

        fig, ax = plt.subplots(figsize=(9.5, 10), subplot_kw={'projection': 'polar'})
        
        # Rotate the plot: 90 degrees at the top (North)
        # Default 0 is East (right). To put 90 at Top, we keep 0 at East.
        ax.set_theta_offset(0) 
        ax.set_theta_direction(1) # Counter-clockwise

        # Plot data with vibrant colors
        ax.plot(theta, polar_0, color='#FF5733', label=r'Elevation Plane $\phi = 0^\circ$ (YZ)', linewidth=2.5)
        ax.plot(theta, polar_90, color='#3357FF', label=r'Azimuth Plane $\Theta = 90^\circ$ (XY)', linewidth=2.5, linestyle='--')
        
        # Grid and Ticks
        # Set ticks every 30 degrees
        angles = np.arange(0, 360, 30)
        ax.set_thetagrids(angles)
        
        # Styling the radial axis (intensity)
        ax.set_rlabel_position(115) 
        ax.tick_params(axis='both', which='major', labelsize=10, grid_alpha=0.5)
        
        # Legend and Title
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True)
        # ax.set_title(f"Radiation Pattern - {path.name}\nMode: {mode.upper()}", fontsize=15, fontweight='bold', pad=20)

        if save_image:
            output_dir = "data/fig_image/"
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f'pattern_{path.name}.pdf'), bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()