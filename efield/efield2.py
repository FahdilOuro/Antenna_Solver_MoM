"""
This code calculates and visualizes the distribution of radiation intensity (U) of an electromagnetic field
that is either radiated or scattered by a surface, over an imaginary sphere surrounding the radiating object.
The sphere serves to simulate the reception of waves at a given distance,
and the computations allow the determination of parameters such as the total radiated power and the gain.
It computes the radiation density and the radiation intensity distributed over the sphere.
"""
import os

import numpy as np
from scipy.io import loadmat, savemat
import plotly.figure_factory as ff

from rwg.rwg2 import DataManager_rwg2
from rwg.rwg4 import DataManager_rwg4
from utils.dipole_parameters import compute_dipole_center_moment, compute_e_h_field

def compute_aspect_ratios(points_data):
    """
    Computes scaling ratios for 3D visualization.

    This function takes as input a set of 3D points (x, y, z) and returns the scaling
    ratios for the x, y, and z axes to ensure a uniform representation during 3D visualization.

    Parameters:
    points_data : tuple or n-d-array of shape (3, N), where N is the number of points.
        Contains the x, y, and z coordinates of the 3D points to be displayed.

    Returns:
    A dictionary with the normalized scaling ratios for each axis ('x', 'y', 'z'),
    ensuring a consistent scale for 3D visualization.
    """

    # Extract x, y, and z coordinates from points_data
    x_, y_, z_ = points_data

    # Compute the overall figure scale by taking the largest difference among the axes
    fig_scale = max(max(x_) - min(x_), max(y_) - min(y_), max(z_) - min(z_))

    # Compute the scaling ratios for each axis relative to the overall scale
    return {
        "x": (max(x_) - min(x_)) / fig_scale,
        "y": (max(y_) - min(y_)) / fig_scale,
        "z": (max(z_) - min(z_)) / fig_scale,
    }

def visualize_surface_current(points_data, triangles_data, radiation_intensity, title="Antennas Surface Current"):
    """
        Visualizes the surface current density using Plotly.

        This function creates a 3D visualization of the surface current density on an antenna model,
        using the Plotly library for interactive presentation. The surface is colored according to
        the radiation intensity, with a colormap to better represent the intensity distribution.

        Parameters:
            * points_data: tuple or n-d-array of shape (3, N), where N is the number of points.
            It contains the x, y, and z coordinates of the 3D vertices of the antenna.
            * triangles_data: n-d-array of shape (3, M), where M is the number of triangles.
            It contains the vertex indices for each triangle of the antenna surface.
            * radiation_intensity: n-d-array, the current density or radiation intensity associated
            with each triangle. This value will be used to color the surface.
            * title: str, title of the visualization (optional). Default is "Antennas Surface Current".

        Returns:
        fig: Plotly object, the 3D figure representing the surface current density colored by radiation intensity.
    """
    # Extract vertex coordinates (x, y, z) from points_data
    x_, y_, z_ = points_data  # X, Y, Z coordinates of the points

    # Create simplices for Plotly (vertex indices of each triangle)
    simplices = triangles_data[:3, :].T  # Transpose to go from [3, n_triangles] to [n_triangles, 3]

    # Compute aspect ratios to adjust the visualization proportions
    aspect_ratios = compute_aspect_ratios(points_data)

    # Create the figure using Plotly's trisurf
    fig = ff.create_trisurf(
        x=x_,                            # X coordinates of vertices
        y=y_,                            # Y coordinates of vertices
        z=z_,                            # Z coordinates of vertices
        simplices=simplices,             # Vertex indices of each triangle
        colormap="Rainbow",              # Colormap for surface coloring
        plot_edges=False,                # Do not display triangle edges
        color_func=radiation_intensity,  # Use normalized current density to color the surface
        show_colorbar=True,              # Display colorbar
        # title=title,                     # Visualization title
        title='',                        # Visualization title
        aspectratio=dict(aspect_ratios), # Adjust aspect ratios for 3D display
    )

    # Return the created Plotly figure
    return fig

def save_gain_power_data(save_folder_name, save_file_name, total_power, gain_linear, gain_logarithmic, efficiency_total):
    """
    Save total power and gain data into a .mat file.

    This function saves the total power results and the linear and logarithmic gains
    into a MATLAB (MAT) file for later use or further analysis.

    Parameters:
        * save_folder_name : str, name of the folder where the file will be saved. If the folder does not exist, it will be created.
        * save_file_name : str, name of the file to save (must include the .mat extension).
        * total_power : float or n-d-array, calculated total power value.
        * gain_linear : float or n-d-array, calculated linear gain (expressed as a multiplicative factor).
        * gain_logarithmic : float or n-d-array, calculated logarithmic gain (expressed in dB).

    Side effects:
        * Creates the specified folder if it does not exist.
        * Saves a .mat file containing power and gain data at the specified location.
    """
    # Build the full path for the file to save
    full_save_path = os.path.join(save_folder_name, save_file_name)

    # Check if the folder exists and create it if necessary
    if not os.path.exists(save_folder_name):  # Check and create folder if needed
        os.makedirs(save_folder_name)
        print(f"Directory '{save_folder_name}' created.")

    # Prepare the data to save in a dictionary
    data_gain_power = {
        'totalPower': total_power,
        'gainLinear': gain_linear,
        'gainLogarithmic': gain_logarithmic,
        'efficiencyTotal': efficiency_total
    }

    # Save the data into the .mat file
    savemat(full_save_path, data_gain_power)
    print(f"Data saved successfully to {full_save_path}")

def load_gain_power_data(filename_to_load):
    """
        Load power and gain data from a .mat file.

        This function loads a MATLAB (MAT) file containing total power, linear gain, and logarithmic gain results,
        retrieving the data associated with these parameters. It also handles possible errors during the data
        loading process.

        Parameter:
        filename_to_load : str, full path of the .mat file to load.

        Returns:
            * total_power : float or n-d-array, total power loaded from the file.
            * gain_linear : float or n-d-array, linear gain loaded from the file.
            * gain_logarithmic : float or n-d-array, logarithmic gain (in dB) loaded from the file.
            * efficiency_total : float or n-d-array, efficiency loaded from the file.

        Exceptions:
            * FileNotFoundError : raised if the specified file does not exist.
            * KeyError : raised if one of the expected keys ('totalPower', 'gainLinear', 'gainLogarithmic', 'efficiencyTotal') is missing in the file.
            * ValueError : raised if the data is malformed or corrupted.
            * General Exception : raised for any other unexpected error.
    """
    try:
        # Check if the file exists before loading
        if not os.path.isfile(filename_to_load):
            raise FileNotFoundError(f"File '{filename_to_load}' does not exist.")

        # Load data from the .mat file
        data = loadmat(filename_to_load)

        # Extract data: total power, linear gain, and logarithmic gain
        total_power = data['totalPower'].squeeze()
        gain_linear = data['gainLinear'].squeeze()
        gain_logarithmic = data['gainLogarithmic'].squeeze()
        efficiency_total = data['efficiencyTotal'].squeeze()

        print(f"Data loaded from {filename_to_load}")

        # Return extracted data
        return total_power, gain_linear, gain_logarithmic, efficiency_total
    
    except FileNotFoundError as e:
        # Handle errors if the file is not found
        print(f"Error: {e}")

    except KeyError as e:
        # Handle errors if a key is missing in the .mat file
        print(f"Key Error: {e}")

    except ValueError as e:
        # Handle errors if the data is malformed
        print(f"Value Error (likely malformed data): {e}")

    except Exception as e:
        # Handle unexpected errors
        print(f"An unexpected error occurred: {e}")

def radiation_intensity_distribution_over_sphere_surface(filename_mesh2_to_load, filename_current_to_load, filename_sphere_to_load, 
                                                         scattering = False, radiation = False, voltage_amplitude=0.5, show=True, save_image=False):
    """
        Calculate and visualize the radiation intensity and gain distribution on the surface of a sphere surrounding an antenna.

        This function loads the necessary data (mesh, currents, sphere), performs electromagnetic field calculations
        for each triangle of the sphere, and computes metrics such as total power, linear gain, and logarithmic gain.
        The results are then saved and visualized.

        Parameters:
            * filename_mesh2_to_load : str
                Path to the file containing the antenna mesh data (triangles, points, etc.).
            * filename_current_to_load : str
                Path to the file containing the current data on the antenna.
            * filename_sphere_to_load : str
                Path to the file containing the sphere data (coordinates and triangles).

        Returns:
        No explicit return. The results are saved to a file and visualized.

        Main steps:
            1. Load input data (mesh, currents, sphere).
            2. Calculate electromagnetic fields on the sphere's triangles.
            3. Compute radiation metrics: total power, linear and logarithmic gain.
            4. Save the computed results.
            5. Visualize the results as a gain distribution over the sphere.
    """

    # Extract the base file name without extension and modify the name
    base_name = os.path.splitext(os.path.basename(filename_mesh2_to_load))[0]
    base_name = base_name.replace('_mesh2', '')

    # Load files containing mesh, current, and sphere data
    data_sphere = loadmat(filename_sphere_to_load)

    _, triangles, edges, *_ = DataManager_rwg2.load_data(filename_mesh2_to_load)

    if scattering:
        frequency, omega, _, _, light_speed_c, eta, _, _, _, current = DataManager_rwg4.load_data(filename_current_to_load, scattering=scattering)
    elif radiation:
        frequency, omega, _, _, light_speed_c, eta, _, current, _, gap_current, *_ = DataManager_rwg4.load_data(filename_current_to_load, radiation=radiation)
    elif (radiation is False and scattering is False) or (radiation is True and scattering is True):
        raise ValueError("Either radiation or scattering must be True, but not both or neither.")

    # Load sphere data
    sphere_points = data_sphere['p'] * 100    # Sphere coordinates are scaled by 100 (radius of 100 m).
    sphere_triangles = data_sphere['t'] - 1   # Convert MATLAB indices (1-based) to Python indices (0-based).

    # Compute wave number k and its complex component
    k = omega / light_speed_c    # Wave number (in rad/m).
    complex_k = 1j * k           # Complex component.

    # Display frequency and wavelength
    print('')
    print(f"Frequency = {frequency} Hz")
    print(f"Wavelength lambda = {light_speed_c / frequency} m")

    # Compute dipoles and dipole moments (complex)
    dipole_center, dipole_moment = compute_dipole_center_moment(triangles, edges, current)

    # Initialization for field and total power calculation
    sphere_total_of_triangles = sphere_triangles.shape[1]
    total_power = 0
    observation_point = np.zeros((3, sphere_total_of_triangles))
    poynting_vector = np.zeros((3, sphere_total_of_triangles))
    norm_observation_point = np.zeros(sphere_total_of_triangles)
    e_field_total = np.zeros((3, sphere_total_of_triangles), dtype=complex)
    h_field_total = np.zeros((3, sphere_total_of_triangles), dtype=complex)
    sphere_triangle_area = np.zeros(sphere_total_of_triangles)
    w = np.zeros(sphere_total_of_triangles)
    u = np.zeros(sphere_total_of_triangles)

    # Loop over each triangle of the sphere to calculate fields and energy
    for triangle_in_sphere in range(sphere_total_of_triangles):
        sphere_triangle = sphere_triangles[:, triangle_in_sphere]
        observation_point[:, triangle_in_sphere] = np.sum(sphere_points[:, sphere_triangle], axis=1) / 3

        (e_field_total[:, triangle_in_sphere],
         h_field_total[:, triangle_in_sphere],
         poynting_vector[:, triangle_in_sphere],
         w[triangle_in_sphere],
         u[triangle_in_sphere],
         norm_observation_point[triangle_in_sphere]) = compute_e_h_field(observation_point[:, triangle_in_sphere],
                                                                        eta,
                                                                        complex_k,
                                                                        dipole_moment,
                                                                        dipole_center)

        vecteur_1 = sphere_points[:, sphere_triangle[0]] - sphere_points[:, sphere_triangle[1]]
        vecteur_2 = sphere_points[:, sphere_triangle[2]] - sphere_points[:, sphere_triangle[1]]
        sphere_triangle_area[triangle_in_sphere] = np.linalg.norm(np.cross(vecteur_1, vecteur_2)) / 2

        # Contribution of each triangle to the total power
        total_power += w[triangle_in_sphere] * sphere_triangle_area[triangle_in_sphere]

    print('')

    # Calculation of the antenna directivity: it is a measure of how focused an antenna's radiation pattern is in a specific direction 
    # compared to an idealized isotropic antenna that radiates equally in all directions.
    # It quantifies the antenna's ability to concentrate radiated power in a particular direction.
    # Here we call it gain
    # Calculation of gain (linear and logarithmic)
    gain_linear = 4 * np.pi * u / total_power
    gain_logarithmic = 10 * np.log10(gain_linear)
    gain_linear_max = 4 * np.pi * np.max(u) / total_power
    gain_logarithmic_max = 10 * np.log10(gain_linear_max)

    print(f"Total Power : {total_power : 4f}")
    print(f"Gain Linear : {gain_linear_max : 4f}")
    print(f"Gain Logarithmic (Max) : {gain_logarithmic_max : 4f} dBi")
    if radiation:
        print(f"\ngap_current = {gap_current}")
        # If gap_current is an array, take the norm or absolute value of the first element
        if isinstance(gap_current, np.ndarray):
            gap_current_abs = np.abs(gap_current)
            if gap_current_abs.size == 1:
                gap_current_val = gap_current_abs.item()
            else:
                gap_current_val = np.linalg.norm(gap_current_abs)
        else:
            gap_current_val = abs(gap_current)
        radiation_resistance = 2 * total_power / gap_current_val**2
        print(f"Radiation Resistance : {radiation_resistance : 4f} Ohms")

        V_gap = voltage_amplitude  # Supply voltage in volts
        P_in = 0.5 * V_gap * gap_current_val
        print(f"Input Power (P_in) : {P_in:.4f} W")

        efficiency_total = total_power / P_in
        print(f"Total Efficiency : {efficiency_total:.4f}")

        if efficiency_total > 1:
            print("Warning: Total Efficiency is greater than 1, which is physically impossible. Please check the calculations.")
            efficiency_total = 1.0  # Limit total efficiency to 1

    # Save the calculated results
    save_gain_power_folder_name = 'data/antennas_gain_power/'
    save_gain_power_file_name = base_name + '_gain_power.mat'
    save_gain_power_data(save_gain_power_folder_name, save_gain_power_file_name, total_power, gain_linear_max, gain_logarithmic_max, efficiency_total)

    # Visualization of the results
    plot_name_gain = base_name + ' gain distribution over a large sphere surface'
    sphere_total_of_points = sphere_points.shape[1]
    poynting_vector_point = np.zeros((3, sphere_total_of_points))
    norm_observation_point = np.zeros(sphere_total_of_points)
    e_field_total_points = np.zeros((3, sphere_total_of_points), dtype=complex)
    h_field_total_points = np.zeros((3, sphere_total_of_points), dtype=complex)
    w_points = np.zeros(sphere_total_of_points)
    u_points = np.zeros(sphere_total_of_points)

    for point_in_sphere in range(sphere_total_of_points):
        observation_point = sphere_points[:, point_in_sphere]

        (e_field_total_points[:, point_in_sphere],
         h_field_total_points[:, point_in_sphere],
         poynting_vector_point[:, point_in_sphere],
         w_points[point_in_sphere],
         u_points[point_in_sphere],
         norm_observation_point[point_in_sphere]) = compute_e_h_field(observation_point,
                                                                      eta,
                                                                      complex_k,
                                                                      dipole_moment,
                                                                      dipole_center)

    u_points_db = 10 * np.log10(4 * np.pi * u_points / total_power)
    threshold_db = max(u_points_db) - 20
    u_points_db = np.maximum(u_points_db[:sphere_total_of_points] - threshold_db, 0.01)
    sphere_points_update = u_points_db * sphere_points / 1000

    if show:
        # Visualization of the logarithmic gain
        fig2 = visualize_surface_current(sphere_points_update, sphere_triangles, gain_logarithmic, plot_name_gain)
        fig2.show()

        if save_image:
            # Output folder path
            output_dir_fig_image = "data/fig_image/"
            
            # Create the folder if it does not exist
            if not os.path.exists(output_dir_fig_image):
                os.makedirs(output_dir_fig_image)
                print(f"Folder created: {output_dir_fig_image}")
            
            # Name of the PDF file to save
            pdf_path = os.path.join(output_dir_fig_image, 'radiation_intensity_distribution' + ".pdf")
            
            # Set transparent background and remove white margins
            fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=0, b=0)
            )

            # Save the figure
            fig2.write_image(pdf_path, format="pdf")
            print(f"\nImage saved as PDF: {pdf_path}\n")