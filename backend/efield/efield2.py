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

from backend.rwg.rwg2 import DataManager_rwg2
from backend.rwg.rwg4 import DataManager_rwg4
from backend.utils.dipole_parameters import *
from backend.utils.gmsh_function import create_hollow_sphere, extract_msh_to_mat

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

def save_gain_power_data(path_mat_gain_power, total_power, gain_linear, gain_logarithmic, efficiency_total):
    """
    Save total power and gain data into a .mat file.

    This function saves the total power results and the linear and logarithmic gains
    into a MATLAB (MAT) file for later use or further analysis.

    Parameters:
        * path_mat_gain_power : str, full path of the .mat file to save.
        * total_power : float or n-d-array, calculated total power value.
        * gain_linear : float or n-d-array, calculated linear gain (expressed as a multiplicative factor).
        * gain_logarithmic : float or n-d-array, calculated logarithmic gain (expressed in dB).

    Side effects:
        * Creates the specified folder if it does not exist.
        * Saves a .mat file containing power and gain data at the specified location.
    """
    # Prepare the data to save in a dictionary
    data_gain_power = {
        'totalPower': total_power,
        'gainLinear': gain_linear,
        'gainLogarithmic': gain_logarithmic,
        'efficiencyTotal': efficiency_total
    }

    # Save the data into the .mat file
    savemat(path_mat_gain_power, data_gain_power)
    print(f"Data saved successfully to {path_mat_gain_power}")

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

def load_or_create_hollow_sphere(msh_path='data/gmsh_files/hollow_sphere.msh', 
                                 mat_path='data/antennas_mesh/hollow_sphere.mat', 
                                 scale_factor=300):
    """
    Ensures the hollow sphere mesh exists and loads its coordinates and connectivity.

    Parameters:
        msh_path (str): Path to the .msh file.
        mat_path (str): Path to the .mat file.
        scale_factor (float): Scaling factor for the sphere radius.

    Returns:
        tuple: (sphere_points, sphere_triangles)
    """
    if not os.path.isfile(mat_path):
        print("Creating hollow sphere mesh and extracting data...")
        if not os.path.isfile(msh_path):
            create_hollow_sphere()
        extract_msh_to_mat(msh_path, mat_path)

    data_sphere = loadmat(mat_path)
    # Scale coordinates and convert to 0-based indexing for Python
    sphere_points = data_sphere['p'] * scale_factor
    sphere_triangles = data_sphere['t'] - 1
    
    return sphere_points, sphere_triangles

def deform_sphere_for_radiation_pattern(sphere_points, gain_db, dynamic_range=20):
    """
    Deforms sphere coordinates based on gain values to create a 3D radiation pattern.

    Parameters:
        sphere_points (ndarray): Original coordinates of the sphere.
        gain_db (ndarray): Gain values in dB for each point.
        dynamic_range (float): Range in dB to display (values below max - range are clipped).

    Returns:
        ndarray: Deformed sphere coordinates.
    """
    # Calculate threshold based on peak gain
    threshold_db = np.max(gain_db) - dynamic_range
    
    # Normalize gain for deformation (clamped at 0.01 for visualization)
    normalized_gain = np.maximum(gain_db - threshold_db, 0.01)
    
    # Apply deformation: scale point vectors by normalized gain
    # Note: division by 1000 is used here to match your original scaling logic
    sphere_points_update = (normalized_gain * sphere_points) / 1000
    
    return sphere_points_update

def load_and_prepare_antenna_data(path, mode):
    """
    Load the hollow sphere, antenna mesh, and current data based on the simulation mode.
    Calculate and return the basic electromagnetic parameters and dipole data.
    """
    # 1. Load sphere geometry
    sphere_points, sphere_triangles = load_or_create_hollow_sphere()
    
    # 2. Load antenna mesh data
    _, triangles, edges, *_ = DataManager_rwg2.load_data(path.mat_mesh2)

    # 3. Load current and specific physical parameters depending on the mode
    if mode == 'scattering':
        frequency, omega, _, _, light_speed_c, eta, _, _, _, current = DataManager_rwg4.load_data(path.mat_current, scattering=True)
        gap_current = 0
    elif mode == 'radiation':
        frequency, omega, _, _, light_speed_c, eta, _, current, _, gap_current, *_ = DataManager_rwg4.load_data(path.mat_current, radiation=True)
    else:
        raise ValueError("Mode must be either 'radiation' or 'scattering'.")
    
    # 4. Compute physical constants (wave number)
    k = omega / light_speed_c
    complex_k = 1j * k
    
    # 5. Compute dipoles and their moments
    dipole_center, dipole_moment = compute_dipole_center_moment(triangles, edges, current)
    
    return sphere_points, sphere_triangles, frequency, light_speed_c, eta, complex_k, dipole_center, dipole_moment, gap_current

def compute_total_radiated_power(sphere_points, sphere_triangles, eta, complex_k, dipole_moment, dipole_center):
    """
    Iterate over all the triangles of the surrounding sphere to integrate the Poynting vector
    and calculate the total radiated power.
    """
    sphere_total_of_triangles = sphere_triangles.shape[1]
    total_power = 0
    u_triangles = np.zeros(sphere_total_of_triangles)
    
    # Loop over each triangle to calculate fields and energy
    for i in range(sphere_total_of_triangles):
        tri = sphere_triangles[:, i]
        obs_p = np.sum(sphere_points[:, tri], axis=1) / 3
        
        # Calculate electromagnetic fields at the triangle center
        _, _, _, w, u_triangles[i], _ = compute_e_h_field(obs_p, eta, complex_k, dipole_moment, dipole_center)
        
        # Calculate the area of the current triangle
        v1 = sphere_points[:, tri[0]] - sphere_points[:, tri[1]]
        v2 = sphere_points[:, tri[2]] - sphere_points[:, tri[1]]
        area = np.linalg.norm(np.cross(v1, v2)) / 2
        
        # Add the contribution of this triangle to the total power
        total_power += w * area
        
    return total_power, u_triangles

def compute_antenna_efficiency_metrics(mode, total_power, gap_current, voltage_amplitude):
    """
    Calculate and display the radiation resistance and total efficiency if in radiation mode.
    """
    efficiency_total = 0
    
    if mode == 'radiation':
        # Safely extract the gap current absolute value
        gap_current_val = np.linalg.norm(gap_current) if isinstance(gap_current, np.ndarray) else abs(gap_current)
        
        if gap_current_val != 0:
            # Calculate metrics
            rad_resistance = 2 * total_power / gap_current_val**2
            p_in = 0.5 * voltage_amplitude * gap_current_val
            efficiency_total = min(total_power / p_in, 1.0) if p_in > 0 else 0
            
            # Print results to console
            print(f"  Radiation Resistance : {rad_resistance:.4f} Ohms")
            print(f"  Total Efficiency : {efficiency_total*100:.2f} %")
            
    return efficiency_total

def render_and_save_pattern(show, save_image, sphere_points_update, sphere_triangles, gain_points_db, plot_title, pdf_filename):
    """
    Handle the Plotly 3D visualization and PDF saving process.
    Converts vertex-based gain into face-based gain to prevent Plotly ValueErrors.
    """
    if show:
        # Calculate mean gain per face for Plotly compatibility
        gain_faces_db = np.mean(gain_points_db[sphere_triangles], axis=0)
        
        # Create and display the 3D figure
        fig = visualize_surface_current(sphere_points_update, sphere_triangles, gain_faces_db, plot_title)
        fig.show()

        # Save the figure to disk if requested
        if save_image:
            output_dir = "data/fig_image/"
            os.makedirs(output_dir, exist_ok=True)
            pdf_path = os.path.join(output_dir, pdf_filename)
            
            # Remove margins and set a transparent background for the PDF
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=0, b=0))
            fig.write_image(pdf_path, format="pdf")
            print(f"Image saved: {pdf_path}")

def radiation_intensity_distribution_over_sphere_surface(path, mode='radiation', voltage_amplitude=1, show=True, save_image=False):
    """
    Calculate and visualize the total radiation intensity and gain distribution on the surface of a sphere.
    """
    print(f"MODE SELECTED: {mode}")

    # 1. Load data and setup parameters
    sphere_points, sphere_triangles, frequency, light_speed_c, eta, complex_k, dipole_center, dipole_moment, gap_current = load_and_prepare_antenna_data(path, mode)
    print(f"Frequency = {frequency:.2e} Hz | Wavelength lambda = {light_speed_c / frequency:.4f} m")

    # 2. Calculate Total Radiated Power
    total_power, u_triangles = compute_total_radiated_power(sphere_points, sphere_triangles, eta, complex_k, dipole_moment, dipole_center)

    # 3. Calculate Antenna Metrics
    gain_linear_max = 4 * np.pi * np.max(u_triangles) / total_power
    gain_logarithmic_max = 10 * np.log10(gain_linear_max)

    print(f"\n[Antenna Results]")
    print(f"  Total Radiated Power : {total_power:.4f} W")
    print(f"  Max Gain : {gain_linear_max:.4f} ({gain_logarithmic_max:.2f} dBi)")

    efficiency_total = compute_antenna_efficiency_metrics(mode, total_power, gap_current, voltage_amplitude)

    # Save calculated metrics
    save_gain_power_data(path.mat_gain_power, total_power, gain_linear_max, gain_logarithmic_max, efficiency_total)

    # 4. Pattern Visualization Prep (Calculate fields at vertices)
    sphere_total_of_points = sphere_points.shape[1]
    u_points = np.zeros(sphere_total_of_points)

    for i in range(sphere_total_of_points):
        obs_point = sphere_points[:, i]
        _, _, _, _, u_points[i], _ = compute_e_h_field(obs_point, eta, complex_k, dipole_moment, dipole_center)

    u_points_db = 10 * np.log10(np.maximum(4 * np.pi * u_points / total_power, 1e-12))

    # Apply 3D deformation
    sphere_points_update = deform_sphere_for_radiation_pattern(sphere_points, u_points_db, dynamic_range=20)

    # 5. Display and Save
    plot_title = f"{path.name} - Gain Distribution (dBi)"
    pdf_filename = f"radiation_pattern_{path.name}.pdf"
    render_and_save_pattern(show, save_image, sphere_points_update, sphere_triangles, u_points_db, plot_title, pdf_filename)

def polar_circular_gain_distribution_over_sphere(path, polar_type='RHCP', mode='radiation', voltage_amplitude=1, show=True, save_image=False):
    """
    Calculate and visualize the circular gain (RHCP or LHCP) distribution over a sphere.
    """
    print(f"\n--- CALCULATING {polar_type} GAIN ({mode.upper()} MODE) ---")

    # 1. Load data and setup parameters
    sphere_points, sphere_triangles, frequency, light_speed_c, eta, complex_k, dipole_center, dipole_moment, gap_current = load_and_prepare_antenna_data(path, mode)
    print(f"Frequency = {frequency:.2e} Hz | Wavelength lambda = {light_speed_c / frequency:.4f} m")

    # 2. Calculate Total Radiated Power (First Pass)
    total_power, _ = compute_total_radiated_power(sphere_points, sphere_triangles, eta, complex_k, dipole_moment, dipole_center)

    # 3. Calculate Circular Gain at each point (Second Pass on Vertices)
    sphere_total_of_points = sphere_points.shape[1]
    u_circular = np.zeros(sphere_total_of_points)

    for i in range(sphere_total_of_points):
        obs_p = sphere_points[:, i]
        r = np.linalg.norm(obs_p)
        
        # Get electromagnetic field components
        e_total, *_ = compute_e_h_field(obs_p, eta, complex_k, dipole_moment, dipole_center)
        e_theta, e_phi = compute_spherical_e_field(obs_p, r, e_total)
        e_rhcp, e_lhcp = compute_circular_components(e_theta, e_phi)
        
        # Select the requested polarization
        e_pol = e_rhcp if polar_type.upper() == 'RHCP' else e_lhcp
        u_circular[i] = (1 / (2 * eta)) * (np.abs(e_pol)**2) * (r**2)
    
    # 4. Metrics & Console Output
    gain_linear = (4 * np.pi * u_circular) / total_power
    gain_db = 10 * np.log10(np.maximum(gain_linear, 1e-12))
    
    gain_linear_max = np.max(gain_linear)
    gain_logarithmic_max = np.max(gain_db)
    
    print(f"Total Power : {total_power:.4f} W")
    print(f"Max {polar_type} Gain : {gain_logarithmic_max:.4f} dBic")

    efficiency_total = compute_antenna_efficiency_metrics(mode, total_power, gap_current, voltage_amplitude)

    # Save calculated metrics
    path_mat_polar_gain_power = path.mat_polar_rhcp_gain_power if polar_type.upper() == 'RHCP' else path.mat_polar_lhcp_gain_power
    save_gain_power_data(path_mat_polar_gain_power, total_power, gain_linear_max, gain_logarithmic_max, efficiency_total)

    # 5. Deformation & Visualization
    sphere_points_update = deform_sphere_for_radiation_pattern(sphere_points, gain_db, dynamic_range=20)
    
    plot_title = f"{path.name} - {polar_type} Gain (dBic)"
    pdf_filename = f"{polar_type}_pattern_{path.name}.pdf"
    render_and_save_pattern(show, save_image, sphere_points_update, sphere_triangles, gain_db, plot_title, pdf_filename)

def display_rhcp_gain(path, mode='radiation'):
    """Alias to quickly display the RHCP gain."""
    return polar_circular_gain_distribution_over_sphere(
        path, 
        polar_type='RHCP', 
        mode=mode
        )

def display_lhcp_gain(path, mode='radiation'):
    """Alias to quickly display the LHCP gain."""
    return polar_circular_gain_distribution_over_sphere(
        path, 
        polar_type='LHCP', 
        mode=mode
        )