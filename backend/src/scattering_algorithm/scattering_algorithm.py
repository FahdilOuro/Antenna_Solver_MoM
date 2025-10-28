"""
    This algorithm simulates the current distribution on an antenna receiving an incident electromagnetic wave.
    It relies on the RWG (Rao-Wilton-Glisson) functions available in the "/rwg" folder.
    The main steps include:
        1. Loading and processing the antenna mesh.
        2. Building the impedance matrix and the required vectors.
        3. Computing the induced current by the incident wave.
        4. Visualizing the surface currents on the antenna.

    Main inputs:
        * mesh1 : File containing the antenna mesh.
        * frequency : Frequency of the incident wave (in Hz).
        * wave_incident_direction : Propagation direction of the incident wave (3D vector).
        * polarization : Polarization of the incident wave (3D vector).

    Main outputs:
        * Visualization of surface currents on the antenna.
        * Saving intermediate data into different folders for further processing.
"""
from backend.rwg.rwg1 import *
from backend.rwg.rwg2 import *
from backend.rwg.rwg3 import *
from backend.rwg.rwg4 import *
from backend.rwg.rwg5 import *

def scattering_algorithm(mesh, frequency, wave_incident_direction, polarization, load_from_matlab=True, show=True):
    """
        Implements the electromagnetic scattering algorithm for an antenna.
    """
    # Load mesh file
    p, t = load_mesh_file(mesh,load_from_matlab)

    # Define points and triangles from the mesh
    points = Points(p)
    triangles = Triangles(t)

    # Filter invalid triangles and compute geometric properties (areas, centers)
    triangles.filter_triangles()
    triangles.calculate_triangles_area_and_center(points)

    # Display the main antenna dimensions
    base_name = os.path.splitext(os.path.basename(mesh))[0]

    # Define edges and compute their lengths
    edges = triangles.get_edges()
    edges.compute_edges_length(points)

    # Filter complex junctions to simplify the mesh structure
    filter_complexes_jonctions(points, triangles, edges)

    # Save processed mesh data
    save_folder_name_mesh1 = 'data/antennas_mesh1/'
    save_file_name_mesh1 = DataManager_rwg1.save_data(mesh, save_folder_name_mesh1, points, triangles, edges)

    # Load saved mesh data
    filename_mesh1_to_load = save_folder_name_mesh1 + save_file_name_mesh1

    # Define and compute barycentric triangles
    barycentric_triangles = Barycentric_triangle()
    barycentric_triangles.calculate_barycentric_center(points, triangles)

    # Compute RHO vectors for the edges
    vecteurs_rho = Vecteurs_Rho()
    vecteurs_rho.calculate_vecteurs_rho(points, triangles, edges, barycentric_triangles)

    # Save barycentric triangles and RHO vectors data
    save_folder_name_mesh2 = 'data/antennas_mesh2/'
    save_file_name_mesh2 = DataManager_rwg2.save_data(filename_mesh1_to_load, save_folder_name_mesh2, barycentric_triangles, vecteurs_rho)

    # Load processed mesh data
    filename_mesh2_to_load = save_folder_name_mesh2 + save_file_name_mesh2

    # Compute electromagnetic constants and impedance matrix Z
    omega, mu, epsilon, light_speed_c, eta, matrice_z = calculate_z_matrice(triangles,
                                                                            edges,
                                                                            barycentric_triangles,
                                                                            vecteurs_rho,
                                                                            frequency)

    # Save impedance data
    save_folder_name_impedance = 'data/antennas_impedance/'
    save_file_name_impedance = DataManager_rwg3.save_data(filename_mesh2_to_load, save_folder_name_impedance, frequency,
                                                          omega, mu, epsilon, light_speed_c, eta, matrice_z)

    # Load impedance data
    filename_impedance = save_folder_name_impedance + save_file_name_impedance

    # Compute the induced current on the antenna by the incident wave
    frequency, omega, mu, epsilon, light_speed_c, eta, voltage, current = calculate_current_scattering(filename_mesh2_to_load, filename_impedance,
                                                                                                       wave_incident_direction, polarization)

    # Save current data
    save_folder_name_current = 'data/antennas_current/'
    save_file_name_current = DataManager_rwg4.save_data_for_scattering(filename_mesh2_to_load, save_folder_name_current, frequency,
                                                        omega, mu, epsilon, light_speed_c, eta, wave_incident_direction,
                                                        polarization, voltage, current)

    # Compute surface currents from the total current
    surface_current_density = calculate_current_density(current, triangles, edges, vecteurs_rho)

    # Visualize surface currents if requested
    if show:
        antennas_name = os.path.splitext(os.path.basename(filename_mesh2_to_load))[0].replace('_mesh2', ' antenna surface current in receiving mode')
        print(f"\n{antennas_name} view is successfully created at frequency {frequency} Hz")
        fig = visualize_surface_current(points, triangles, surface_current_density, title=antennas_name)
        fig.show()