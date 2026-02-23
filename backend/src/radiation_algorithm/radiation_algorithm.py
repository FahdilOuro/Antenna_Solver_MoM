from backend.rwg.rwg1 import *
from backend.rwg.rwg2 import *
from backend.rwg.rwg3 import *
from backend.rwg.rwg4 import *
from backend.rwg.rwg5 import *
from backend.rwg.rwg6 import plot_surface_current_distribution


'''def radiation_algorithm(mesh, frequency, feed_point, voltage_amplitude=1, port_type=0, monopole=False, 
                        simulate_array_antenna=False, show=True, save_image=False,
                        load_lumped_elements=False, LoadPoint=None, LoadValue=None, LoadDir=None):'''
def radiation_algorithm(mesh, frequency, feed_point, voltage_amplitude=1, excitation_unit_vector=None, show=True, save_image=False,
                        load_lumped_elements=False, LoadPoint=None, LoadValue=None, LoadDir=None):
    
    if (not load_lumped_elements and (LoadPoint is not None or LoadValue is not None or LoadDir is not None)) or \
        (load_lumped_elements and (LoadPoint is None or LoadValue is None or LoadDir is None)):
         raise ValueError("Incoherent parameters: If 'load_lumped_elements' is False, 'LoadPoint', 'LoadValue', and 'LoadDir' must all be None. If 'load_lumped_elements' is True, all three must be provided (not None).")
    
    # Load the mesh file
    p, t = load_mesh_file(mesh)

    # Define points and triangles from the mesh
    points = Points(p)
    triangles = Triangles(t)

    # Compute geometric properties (areas, centers)
    triangles.calculate_triangles_area_and_center(points)

    # Display main dimensions of the antenna
    base_name = os.path.splitext(os.path.basename(mesh))[0]

    # Define edges and compute their lengths
    edges = triangles.get_edges()

    filter_complexes_jonctions(points, triangles, edges)  # Filter complex junctions to simplify the mesh structure

    edges.compute_edges_length(points)

    # Save processed mesh data
    save_folder_name_mesh1 = 'data/antennas_mesh1/'
    save_file_name_mesh1 = DataManager_rwg1.save_data(mesh, save_folder_name_mesh1, points, triangles, edges)

    # Load saved data
    filename_mesh1_to_load = save_folder_name_mesh1 + save_file_name_mesh1

    # Definition and calculation of barycentric triangles
    barycentric_triangles = Barycentric_triangle()
    barycentric_triangles.calculate_barycentric_center(points, triangles)

    # Calculation of RHO vectors for edges
    vecteurs_rho = Vecteurs_Rho()
    vecteurs_rho.calculate_vecteurs_rho(points, triangles, edges, barycentric_triangles)

    # Save barycentric triangles and RHO vectors data
    save_folder_name_mesh2 = 'data/antennas_mesh2/'
    save_file_name_mesh2 = DataManager_rwg2.save_data(filename_mesh1_to_load, save_folder_name_mesh2, barycentric_triangles, vecteurs_rho)

    # Load data for the processed mesh
    filename_mesh2_to_load = save_folder_name_mesh2 + save_file_name_mesh2

    # Calculation of electromagnetic constants and impedance matrix Z
    if load_lumped_elements:
        omega, mu, epsilon, light_speed_c, eta, matrice_z, _ = calculate_z_matrice_lumped_elements(points,
                                                                                                   triangles,
                                                                                                   edges,
                                                                                                   barycentric_triangles,
                                                                                                   vecteurs_rho,
                                                                                                   frequency,
                                                                                                   LoadPoint,
                                                                                                   LoadValue,
                                                                                                   LoadDir)
    else:
        omega, mu, epsilon, light_speed_c, eta, matrice_z = calculate_z_matrice(triangles,
                                                                                edges,
                                                                                barycentric_triangles,
                                                                                vecteurs_rho,
                                                                                frequency)

    # Save impedance data
    save_folder_name_impedance = 'data/antennas_impedance/'
    save_file_name_impedance = DataManager_rwg3.save_data(filename_mesh2_to_load, save_folder_name_impedance, frequency, omega, mu, epsilon, light_speed_c, eta, matrice_z)

    # Load impedance data
    filename_impedance = save_folder_name_impedance + save_file_name_impedance

    # Calculate the induced current on the antenna by the incident wave
    """if port_type == 0:
        port_type = 'unique_point'
    elif port_type == 1:
        port_type = 'boundary_line'
    elif port_type == 2:
        port_type = 'polygonal_area'
    elif port_type == 3:
        port_type = 'Old_feed_port'
    else:
        raise ValueError("Invalid port_type. Must be 0 (unique_point), 1 (boundary_line), or 2 (polygonal_area).")

    frequency, omega, mu, epsilon, light_speed_c, eta, voltage, current, gap_current, gap_voltage, impedance, feed_power, index_feeding_edges = \
        calculate_current_radiation(
            filename_mesh2_to_load, 
            filename_impedance, 
            feed_point, 
            voltage_amplitude, 
            port_type, 
            monopole=monopole, 
            simulate_array_antenna=simulate_array_antenna)"""
    
    frequency, omega, mu, epsilon, light_speed_c, eta, voltage, current = \
    calculate_current_radiation(filename_mesh2_to_load, filename_impedance, feed_point, voltage_amplitude, excitation_unit_vector)

    gap_current = 0.0
    gap_voltage = 0.0
    impedance = 0
    feed_power = 0.0

    # printing gap current gap voltage impedance and feed power (scientific notation with 4 decimals)
    print(f"Gap Current: {gap_current:.4e}")
    print(f"Gap Voltage: {gap_voltage:.4e}")
    print(f"Input Impedance at the feed point: {impedance:.4e} Ohms")
    print(f"Feed Power: {feed_power:.4e} Watts")
    # Save current data
    save_folder_name_current = 'data/antennas_current/'
    save_file_name_current = DataManager_rwg4.save_data_for_radiation(filename_mesh2_to_load, save_folder_name_current, frequency, omega, mu, epsilon, light_speed_c, eta, voltage, current, gap_current, gap_voltage, impedance, feed_power)

    # Compute surface currents from the total current
    surface_current_density = calculate_current_density(current, triangles, edges, vecteurs_rho)

    # Visualization of surface currents
    if show:
        antennas_name = os.path.splitext(os.path.basename(filename_mesh2_to_load))[0].replace('_mesh2', ' antenna surface current in radiation mode')
        fig = visualize_surface_current(points, triangles, surface_current_density, feed_point, antennas_name)
        fig.show()

        if save_image:
            # Output folder path
            output_dir_fig_image = "data/fig_image/"
            
            # Create the folder if it does not exist
            if not os.path.exists(output_dir_fig_image):
                os.makedirs(output_dir_fig_image)
            
            # Name of the PDF file to save
            pdf_path = os.path.join(output_dir_fig_image, antennas_name.replace(" ", "_") + ".pdf")
            
            # Set transparent background and remove white margins
            fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=0, b=0)
            )

            # Save the figure
            fig.write_image(pdf_path, format="pdf")
            print(f"\nImage saved in PDF format: {pdf_path}\n")

    return matrice_z, voltage, current, surface_current_density