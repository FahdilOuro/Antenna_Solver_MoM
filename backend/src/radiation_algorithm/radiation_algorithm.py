from backend.rwg.rwg1 import *
from backend.rwg.rwg2 import *
from backend.rwg.rwg3 import *
from backend.rwg.rwg4 import *
from backend.rwg.rwg5 import *

from backend.utils.gmsh_function import *


def radiation_algorithm(path, frequency, feed_point, voltage_amplitude=1, excitation_unit_vector=None, gap_width=0.05, voltage_phase=None,
                        show=True, save_image=False,
                        load_lumped_elements=False, LoadPoint=None, LoadValue=None, LoadDir=None):
    
    if (not load_lumped_elements and (LoadPoint is not None or LoadValue is not None or LoadDir is not None)) or \
        (load_lumped_elements and (LoadPoint is None or LoadValue is None or LoadDir is None)):
         raise ValueError("Incoherent parameters: If 'load_lumped_elements' is False, " \
         "'LoadPoint', 'LoadValue', and 'LoadDir' must all be None. If 'load_lumped_elements' is True, all three must be provided (not None).")
    
    extract_msh_to_mat(path.msh, path.mat)
    
    # Load the mesh file
    p, t = load_mesh_file(path.mat)

    # Define points and triangles from the mesh
    points = Points(p)
    triangles = Triangles(t)

    # Compute geometric properties (areas, centers)
    triangles.calculate_triangles_area_and_center(points)

    # Define edges and compute their lengths
    edges = triangles.get_edges()

    filter_complexes_jonctions(points, triangles, edges)  # Filter complex junctions to simplify the mesh structure

    edges.compute_edges_length(points)

    # Save processed mesh data
    DataManager_rwg1.save_data(path, points, triangles, edges)

    # Definition and calculation of barycentric triangles
    barycentric_triangles = Barycentric_triangle()
    barycentric_triangles.calculate_barycentric_center(points, triangles)

    # Calculation of RHO vectors for edges
    vecteurs_rho = Vecteurs_Rho()
    vecteurs_rho.calculate_vecteurs_rho(points, triangles, edges, barycentric_triangles)

    # Save barycentric triangles and RHO vectors data
    DataManager_rwg2.save_data(path, barycentric_triangles, vecteurs_rho)

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
    DataManager_rwg3.save_data(path, frequency, omega, mu, epsilon, light_speed_c, eta, matrice_z)
    
    '''frequency, omega, mu, epsilon, light_speed_c, eta, voltage, current, gap_current, source_voltage, impedance, feed_power = \
    calculate_current_radiation(path, feed_point, voltage_amplitude, excitation_unit_vector, gap_width, voltage_phase)'''
    frequency, voltage, current, port_results = \
    calculate_current_radiation(path, feed_point, voltage_amplitude, excitation_unit_vector, gap_width, voltage_phase)

    # Extract port-specific data into numpy arrays for structured saving
    # This converts the list of dictionaries into clean vectors
    gap_currents = np.array([p['gap_current'] for p in port_results])
    gap_voltages = np.array([p['source_voltage'] for p in port_results])
    impedances   = np.array([p['impedance'] for p in port_results])
    feed_powers  = np.array([p['power'] for p in port_results])

    # print(f"Impedances : {impedances[0]}")

    # 3. Save everything using the updated DataManager
    DataManager_rwg4.save_data_for_radiation(path, frequency, omega, mu, epsilon, light_speed_c, eta, 
                                             voltage, current, gap_currents, gap_voltages, impedances, feed_powers)

    # Compute surface currents from the total current
    surface_current_density = calculate_current_density(current, triangles, edges, vecteurs_rho)

    # Visualization of surface currents
    if show:
        fig = visualize_surface_current(points, triangles, surface_current_density, feed_point, path.name)
        fig.show()

        if save_image:
            # Output folder path
            output_dir_fig_image = "data/fig_image/"
            
            # Create the folder if it does not exist
            if not os.path.exists(output_dir_fig_image):
                os.makedirs(output_dir_fig_image)
            
            # Name of the PDF file to save
            antennas_name = path.name + ' antenna surface current in radiation mode'
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

    return matrice_z, voltage, current, surface_current_density, gap_currents, gap_voltages, impedances, feed_powers