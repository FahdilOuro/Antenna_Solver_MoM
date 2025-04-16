from rwg.rwg1 import *
from rwg.rwg2 import *
from rwg.rwg3 import *
from rwg.rwg4 import *
from rwg.rwg5 import *
from utils.refinement_function import *


def radiation_algorithm(mesh1, frequency, feed_point, voltage_amplitude=1, load_from_matlab=True, monopole=False):
    # Chargement du fichier de maillage
    p, t = load_mesh_file(mesh1, load_from_matlab)

    # Définition des points et triangles à partir du maillage
    points = Points(p)
    triangles = Triangles(t)

    # Filtrage des triangles invalides et calcul des propriétés géométriques (aires, centres)
    triangles.filter_triangles()
    triangles.calculate_triangles_area_and_center(points)

    # Affiche les dimensions principales de l'antenne
    base_name = os.path.splitext(os.path.basename(mesh1))[0]

    # Définition des arêtes et calcul de leurs longueurs
    edges = triangles.get_edges()
    filter_complexes_jonctions(points, triangles, edges)          # Filtrage des jonctions complexes pour simplifier la structure du maillage

    print(f"\nNombre d'elements de maillage (edges) = {edges.total_number_of_edges}\n")
    print(f"\nNombre de triangles = {triangles.total_of_triangles}\n")

    edges.compute_edges_length(points)
    
    index = 0
    for area in triangles.triangles_area:
        if area == 0:
            print(area)
            print("Aire du triangle nulle a la colonne :", index)
        index += 1

    # Sauvegarde des données du maillage traité
    save_folder_name_mesh1 = 'data/antennas_mesh1/'
    save_file_name_mesh1 = DataManager_rwg1.save_data(mesh1, save_folder_name_mesh1, points, triangles, edges)

    # Chargement des données sauvegardées
    filename_mesh1_to_load = save_folder_name_mesh1 + save_file_name_mesh1

    # Définition et calcul des triangles barycentriques
    barycentric_triangles = Barycentric_triangle()
    barycentric_triangles.calculate_barycentric_center(points, triangles)

    # Calcul des vecteurs RHO pour les arêtes
    vecteurs_rho = Vecteurs_Rho()
    vecteurs_rho.calculate_vecteurs_rho(points, triangles, edges, barycentric_triangles)

    # Sauvegarde des données des triangles barycentriques et vecteurs RHO
    save_folder_name_mesh2 = 'data/antennas_mesh2/'
    save_file_name_mesh2 = DataManager_rwg2.save_data(filename_mesh1_to_load, save_folder_name_mesh2, barycentric_triangles, vecteurs_rho)

    # Chargement des données pour le maillage traité
    filename_mesh2_to_load = save_folder_name_mesh2 + save_file_name_mesh2

    # Calcul des constantes électromagnétiques et de la matrice d'impédance Z
    omega, mu, epsilon, light_speed_c, eta, matrice_z = calculate_z_matrice(triangles,
                                                                            edges,
                                                                            barycentric_triangles,
                                                                            vecteurs_rho,
                                                                            frequency)

    # Sauvegarde des données d'impédance
    save_folder_name_impedance = 'data/antennas_impedance/'
    save_file_name_impedance = DataManager_rwg3.save_data(filename_mesh2_to_load, save_folder_name_impedance, frequency, omega, mu, epsilon, light_speed_c, eta, matrice_z)

    # Chargement des données d'impédance
    filename_impedance = save_folder_name_impedance + save_file_name_impedance

    # Calcul du courant induit sur l'antenne par l'onde incidente
    frequency, omega, mu, epsilon, light_speed_c, eta, voltage, current, gap_current, gap_voltage, impedance, feed_power = calculate_current_radiation(filename_mesh2_to_load, filename_impedance, feed_point, voltage_amplitude, monopole)

    print(f"\nLa valeur de l'impédance d'entrée de l'antenne {base_name} = {impedance.real : .7f} {"+" if impedance.imag >= 0 else "-"}{abs(impedance.imag) : .7f}i Ohm")
    print(f"Gap current of {base_name} = {gap_current}")
    print(f"Gap voltage of {base_name} = {gap_voltage}")
    print(f"La valeur de feed_power  = {feed_power}\n")

    # Sauvegarde des données de courant
    save_folder_name_current = 'data/antennas_current/'
    save_file_name_current = DataManager_rwg4.save_data_for_radiation(filename_mesh2_to_load, save_folder_name_current, frequency, omega, mu, epsilon, light_speed_c, eta, voltage, current, gap_current, gap_voltage, impedance, feed_power)
    print(f"Sauvegarde du fichier : {save_file_name_current} effectué avec succès !")

    print(f"Fréquence de rayonnement de l'antenne : {frequency} Hz")

    # Calcul des courants de surface à partir du courant total
    surface_current_density = calculate_current_density(current, triangles, edges, vecteurs_rho)

    """selected_triangles = calculate_seuil_surface_current_density(surface_current_density)

    print(f"\nshape de selected_triangle = {selected_triangles.shape}\n")
    print(f"\nselected_triangle = {selected_triangles}\n")
    
    refinement_nodes = get_selected_edges(edges, selected_triangles)

    '''print(f"\nshape de refinement_nodes = {refinement_nodes.shape}\n")
    print(f"\nrefinement_nodes = {refinement_nodes}\n")'''

    high_current_points_list = get_edge_midpoints(points, refinement_nodes)"""
    
    """print(f"\nshape de high_current_points_list = {high_current_points_list.shape}\n")
    print(f"\ntype de high_current_points_list = {type(high_current_points_list)}\n")
    print(f"\ntype de high_current_points_list = {high_current_points_list}\n")"""

    # Visualisation des courants de surface
    antennas_name = os.path.splitext(os.path.basename(filename_mesh2_to_load))[0].replace('_mesh2', ' antenna surface current in receiving mode')
    print(f"{antennas_name} view is successfully created at frequency {frequency} Hz")
    fig = visualize_surface_current(points, triangles, surface_current_density, antennas_name)
    fig.show()

    return impedance, surface_current_density