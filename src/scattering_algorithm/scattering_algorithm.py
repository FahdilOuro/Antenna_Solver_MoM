"""
    Cet algorithme permet de simuler la distribution de courant sur une antenne recevant une onde électromagnétique incidente.
    Il s'appuie sur les fonctions RWG (Rao-Wilton-Glisson) disponibles dans le dossier "/rwg".
    Les étapes principales comprennent :
        1. Le chargement et le traitement du maillage de l'antenne.
        2. La construction de la matrice d'impédance et des vecteurs nécessaires.
        3. Le calcul du courant induit par l'onde incidente.
        4. La visualisation des courants de surface sur l'antenne.

    Entrées principales :
        * mesh1 : Fichier contenant le maillage de l'antenne.
        * frequency : Fréquence de l'onde incidente (en Hz).
        * wave_incident_direction : Direction de propagation de l'onde incidente (vecteur 3D).
        * polarization : Polarisation de l'onde incidente (vecteur 3D).

    Sorties principales :
        * Visualisation des courants de surface sur l'antenne.
        * Sauvegarde des données intermédiaires dans différents dossiers pour un traitement ultérieur.
"""
from rwg.rwg1 import *
from rwg.rwg2 import *
from rwg.rwg3 import *
from rwg.rwg4 import *
from rwg.rwg5 import *

def scattering_algorithm(mesh, frequency, wave_incident_direction, polarization, load_from_matlab=True):
    """
        Implémente l'algorithme de diffusion électromagnétique pour une antenne.
    """
    # Chargement du fichier de maillage
    p, t = load_mesh_file(mesh,load_from_matlab)

    # Définition des points et triangles à partir du maillage
    points = Points(p)
    triangles = Triangles(t)

    # Filtrage des triangles invalides et calcul des propriétés géométriques (aires, centres)
    triangles.filter_triangles()
    triangles.calculate_triangles_area_and_center(points)

    # Affiche les dimensions principales de l'antenne
    base_name = os.path.splitext(os.path.basename(mesh))[0]
    print(f"length of antenna {base_name} = {points.length} meter")
    print(f"width of antenna {base_name} = {points.width} meter")
    print(f"height of antenna {base_name} = {points.height} meter")

    # Définition des arêtes et calcul de leurs longueurs
    edges = triangles.get_edges()
    edges.compute_edges_length(points)

    # Filtrage des jonctions complexes pour simplifier la structure du maillage
    filter_complexes_jonctions(triangles, edges)

    # Sauvegarde des données du maillage traité
    save_folder_name_mesh1 = 'data/antennas_mesh1/'
    save_file_name_mesh1 = DataManager_rwg1.save_data(mesh, save_folder_name_mesh1, points, triangles, edges)

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
    save_file_name_impedance = DataManager_rwg3.save_data(filename_mesh2_to_load, save_folder_name_impedance, frequency,
                                                          omega, mu, epsilon, light_speed_c, eta, matrice_z)

    # Chargement des données d'impédance
    filename_impedance = save_folder_name_impedance + save_file_name_impedance

    # Calcul du courant induit sur l'antenne par l'onde incidente
    frequency, omega, mu, epsilon, light_speed_c, eta, voltage, current = calculate_current_scattering(filename_mesh2_to_load, filename_impedance,
                                                                                                       wave_incident_direction, polarization)

    # Sauvegarde des données de courant
    save_folder_name_current = 'data/antennas_current/'
    save_file_name_current = DataManager_rwg4.save_data_for_scattering(filename_mesh2_to_load, save_folder_name_current, frequency,
                                                        omega, mu, epsilon, light_speed_c, eta, wave_incident_direction,
                                                        polarization, voltage, current)
    print(f"Sauvegarde du fichier : {save_file_name_current} effectué avec succès !")

    print(f"Fréquence de l'onde incidente : {frequency} Hz")

    # Calcul des courants de surface à partir du courant total
    surface_current_density = calculate_current_density(current, triangles, edges, vecteurs_rho)

    # Visualisation des courants de surface
    antennas_name = os.path.splitext(os.path.basename(filename_mesh2_to_load))[0].replace('_mesh2', ' antenna surface current in receiving mode')
    print(f"{antennas_name} view is successfully created at frequency {frequency} Hz")
    fig = visualize_surface_current(points, triangles, surface_current_density, antennas_name)
    fig.show()