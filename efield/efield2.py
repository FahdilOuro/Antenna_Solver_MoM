"""
 Ce code calcule et visualise la répartition de l'intensité de radiation (U) d'un champ électromagnétique rayonné ou diffusé par une surface,
 sur une sphère imaginaire qui entoure l'objet rayonnant. La sphère sert à simuler la réception des ondes à une distance donnée,
 et les calculs permettent de déterminer des paramètres comme la puissance totale rayonnée et le gain.
 Calcul la densité de radiation et l'intensité de radiation distribués sur la sphere
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
        Calcule les rapports d'échelle pour l'affichage 3D.

        Cette fonction prend en entrée un ensemble de points 3D (x, y, z), et retourne les rapports d'échelle
        pour les axes x, y et z afin de garantir une représentation uniforme lors de la visualisation 3D.

        Paramètres :
        points_data : tuple ou n-d-array de forme (3, N), où N est le nombre de points.
          Il contient les coordonnées x, y et z des points 3D à afficher.

        Retourne :
        Un dictionnaire avec les rapports d'échelle normalisés pour chaque axe ('x', 'y', 'z') afin d'ajuster
          l'affichage 3D avec une échelle uniforme.
    """

    # Extraction des coordonnées x, y et z à partir de points_data
    x_, y_, z_ = points_data

    # Calcul de l'échelle globale (figure scale) en prenant la plus grande différence entre les axes
    fig_scale = max(max(x_) - min(x_), max(y_) - min(y_), max(z_) - min(z_))

    # Calcul des rapports d'échelle pour chaque axe par rapport à l'échelle globale
    return {
        "x": (max(x_) - min(x_)) / fig_scale,
        "y": (max(y_) - min(y_)) / fig_scale,
        "z": (max(z_) - min(z_)) / fig_scale,
    }

def visualize_surface_current(points_data, triangles_data, radiation_intensity, title="Antennas Surface Current"):
    """
        Visualise la densité de courant surfacique en utilisant Plotly.

        Cette fonction permet de créer une visualisation 3D de la densité de courant surfacique sur un modèle d'antenne,
        en utilisant la bibliothèque Plotly pour une présentation interactive. La surface est coloriée en fonction de
        l'intensité de radiation, avec un colormap pour mieux représenter la distribution des intensités.

        Paramètres :
            * points_data : tuple ou n-d-array de forme (3, N), où N est le nombre de points.
              Il contient les coordonnées x, y et z des points 3D des sommets de l'antenne.
            * triangles_data : n-d-array de forme (3, M), où M est le nombre de triangles.
              Il contient les indices des sommets pour chaque triangle de la surface de l'antenne.
            * radiation_intensity : n-d-array, la densité de courant ou l'intensité de radiation associée
              à chaque triangle. Cette valeur sera utilisée pour colorier la surface.
            * title : str, titre de la visualisation (optionnel). Par défaut, il est défini sur "Antennas Surface Current".

        Retourne :
        fig : Objet Plotly, la figure 3D représentant la densité de courant surfacique colorée par l'intensité de radiation.
    """
    # Extraction des coordonnées des sommets (x, y, z) à partir de points_data
    x_, y_, z_ = points_data  # Coordonnées X, Y, Z des points

    # Création des simplices pour Plotly (les indices des sommets de chaque triangle)
    simplices = triangles_data[:3, :].T  # Transpose pour passer de [3, n_triangles] à [n_triangles, 3]

    # Calcul des rapports d'échelle pour ajuster les proportions de la visualisation
    aspect_ratios = compute_aspect_ratios(points_data)

    # Création de la figure avec trisurf de Plotly
    fig = ff.create_trisurf(
        x=x_,                            # Coordonnées X des sommets
        y=y_,                            # Coordonnées Y des sommets
        z=z_,                            # Coordonnées Z des sommets
        simplices=simplices,             # Indices des sommets de chaque triangle
        colormap="Rainbow",              # Colormap pour la coloration de la surface
        plot_edges=False,                # Ne pas afficher les bords des triangles
        color_func=radiation_intensity,  # Utilisation de la densité de courant normalisée pour colorer
        show_colorbar=True,              # Affichage de la barre de couleurs
        title=title,                     # Titre de la visualisation
        aspectratio=dict(aspect_ratios), # Ajustement des rapports d'échelle pour l'affichage 3D
    )

    # Retour de la figure Plotly créée
    return fig

def save_gain_power_data(save_folder_name, save_file_name, total_power, gain_linear, gain_logarithmic):
    """
    Sauvegarde les données de puissance totale et de gain dans un fichier .mat.

    Cette fonction permet de sauvegarder les résultats de puissance totale et les gains linéaire et logarithmique dans un fichier
    MATLAB (MAT) pour une utilisation ultérieure ou une analyse complémentaire.

    Paramètres :
        * save_folder_name : str, le nom du dossier où le fichier sera sauvegardé. Si le dossier n'existe pas, il sera créé.
        * save_file_name : str, le nom du fichier à sauvegarder (doit inclure l'extension .mat).
        * total_power : float ou n-d-array, la valeur de la puissance totale calculée.
        * gain_linear : float ou n-d-array, le gain linéaire calculé (exprimé en facteur multiplicatif).
        * gain_logarithmic : float ou n-d-array, le gain logarithmique calculé (exprimé en dB).

    Effet de bord :
        * Crée le dossier spécifié s'il n'existe pas.
        * Sauvegarde un fichier .mat contenant les données de puissance et de gain à l'emplacement spécifié.
    """
    # Construction du chemin complet pour le fichier à sauvegarder
    full_save_path = os.path.join(save_folder_name, save_file_name)

    # Vérification si le dossier existe et création si nécessaire
    if not os.path.exists(save_folder_name):  # Vérification et création du dossier si nécessaire
        os.makedirs(save_folder_name)
        print(f"Directory '{save_folder_name}' created.")

    # Préparation des données à sauvegarder dans un dictionnaire
    data_gain_power = {
        'totalPower': total_power,
        'gainLinear': gain_linear,
        'gainLogarithmic': gain_logarithmic
    }

    # Sauvegarde des données dans le fichier .mat
    savemat(full_save_path, data_gain_power)
    print(f"Data saved successfully to {full_save_path}")

def load_gain_power_data(filename_to_load):
    """
        Charge les données de puissance et de gain à partir d'un fichier .mat.

        Cette fonction charge un fichier MATLAB (MAT) contenant les résultats de puissance et de gain linéaire et logarithmique,
        en récupérant les données associées à ces paramètres. Elle gère également les erreurs possibles durant le processus de
        chargement des données.

        Paramètre :
        filename_to_load : str, le chemin complet du fichier .mat à charger.

        Retour :
            * total_power : float ou n-d-array, la puissance totale chargée depuis le fichier.
            * gain_linear : float ou n-d-array, le gain linéaire chargé depuis le fichier.
            * gain_logarithmic : float ou n-d-array, le gain logarithmique (en dB) chargé depuis le fichier.

        Exceptions :
            * FileNotFoundError : levée si le fichier spécifié n'existe pas.
            * KeyError : levée si l'une des clés attendues ('totalPower', 'gainLinear', 'gainLogarithmic') est manquante dans le fichier.
            * ValueError : levée si les données sont malformées ou corrompues.
            * Exception générale : levée pour toute autre erreur inattendue.
    """
    try:
        # Vérification si le fichier existe avant de le charger
        if not os.path.isfile(filename_to_load):
            raise FileNotFoundError(f"File '{filename_to_load}' does not exist.")

        # Chargement des données depuis le fichier .mat
        data = loadmat(filename_to_load)

        # Extraction des données : puissance totale, gain linéaire et gain logarithmique
        total_power = data['totalPower'].squeeze()
        gain_linear = data['gainLinear'].squeeze()
        gain_logarithmic = data['gainLogarithmic'].squeeze()

        print(f"Data loaded from {filename_to_load}")

        # Retour des données extraites
        return total_power, gain_linear, gain_logarithmic
    except FileNotFoundError as e:
        # Gestion des erreurs si le fichier n'est pas trouvé
        print(f"Error: {e}")

    except KeyError as e:
        # Gestion des erreurs si une clé est manquante dans le fichier .mat
        print(f"Key Error: {e}")

    except ValueError as e:
        # Gestion des erreurs si les données sont malformées
        print(f"Value Error (likely malformed data): {e}")

    except Exception as e:
        # Gestion des erreurs inattendues
        print(f"An unexpected error occurred: {e}")

def radiation_intensity_distribution_over_sphere_surface(filename_mesh2_to_load, filename_current_to_load, filename_sphere_to_load, scattering = False, radiation = False):
    """
        Calcule et visualise la distribution d'intensité de radiation et de gain sur la surface d'une sphère entourant une antenne.

        Cette fonction charge les données nécessaires (maillage, courants, sphère), effectue des calculs de champ électromagnétique
        pour chaque triangle de la sphère, et calcule des métriques telles que la puissance totale, le gain linéaire et logarithmique.
        Les résultats sont ensuite sauvegardés et visualisés.

        Paramètres :
            * filename_mesh2_to_load : str
                Chemin du fichier contenant les données de maillage de l'antenne (triangles, points, etc.).
            * filename_current_to_load : str
                Chemin du fichier contenant les données de courant sur l'antenne.
            * filename_sphere_to_load : str
                Chemin du fichier contenant les données de la sphère (coordonnées et triangles).

        Retour :
        Aucun retour explicite. Les résultats sont sauvegardés dans un fichier et visualisés.

        Étapes principales :
            1. Chargement des données d'entrée (maillage, courants, sphère).
            2. Calcul des champs électromagnétiques sur les triangles de la sphère.
            3. Calcul des métriques de radiation : puissance totale, gain linéaire et logarithmique.
            4. Sauvegarde des résultats calculés.
            5. Visualisation des résultats sous forme de distribution de gain sur la sphère.
    """

    #  Extraction du nom de base du fichier sans l'extension et modification du nom
    base_name = os.path.splitext(os.path.basename(filename_mesh2_to_load))[0]
    base_name = base_name.replace('_mesh2', '')

    # Chargement des fichiers contenant les données de maillage, courants et sphère
    data_sphere = loadmat(filename_sphere_to_load)

    _, triangles, edges, *_ = DataManager_rwg2.load_data(filename_mesh2_to_load)

    if scattering :
        frequency, omega, _, _, light_speed_c, eta, _, _, _, current = DataManager_rwg4.load_data(filename_current_to_load, scattering=scattering)
    elif radiation:
        frequency, omega, _, _, light_speed_c, eta, _, current, _, gap_current, *_ = DataManager_rwg4.load_data(filename_current_to_load, radiation=radiation)

    # Chargement des données de la sphère
    sphere_points = data_sphere['p'] * 100    # Les coordonnées de la sphère sont multipliées par 100 (rayon de 100 m).
    sphere_triangles = data_sphere['t'] - 1   # Conversion des indices MATLAB (1-based) en indices Python (0-based).

    # Calcul du nombre d'onde k et de sa composante complexe
    k = omega / light_speed_c    # Nombre d'onde (en rad/m).
    complex_k = 1j * k           # Composante complexe.

    # Affichage de la fréquence et de la longueur d'onde
    print('')
    print(f"Frequency = {frequency} Hz")
    print(f"Longueur d'onde lambda = {light_speed_c / frequency} m")

    # Calcul des dipôles et des moments dipolaires (en complexe)
    dipole_center, dipole_moment = compute_dipole_center_moment(triangles, edges, current)

    # Initialisation pour le calcul des champs et de la puissance totale
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

    # Boucle sur chaque triangle de la sphère pour calculer les champs et l'énergie
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

        # Contribution de chaque triangle à la puissance totale
        total_power += w[triangle_in_sphere] * sphere_triangle_area[triangle_in_sphere]

    print('')

    # Calcul of the antenna directivity : is a measure of how focused an antenna's radiation pattern is in a specific direction compared to an idealized isotropic antenna that radiates equally in all directions.
    # It quantifies the antenna's ability to concentrate radiated power in a particular direction.
    # Here we call it gain
    # Calcul du gain (linéaire et logarithmique)
    gain_linear = 4 * np.pi * u / total_power
    gain_logarithmic = 10 * np.log10(gain_linear)
    gain_linear_max = 4 * np.pi * np.max(u) / total_power
    gain_logarithmic_max = 10 * np.log10(gain_linear_max)

    print(f"Total Power : {total_power : 4f}")
    print(f"Gain Linear : {gain_linear_max : 4f}")
    print(f"Gain Logarithmic : {gain_logarithmic_max : 4f} dB")
    if gap_current:
        radiation_resistance = 2 * total_power / abs(gap_current)**2
        print(f"Radiation Resistance : {radiation_resistance : 4f} Ohms")

    # Sauvegarde des résultats calculés
    save_gain_power_folder_name = 'data/antennas_gain_power/'
    save_gain_power_file_name = base_name + '_gain_power.mat'
    save_gain_power_data(save_gain_power_folder_name, save_gain_power_file_name, total_power, gain_linear_max, gain_logarithmic_max)

    # Visualisation des résultats
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
    seuil_db = max(u_points_db) - 20
    u_points_db = np.maximum(u_points_db[:sphere_total_of_points] - seuil_db, 0.01)
    sphere_points_update = u_points_db * sphere_points / 1000

    # Affichage de l'intensité de radiation
    # fig1 = visualize_surface_current(sphere_points, sphere_triangles, u_normalize, plot_name_intensity)
    # fig1.show()

    # Visualisation du gain logarithmique
    fig2 = visualize_surface_current(sphere_points_update, sphere_triangles, gain_logarithmic, plot_name_gain)
    fig2.show()