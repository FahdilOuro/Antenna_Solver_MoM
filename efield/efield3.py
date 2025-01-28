import os

import numpy as np
from matplotlib import pyplot as plt

from efield.efield2 import load_gain_power_data
from rwg.rwg2 import DataManager_rwg2
from rwg.rwg4 import DataManager_rwg4
from utils.dipole_parameters import compute_dipole_center_moment, compute_e_h_field


def compute_observation_points(r, angle, phi):
    """
        Calcule les points d'observation sur une sphère de rayon donné.

        Ce calcul est basé sur les coordonnées sphériques (r, angle, phi) pour obtenir les coordonnées cartésiennes (x, y, z).

        Paramètres :
            * r : Rayon de la sphère (float).
            * angle : Liste des angles d'élévation (theta) en radians (1D array).
            * phi : Angle d'azimut constant (float).

        Retourne :
        np.n-d-array : Tableau Nx3 contenant les coordonnées cartésiennes des points d'observation.
    """
    x = r * np.sin(angle) * np.cos(phi)
    y = r * np.sin(angle) * np.sin(phi)
    z = r * np.cos(angle)
    return np.vstack((x, y, z)).T  # Retourne une liste (Nx3) des coordonnées


# Points pour deux valeurs d'azimut spécifiques (phi = 0° et phi = 90°)
phi_0 = 0                # Azimut de 0 radians
phi_90 = 0.5 * np.pi     # Azimut de 90°


def compute_polar(observation_point_list_phi, numbers_of_points, eta, complex_k, dipole_moment, dipole_center, total_power):
    """
    Calcule la répartition de l'intensité du champ (en dB) sur un plan polaire donné.

    Paramètres :
        * observation_point_list_phi : Liste des points d'observation (Nx3 n-d-array).
        * numbers_of_points : Nombre total de points d'observation (int).
        * eta : Impédance du milieu (float).
        * complex_k : Nombre d'onde complexe (1j * k) (complex).
        * dipole_moment : Moments dipolaires (complex n-d-array).
        * dipole_center : Centres des dipôles (n-d-array).
        * total_power : Puissance totale rayonnée par l'antenne (float).

    Retourne :
    np.n-d-array : Diagramme polaire de l'intensité normalisée en dB (1D array).
    """
    e_field_total = np.zeros((3, numbers_of_points), dtype=complex)  # Champ électrique
    h_field_total = np.zeros((3, numbers_of_points), dtype=complex)  # Champ magnétique
    poynting_vector = np.zeros((3, numbers_of_points))  # Vecteur de Poynting
    w = np.zeros(numbers_of_points)  # Densité d'énergie
    u = np.zeros(numbers_of_points)  # Densité de puissance

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

    polar = 10 * np.log10(4 * np.pi * u / total_power)  # Conversion en dB
    return polar

def antenna_directivity_pattern(filename_mesh2_to_load, filename_current_to_load, filename_gain_power_to_load, scattering = False, radiation = False):
    """
        Génère le diagramme de directivité d'une antenne dans les plans Phi = 0° et Phi = 90°.

        Cette fonction charge les données nécessaires (maillage, courants, puissance rayonnée),
        calcule les diagrammes polaires d'intensité, et affiche les résultats.

        Paramètres :
            * filename_mesh2_to_load : Chemin du fichier contenant le maillage de l'antenne.
            * filename_current_to_load : Chemin du fichier contenant les courants sur l'antenne.
            * filename_gain_power_to_load : Chemin du fichier contenant les données de gain et de puissance.
    """
    # Extraction et modification du nom de base du fichier
    base_name = os.path.splitext(os.path.basename(filename_mesh2_to_load))[0]
    base_name = base_name.replace('_mesh2', '')

    # Chargement des données nécessaires
    _, triangles, edges, *_ = DataManager_rwg2.load_data(filename_mesh2_to_load)

    if scattering :
        frequency, omega, _, _, light_speed_c, eta, _, _, _, current = DataManager_rwg4.load_data(filename_current_to_load, scattering=scattering)
    elif radiation:
        frequency, omega, _, _, light_speed_c, eta, _, current, *_ = DataManager_rwg4.load_data(filename_current_to_load, radiation=radiation)

    total_power, *_ = load_gain_power_data(filename_gain_power_to_load)

    # Calcul des paramètres fondamentaux
    k = omega / light_speed_c    # Nombre d'onde (en rad/m)
    complex_k = 1j * k           # Composante complexe du nombre d'onde
    dipole_center, dipole_moment = compute_dipole_center_moment(triangles, edges, current)  # Moments dipolaires

    numbers_of_points = 100    # Nombre de points sur chaque plan
    radius = 100               # Rayon de la sphère d'observation

    # Calcul des points d'observation pour Phi = 0° et Phi = 90°
    theta = np.linspace(0, 2 * np.pi, numbers_of_points)    # Angles theta (0 à 360°)

    observation_point_list_phi0 = compute_observation_points(radius, theta, phi_0)
    observation_point_list_phi90 = compute_observation_points(radius, theta, phi_90)

    # Calcul des diagrammes polaires d'intensité
    polar_0 = compute_polar(observation_point_list_phi0, numbers_of_points, eta, complex_k, dipole_moment, dipole_center, total_power)
    polar_90 = compute_polar(observation_point_list_phi90, numbers_of_points, eta, complex_k, dipole_moment, dipole_center, total_power)

    # Visualisation du diagramme polaire
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta, polar_0, color='red', label='Phi = 0°')
    ax.plot(theta, polar_90, color='blue', label='Phi = 90°')

    # Configuration des axes et légendes
    ax.set_theta_zero_location("N")    # 0° au nord
    ax.set_theta_direction(-1)         # Sens horaire pour les angles
    ax.set_rlabel_position(-22.5)      # Position des étiquettes radiales
    ax.text(0, max(polar_0) + 5, "z", ha='center', va='bottom', fontsize=10, color='red')
    ax.legend()
    ax.grid(True)
    ax.set_title(base_name + " E-field pattern in Phi = 0° and 90° plane", va='bottom')
    plt.show()