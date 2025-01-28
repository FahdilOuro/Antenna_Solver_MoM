"""
Ce code simule les champs électriques et magnétiques générés par une surface métallique à partir de courants surfaciques en un point de l'espace.
Il utilise des dipôles équivalents pour représenter ces courants.
Calcule les contributions des champs électrique et magnétique pour chaque dipôle au point d'observation
"""
import os

import numpy as np

from rwg.rwg2 import DataManager_rwg2
from rwg.rwg4 import DataManager_rwg4
from utils.dipole_parameters import compute_dipole_center_moment, compute_e_h_field

def calculate_electric_magnetic_field_at_point(filename_mesh2_to_load, filename_current_to_load, observation_point, scattering = False, radiation = False):
    """
        Calcule et affiche les champs électriques, magnétiques, le vecteur de Poynting, l'énergie et la section efficace radar (RCS)
        à un point d'observation spécifié, à partir des données de maillage et des courants chargés depuis des fichiers .mat.

        Paramètres :
            * filename_mesh2_to_load : str, chemin du fichier de maillage (MAT).
            * filename_current_to_load : str, chemin du fichier contenant les courants et autres données associées (MAT).
            * observation_point : tuple ou n-d-array, coordonnées du point d'observation où les champs seront calculés.

        Fonctionnement :
            1. Extraction du nom de base du fichier de maillage sans extension et modification du nom.
            2. Chargement des données de maillage et des courants depuis les fichiers .mat.
            3. Calcul du nombre d'onde et de sa composante complexe.
            4. Calcul des caractéristiques des dipôles sur le maillage (centres et moments dipolaires).
            5. Calcul des champs électriques et magnétiques totaux à partir du point d'observation et des moments dipolaires.
            6. Affichage des résultats pour les champs, le vecteur de Poynting, l'énergie, l'énergie par unité d'angle solide et la RCS.

        Retour :
        Aucune valeur retournée, mais les résultats sont affichés à la console pour analyse.

        Exemple :
        Cette fonction permet de calculer et d'afficher les différents paramètres liés aux champs électromagnétiques
        à un point d'observation donné, en utilisant un maillage 3D et des courants préalablement simulés.

        Notes :
        La RCS (Radar Cross Section) est une mesure de la capacité d'un objet à réfléchir des ondes électromagnétiques,
          souvent utilisée pour caractériser la taille apparente d'un objet en radar.
    """

    # 1. Extraction du nom de base du fichier sans l'extension et modification du nom
    base_name = os.path.splitext(os.path.basename(filename_mesh2_to_load))[0]
    base_name = base_name.replace('_mesh2', '')

    # 2. Chargement des données de maillage et des courants à partir des fichiers MAT
    _, triangles, edges, *_ = DataManager_rwg2.load_data(filename_mesh2_to_load)

    if scattering :
        frequency, omega, _, _, light_speed_c, eta, _, _, _, current = DataManager_rwg4.load_data(filename_current_to_load, scattering=scattering)
    elif radiation:
        frequency, omega, _, _, light_speed_c, eta, _, current, *_ = DataManager_rwg4.load_data(filename_current_to_load, radiation=radiation)

    # 3. Calcul du nombre d'onde k et de sa composante complexe
    k = omega / light_speed_c    # Nombre d'onde (en rad/m)
    complex_k = 1j * k           # Composante complexe du nombre d'onde

    # 4. Affichage des informations de base
    print('')
    print(f"Frequency = {frequency} Hz")
    print(f"Longueur d'onde lambda = {light_speed_c / frequency} m")

    # 5. Calcul des dipôles et des moments dipolaires
    dipole_center, dipole_moment = compute_dipole_center_moment(triangles, edges, current)

    # 6. Calcul des champs électriques et magnétiques totaux à partir du point d'observation
    e_field_total, h_field_total, poynting_vector, w, u, norm_observation_point = compute_e_h_field(observation_point, eta, complex_k, dipole_moment, dipole_center)

    # 7. Affichage du point d'observation
    print(f"Le point d'observation est : {observation_point}")

    print('')

    # 8. Affichage des résultats du champ électrique total
    print(f"e_field_total of {base_name} at the observation point {observation_point} is :")
    print(f"{e_field_total[0].real : .7f} {"+" if e_field_total[0].imag >= 0 else "-"}{abs(e_field_total[0].imag) : .7f}i V/m")
    print(f"{e_field_total[1].real : .7f} {"+" if e_field_total[1].imag >= 0 else "-"}{abs(e_field_total[1].imag) : .7f}i V/m")
    print(f"{e_field_total[2].real : .7f} {"+" if e_field_total[1].imag >= 0 else "-"}{abs(e_field_total[2].imag) : .7f}i V/m")

    print('')

    # 9. Affichage des résultats du champ magnétique total
    print(f"h_field_total of {base_name} at the observation point {observation_point} is :")
    print(f"{h_field_total[0].real : .7f} {"+" if h_field_total[0].imag >= 0 else "-"}{abs(h_field_total[0].imag) : .7f}i A/m")
    print(f"{h_field_total[1].real : .7f} {"+" if h_field_total[1].imag >= 0 else "-"}{abs(h_field_total[1].imag) : .7f}i A/m")
    print(f"{h_field_total[2].real : .7f} {"+" if h_field_total[2].imag >= 0 else "-"}{abs(h_field_total[2].imag) : .7f}i A/m")

    print('')
    print("Poynting vector is equal to : ")
    print(f"{poynting_vector[0] : 8f} W/m^2")
    print(f"{poynting_vector[1] : 8f} W/m^2")
    print(f"{poynting_vector[2] : 8f} W/m^2")

    print('')
    print(f"w = {w} W/m^2")

    print('')
    print(f"u = {u} W/unit solid angle")

    # La section efficace radar (RCS) est une mesure de la capacité d'un objet à réfléchir ou diffuser les ondes électromagnétiques.
    # C'est une quantité utilisée principalement dans les applications de radar pour décrire l'échelle de l'objet en termes de diffusion radar.
    # 10. Calcul de la section efficace radar (RCS)
    e_field_dot_conj = np.sum(np.real(e_field_total * np.conj(e_field_total)))
    rcs = 4 * np.pi * (norm_observation_point ** 2) * e_field_dot_conj             # for scattering

    print('')
    print(f"RCS = {rcs}")