"""
    Ce module implémente le calcul des champs électromagnétiques et des propriétés associées
    à partir de données de triangles, arêtes, et courants pour une antenne.

    Fonctionnalités principales :
        1. Calcul des centres et moments dipolaires associés aux arêtes d'un maillage triangulaire.
        2. Détermination des champs électriques (E) et magnétiques (H) radiés et dispersés en un point d'observation.
        3. Calcul de la densité de puissance (vecteur de Poynting), de la densité de radiation et de l'intensité de radiation.

    Entrées principales :
        * triangles_data : Contient les données des triangles, y compris leurs centres et indices liés aux arêtes.
        * edges_data : Contient les longueurs et le nombre total d'arêtes.
        * current_data : Tableau des courants électriques associés aux arêtes du maillage.
        * observation_point : Point dans l'espace, où les champs seront calculés (vecteur 3D).
        * eta : Impédance caractéristique du milieu.
        * complex_k : Nombre d'onde complexe du milieu.
"""
import numpy as np

from utils.point_field import radiated_scattered_field_at_a_point


def compute_dipole_center_moment(triangles_data, edges_data, current_data):
    """
        Calcule les centres et moments dipolaires associés aux arêtes d'un maillage.

        Paramètres :
            * triangles_data : Objet contenant les données des triangles, y compris leurs centres.
            * edges_data : Objet contenant les informations sur les arêtes (nombre total, longueurs, etc.).
            * current_data : Tableau contenant les courants électriques sur chaque arête.

        Retourne :
         * dipole_center : Coordonnées des centres des dipôles associés aux arêtes (matrice 3xN).
         * dipole_moment : Moments dipolaires complexes pour chaque arête (matrice 3xN).
    """
    # Initialisation des centres et moments dipolaires
    dipole_center = np.zeros((3, edges_data.total_number_of_edges))
    dipole_moment = np.zeros((3, edges_data.total_number_of_edges), dtype=complex)

    # Calcul des centres et moments pour chaque arête
    for edge in range(edges_data.total_number_of_edges):
        # Coordonnées des centres des triangles associés à l'arête
        point_plus_center = triangles_data.triangles_center[:, triangles_data.triangles_plus[edge]]
        point_minus_center = triangles_data.triangles_center[:, triangles_data.triangles_minus[edge]]

        # Coordonnées des centres des triangles associés à l'arête
        dipole_center[:, edge] = 0.5 * (point_plus_center + point_minus_center)

        # Moment dipolaire : longueur de l'arête × courant × différence des centres des triangles
        dipole_moment[:, edge] = edges_data.edges_length[edge] * current_data[edge] * (-point_plus_center + point_minus_center)

    return dipole_center, dipole_moment

def compute_e_h_field(observation_point, eta, complex_k, dipole_moment, dipole_center):
    """
        Calcule les champs électriques et magnétiques radiés et dispersés au point d'observation,
        ainsi que des quantités associées comme le vecteur de Poynting et l'intensité de radiation.

        Paramètres :
            * observation_point : Coordonnées du point d'observation (vecteur 3D).
            * eta : Impédance caractéristique du milieu.
            * complex_k : Nombre d'onde complexe.
            * dipole_moment : Moments dipolaires associés aux arêtes (matrice 3xN).
            * dipole_center : Centres des dipôles associés aux arêtes (matrice 3xN).

        Retourne :
         * e_field_total : Champ électrique total au point d'observation (vecteur 3D).
         * h_field_total : Champ magnétique total au point d'observation (vecteur 3D).
         * poynting_vector : Vecteur de Poynting représentant la densité de puissance transportée (vecteur 3D).
         * w : Densité de radiation (puissance par unité de surface).
         * u : Intensité de radiation (puissance par unité d'angle solide).
         * norm_observation_point : Distance entre le point d'observation et l'origine.
    """
    # Calcul des champs E et H au point d'observation à partir des moments dipolaires
    e_field, h_field = radiated_scattered_field_at_a_point(observation_point, eta, complex_k, dipole_moment, dipole_center)

    # Sommation des contributions des dipôles pour obtenir les champs totaux
    e_field_total = np.sum(e_field, axis=1)
    h_field_total = np.sum(h_field, axis=1)

    # Calcul du vecteur de Poynting (densité de puissance transportée par les ondes EM)
    poynting_vector = np.real(0.5 * (np.cross(e_field_total.flatten(), np.conj(h_field_total).flatten())))

    # Norme de la position du point d'observation
    norm_observation_point = np.linalg.norm(observation_point)

    # Densité de radiation : norme du vecteur de Poynting
    w = np.linalg.norm(poynting_vector)

    # Intensité de radiation : densité de radiation pondérée par le carré de la distance
    u = (norm_observation_point ** 2) * w

    return e_field_total, h_field_total, poynting_vector, w, u, norm_observation_point