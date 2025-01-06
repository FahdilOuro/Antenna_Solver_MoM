"""
    Ce module implémente le calcul de la matrice d'impédance Z pour un système d'arêtes et de triangles,
    basé sur la méthode des moments (MoM) en électromagnétisme.

    Fonctionnalités principales :
        1. Construction de la matrice Z, représentant les interactions d'impédance entre les arêtes.
        2. Utilisation des centres barycentriques, vecteurs rho, et contributions des triangles adjacents.
        3. Calcul basé sur une fonction g_mn(r') et des termes de couplage scalaire.

    Entrées principales :
        * edges_data : Données sur les arêtes, incluant leur nombre total et leurs longueurs.
        * triangles_data : Données sur les triangles, incluant leurs centres et les indices des triangles adjacents aux arêtes.
        * barycentric_triangles_data : Centres barycentriques des triangles.
        * vecteurs_rho_data : Vecteurs rho associés aux arêtes et triangles barycentriques.
        * parameter_k : Nombre d'onde complexe du milieu.
        * factor_a : Facteur de pondération pour les contributions vectorielles A_{mn}.
        * factor_fi : Facteur de pondération pour les contributions scalaires Phi_{mn}.

    Sortie :
    matrice_z : Matrice d'impédance Z (complexe), de dimension [nombre total d'arêtes, nombre total d'arêtes].
"""
import numpy as np

def impedance_matrice_z(edges_data, triangles_data, barycentric_triangles_data, vecteurs_rho_data, parameter_k, factor_a, factor_fi):
    """
    Calcule la matrice d'impédance Z pour les interactions entre les arêtes du maillage.

    Paramètres :
        * edges_data : Objet contenant les données sur les arêtes (longueurs, nombre total, etc.).
        * triangles_data : Objet contenant les données sur les triangles (centres, triangles adjacents, etc.).
        * barycentric_triangles_data : Centres barycentriques des triangles.
        * vecteurs_rho_data : Données sur les vecteurs rho.
        * parameter_k : Nombre d'onde complexe du milieu.
        * factor_a : Facteur pour les contributions vectorielles.
        * factor_fi : Facteur pour les contributions scalaires.

    Retourne :
    matrice_z : Matrice d'impédance Z complexe.
    """
    # Initialisation des variables globales et des données nécessaires
    total_number_of_edges = edges_data.total_number_of_edges
    total_of_triangles = triangles_data.total_of_triangles
    triangles_plus = triangles_data.triangles_plus
    triangles_minus = triangles_data.triangles_minus
    triangles_center = triangles_data.triangles_center
    edges_length = edges_data.edges_length
    barycentric_triangle_center = barycentric_triangles_data.barycentric_triangle_center
    vecteur_rho_plus = vecteurs_rho_data.vecteur_rho_plus
    vecteur_rho_minus = vecteurs_rho_data.vecteur_rho_minus
    vecteur_rho_barycentric_plus = vecteurs_rho_data.vecteur_rho_barycentric_plus
    vecteur_rho_barycentric_minus = vecteurs_rho_data.vecteur_rho_barycentric_minus

    # Préparation des vecteurs rho pour les calculs
    vecteur_rho_plus_tiled = np.tile(vecteur_rho_plus[:, None, :], (1, 9, 1))    # Dimension [3, 9, total_number_of_edges]
    vecteur_rho_minus_tiled = np.tile(vecteur_rho_minus[:, None, :], (1, 9, 1))  # Dimension [3, 9, total_number_of_edges]

    # Initialisation de la matrice d'impédance Z
    matrice_z = np.zeros((total_number_of_edges, total_number_of_edges), dtype=complex)   # Dimension [total_number_of_edges, total_number_of_edges]

    # Boucle sur les triangles pour calculer les interactions
    for triangle in range(total_of_triangles):
        # Identification des contributions des triangles plus et moins
        positions_plus = triangles_plus == triangle
        positions_minus = triangles_minus == triangle

        # Calcul de la fonction g_mn(r'); l'indice m correspond au triangle que le programme traite et les indices n correspondent à tous les triangles de 0 à l'indice total_of_triangles - 1
        distances = barycentric_triangle_center - triangles_center[:, triangle][:, None, None]   # Dimension [3, 9, total_of_triangles]
        norm_of_distances = np.sqrt(np.sum(distances**2, axis=0, keepdims=True))                 # Dimension [1, 9, total_of_triangles]
        g_function = np.exp(-parameter_k * norm_of_distances) / norm_of_distances                # Dimension [1, 9, total_of_triangles]

        g_function_plus = g_function[:, :, triangles_plus]                                       # Dimension [1, 9, total_number_of_edges]
        g_function_minus = g_function[:, :, triangles_minus]                                     # Dimension [1, 9, total_number_of_edges]

        # Contribution scalaire fi pour Phi_mn
        fi = np.sum(g_function_plus, axis=1, keepdims=True) - np.sum(g_function_minus, axis=1, keepdims=True)      # Dimension [1, 1, total_number_of_edges]

        impedance_coupling_zf = factor_fi.reshape(-1, 1) * fi.squeeze().reshape(-1, 1)                             # Dimension [total_number_of_edges, 1]

        # Fonction pour mettre à jour Z en fonction de rho et A_mn
        def update(the_position, vecteur_rho_barycentric_p_m, sign):
            vecteur_rho_barycentric = np.tile(vecteur_rho_barycentric_p_m[:, :, the_position][:, :, None], (1, 1, total_number_of_edges))   # Dimension [3, 9, total_number_of_edges]
            a_contribution = (np.sum(g_function_plus * np.sum(vecteur_rho_barycentric * vecteur_rho_plus_tiled, axis=0), axis=0)
                              +
                              np.sum(g_function_minus * np.sum(vecteur_rho_barycentric * vecteur_rho_minus_tiled, axis=0), axis=0))     # Dimension [9, total_number_of_edges]
            z1 = factor_a * a_contribution[:, None]
            z1_reshaped = z1.squeeze(axis=1).sum(axis=0)  # Suppression de l'axe inutile et réduction pour correspondre à (total_number_of_edges)
            matrice_z[:, the_position] += edges_length[the_position] * (z1_reshaped + sign * impedance_coupling_zf.squeeze())

        # Calcul des contributions pour les triangles plus et moins
        for position in np.where(positions_plus)[0]:
            update(position, vecteur_rho_barycentric_plus, +1)
        for position in np.where(positions_minus)[0]:
            update(position, vecteur_rho_barycentric_minus, -1)

    return matrice_z