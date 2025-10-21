import os
import time

import numpy as np
from scipy.io import loadmat, savemat


def impedance_matrice_z(EdgeLength, EdgesTotal, TrianglesTotal, TrianglePlus, TriangleMinus, Center, Center_, RHO_Plus, RHO_Minus, RHO__Plus, RHO__Minus, parameter_k, factor_a, factor_fi):
    total_number_of_edges = EdgesTotal
    total_of_triangles = TrianglesTotal
    triangles_plus = TrianglePlus
    triangles_minus = TriangleMinus
    triangles_center = Center
    edges_length = EdgeLength
    barycentric_triangle_center = Center_
    vecteur_rho_plus = RHO_Plus
    vecteur_rho_minus = RHO_Minus
    vecteur_rho_barycentric_plus = RHO__Plus
    vecteur_rho_barycentric_minus = RHO__Minus

    vecteur_rho_plus_tiled = np.tile(vecteur_rho_plus[:, None, :], (1, 9, 1))    # Dimension [3, 9, total_number_of_edges]
    vecteur_rho_minus_tiled = np.tile(vecteur_rho_minus[:, None, :], (1, 9, 1))  # Dimension [3, 9, total_number_of_edges]

    # Initialisation de la matrice d'impédance Z
    matrice_z = np.zeros((total_number_of_edges, total_number_of_edges), dtype=complex)   # Dimension [total_number_of_edges, total_number_of_edges]

    # Boucle sur les triangles pour calculer les interactions
    for triangle in range(total_of_triangles):
        # Identification des contributions des triangles plus et moins
        positions_plus = triangles_plus == triangle
        positions_minus = triangles_minus == triangle

        # Calcul de la fonction g_mn(r')
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
            z1_reshaped = z1.squeeze(axis=1).sum(axis=0)  # Suppression de l'axe inutile et réduction pour correspondre à (176,)
            matrice_z[:, the_position] += edges_length[the_position] * (z1_reshaped + sign * impedance_coupling_zf.squeeze())

        # Calcul des contributions pour les triangles plus et moins
        for position in np.where(positions_plus)[0]:
            update(position, vecteur_rho_barycentric_plus, +1)
        for position in np.where(positions_minus)[0]:
            update(position, vecteur_rho_barycentric_minus, -1)

    return matrice_z

filename_matlab_plate_to_load = 'data/antennas_mesh2/plate_from_matlab_mesh2.mat'
data = loadmat(filename_matlab_plate_to_load)
p = data['p'].squeeze()
t = data['t'].squeeze()
TrianglesTotal = data['TrianglesTotal'].squeeze()
EdgesTotal = data['EdgesTotal'].squeeze()
Edge_ = data['Edge_'].squeeze()
TrianglePlus  = data['TrianglePlus'].squeeze() - 1
TriangleMinus = data['TriangleMinus'].squeeze() - 1
EdgeLength = data['EdgeLength'].squeeze()
EdgeIndicator = data['EdgeIndicator'].squeeze()
Area = data['Area'].squeeze()
RHO_Plus = data['RHO_Plus'].squeeze()
RHO_Minus = data['RHO_Minus'].squeeze()
RHO__Plus = data['RHO__Plus'].squeeze()
RHO__Minus = data['RHO__Minus'].squeeze()
Center = data['Center'].squeeze()
Center_ = data['Center_'].squeeze()

# Définition des paramètres électromagnétiques
frequency = 3e8 # La fréquence du champ électromagnétique (300 MHz)
epsilon   = 8.854e-12          # La permittivité du vide (F/m)
mu        = 1.257e-6           # La perméabilité magnétique du vide (H/m)

# Calcul des constantes électromagnétiques
light_speed_c = 1 / np.sqrt(epsilon * mu)  # Vitesse de la lumière dans le vide
eta = np.sqrt(mu / epsilon)               # Impédance caractéristique de l'espace libre
omega = 2 * np.pi * frequency             # Pulsation angulaire
k = omega / light_speed_c                 # Nombre d'onde
parameter_k = 1j * k                      # Nombre d'onde complexe

# Définition des facteurs pour accélérer les calculs
constant_1   = mu / (4 * np.pi)
constant_2   = 1 / (1j * 4 * np.pi * omega * epsilon)
factor      = 1 / 9

factor_a     = factor * (1j * omega * EdgeLength / 4) * constant_1
factor_fi    = factor * EdgeLength * constant_2

# Chronométrage
start_time = time.time()

# Calcul de la matrice d'impédance
matrice_z = impedance_matrice_z(EdgeLength, EdgesTotal, TrianglesTotal, TrianglePlus, TriangleMinus, Center, Center_, RHO_Plus, RHO_Minus, RHO__Plus, RHO__Minus, parameter_k, factor_a, factor_fi)

# Fin du chronométrage
elapsed_time = time.time() - start_time
print(f"Temps écoulé pour le calcul de la matrice Z : {elapsed_time:.6f} secondes")
data = {
    'f': frequency,
    'omega': omega,
    'mu_': mu,
    'epsilon_': epsilon,
    'c_': light_speed_c,
    'eta_': eta,
    'Z': matrice_z,
}
# Sauvegarder toutes ces données dans un fichier .mat
save_file_name = 'plate_impedance_with_matlab_file.mat'
save_folder_name = 'data/antennas_impedance/'
full_save_path = os.path.join(save_folder_name, save_file_name)
if not os.path.exists(save_folder_name):           # Vérification et création du dossier si nécessaire
    os.makedirs(save_folder_name)
    print(f"Directory '{save_folder_name}' created.")
savemat(full_save_path, data)
print(f"Data saved successfully to {full_save_path}")
