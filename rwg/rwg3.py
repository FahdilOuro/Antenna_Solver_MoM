import os
import time
from scipy.io import savemat, loadmat
from utils.impmet import *

def calculate_z_matrice(triangles, edges, barycentric_triangles, vecteurs_rho, frequency):
    """
        Calcule la matrice d'impédance électromagnétique pour un système donné.

        Paramètres :
            * triangles : Objet représentant les triangles du maillage.
            * edges : Objet représentant les arêtes du maillage.
            * barycentric_triangles : Objet contenant les données barycentriques des triangles.
            * vecteurs_rho : Objet contenant les vecteurs Rho associés aux triangles.
            * frequency : Fréquence du signal électromagnétique (en Hz).

        Retourne :
            * omega : Pulsation angulaire (rad/s).
            * mu : Perméabilité magnétique du vide (H/m).
            * epsilon : Permittivité du vide (F/m).
            * light_speed_c : Vitesse de la lumière dans le vide (m/s).
            * eta : Impédance caractéristique de l'espace libre (Ω).
            * matrice_z : Matrice d'impédance calculée.
    """

    # Définition des paramètres électromagnétiques
    epsilon = 8.854e-12  # Permittivité du vide (F/m)
    mu = 1.257e-6        # Perméabilité magnétique du vide (H/m)

    # Calcul des constantes électromagnétiques
    light_speed_c = 1 / np.sqrt(epsilon * mu)  # Vitesse de la lumière dans le vide (m/s)
    eta = np.sqrt(mu / epsilon)                # Impédance caractéristique de l'espace libre (Ω)
    omega = 2 * np.pi * frequency              # Pulsation angulaire (rad/s)
    k = omega / light_speed_c                  # Nombre d'onde (rad/m)
    complexe_k = 1j * k                        # Nombre d'onde complexe pour les calculs

    # Définition des facteurs pour optimiser les calculs
    constant_1   = mu / (4 * np.pi)                        # Constante pour le calcul des champs magnétiques
    constant_2   = 1 / (1j * 4 * np.pi * omega * epsilon)  # Constante pour le calcul des champs électriques
    factor      = 1 / 9                                    # Facteur de pondération pour les calculs intégrés

    # Facteurs spécifiques liés aux arêtes
    factor_a     = factor * (1j * omega * edges.edges_length / 4) * constant_1  # Facteur pour le calcul des potentiels vectoriels
    factor_fi    = factor * edges.edges_length * constant_2                     # Facteur pour le calcul des potentiels scalaires

    # Calcul de la matrice d'impédance
    matrice_z = impedance_matrice_z(edges, triangles, barycentric_triangles, vecteurs_rho, complexe_k, factor_a, factor_fi)

    return omega, mu, epsilon, light_speed_c, eta, matrice_z

def compute_lumped_impedance(load_values, omega):
    """
    Calcule Z = jωL + 1/(jωC) + R à partir de load_values (3, N),
    avec traitement spécial :
    - C = None ⇒ remplacé par très grande valeur (C = 1e64)
    - C = 0    ⇒ lève une erreur explicite
    - L, R : None ou 0 ⇒ traités comme 0 (aucune inductance ou résistance)
    """
    N = load_values.shape[1]

    # Initialisation
    L_vals = np.zeros(N, dtype=np.complex128)
    C_vals = np.empty(N, dtype=np.complex128)
    R_vals = np.zeros(N, dtype=np.complex128)

    for i in range(N):
        L = load_values[0, i]
        C = load_values[1, i]
        R = load_values[2, i]

        # L
        if L not in (None, 0):
            L_vals[i] = L

        # C
        if C is None:
            C_vals[i] = 1e64  # Simule C très grand ⇒ Z_C ≈ 0
        elif C == 0:
            raise ValueError(f"Valeur de C invalide (0) au point {i}. Utilisez `None` pour ignorer C.")
        else:
            C_vals[i] = C

        # R
        if R not in (None, 0):
            R_vals[i] = R

    Z_L = 1j * omega * L_vals
    with np.errstate(divide='ignore', invalid='ignore'):
        Z_C = 1 / (1j * omega * C_vals)
    Z_R = R_vals

    DeltaZ = Z_L + Z_C + Z_R

    return DeltaZ

def calculate_z_matrice_lumped_elements(points, triangles, edges, barycentric_triangles, vecteurs_rho, frequency, LoadPoint, LoadValue, LoadDir):
    """
        Calcule la matrice d'impédance électromagnétique pour un système donné.

        Paramètres :
            * triangles : Objet représentant les triangles du maillage.
            * edges : Objet représentant les arêtes du maillage.
            * barycentric_triangles : Objet contenant les données barycentriques des triangles.
            * vecteurs_rho : Objet contenant les vecteurs Rho associés aux triangles.
            * frequency : Fréquence du signal électromagnétique (en Hz).

        Retourne :
            * omega : Pulsation angulaire (rad/s).
            * mu : Perméabilité magnétique du vide (H/m).
            * epsilon : Permittivité du vide (F/m).
            * light_speed_c : Vitesse de la lumière dans le vide (m/s).
            * eta : Impédance caractéristique de l'espace libre (Ω).
            * matrice_z : Matrice d'impédance calculée.
    """

    # Définition des paramètres électromagnétiques
    epsilon = 8.854e-12  # Permittivité du vide (F/m)
    mu = 1.257e-6        # Perméabilité magnétique du vide (H/m)

    # Calcul des constantes électromagnétiques
    light_speed_c = 1 / np.sqrt(epsilon * mu)  # Vitesse de la lumière dans le vide (m/s)
    eta = np.sqrt(mu / epsilon)                # Impédance caractéristique de l'espace libre (Ω)
    omega = 2 * np.pi * frequency              # Pulsation angulaire (rad/s)
    k = omega / light_speed_c                  # Nombre d'onde (rad/m)
    complexe_k = 1j * k                        # Nombre d'onde complexe pour les calculs

    # Définition des facteurs pour optimiser les calculs
    constant_1   = mu / (4 * np.pi)                        # Constante pour le calcul des champs magnétiques
    constant_2   = 1 / (1j * 4 * np.pi * omega * epsilon)  # Constante pour le calcul des champs électriques
    factor      = 1 / 9                                    # Facteur de pondération pour les calculs intégrés

    # Facteurs spécifiques liés aux arêtes
    factor_a     = factor * (1j * omega * edges.edges_length / 4) * constant_1  # Facteur pour le calcul des potentiels vectoriels
    factor_fi    = factor * edges.edges_length * constant_2                     # Facteur pour le calcul des potentiels scalaires

    # Calcul de la matrice d'impédance
    matrice_z = impedance_matrice_z(edges, triangles, barycentric_triangles, vecteurs_rho, complexe_k, factor_a, factor_fi)

    # Lumped impedance implementation
    # The informaion needed for lumped elements :
    #       LoadPoint                       Lumped element locations
    #       LoadValue                       Vector of L, C, and R
    #       LoadDir                         "Direction" of lumped element

    LoadPoint = LoadPoint.T  # (3, LNumber)
    LoadValue = LoadValue.T  # (3, LNumber)
    LoadDir   = LoadDir.T    # (3, LNumber)

    LNumber = LoadPoint.shape[1]

    DeltaZ = compute_lumped_impedance(LoadValue, omega)
    # print(f"DeltaZ = {DeltaZ}")

    ImpArray = []
    tol = 1e-3  # Tolerance for orientation

    for k in range(LNumber):
        EdgeCenters = 0.5 * (points.points[:, edges.first_points] + points.points[:, edges.second_points])  # (3, EdgesTotal)
        EdgeVectors = (points.points[:, edges.first_points] - points.points[:, edges.second_points]) / edges.edges_length[np.newaxis, :]

        diff = EdgeCenters - LoadPoint[:, k][:, np.newaxis]
        Dist = np.linalg.norm(diff, axis=0)
        Orien = np.abs(np.einsum('ij,i->j', EdgeVectors, LoadDir[:, k]))

        index = np.argsort(Dist)

        for idx in index:
            if Orien[idx] < tol:
                ImpArray.append(idx)
                matrice_z[idx, idx] += edges.edges_length[idx]**2 * DeltaZ[k]
                break

    ImpArray = np.array(ImpArray)  # Convert to numpy array for consistency

    return omega, mu, epsilon, light_speed_c, eta, matrice_z, ImpArray

class DataManager_rwg3:
    """
        Une classe pour gérer la sauvegarde et le chargement des données électromagnétiques liées à la matrice d'impédance.

        Cette classe fournit deux méthodes principales :
            * save_data : pour sauvegarder les données calculées dans un fichier .mat.
            * load_data : pour charger des données sauvegardées à partir d'un fichier .mat.
    """

    @staticmethod
    def save_data(filename_mesh2, save_folder_name, frequency, omega, mu, epsilon, light_speed_c, eta, matrice_z):
        """
            Sauvegarde les données calculées dans un fichier .mat.

            Paramètres :
                * filename_mesh2 : str, nom du fichier source contenant les données maillées de base.
                * save_folder_name : str, nom du dossier où les données doivent être sauvegardées.
                * frequency : float, fréquence utilisée pour le calcul électromagnétique (Hz).
                * omega : float, pulsation angulaire (rad/s).
                * mu : float, perméabilité magnétique du vide (H/m).
                * epsilon : float, permittivité du vide (F/m).
                * light_speed_c : float, vitesse de la lumière dans le vide (m/s).
                * eta : float, impédance caractéristique de l'espace libre (Ω).
                * matrice_z : n-d-array, matrice d'impédance calculée.

            Retourne :
            save_file_name : str, nom du fichier sauvegardé.

            Comportement :
                * Ajoute un suffixe '_impedance' au nom de base du fichier source.
                * Sauvegarde les données dans un fichier .mat au chemin spécifié.
                * Crée le dossier spécifié s'il n'existe pas encore.
        """
        # Préparer les données pour la sauvegarde
        data = {
            'frequency': frequency,
            'omega': omega,
            'mu': mu,
            'epsilon': epsilon,
            'light_speed_c': light_speed_c,
            'eta': eta,
            'matrice_z': matrice_z,
        }
        # Construction du nom du fichier de sauvegarde
        base_name = os.path.splitext(os.path.basename(filename_mesh2))[0]
        base_name = base_name.replace('_mesh2', '')   # Suppression de l'ancien suffixe '_mesh2'
        save_file_name = base_name + '_impedance.mat'  # Ajout du nouveau suffixe '_impedance'
        full_save_path = os.path.join(save_folder_name, save_file_name)

        # Vérification et création du dossier si nécessaire
        if not os.path.exists(save_folder_name):           # Vérification et création du dossier si nécessaire
            os.makedirs(save_folder_name)
            # print(f"Directory '{save_folder_name}' created.")

        # Sauvegarde des données dans un fichier .mat
        savemat(full_save_path, data)
        # print(f"Data saved successfully to {full_save_path}")
        return save_file_name

    @staticmethod
    def load_data(filename):
        """
            Charge les données d'un fichier .mat.

            Paramètres :
            filename : str, chemin du fichier .mat à charger.

            Retourne :
            tuple contenant les données suivantes :
              frequency (float), omega (float), mu (float), epsilon (float),
              light_speed_c (float), eta (float), matrice_z (n-d-array).

            Comportement :
            * Vérifie l'existence du fichier avant de le charger.
            * Charge les données en s'assurant que les dimensions des tableaux sont correctes.
            * Gère les exceptions courantes, comme un fichier introuvable ou des données mal formées.
        """
        try:
            # Vérification si le fichier existe
            if not os.path.isfile(filename):
                raise FileNotFoundError(f"File '{filename}' does not exist.")

            # Chargement des données
            data = loadmat(filename)
            frequency = data['frequency'].squeeze()
            omega = data['omega'].squeeze()
            mu = data['mu'].squeeze()
            epsilon = data['epsilon'].squeeze()
            light_speed_c = data['light_speed_c'].squeeze()
            eta = data['eta'].squeeze()
            matrice_z = data['matrice_z'].squeeze()
            # print(f"Data loaded from {filename}")
            return frequency, omega, mu, epsilon, light_speed_c, eta, matrice_z
        except FileNotFoundError as e:
            print(f"Error: {e}")
        except KeyError as e:
            print(f"Key Error: {e}")
        except ValueError as e:
            print(f"Value Error (likely malformed data): {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")