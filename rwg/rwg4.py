import os
import time

import numpy as np
from scipy.io import savemat, loadmat

from rwg.rwg2 import DataManager_rwg2
from rwg.rwg3 import DataManager_rwg3


# Définition du champ incident
# Example: wave_incident_direction = [0, 0, -1] signifie que le champ incident arrive dans la direction "-z"
# Example: polarization = [1, 0, 0] signifie le champ incident est polarisé dans la direction "x"

def calculate_current_scattering(filename_mesh_2, filename_impedance, wave_incident_direction, polarization):
    """
        Calcule le courant et le vecteur de tension résultant de la diffusion d'une onde incidente sur une structure.

        Cette fonction utilise des données maillées et des données d'impédance calculées pour résoudre les équations
        de la méthode des moments (MoM), modélisant la réponse électromagnétique d'une structure.

        Paramètres :
            * filename_mesh_2 : str, chemin vers le fichier contenant les données maillées (fichier _mesh2).
            * filename_impedance : str, chemin vers le fichier contenant les données d'impédance (_impedance).
            * wave_incident_direction : n-d-array (3,), direction de propagation de l'onde incidente (vecteur unitaire).
            * polarization : n-d-array (3,), vecteur décrivant la polarisation du champ électrique incident
              (par exemple, le sens 'x' ou 'y').

        Retourne :
            * frequency : float, fréquence utilisée dans le calcul électromagnétique (Hz).
            * omega : float, pulsation angulaire associée (rad/s).
            * mu : float, perméabilité magnétique du vide (H/m).
            * epsilon : float, permittivité du vide (F/m).
            * light_speed_c : float, vitesse de la lumière dans le vide (m/s).
            * eta : float, impédance caractéristique de l'espace libre (Ω).
            * voltage : n-d-array, vecteur de tension résultant des équations de MoM (Z * I = V).
            * current : n-d-array, vecteur courant solution des équations de MoM.

        Comportement :
            1. Charge les données maillées et les données d'impédance des fichiers spécifiés.
            2. Calcule le vecteur d'onde `kv` à partir de la direction de l'onde incidente et du nombre d'onde 'k'.
            3. Initialise un vecteur 'voltage' (second membre des équations de MoM) à partir des contributions des arêtes
               et des produits scalaires liés aux triangles associés.
            4. Résout le système d'équations de MoM pour obtenir le vecteur courant 'current'.
            5. Affiche le temps de calcul pour la résolution du système linéaire.

        Notes :
            * La méthode repose sur la précision des données maillées et des données d'impédance fournies.
            * La direction de l'onde incidente ('wave_incident_direction') et la polarisation doivent être correctement
              normalisées pour garantir des résultats cohérents.
    """
    # Chargement des données maillées (points, triangles, arêtes, barycentres, vecteurs rho)
    _, triangles, edges, _, vecteurs_rho = DataManager_rwg2.load_data(filename_mesh_2)
    # Chargement des données d'impédance (fréquence, paramètres électromagnétiques, matrice Z)
    frequency, omega, mu, epsilon, light_speed_c, eta, matrice_z = DataManager_rwg3.load_data(filename_impedance)

    # Calcul des constantes physiques
    k = omega / light_speed_c               # Nombre d'onde
    kv = k * wave_incident_direction        # Vecteur d'onde incident

    # Initialisation du vecteur de tension (second membre des équations de MoM)
    voltage = np.zeros(edges.total_number_of_edges, dtype=complex)
    
    # === Préparation des centres de triangles associés aux arêtes ===
    centers_plus = triangles.triangles_center[:, triangles.triangles_plus]    # (3, N_edges)
    centers_minus = triangles.triangles_center[:, triangles.triangles_minus]  # (3, N_edges)

    # === Calcul des produits scalaires kv . r_plus et kv . r_minus ===
    scalar_product_plus = np.einsum('i,ij->j', kv, centers_plus)   # (N_edges,)
    scalar_product_minus = np.einsum('i,ij->j', kv, centers_minus) # (N_edges,)

    # === Calcul des facteurs d'onde complexes (em_plus et em_minus) ===
    # Broadcasting : (3, 1) * (1, N_edges) -> (3, N_edges)
    em_plus = polarization[:, None] * np.exp(-1j * scalar_product_plus)[None, :]
    em_minus = polarization[:, None] * np.exp(-1j * scalar_product_minus)[None, :]

    # === Produits scalaires avec vecteurs rho ===
    # em_plus et vecteur_rho_plus sont tous deux (3, N_edges)
    scalar_plus = np.einsum('ij,ij->j', em_plus, vecteurs_rho.vecteur_rho_plus)   # (N_edges,)
    scalar_minus = np.einsum('ij,ij->j', em_minus, vecteurs_rho.vecteur_rho_minus) # (N_edges,)

    # === Assemblage final du vecteur "voltage" ===
    voltage = edges.edges_length * 0.5 * (scalar_plus + scalar_minus)  # (N_edges,)

    # Chronométrage du calcul
    start_time = time.time()

    # Résolution du système linéaire (Z * I = V) pour obtenir le vecteur courant
    current = np.linalg.solve(matrice_z, voltage)

    # Mesure du temps écoulé pour la résolution
    # elapsed_time = time.time() - start_time
    # print(f"Temps écoulé pour le calcul du courant : {elapsed_time:.6f} secondes")

    # Retourner les résultats principaux
    return frequency, omega, mu, epsilon, light_speed_c, eta, voltage, current

def find_feed_edges(points, edges, feed_point, monopole=False):
    # --- Normalisation de feed_point / Feed ---
    Feed = np.asarray(feed_point.T)
    if Feed.ndim == 1:
        Feed = Feed.reshape(3, 1)
    elif Feed.ndim == 2 and Feed.shape[0] == 3:
        pass
    else:
        raise ValueError("feed_point doit être de forme (3,) ou (3, N)")
    
    # --- Calcul vectorisé des centres géométriques des arêtes ---
    centers = 0.5 * (points.points[:, edges.first_points] + points.points[:, edges.second_points])  # (3, E)

    # --- Calcul des distances entre chaque feed point et chaque arête ---
    diff = centers[:, :, np.newaxis] - Feed[:, np.newaxis, :]  # (3, E, N)
    dist_squared = np.sum(diff ** 2, axis=0)                   # (E, N)

    # --- Sélection des arêtes à alimenter ---
    if monopole:
        index_feeding_edges = np.argsort(dist_squared, axis=0)[:2, :]  # (2, N)
        index_feeding_edges = index_feeding_edges.flatten(order='F')   # (2*N,)
        # print(f"index_feeding_edges = {index_feeding_edges}")
    else:
        index_feeding_edges = np.argmin(dist_squared, axis=0)          # (N,)
        # print(f"index_feeding_edges = {index_feeding_edges}")
    
    return index_feeding_edges

def calculate_current_radiation(filename_mesh_2, filename_impedance, feed_point, voltage_amplitude, monopole=False, simulate_array_antenna=False):
    """
        Calcule les courants, l'impédance d'entrée et la puissance rayonnée d'une antenne.

        Cette fonction utilise les données maillées et les données d'impédance pour résoudre les équations
        de la méthode des moments (MoM). Elle simule l'effet d'un point d'alimentation sur l'antenne et en
        déduit ses paramètres de fonctionnement.

        Paramètres :
            * filename_mesh_2 : str, chemin vers le fichier contenant les données maillées (_mesh2).
            * filename_impedance : str, chemin vers le fichier contenant les données d'impédance (_impedance).
            * feed_point : n-d-array (3,), coordonnées du point d'alimentation sur l'antenne.
            * voltage_amplitude : float, amplitude du signal appliqué au point d'alimentation.

        Retourne :
            * frequency : float, fréquence de fonctionnement (Hz).
            * omega : float, pulsation angulaire (rad/s).
            * mu : float, perméabilité magnétique du vide (H/m).
            * epsilon : float, permittivité du vide (F/m).
            * light_speed_c : float, vitesse de la lumière dans le vide (m/s).
            * eta : float, impédance caractéristique de l'espace libre (Ω).
            * voltage : n-d-array, vecteur de tension appliqué aux arêtes.
            * current : n-d-array, vecteur courant résultant de la résolution des équations de MoM.
            * impedance : complex, impédance d'entrée calculée au point d'alimentation (Ω).
            * feed_power : float, puissance active fournie à l'antenne (W).

        Comportement :
            1. Charge les données maillées et d'impédance nécessaires au calcul.
            2. Identifie l'arête la plus proche du point d'alimentation (feed_point).
            3. Définit le vecteur de tension (voltage) avec une excitation appliquée à l'arête alimentée.
            4. Résout les équations de MoM pour obtenir les courants circulant dans le réseau.
            5. Calcule les paramètres électriques de l'antenne, notamment :
               * L'impédance d'entrée au point d'alimentation.
               * La puissance active transmise à l'antenne.

        Notes :
            * Le point d'alimentation (feed_point) doit être situé à proximité de l'une des arêtes du maillage.
            * Les données d'impédance et maillage doivent correspondre pour garantir des calculs cohérents.
            * La résolution du système linéaire repose sur une matrice d'impédance correctement formée.
    """
    # Chargement des données maillées et d'impédance
    points, _, edges, *_ = DataManager_rwg2.load_data(filename_mesh_2)
    frequency, omega, mu, epsilon, light_speed_c, eta, matrice_z = DataManager_rwg3.load_data(filename_impedance)

    # Initialisation du vecteur de tension
    voltage = np.zeros(edges.total_number_of_edges, dtype=complex)

    index_feeding_edges = find_feed_edges(points, edges, feed_point, monopole)

    # --- Application de la tension avec phase progressive si simulate_array ---
    if simulate_array_antenna:
        N = len(index_feeding_edges)
        phase = 0  # ex: 0 (broadside), -2 * np.pi / 3 (end-fire), etc.
        phase_shift = np.exp(1j * phase * np.arange(N))
        voltage[index_feeding_edges] = voltage_amplitude * edges.edges_length[index_feeding_edges] * phase_shift
        # print("voltage[index_feeding_edges] = ", voltage[index_feeding_edges])
    else:
        voltage[index_feeding_edges] = voltage_amplitude * edges.edges_length[index_feeding_edges]

        # print("voltage[index_feeding_edges] = ", voltage[index_feeding_edges])

    # --- Résolution du système linéaire (Z * I = V) ---
    current = np.linalg.solve(matrice_z, voltage)

    # --- Impédance / puissance ---
    if simulate_array_antenna:
        edge_lengths = edges.edges_length[index_feeding_edges]
        current_vals = current[index_feeding_edges]
        voltage_vals = voltage[index_feeding_edges]

        gap_current = current_vals * edge_lengths
        gap_voltage = voltage_vals / edge_lengths
        impedance = gap_voltage / gap_current
        feed_power = 0.5 * np.real(gap_current * np.conj(gap_voltage))
    else:
        gap_current = np.sum(current[index_feeding_edges] * edges.edges_length[index_feeding_edges])
        gap_voltage = np.mean(voltage[index_feeding_edges] / edges.edges_length[index_feeding_edges])
        impedance = gap_voltage / gap_current
        feed_power = 0.5 * np.real(gap_current * np.conj(gap_voltage))

    # Retourner les résultats
    return frequency, omega, mu, epsilon, light_speed_c, eta, voltage, current, gap_current, gap_voltage, impedance, feed_power, index_feeding_edges


class DataManager_rwg4:
    """
        Une classe pour gérer la sauvegarde et le chargement des données liées aux problèmes
        d'ondes électromagnétiques, tels que la diffusion ou la radiation, en utilisant des fichiers MATLAB.

        Méthodes :
            * save_data_fro_scattering : Sauvegarde des données liées à la diffusion des ondes.
            * save_data_for_radiation : Sauvegarde des données liées à la radiation.
            * load_data : Chargement des données à partir d'un fichier MATLAB.
    """
    @staticmethod
    def save_data_for_scattering(filename_mesh2, save_folder_name, frequency,
                                 omega, mu, epsilon, light_speed_c, eta,
                                 wave_incident_direction, polarization, voltage, current):
        """
            Sauvegarde les données liées à la diffusion d'ondes électromagnétiques dans un fichier MATLAB.

            Paramètres :
                * filename_mesh2 (str) : Nom du fichier de maillage utilisé pour la simulation.
                * save_folder_name (str) : Répertoire où les données seront sauvegardées.
                * frequency (float) : Fréquence d'onde.
                * omega (float) : Pulsation angulaire
                * mu (float) : Perméabilité magnétique du milieu.
                * epsilon (float) : Permittivité électrique du milieu.
                * light_speed_c (float) : Vitesse de la lumière dans le milieu.
                * eta (float) : Impédance du milieu.
                * wave_incident_direction (np.n-d-array) : Direction de l'onde incidente.
                * polarization (np.n-d-array) : Polarisation de l'onde incidente.
                * voltage (np.n-d-array) : Tensions simulées.
                * current (np.n-d-array) : Courants simulés.

            Retourne :
            save_file_name (str) : Nom du fichier de sauvegarde généré.
        """
        # Construction du nom de fichier
        base_name = os.path.splitext(os.path.basename(filename_mesh2))[0]
        base_name = base_name.replace('_mesh2', '')  # Suppression de la partie '_mesh2'
        save_file_name = base_name + '_current.mat'  # Ajout de '_current' au nom
        full_save_path = os.path.join(save_folder_name, save_file_name)

        # Vérification et création du répertoire si nécessaire
        if not os.path.exists(save_folder_name):
            os.makedirs(save_folder_name)
            # print(f"Directory '{save_folder_name}' created.")

        # Sauvegarde des données avec la direction et la polarisation de l'incident de l'onde
        data = {
            'frequency': frequency,
            'omega': omega,
            'mu': mu,
            'epsilon': epsilon,
            'light_speed_c': light_speed_c,
            'eta': eta,
            'wave_incident_direction': wave_incident_direction,
            'polarization': polarization,
            'voltage': voltage,
            'current': current
        }

        # Sauvegarde des données
        savemat(full_save_path, data)

        # print(f"Data saved successfully to {full_save_path}")

        return save_file_name

    @staticmethod
    def save_data_for_radiation(filename_mesh2, save_folder_name, frequency, omega,
                                mu, epsilon, light_speed_c, eta,
                                voltage, current, gap_current, gap_voltage, impedance, feed_power):
        """
            Sauvegarde les données liées à la radiation des ondes électromagnétiques dans un fichier MATLAB.

            Paramètres :
                (Identiques à ceux de 'save_data_fro_scattering', avec en plus :)
                * impedance (np.n-d-array) : Impédance mesurée.
                * feed_power (np.n-d-array) : Puissance d'alimentation.

            Retourne :
            save_file_name (str) : Nom du fichier de sauvegarde généré.
        """
        # Construction du nom de fichier
        base_name = os.path.splitext(os.path.basename(filename_mesh2))[0]
        base_name = base_name.replace('_mesh2', '')  # Suppression de la partie '_mesh2'
        save_file_name = base_name + '_current.mat'  # Ajout de '_current' au nom
        full_save_path = os.path.join(save_folder_name, save_file_name)

        # Vérification et création du répertoire si nécessaire
        if not os.path.exists(save_folder_name):  # Vérification et création du dossier si nécessaire
            os.makedirs(save_folder_name)
            # print(f"Directory '{save_folder_name}' created.")

        # Sauvegarde des données avec le courant, l'impédance et la puissance d'alimentation
        data = {
            'frequency': frequency,
            'omega': omega,
            'mu': mu,
            'epsilon': epsilon,
            'light_speed_c': light_speed_c,
            'eta': eta,
            'voltage': voltage,
            'current': current,
            'gap_current' : gap_current,
            'gap_voltage' : gap_voltage,
            'impedance': impedance,
            'feed_power': feed_power
        }

        # Sauvegarde des données
        savemat(full_save_path, data)

        # print(f"Data saved successfully to {full_save_path}")

        return save_file_name

    @staticmethod
    def load_data(filename, radiation = False, scattering = False):
        """
            Charge des données à partir d'un fichier MATLAB.

            Paramètres :
            filename (str) : Chemin complet vers le fichier à charger.

            Retourne :
            tuple : Contenu des données chargées, dépendant des clés présentes dans le fichier.

            Exceptions gérées :
                * FileNotFoundError : Si le fichier spécifié n'existe pas.
                * KeyError : Si des clés attendues sont manquantes dans le fichier.
                * ValueError` : Si les données sont mal formatées.
        """
        try:
            # Vérifie l'existence du fichier
            if not os.path.isfile(filename):
                raise FileNotFoundError(f"File '{filename}' does not exist.")

            # Extraction des données principales
            data = loadmat(filename)
            frequency = data['frequency'].squeeze()
            omega = data['omega'].squeeze()
            mu = data['mu'].squeeze()
            epsilon = data['epsilon'].squeeze()
            light_speed_c = data['light_speed_c'].squeeze()
            eta = data['eta'].squeeze()
            voltage = data['voltage'].squeeze()
            current = data['current'].squeeze()

            # Extraction des champs spécifiques
            if 'wave_incident_direction' in data and 'polarization' in data and scattering:
                wave_incident_direction = data['wave_incident_direction'].squeeze()
                polarization = data['polarization'].squeeze()
                return frequency, omega, mu, epsilon, light_speed_c, eta, wave_incident_direction, polarization, voltage, current

            if 'feed_power' in data and 'impedance' in data and 'gap_voltage' in data and 'gap_current' in data and radiation:
                impedance = data['voltage'].squeeze()
                feed_power = data['current'].squeeze()
                gap_voltage = data['gap_voltage'].squeeze()
                # print("on est ici dans 2")
                gap_current = data['gap_current'].squeeze()
                return frequency, omega, mu, epsilon, light_speed_c, eta, voltage, current, gap_voltage, gap_current, impedance, feed_power
            if not scattering and not radiation:
                raise ValueError("Erreur : 'scattering' et 'radiation' ne peuvent pas être tous les deux False. Precision a ajouter")

        

        except FileNotFoundError as e:
            print(f"Error: {e}")
        except KeyError as e:
            print(f"Key Error: {e}")
        except ValueError as e:
            print(f"Value Error (likely malformed data): {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")