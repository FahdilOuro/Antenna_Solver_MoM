import os
import numpy as np
from scipy.io import savemat, loadmat

from rwg.rwg1 import Points, Triangles, Edges


class Barycentric_triangle:
    """
        Classe pour calculer et stocker les centres barycentriques d'un maillage triangulaire.

        Cette classe utilise les coordonnées des sommets des triangles et leurs centres géométriques pour
        calculer les centres barycentriques associés aux subdivisions des triangles en neuf sous-triangles.
    """

    def __init__(self):
        """
            Initialise l'objet avec un attribut pour stocker les centres barycentriques.
        """
        self.barycentric_triangle_center = None
    
    # version 2 :
    def calculate_barycentric_center(self, point_data, triangles_data):
        points = point_data.points                             # (3, M)
        triangles = triangles_data.triangles                   # (3, N)
        triangles_center = triangles_data.triangles_center     # (3, N)
        total_of_triangles = triangles_data.total_of_triangles # N

        # Récupération des points des sommets pour tous les triangles
        pt1 = points[:, triangles[0]]  # (3, N)
        pt2 = points[:, triangles[1]]  # (3, N)
        pt3 = points[:, triangles[2]]  # (3, N)

        # Calcul des vecteurs côtés
        v12 = pt2 - pt1  # (3, N)
        v23 = pt3 - pt2  # (3, N)
        v13 = pt3 - pt1  # (3, N)

        # Points aux fractions 1/3 et 2/3
        pt12_1 = pt1 + (1/3) * v12
        pt12_2 = pt1 + (2/3) * v12
        pt23_1 = pt2 + (1/3) * v23
        pt23_2 = pt2 + (2/3) * v23
        pt13_1 = pt1 + (1/3) * v13
        pt13_2 = pt1 + (2/3) * v13

        c = triangles_center  # alias plus court

        # Calcul des 9 centres barycentriques
        bary_1 = (pt12_1 + pt13_1 + pt1) / 3
        bary_2 = (pt12_1 + pt12_2 + c) / 3
        bary_3 = (pt12_2 + pt23_1 + pt2) / 3
        bary_4 = (pt12_2 + pt23_1 + c) / 3
        bary_5 = (pt23_1 + pt23_2 + c) / 3
        bary_6 = (pt12_1 + pt13_1 + c) / 3
        bary_7 = (pt13_1 + pt13_2 + c) / 3
        bary_8 = (pt23_2 + pt13_2 + c) / 3
        bary_9 = (pt23_2 + pt13_2 + pt3) / 3

        # Empilement dans un tableau final (3, 9, N)
        self.barycentric_triangle_center = np.stack([
            bary_1, bary_2, bary_3, bary_4, bary_5,
            bary_6, bary_7, bary_8, bary_9
        ], axis=1)  # shape (3, 9, N)

    def set_barycentric_center(self, barycentric_triangle_center):
        """
            Définit manuellement les centres barycentriques.

            Paramètres :
            barycentric_triangle_center (n-d-array) : Tableau 3 x 9 x N contenant les centres barycentriques à définir.
        """
        self.barycentric_triangle_center = barycentric_triangle_center


class Vecteurs_Rho:
    """
        Classe pour calculer et gérer les vecteurs Rho associés aux triangles plus et moins des arêtes dans un maillage.

        Les vecteurs Rho représentent des vecteurs reliant un point spécifique d'un triangle (opposé à l'arête considérée)
        à son centre géométrique ou à ses centres barycentriques.
    """

    def __init__(self):
        """
            Initialise les attributs pour stocker les vecteurs Rho.
        """
        self.vecteur_rho_plus = None
        self.vecteur_rho_minus = None
        self.vecteur_rho_barycentric_plus = None
        self.vecteur_rho_barycentric_minus = None
    
    # version 4
    def calculate_vecteurs_rho(self, points_data, triangles_data, edges_data, barycentric_triangle_data):
        points = points_data.points  # (3, N_points)
        triangles = triangles_data.triangles  # (3, N_triangles)
        triangles_center = triangles_data.triangles_center  # (3, N_triangles)
        triangles_plus = triangles_data.triangles_plus  # (N_edges,)
        triangles_minus = triangles_data.triangles_minus  # (N_edges,)
        barycentric_triangle_center = barycentric_triangle_data.barycentric_triangle_center  # (3, 9, N_triangles)

        edges_first_points = edges_data.first_points  # (N_edges,)
        edges_second_points = edges_data.second_points  # (N_edges,)
        total_number_of_edges = edges_data.total_number_of_edges  # int

        self.vecteur_rho_plus = np.zeros((3, total_number_of_edges))
        self.vecteur_rho_minus = np.zeros((3, total_number_of_edges))
        self.vecteur_rho_barycentric_plus = np.zeros((3, 9, total_number_of_edges))
        self.vecteur_rho_barycentric_minus = np.zeros((3, 9, total_number_of_edges))

        # --- Traitement vectorisé pour les triangles "plus" ---
        triangles_plus_sommets = triangles[:, triangles_plus]  # (3, N_edges)
        edges_fp = edges_first_points
        edges_sp = edges_second_points

        # Détection du sommet opposé
        mask_plus = (triangles_plus_sommets != edges_fp) & (triangles_plus_sommets != edges_sp)  # (3, N_edges)
        # Pour chaque arête, trouver l'indice du sommet opposé
        indices_opposes_plus = np.argmax(mask_plus, axis=0)  # (N_edges,)
        index_point_vecteur_plus = triangles_plus_sommets[indices_opposes_plus, np.arange(total_number_of_edges)]  # (N_edges,)
        point_vecteurs_plus = points[:, index_point_vecteur_plus]  # (3, N_edges)

        # Calcul des vecteurs Rho "plus"
        self.vecteur_rho_plus = triangles_center[:, triangles_plus] - point_vecteurs_plus
        self.vecteur_rho_barycentric_plus = barycentric_triangle_center[:, :, triangles_plus] - point_vecteurs_plus[:, None, :]

        # --- Traitement vectorisé pour les triangles "moins" ---
        triangles_minus_sommets = triangles[:, triangles_minus]  # (3, N_edges)

        mask_minus = (triangles_minus_sommets != edges_fp) & (triangles_minus_sommets != edges_sp)
        indices_opposes_minus = np.argmax(mask_minus, axis=0)
        index_point_vecteur_minus = triangles_minus_sommets[indices_opposes_minus, np.arange(total_number_of_edges)]
        point_vecteurs_minus = points[:, index_point_vecteur_minus]

        # Calcul des vecteurs Rho "moins"
        self.vecteur_rho_minus = point_vecteurs_minus - triangles_center[:, triangles_minus]
        self.vecteur_rho_barycentric_minus = point_vecteurs_minus[:, None, :] - barycentric_triangle_center[:, :, triangles_minus]

    def set_vecteurs_rho(self, vecteur_rho_plus, vecteur_rho_minus, vecteur_rho_barycentric_plus, vecteur_rho_barycentric_minus):
        """
            Définit manuellement les vecteurs Rho.

            Paramètres :
                * vecteur_rho_plus (n-d-array) : Vecteurs Rho pour les triangles "plus".
                * vecteur_rho_minus (n-d-array) : Vecteurs Rho pour les triangles "moins".
                * vecteur_rho_barycentric_plus (n-d-array) : Vecteurs barycentriques pour les triangles "plus".
                * vecteur_rho_barycentric_minus (n-d-array) : Vecteurs barycentriques pour les triangles "moins".
        """
        self.vecteur_rho_plus = vecteur_rho_plus
        self.vecteur_rho_minus = vecteur_rho_minus
        self.vecteur_rho_barycentric_plus = vecteur_rho_barycentric_plus
        self.vecteur_rho_barycentric_minus = vecteur_rho_barycentric_minus


class DataManager_rwg2:
    """
        Classe pour sauvegarder et charger des données liées à un maillage et ses propriétés dans des fichiers MAT.

        Fournit des méthodes statiques pour :
            * Sauvegarder les données enrichies dans un fichier MAT.
            * Charger les données à partir d'un fichier MAT existant.
    """
    @staticmethod
    def save_data(filename_mesh1, save_folder_name, barycentric_triangle_data, vecteurs_rho_data):
        """
            Sauvegarde les données dans un fichier MAT après les avoir enrichies.

            Paramètres :
                * filename_mesh1 (str) : Chemin du fichier MAT initial contenant les données de maillage.
                * save_folder_name (str) : Nom du dossier où le fichier enrichi sera sauvegardé.
                * barycentric_triangle_data (Barycentric_triangle) : Données barycentriques du triangle.
                * vecteurs_rho_data (Vecteurs_Rho) : Données des vecteurs Rho.

            Retourne :
            save_file_name (str) : Nom du fichier MAT sauvegardé.
        """
        # Chargement des données initiales
        data = loadmat(filename_mesh1)

        # Ajout des nouvelles données
        new_data = {
            'barycentric_triangle_center' : barycentric_triangle_data.barycentric_triangle_center,
            'vecteur_rho_plus' : vecteurs_rho_data.vecteur_rho_plus,
            'vecteur_rho_minus' : vecteurs_rho_data.vecteur_rho_minus,
            'vecteur_rho_barycentric_plus' : vecteurs_rho_data.vecteur_rho_barycentric_plus,
            'vecteur_rho_barycentric_minus' : vecteurs_rho_data.vecteur_rho_barycentric_plus,
        }
        data.update(new_data)


        # Génération du nom du fichier de sauvegarde
        base_name = os.path.splitext(os.path.basename(filename_mesh1))[0]
        base_name = base_name.replace('_mesh1', '')  # Suppression de '_mesh1'
        # Ajout de la partie '_mesh2'
        save_file_name = base_name + '_mesh2.mat'
        full_save_path = os.path.join(save_folder_name, save_file_name)

        # Création du dossier si nécessaire
        if not os.path.exists(save_folder_name): # Vérification et création du dossier si nécessaire
            os.makedirs(save_folder_name)
            # print(f"Directory '{save_folder_name}' created.")

        # Sauvegarde des données dans un fichier MAT
        savemat(full_save_path, data)
        # print(f"Data saved successfully to {full_save_path}")
        return save_file_name

    @staticmethod
    def load_data(filename):
        """
            Charge les données d'un fichier MAT et initialise les objets associés.

            Paramètres :
            filename (str) : Chemin du fichier MAT contenant les données.

            Retourne :
            (tuple) : Contient les objets Points, Triangles, Edges, Barycentric_triangle et Vecteurs_Rho.

            Exception :
                * FileNotFoundError : Si le fichier n'existe pas.
                * KeyError : Si une clé attendue est absente des données.
                * ValueError : Si les données sont mal formées.
        """
        try:
            # Vérification de l'existence du fichier
            if not os.path.isfile(filename):
                raise FileNotFoundError(f"File '{filename}' does not exist.")

            # Chargement des données
            data = loadmat(filename)


            # Initialisation des objets avec les données chargées
            points = Points(points_data=data['points'].squeeze())
            triangles = Triangles(triangles_data=data['triangles'].squeeze())
            edges = Edges(first_points=data['edge_first_points'].squeeze(), second_points=data['edge_second_points'].squeeze())
            triangles.set_triangles_plus_minus(triangles_plus=data['triangles_plus'].squeeze(), triangles_minus=data['triangles_minus'].squeeze())
            triangles.set_triangles_area_and_center(triangles_area=data['triangles_area'].squeeze(), triangles_center=data['triangles_center'].squeeze())
            edges.set_edge_length(edge_length=data['edges_length'].squeeze())
            barycentric_triangle = Barycentric_triangle()
            barycentric_triangle.set_barycentric_center(barycentric_triangle_center=data['barycentric_triangle_center'].squeeze())
            vecteurs_rho = Vecteurs_Rho()
            vecteurs_rho.set_vecteurs_rho(vecteur_rho_plus=data['vecteur_rho_plus'].squeeze(),
                                          vecteur_rho_minus=data['vecteur_rho_minus'].squeeze(),
                                          vecteur_rho_barycentric_plus=data['vecteur_rho_barycentric_plus'].squeeze(),
                                          vecteur_rho_barycentric_minus=data['vecteur_rho_barycentric_minus'].squeeze())
            # print(f"Data loaded from {filename}")
            return points, triangles, edges, barycentric_triangle, vecteurs_rho
        except FileNotFoundError as e:
            print(f"Error: {e}")
        except KeyError as e:
            print(f"Key Error: {e}")
        except ValueError as e:
            print(f"Value Error (likely malformed data): {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")