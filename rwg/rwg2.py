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

    def calculate_barycentric_center(self, point_data, triangles_data):
        """
            Calcule les centres barycentriques pour chaque triangle du maillage.

            Cette méthode divise chaque triangle en neuf sous-triangles, calcule les centres de chacun,
            et stocke ces centres dans une matrice 3 x 9 x N (où N est le nombre total de triangles).

            Paramètres :
                * point_data (Points) : Objet contenant les coordonnées des points du maillage (3 x M).
                * triangles_data (Triangles) : Objet contenant les indices des sommets des triangles et leurs centres (3 x N).
        """
        points = point_data.points  # Coordonnées des points du maillage
        triangles = triangles_data.triangles  # Indices des sommets des triangles
        triangles_center = triangles_data.triangles_center  # Centres géométriques des triangles
        total_of_triangles = triangles_data.total_of_triangles  # Nombre total de triangles

        # Initialisation de la matrice pour stocker les centres barycentriques
        self.barycentric_triangle_center = np.zeros((3, 9, total_of_triangles))  # Initialisation

        for triangle in range(total_of_triangles):
            # Extraction des indices des sommets du triangle courant
            sommet_triangle_1 = triangles[0, triangle]
            sommet_triangle_2 = triangles[1, triangle]
            sommet_triangle_3 = triangles[2, triangle]
            centre_triangle = triangles_center[:, triangle]

            # Coordonnées des sommets
            point_triangle_1 = points[:, sommet_triangle_1]
            point_triangle_2 = points[:, sommet_triangle_2]
            point_triangle_3 = points[:, sommet_triangle_3]

            # Calcul des côtés du triangle
            cote_1_2 = point_triangle_2 - point_triangle_1
            cote_2_3 = point_triangle_3 - point_triangle_2
            cote_1_3 = point_triangle_3 - point_triangle_1

            # Points intermédiaires sur les côtés du triangle (1/3 et 2/3 des longueurs)
            cote12_point_1 = point_triangle_1 + (1 / 3) * cote_1_2
            cote12_point_2 = point_triangle_1 + (2 / 3) * cote_1_2
            cote23_point_1 = point_triangle_2 + (1 / 3) * cote_2_3
            cote23_point_2 = point_triangle_2 + (2 / 3) * cote_2_3
            cote13_point_1 = point_triangle_1 + (1 / 3) * cote_1_3
            cote13_point_2 = point_triangle_1 + (2 / 3) * cote_1_3

            # Calcul des centres barycentriques pour les neuf sous-triangles
            barycentric_triangle_center_1 = (cote12_point_1 + cote13_point_1 + point_triangle_1) / 3
            barycentric_triangle_center_2 = (cote12_point_1 + cote12_point_2 + centre_triangle) / 3
            barycentric_triangle_center_3 = (cote12_point_2 + cote23_point_1 + point_triangle_2) / 3
            barycentric_triangle_center_4 = (cote12_point_2 + cote23_point_1 + centre_triangle) / 3
            barycentric_triangle_center_5 = (cote23_point_1 + cote23_point_2 + centre_triangle) / 3
            barycentric_triangle_center_6 = (cote12_point_1 + cote13_point_1 + centre_triangle) / 3
            barycentric_triangle_center_7 = (cote13_point_1 + cote13_point_2 + centre_triangle) / 3
            barycentric_triangle_center_8 = (cote23_point_2 + cote13_point_2 + centre_triangle) / 3
            barycentric_triangle_center_9 = (cote23_point_2 + cote13_point_2 + point_triangle_3) / 3

            # Stockage des centres barycentriques dans la matrice
            self.barycentric_triangle_center[:, :, triangle] = np.array([
                barycentric_triangle_center_1, barycentric_triangle_center_2, barycentric_triangle_center_3,
                barycentric_triangle_center_4, barycentric_triangle_center_5, barycentric_triangle_center_6,
                barycentric_triangle_center_7, barycentric_triangle_center_8, barycentric_triangle_center_9
            ]).T

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

    def calculate_vecteurs_rho(self, points_data, triangles_data, edges_data, barycentric_triangle_data):
        """
            Calcule les vecteurs Rho pour chaque arête du maillage.

            Les vecteurs Rho sont définis pour les triangles "plus" et "moins" associés à chaque arête :
            - `vecteur_rho_plus` : vecteur du point opposé à l'arête dans le triangle "plus" au centre du triangle.
            - `vecteur_rho_minus` : vecteur du point opposé à l'arête dans le triangle "moins" au centre du triangle.
            - `vecteur_rho_barycentric_plus` : vecteurs pour les sous-triangles barycentriques dans le triangle "plus".
            - `vecteur_rho_barycentric_minus` : vecteurs pour les sous-triangles barycentriques dans le triangle "moins".

            Paramètres :
                * points_data (Points) : Objet contenant les coordonnées des points du maillage (3 x M).
                * triangles_data (Triangles) : Objet contenant les triangles et leurs propriétés.
                * edges_data (Edges) : Objet contenant les informations des arêtes du maillage.
                * barycentric_triangle_data (Barycentric_triangle) : Objet contenant les centres barycentriques.
        """
        points = points_data.points  # Coordonnées des points
        triangles = triangles_data.triangles  # Indices des sommets des triangles
        edges_first_points = edges_data.first_points  # Indices des premiers points des arêtes
        edges_second_points = edges_data.second_points  # Indices des seconds points des arêtes
        total_number_of_edges = edges_data.total_number_of_edges  # Nombre total d'arêtes
        triangles_center = triangles_data.triangles_center  # Centres géométriques des triangles
        triangles_plus = triangles_data.triangles_plus  # Triangles "plus" associés aux arêtes
        triangles_minus = triangles_data.triangles_minus  # Triangles "moins" associés aux arêtes
        barycentric_triangle_center = barycentric_triangle_data.barycentric_triangle_center  # Centres barycentriques

        # Initialisation des matrices pour stocker les vecteurs Rho
        self.vecteur_rho_plus = np.zeros((3, total_number_of_edges))
        self.vecteur_rho_minus = np.zeros((3, total_number_of_edges))
        self.vecteur_rho_barycentric_plus = np.zeros((3, 9, total_number_of_edges))
        self.vecteur_rho_barycentric_minus = np.zeros((3, 9, total_number_of_edges))

        # Calcul des vecteurs Rho pour les triangles "plus"
        for edge in range(total_number_of_edges):
            index_point_vecteur = 0
            index_triangle_plus = triangles_plus[edge]

            # Identification du point opposé à l'arête dans le triangle "plus"
            sommet_triangle_plus_1 = triangles[0, index_triangle_plus]
            sommet_triangle_plus_2 = triangles[1, index_triangle_plus]
            sommet_triangle_plus_3 = triangles[2, index_triangle_plus]
            if np.all(sommet_triangle_plus_1 != edges_first_points[edge]) and np.all(sommet_triangle_plus_1 != edges_second_points[edge]):
                index_point_vecteur = sommet_triangle_plus_1
            elif np.all(sommet_triangle_plus_2 != edges_first_points[edge]) and np.all(sommet_triangle_plus_2 != edges_second_points[edge]):
                index_point_vecteur = sommet_triangle_plus_2
            elif np.all(sommet_triangle_plus_3 != edges_first_points[edge]) and np.all(sommet_triangle_plus_3 != edges_second_points[edge]):
                index_point_vecteur = sommet_triangle_plus_3

            point_vecteur = points[:, index_point_vecteur]

            # Calcul du vecteur Rho pour le triangle "plus"
            self.vecteur_rho_plus[:, edge] = + triangles_center[:, index_triangle_plus] - point_vecteur
            self.vecteur_rho_barycentric_plus[:, :, edge] = + barycentric_triangle_center[:, :, index_triangle_plus] - np.tile(point_vecteur, (9, 1)).T

        # Calcul des vecteurs Rho pour les triangles "moins"
        for edge in range(total_number_of_edges):
            index_point_vecteur = 0
            index_triangle_minus = triangles_minus[edge]

            # Identification du point opposé à l'arête dans le triangle "moins"
            sommet_triangle_minus_1 = triangles[0, index_triangle_minus]
            sommet_triangle_minus_2 = triangles[1, index_triangle_minus]
            sommet_triangle_minus_3 = triangles[2, index_triangle_minus]
            if np.all(sommet_triangle_minus_1 != edges_first_points[edge]) and np.all(sommet_triangle_minus_1 != edges_second_points[edge]):
                index_point_vecteur = sommet_triangle_minus_1
            elif np.all(sommet_triangle_minus_2 != edges_first_points[edge]) and np.all(sommet_triangle_minus_2 != edges_second_points[edge]):
                index_point_vecteur = sommet_triangle_minus_2
            elif np.all(sommet_triangle_minus_3 != edges_first_points[edge]) and np.all(sommet_triangle_minus_3 != edges_second_points[edge]):
                index_point_vecteur = sommet_triangle_minus_3

            point_vecteur = points[:, index_point_vecteur]

            # Calcul du vecteur Rho pour le triangle "moins"
            self.vecteur_rho_minus[:, edge] = - triangles_center[:, index_triangle_minus] + point_vecteur
            self.vecteur_rho_barycentric_minus[:, :, edge] = - barycentric_triangle_center[:, :, index_triangle_minus] + np.tile(point_vecteur, (9, 1)).T

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
            print(f"Directory '{save_folder_name}' created.")

        # Sauvegarde des données dans un fichier MAT
        savemat(full_save_path, data)
        print(f"Data saved successfully to {full_save_path}")
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
            points_feed = data['points_feed'].squeeze()
            triangles_feed = data['triangles_feed'].squeeze()
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
            print(f"Data loaded from {filename}")
            return points, triangles, points_feed, triangles_feed, edges, barycentric_triangle, vecteurs_rho
        except FileNotFoundError as e:
            print(f"Error: {e}")
        except KeyError as e:
            print(f"Key Error: {e}")
        except ValueError as e:
            print(f"Value Error (likely malformed data): {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")