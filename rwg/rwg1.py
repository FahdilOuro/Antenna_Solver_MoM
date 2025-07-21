import os
import numpy as np
from itertools import combinations
from collections import defaultdict
from scipy.io import savemat, loadmat


class Points:
    """
        Classe représentant un ensemble de points dans un espace tridimensionnel.

        Cette classe permet de stocker des points 3D et de fournir des informations
        sur les dimensions (longueur, largeur, hauteur) de l'espace occupé par ces points.

        Attributs :
            * points (n-d-array) : Un tableau numpy contenant les coordonnées des points sous forme (3, N),
                                où N est le nombre total de points dans l'espace 3D.
            * total_of_points (int) : Le nombre total de points présents dans l'ensemble.
            * length (float) : La longueur (dimension maximale sur l'axe X) de l'ensemble de points.
            * width (float) : La largeur (dimension maximale sur l'axe Y) de l'ensemble de points.
            * height (float) : La hauteur (dimension maximale sur l'axe Z) de l'ensemble de points.
    """

    def __init__(self, points_data):
        """
            Initialise un objet de la classe Points avec les données de coordonnées.

            Paramètres :
            points_data (n-d-array) : Un tableau numpy de forme (3, N) représentant les coordonnées
                                     des N points dans l'espace 3D. La première ligne contient
                                     les coordonnées X, la deuxième Y et la troisième Z des points.
        """
        self.points = points_data
        # Le nombre total de points est défini par le nombre de colonnes du tableau (points_data).
        self.total_of_points = self.points.shape[1]

        # Calcul des dimensions de l'espace occupé par les points
        self.length = max(points_data[0]) - min(points_data[0])  # Différence entre les valeurs maximales et minimales sur l'axe X
        self.width = max(points_data[1]) - min(points_data[1])   # Différence entre les valeurs maximales et minimales sur l'axe Y
        self.height = max(points_data[2]) - min(points_data[2])  # Différence entre les valeurs maximales et minimales sur l'axe Z

    def get_point_coordinates(self, index):
        """
            Récupère les coordonnées d'un point donné par son index.

            Paramètre :
            index (int) : L'index du point dont les coordonnées doivent être retournées.

            Retour :
            n-d-array : Un tableau contenant les coordonnées (X, Y, Z) du point spécifié.
        """
        return self.points[index]


class Triangles:
    """
        Classe représentant un ensemble de triangles dans un maillage 3D.

        Cette classe permet de stocker des triangles, de calculer leurs propriétés géométriques
        (aires, centres) et de détecter des arêtes communes entre les triangles.

        Attributs :
            * triangles (n-d-array) : Un tableau numpy de forme (3, N) représentant les indices des sommets
                                   de chaque triangle dans l'espace 3D.
            * total_of_triangles (int) : Le nombre total de triangles dans l'ensemble.
            * triangles_area (n-d-array) : Un tableau contenant les aires de chaque triangle.
            * triangles_center (n-d-array) : Un tableau contenant les coordonnées du centre de chaque triangle.
            * triangles_plus (n-d-array) : Un tableau contenant les indices des triangles qui partagent une arête commune.
            * triangles_minus (n-d-array) : Un tableau contenant les indices des triangles qui partagent une arête commune.
    """

    def __init__(self, triangles_data):
        """
            Initialise un objet de la classe Triangles avec les données des triangles.

            Paramètre :
            triangles_data (n-d-array) : Un tableau numpy de forme (3, N) représentant les indices
                                        des sommets de chaque triangle. Chaque colonne représente
                                        un triangle et contient trois indices pour les sommets du triangle.
        """
        self.triangles = triangles_data
        self.triangles = self.triangles.astype(int)  # Conversion explicite pour éviter les erreurs
        self.total_of_triangles = triangles_data.shape[1]
        self.triangles_area = None
        self.triangles_center = None
        self.triangles_plus = None
        self.triangles_minus = None

    def filter_triangles(self):
        """
        Filtre les triangles dont la quatrième ligne est > 1.
        """
        if self.triangles.shape[0] < 4:
            raise ValueError("Les données de triangles doivent avoir au moins 4 lignes.")
        # Filtrage des triangles valides en fonction de la quatrième ligne
        valid_indices = np.where(self.triangles[3, :] <= 1)[0]
        self.triangles = self.triangles[:, valid_indices].astype(int)
        self.total_of_triangles = self.triangles.shape[1]

    def calculate_triangles_area_and_center(self, points_data):
        """
            Calcule les aires et les centres de tous les triangles.

            Cette méthode utilise les coordonnées des points du maillage pour calculer l'aire et le centre de chaque triangle.
            L'aire est calculée en utilisant le produit vectoriel des vecteurs formés par les sommets du triangle.
            Le centre est calculé comme la moyenne des coordonnées des trois sommets du triangle.

            Paramètre :
            points_data (Points) : Un objet de la classe Points qui contient les coordonnées des points du maillage.
        """
        if self.triangles_area is None and self.triangles_center is None:
            points = points_data.points
            # Initialisation des tableaux pour les aires et les centres des triangles
            self.triangles_area = np.zeros(self.total_of_triangles)
            self.triangles_center = np.zeros((3, self.total_of_triangles))
            for index_triangle in range(self.total_of_triangles):
                triangle = self.triangles[:3, index_triangle]               # On enregistre l'indice des trois sommets du triangle
                # Vecteurs pour calculer l'aire avec le produit vectoriel
                vecteur_1 = points[:, triangle[0]] - points[:, triangle[1]]
                vecteur_2 = points[:, triangle[2]] - points[:, triangle[1]]
                # Aire du triangle (produit vectoriel divisé par 2)
                self.triangles_area[index_triangle] = np.linalg.norm(np.cross(vecteur_1, vecteur_2)) / 2
                # Centre du triangle (moyenne des coordonnées des sommets)
                self.triangles_center[:, index_triangle] = np.sum(points[:, triangle], axis=1) / 3

    def set_triangles_area_and_center(self, triangles_area, triangles_center):
        """
            Définit les aires et centres des triangles manuellement.

            Paramètres :
                * triangles_area (n-d-array) : Tableau des aires des triangles.
                * triangles_center (n-d-array) : Tableau des centres des triangles.
        """
        self.triangles_area = triangles_area
        self.triangles_center = triangles_center

    def get_edges(self):
        """
            Détecte les arêtes communes entre les triangles et détermine les relations de triangle "plus" et "minus".
            
            Cette méthode analyse les triangles pour trouver les arêtes communes et les classer en paires
            de triangles ayant une arête partagée. Les indices des triangles ayant des arêtes communes sont
            enregistrés dans les tableaux triangles_plus et triangles_minus. 

            Note : enregistre les arrêtes partagées entre plus de deux triangles.

            Retour :
            Edges : Un objet de la classe Edges représentant les arêtes communes entre triangles.
        """
        triangles = self.triangles[:3].T  # (n_triangles, 3)
        edge_dict = defaultdict(list)

        # Générer les arêtes pour chaque triangle
        tri_indices = np.arange(triangles.shape[0])
        edges = np.stack([
            np.sort(triangles[:, [0, 1]], axis=1),
            np.sort(triangles[:, [1, 2]], axis=1),
            np.sort(triangles[:, [2, 0]], axis=1)
        ], axis=1)  # shape (n_triangles, 3, 2)

        # Ajouter chaque arête dans un dictionnaire de la forme (n1, n2) -> [tri1, tri2, ...]
        for tri_idx, tri_edges in zip(tri_indices, edges):
            for edge in tri_edges:
                edge_key = tuple(edge)
                edge_dict[edge_key].append(tri_idx)

        edge_points = []
        triangles_plus = []
        triangles_minus = []

        for edge, tris in edge_dict.items():
            # Pour chaque paire unique de triangles partageant l’arête
            for t1, t2 in combinations(tris, 2):
                edge_points.append(edge)
                triangles_plus.append(t1)
                triangles_minus.append(t2)

        self.triangles_plus = np.array(triangles_plus)
        self.triangles_minus = np.array(triangles_minus)
        edge_array = np.array(edge_points).T  # shape (2, n_edges)

        return Edges(edge_array[0], edge_array[1])

    
    def set_triangles_plus_minus(self, triangles_plus, triangles_minus):
        """
            Définit manuellement les triangles qui partagent des arêtes communes.

            Paramètres :
            * triangles_plus (n-d-array) : Indices des triangles ayant des arêtes communes dans l'ordre "plus".
            * triangles_minus (n-d-array) : Indices des triangles ayant des arêtes communes dans l'ordre "minus".
        """
        self.triangles_plus = triangles_plus
        self.triangles_minus = triangles_minus


class Edges:
    """
        Classe représentant un ensemble d'arêtes dans un maillage 3D.

        Cette classe permet de stocker les arêtes du maillage définies par des paires de points
        et de calculer la longueur de chaque arête.

        Attributs :
            * first_points (n-d-array) : Un tableau numpy contenant les indices des premiers points de chaque arête.
            * second_points (n-d-array) : Un tableau numpy contenant les indices des deuxièmes points de chaque arête.
            * edges_length (n-d-array) : Un tableau contenant la longueur de chaque arête.
            * total_number_of_edges (int) : Le nombre total d'arêtes dans l'ensemble.
    """

    def __init__(self, first_points, second_points):
        """
                Initialise un objet de la classe Edges avec les indices des points définissant les arêtes.

                Paramètres :
                    * first_points (n-d-array) : Tableau contenant les indices des premiers points de chaque arête.
                    * second_points (n-d-array) : Tableau contenant les indices des deuxièmes points de chaque arête.
        """
        self.first_points = first_points
        self.second_points = second_points
        self.edges_length = None
        self.total_number_of_edges = first_points.shape[0]

    def compute_edges_length(self, point_data):
        """
            Calcule les longueurs de toutes les arêtes.

            Cette méthode utilise les coordonnées des points du maillage pour calculer la longueur
            de chaque arête en utilisant la norme euclidienne entre les points définissant l'arête.

            Paramètre :
            point_data (Points) : Un objet de la classe Points contenant les coordonnées des points du maillage.
        """
        points = point_data.points
        edges_length = []
        for edge in range(self.total_number_of_edges):
            # Calcul de la longueur de l'arête en utilisant la norme euclidienne entre les deux points
            edge_length = np.linalg.norm(points[:, self.first_points[edge]] - points[:, self.second_points[edge]])
            edges_length.append(edge_length)
        self.edges_length = np.array(edges_length)       

    def set_edges(self, first_points, second_points):
        """
            Définit manuellement les points qui définissent les arêtes.

            Paramètres :
                * first_points (n-d-array) : Tableau des indices des premiers points de chaque arête.
                * second_points (n-d-array) : Tableau des indices des deuxièmes points de chaque arête.
        """
        self.first_points = first_points
        self.second_points = second_points
        self.total_number_of_edges = first_points.shape[0]

    def set_edge_length(self, edge_length):
        """
            Définit manuellement les longueurs des arêtes.

            Paramètre :
            edge_length (n-d-array) : Tableau des longueurs des arêtes.
        """
        self.edges_length = edge_length


def load_mesh_file(filename, load_from_matlab = True):
    """
        Charge un fichier MAT contenant un maillage et retourne les points et triangles du maillage.

        Cette fonction charge un fichier MATLAB MAT et extrait les données des points et des triangles.
        Si le fichier provient d'un fichier MATLAB, elle ajuste les indices des triangles (qui commencent
        souvent à '1' dans MATLAB) en les convertissant à un format commençant à 0.

        Paramètres :
            * filename (str) : Le nom du fichier .mat à charger.
            * load_from_matlab (bool) : Si True, les indices des triangles seront ajustés pour commencer à 0
                                       (MATLAB commence les indices à 1, mais en Python, ils commencent à 0).

        Retour :
            * points (n-d-array) : Un tableau numpy de forme (3, N) contenant les coordonnées des points du maillage.
            * triangles (n-d-array) : Un tableau numpy de forme (4, M) contenant les indices des sommets des triangles.

        Levée des exceptions :
        FileNotFoundError : Si le fichier n'existe pas.
        RuntimeError : Si une erreur survient lors du chargement du fichier.
        ValueError : Si le fichier ne contient pas les variables 'p' (points) et 't' (triangles).
    """
    try:
        mesh = loadmat(filename)  # Charge le fichier MAT
    except FileNotFoundError:
        raise FileNotFoundError(f"Le fichier '{filename}' est introuvable.")  # Erreur si le fichier n'existe pas
    except Exception as error:
        raise RuntimeError(f"Erreur lors du chargement du fichier : {error}")  # Erreur générale lors du chargement

    # Validation des variables requises dans le fichier MAT
    if 'p' not in mesh or 't' not in mesh:
        raise ValueError("Le fichier doit contenir les variables 'p' (points) et 't' (triangles).")  # Vérification des données

    points = mesh['p']  # Extraire les points du maillage (3 x N)
    triangles = mesh['t']  # Extraire les triangles du maillage (4 x M)

    # Si le fichier provient de MATLAB, ajuster les indices des triangles pour commencer à 0 (au lieu de 1).
    if load_from_matlab:
        triangles[:3] = triangles[:3] - 1  # Les indices dans MATLAB commencent à '1', donc on les convertit à '0'

    return points, triangles  # Retourne les données extraites

def filter_complexes_jonctions(point_data, triangle_data, edge_data):
    """
    Supprime les arêtes issues de jonctions en T : lorsque trois triangles partagent la même arête.
    Cela correspond à des incohérences dans un maillage (non-manifold ou artefacts topologiques).

    Paramètres :
        point_data : données des points (utilisé ici pour recalculer les longueurs d’arêtes après suppression)
        triangle_data : contient les triangles et les liens triangle_plus / triangle_minus
        edge_data : contient les arêtes (first_points, second_points)

    Comportement :
        Supprime les arêtes identiques (dans un sens ou dans l’autre) lorsqu’elles apparaissent trois fois.
    """
    triangles = triangle_data.triangles
    triangles_plus = triangle_data.triangles_plus
    triangles_minus = triangle_data.triangles_minus

    # Représentation des arêtes
    edges = np.vstack((edge_data.first_points, edge_data.second_points))  # shape (2, n_edges)
    edges_inv = edges[::-1, :]  # arêtes inversées

    remove = []

    for i in range(edge_data.total_number_of_edges):
        current_edge = edges[:, i].reshape(2, 1)
        # On répète l’arête courante autant de fois qu’il y a d’arêtes
        repeated_edge = np.tile(current_edge, (1, edges.shape[1]))

        # Cherche les arêtes identiques à current_edge (dans le même sens ou inverse)
        is_same_forward = np.all(edges == repeated_edge, axis=0)
        is_same_backward = np.all(edges_inv == repeated_edge, axis=0)
        same_edges_indices = np.where(is_same_forward | is_same_backward)[0]

        # Si cette arête est identifiée trois fois : jonction en T
        if len(same_edges_indices) == 3:
            # Vérifie que les labels des triangles associés sont identiques
            t_plus_labels = triangles[3, triangles_plus[same_edges_indices]]
            t_minus_labels = triangles[3, triangles_minus[same_edges_indices]]
            same_label = t_plus_labels == t_minus_labels

            # On retire uniquement les arêtes dont les deux triangles ont le même label
            to_remove = same_edges_indices[np.where(same_label)[0]]
            remove.extend(to_remove)

    if remove:
        edges = np.delete(edges, remove, axis=1)
        triangles_plus = np.delete(triangles_plus, remove)
        triangles_minus = np.delete(triangles_minus, remove)

        edge_data.set_edges(edges[0], edges[1])
        triangle_data.set_triangles_plus_minus(triangles_plus, triangles_minus)
        edge_data.compute_edges_length(point_data)

        # print(f"Suppression de {len(remove)} jonctions en T.")
    """ else:
        print("Aucune jonction complexe trouvée.") """


class DataManager_rwg1:
    @staticmethod
    def save_data(filename, save_folder_name, points_data, triangles_data, edges_data):
        """
            Sauvegarde les données (points, triangles, arêtes, etc.) dans un fichier MAT

            Cette méthode prend les données du maillage, les organise dans un dictionnaire,
            puis les sauvegarde sous forme de fichier MAT dans le dossier spécifié.

            Paramètres :
                * filename (str) : Le nom du fichier d'origine (utilisé pour générer le nom de sauvegarde).
                * save_folder_name (str) : Le dossier ou le fichier de sauvegarde sera stocké.
                * points_data (Points) : Un objet de la classe Points contenant les données des points du maillage.
                * triangles_data (Triangles) : Un objet de la classe Triangles contenant les données des triangles du maillage.
                * edges_data (Edges) : Un objet de la classe Edges contenant les données des arêtes du maillage.

            Retourne :
            str : Le nom du fichier sauvegardé.
        """
        mesh = loadmat(filename)  # Charge le fichier MAT

        # Crée un dictionnaire contenant toutes les données à sauvegarder
        data = {
            'points' : points_data.points,
            'triangles' : triangles_data.triangles,
            'edge_first_points' : edges_data.first_points,
            'edge_second_points' : edges_data.second_points,
            'triangles_plus' : triangles_data.triangles_plus,
            'triangles_minus' : triangles_data.triangles_minus,
            'edges_length' : edges_data.edges_length,
            'triangles_area' : triangles_data.triangles_area,
            'triangles_center' : triangles_data.triangles_center
        }

        # Génère le nom du fichier de sauvegarde à partir du nom d'origine
        base_name = os.path.splitext(os.path.basename(filename))[0]  # Retire l'extension du fichier original
        save_file_name = base_name + '_mesh1.mat'
        full_save_path = os.path.join(save_folder_name, save_file_name)  # Chemin complet pour la sauvegarde

        # Vérifie si le dossier existe, sinon crée le dossier
        if not os.path.exists(save_folder_name): # Vérification et création du dossier si nécessaire
            os.makedirs(save_folder_name)
            # print(f"Directory '{save_folder_name}' created.")

        # Sauvegarde les données dans le fichier MAT
        savemat(full_save_path, data)
        # print(f"Data saved successfully to {full_save_path}")

        # Retourne le nom du fichier sauvegardé
        return save_file_name

    @staticmethod
    def load_data(filename):
        """
            Charge les données d'un fichier MAT et retourne les objets Points, Triangles et Edges correspondants.

            Cette méthode charge les données du fichier MAT, les décompresse et crée des objets
            pour les points, les triangles et les arêtes, puis les initialise avec les données chargées.

            Paramètres :
            filename (str) : Le nom du fichier à charger.

            Retourne :
            tuple : Un tuple contenant les objets Points, Triangles, et Edges.
        """
        try:
            # Vérifier si le fichier existe avant de le charger
            if not os.path.isfile(filename):
                raise FileNotFoundError(f"File '{filename}' does not exist.")

            # Charge le fichier .mat
            data = loadmat(filename)

            # Lors du chargement des données, les dimensions peuvent être compressées (avec 'squeeze')
            # On crée les objets en passant les données extraites.
            points = Points(points_data=data['points'].squeeze())
            triangles = Triangles(triangles_data=data['triangles'].squeeze())
            edges = Edges(first_points=data['edge_first_points'].squeeze(), second_points=data['edge_second_points'].squeeze())
            triangles.set_triangles_plus_minus(triangles_plus=data['triangles_plus'].squeeze(), triangles_minus=data['triangles_minus'].squeeze())
            triangles.set_triangles_area_and_center(triangles_area=data['triangles_area'].squeeze(), triangles_center=data['triangles_center'].squeeze())
            edges.set_edge_length(edge_length=data['edges_length'].squeeze())

            # Affiche un message confirmant le succès du chargement des données
            # print(f"Data loaded from {filename}")

            # Retourne les objets créés
            return points, triangles, edges

        except FileNotFoundError as e:
            # Gère l'exception si le fichier n'existe pas
            print(f"Error: {e}")
        except KeyError as e:
            # Gère l'exception si une clé est manquante dans les données du fichier
            print(f"Key Error: {e}")
        except ValueError as e:
            # Gère les erreurs de données malformées ou incorrectes
            print(f"Value Error (likely malformed data): {e}")
        except Exception as e:
            # Gère toute autre exception non prévue
            print(f"An unexpected error occurred: {e}")