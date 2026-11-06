import numpy as np
from scipy.io import savemat, loadmat

from solver.rwg.rwg1 import Points, Triangles, Edges
from solver.utils.mesh_io import MeshDataError, load_matfile_or_raise


class Vecteurs_Rho:
    """
    Class to calculate and manage the Rho vectors associated with the "plus" and "minus" triangles of edges in a mesh.

    The Rho vectors represent vectors connecting a specific point of a triangle (opposite to the considered edge)
    to its geometric center.
    """

    def __init__(self):
        """
        Initializes the attributes to store the Rho vectors.
        """
        self.vecteur_rho_plus = None
        self.vecteur_rho_minus = None

    def calculate_vecteurs_rho(self, points_data, triangles_data, edges_data):
        points = points_data.points  # (3, N_points)
        triangles = triangles_data.triangles  # (3, N_triangles)
        triangles_center = triangles_data.triangles_center  # (3, N_triangles)
        triangles_plus = triangles_data.triangles_plus  # (N_edges,)
        triangles_minus = triangles_data.triangles_minus  # (N_edges,)

        edges_first_points = edges_data.first_points  # (N_edges,)
        edges_second_points = edges_data.second_points  # (N_edges,)
        total_number_of_edges = edges_data.total_number_of_edges  # int

        self.vecteur_rho_plus = np.zeros((3, total_number_of_edges))
        self.vecteur_rho_minus = np.zeros((3, total_number_of_edges))

        # --- Vectorized processing for “plus” triangles ---
        triangles_plus_sommets = triangles[:, triangles_plus]  # (3, N_edges)

        # Detection of the opposite vertex
        mask_plus = (triangles_plus_sommets != edges_first_points) & (triangles_plus_sommets != edges_second_points)  # (3, N_edges)
        # For each edge, find the index of the opposite vertex
        indices_opposes_plus = np.argmax(mask_plus, axis=0)  # (N_edges,)
        index_point_vecteur_plus = triangles_plus_sommets[indices_opposes_plus, np.arange(total_number_of_edges)]  # (N_edges,)
        point_vecteurs_plus = points[:, index_point_vecteur_plus]  # (3, N_edges)

        # Calculation of Rho “plus” vectors
        self.vecteur_rho_plus = triangles_center[:, triangles_plus] - point_vecteurs_plus

        # --- Vectorized processing for “minus” triangles ---
        triangles_minus_sommets = triangles[:, triangles_minus]  # (3, N_edges)

        mask_minus = (triangles_minus_sommets != edges_first_points) & (triangles_minus_sommets != edges_second_points)
        indices_opposes_minus = np.argmax(mask_minus, axis=0)
        index_point_vecteur_minus = triangles_minus_sommets[indices_opposes_minus, np.arange(total_number_of_edges)]
        point_vecteurs_minus = points[:, index_point_vecteur_minus]

        # Calculation of Rho “minus” vectors
        self.vecteur_rho_minus = point_vecteurs_minus - triangles_center[:, triangles_minus]

    def set_vecteurs_rho(self, vecteur_rho_plus, vecteur_rho_minus):
        """
            Manually sets the Rho vectors.

            Parameters:
                * vecteur_rho_plus (n-d-array) : Rho vectors for the "plus" triangles.
                * vecteur_rho_minus (n-d-array) : Rho vectors for the "minus" triangles.
        """
        self.vecteur_rho_plus = vecteur_rho_plus
        self.vecteur_rho_minus = vecteur_rho_minus


class DataManager_rwg2:
    """
        Class to save and load mesh-related data and its properties in MAT files.

        Provides static methods to:
            * Save enriched data into a MAT file.
            * Load data from an existing MAT file.
    """
    @staticmethod
    def save_data(path, vecteurs_rho_data):
        """
            Saves the data into a MAT file after enriching it.

            Parameters:
                * path (Path) : Path object containing the file path information.
                * vecteurs_rho_data (Vecteurs_Rho) : Rho vectors data.

            Returns:
                save_file_name (str) : Name of the saved MAT file.
        """
        # Load the initial data
        data = loadmat(path.mat_mesh1)

        # Add the new data
        new_data = {
            'vecteur_rho_plus' : vecteurs_rho_data.vecteur_rho_plus,
            'vecteur_rho_minus' : vecteurs_rho_data.vecteur_rho_minus,
        }
        data.update(new_data)

        # Save the data into the MAT file
        savemat(path.mat_mesh2, data)

    @staticmethod
    def load_data(filename):
        """
            Loads data from a MAT file and initializes the associated objects.

            Parameters:
                filename (str) : Path to the MAT file containing the data.

            Returns:
                (tuple) : Contains the objects Points, Triangles, Edges, and Vecteurs_Rho.

            Exceptions:
                * FileNotFoundError : If the file does not exist.
                * KeyError : If an expected key is missing in the data.
                * ValueError : If the data is malformed.
        """
        data = load_matfile_or_raise(filename)

        try:
            # Initialize objects with the loaded data
            points = Points(points_data=data['points'].squeeze())
            triangles = Triangles(triangles_data=data['triangles'].squeeze())
            edges = Edges(first_points=data['edge_first_points'].squeeze(), second_points=data['edge_second_points'].squeeze())
            triangles.set_triangles_plus_minus(triangles_plus=data['triangles_plus'].squeeze(), triangles_minus=data['triangles_minus'].squeeze())
            triangles.set_triangles_area_and_center(triangles_area=data['triangles_area'].squeeze(), triangles_center=data['triangles_center'].squeeze())
            edges.set_edge_length(edge_length=data['edges_length'].squeeze())
            vecteurs_rho = Vecteurs_Rho()
            vecteurs_rho.set_vecteurs_rho(
                vecteur_rho_plus=data['vecteur_rho_plus'].squeeze(),
                vecteur_rho_minus=data['vecteur_rho_minus'].squeeze(),
            )
            return points, triangles, edges, vecteurs_rho

        except KeyError as e:
            raise MeshDataError(
                f"'{filename}' is missing expected key {e}. It may have been "
                f"generated by an older/incompatible version of the pipeline."
            ) from e
        except ValueError as e:
            raise MeshDataError(f"'{filename}' contains malformed data: {e}") from e