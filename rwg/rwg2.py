import os
import numpy as np
from scipy.io import savemat, loadmat

from rwg.rwg1 import Points, Triangles, Edges


class Barycentric_triangle:
    """
    Class to calculate and store barycentric centers of a triangular mesh.

    This class uses the vertex coordinates of triangles and their geometric centers to
    compute the barycentric centers associated with subdividing each triangle into nine sub-triangles.
    """

    def __init__(self):
        """
        Initializes the object with an attribute to store barycentric centers.
        """
        self.barycentric_triangle_center = None
    
    def calculate_barycentric_center(self, point_data, triangles_data):
        """
        Calculates the barycentric centers for all triangles in the mesh.

        Parameters:
            * point_data: Points object containing the coordinates of the mesh points.
            * triangles_data: Triangles object containing the triangle indices and their geometric centers.

        This method computes points at fractions 1/3 and 2/3 along the triangle edges,
        and then calculates the nine barycentric centers for each triangle, stacking
        the results in a 3×9×N array.
        """
        points = point_data.points                             # (3, M)
        triangles = triangles_data.triangles                   # (3, N)
        triangles_center = triangles_data.triangles_center     # (3, N)
        total_of_triangles = triangles_data.total_of_triangles # N

        # Retrieve vertex points for all triangles
        pt1 = points[:, triangles[0]]  # (3, N)
        pt2 = points[:, triangles[1]]  # (3, N)
        pt3 = points[:, triangles[2]]  # (3, N)

        # Compute edge vectors
        v12 = pt2 - pt1  # (3, N)
        v23 = pt3 - pt2  # (3, N)
        v13 = pt3 - pt1  # (3, N)

        # Points at fractions 1/3 and 2/3 along each edge
        pt12_1 = pt1 + (1/3) * v12
        pt12_2 = pt1 + (2/3) * v12
        pt23_1 = pt2 + (1/3) * v23
        pt23_2 = pt2 + (2/3) * v23
        pt13_1 = pt1 + (1/3) * v13
        pt13_2 = pt1 + (2/3) * v13

        c = triangles_center  # shorter alias

        # Compute the 9 barycentric centers
        bary_1 = (pt12_1 + pt13_1 + pt1) / 3
        bary_2 = (pt12_1 + pt12_2 + c) / 3
        bary_3 = (pt12_2 + pt23_1 + pt2) / 3
        bary_4 = (pt12_2 + pt23_1 + c) / 3
        bary_5 = (pt23_1 + pt23_2 + c) / 3
        bary_6 = (pt12_1 + pt13_1 + c) / 3
        bary_7 = (pt13_1 + pt13_2 + c) / 3
        bary_8 = (pt23_2 + pt13_2 + c) / 3
        bary_9 = (pt23_2 + pt13_2 + pt3) / 3

        # Stack into the final array (3, 9, N)
        self.barycentric_triangle_center = np.stack([
            bary_1, bary_2, bary_3, bary_4, bary_5,
            bary_6, bary_7, bary_8, bary_9
        ], axis=1)  # shape (3, 9, N)

    def set_barycentric_center(self, barycentric_triangle_center):
        """
        Manually sets the barycentric centers.

        Parameters:
            barycentric_triangle_center (n-d-array): 3 x 9 x N array containing the barycentric centers to set.
        """
        self.barycentric_triangle_center = barycentric_triangle_center


class Vecteurs_Rho:
    """
    Class to calculate and manage the Rho vectors associated with the "plus" and "minus" triangles of edges in a mesh.

    The Rho vectors represent vectors connecting a specific point of a triangle (opposite to the considered edge)
    to its geometric center or to its barycentric centers.
    """

    def __init__(self):
        """
        Initializes the attributes to store the Rho vectors.
        """
        self.vecteur_rho_plus = None
        self.vecteur_rho_minus = None
        self.vecteur_rho_barycentric_plus = None
        self.vecteur_rho_barycentric_minus = None
    
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

        # --- Vectorized processing for “plus” triangles ---
        triangles_plus_sommets = triangles[:, triangles_plus]  # (3, N_edges)
        edges_fp = edges_first_points
        edges_sp = edges_second_points

        # Detection of the opposite vertex
        mask_plus = (triangles_plus_sommets != edges_fp) & (triangles_plus_sommets != edges_sp)  # (3, N_edges)
        # For each edge, find the index of the opposite vertex
        indices_opposes_plus = np.argmax(mask_plus, axis=0)  # (N_edges,)
        index_point_vecteur_plus = triangles_plus_sommets[indices_opposes_plus, np.arange(total_number_of_edges)]  # (N_edges,)
        point_vecteurs_plus = points[:, index_point_vecteur_plus]  # (3, N_edges)

        # Calculation of Rho “plus” vectors
        self.vecteur_rho_plus = triangles_center[:, triangles_plus] - point_vecteurs_plus
        self.vecteur_rho_barycentric_plus = barycentric_triangle_center[:, :, triangles_plus] - point_vecteurs_plus[:, None, :]

        # --- Vectorized processing for “minus” triangles ---
        triangles_minus_sommets = triangles[:, triangles_minus]  # (3, N_edges)

        mask_minus = (triangles_minus_sommets != edges_fp) & (triangles_minus_sommets != edges_sp)
        indices_opposes_minus = np.argmax(mask_minus, axis=0)
        index_point_vecteur_minus = triangles_minus_sommets[indices_opposes_minus, np.arange(total_number_of_edges)]
        point_vecteurs_minus = points[:, index_point_vecteur_minus]

        # Calculation of Rho “minus” vectors
        self.vecteur_rho_minus = point_vecteurs_minus - triangles_center[:, triangles_minus]
        self.vecteur_rho_barycentric_minus = point_vecteurs_minus[:, None, :] - barycentric_triangle_center[:, :, triangles_minus]

    def set_vecteurs_rho(self, vecteur_rho_plus, vecteur_rho_minus, vecteur_rho_barycentric_plus, vecteur_rho_barycentric_minus):
        """
            Manually sets the Rho vectors.

            Parameters:
                * vecteur_rho_plus (n-d-array) : Rho vectors for the "plus" triangles.
                * vecteur_rho_minus (n-d-array) : Rho vectors for the "minus" triangles.
                * vecteur_rho_barycentric_plus (n-d-array) : Barycentric vectors for the "plus" triangles.
                * vecteur_rho_barycentric_minus (n-d-array) : Barycentric vectors for the "minus" triangles.
        """
        self.vecteur_rho_plus = vecteur_rho_plus
        self.vecteur_rho_minus = vecteur_rho_minus
        self.vecteur_rho_barycentric_plus = vecteur_rho_barycentric_plus
        self.vecteur_rho_barycentric_minus = vecteur_rho_barycentric_minus


class DataManager_rwg2:
    """
        Class to save and load mesh-related data and its properties in MAT files.

        Provides static methods to:
            * Save enriched data into a MAT file.
            * Load data from an existing MAT file.
    """
    @staticmethod
    def save_data(filename_mesh1, save_folder_name, barycentric_triangle_data, vecteurs_rho_data):
        """
            Saves the data into a MAT file after enriching it.

            Parameters:
                * filename_mesh1 (str) : Path to the initial MAT file containing mesh data.
                * save_folder_name (str) : Name of the folder where the enriched file will be saved.
                * barycentric_triangle_data (Barycentric_triangle) : Barycentric triangle data.
                * vecteurs_rho_data (Vecteurs_Rho) : Rho vectors data.

            Returns:
                save_file_name (str) : Name of the saved MAT file.
        """
        # Load the initial data
        data = loadmat(filename_mesh1)

        # Add the new data
        new_data = {
            'barycentric_triangle_center' : barycentric_triangle_data.barycentric_triangle_center,
            'vecteur_rho_plus' : vecteurs_rho_data.vecteur_rho_plus,
            'vecteur_rho_minus' : vecteurs_rho_data.vecteur_rho_minus,
            'vecteur_rho_barycentric_plus' : vecteurs_rho_data.vecteur_rho_barycentric_plus,
            'vecteur_rho_barycentric_minus' : vecteurs_rho_data.vecteur_rho_barycentric_plus,
        }
        data.update(new_data)

        # Generate the save file name
        base_name = os.path.splitext(os.path.basename(filename_mesh1))[0]
        base_name = base_name.replace('_mesh1', '')  # Remove '_mesh1'
        save_file_name = base_name + '_mesh2.mat'    # Add '_mesh2'
        full_save_path = os.path.join(save_folder_name, save_file_name)

        # Create folder if needed
        if not os.path.exists(save_folder_name):
            os.makedirs(save_folder_name)

        # Save the data into the MAT file
        savemat(full_save_path, data)
        return save_file_name

    @staticmethod
    def load_data(filename):
        """
            Loads data from a MAT file and initializes the associated objects.

            Parameters:
                filename (str) : Path to the MAT file containing the data.

            Returns:
                (tuple) : Contains the objects Points, Triangles, Edges, Barycentric_triangle, and Vecteurs_Rho.

            Exceptions:
                * FileNotFoundError : If the file does not exist.
                * KeyError : If an expected key is missing in the data.
                * ValueError : If the data is malformed.
        """
        try:
            # Check if the file exists
            if not os.path.isfile(filename):
                raise FileNotFoundError(f"File '{filename}' does not exist.")

            # Load the data
            data = loadmat(filename)

            # Initialize objects with the loaded data
            points = Points(points_data=data['points'].squeeze())
            triangles = Triangles(triangles_data=data['triangles'].squeeze())
            edges = Edges(first_points=data['edge_first_points'].squeeze(), second_points=data['edge_second_points'].squeeze())
            triangles.set_triangles_plus_minus(triangles_plus=data['triangles_plus'].squeeze(), triangles_minus=data['triangles_minus'].squeeze())
            triangles.set_triangles_area_and_center(triangles_area=data['triangles_area'].squeeze(), triangles_center=data['triangles_center'].squeeze())
            edges.set_edge_length(edge_length=data['edges_length'].squeeze())
            barycentric_triangle = Barycentric_triangle()
            barycentric_triangle.set_barycentric_center(barycentric_triangle_center=data['barycentric_triangle_center'].squeeze())
            vecteurs_rho = Vecteurs_Rho()
            vecteurs_rho.set_vecteurs_rho(
                vecteur_rho_plus=data['vecteur_rho_plus'].squeeze(),
                vecteur_rho_minus=data['vecteur_rho_minus'].squeeze(),
                vecteur_rho_barycentric_plus=data['vecteur_rho_barycentric_plus'].squeeze(),
                vecteur_rho_barycentric_minus=data['vecteur_rho_barycentric_minus'].squeeze()
            )
            return points, triangles, edges, barycentric_triangle, vecteurs_rho

        except FileNotFoundError as e:
            print(f"Error: {e}")
        except KeyError as e:
            print(f"Key Error: {e}")
        except ValueError as e:
            print(f"Value Error (likely malformed data): {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")