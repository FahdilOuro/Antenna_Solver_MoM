import os
import numpy as np
from itertools import combinations
from collections import defaultdict
from scipy.io import savemat, loadmat


class Points:
    """
        Class representing a set of points in a three-dimensional space.

        This class stores 3D points and provides information
        about the dimensions (length, width, height) of the space occupied by these points.

        Attributes:
            * points (ndarray): A numpy array containing the coordinates of the points in shape (3, N),
                                where N is the total number of points in 3D space.
            * total_of_points (int): The total number of points in the set.
            * length (float): The length (maximum dimension along the X axis) of the point set.
            * width (float): The width (maximum dimension along the Y axis) of the point set.
            * height (float): The height (maximum dimension along the Z axis) of the point set.
    """

    def __init__(self, points_data):
        """
            Initialize a Points object with coordinate data.

            Parameters:
                points_data (ndarray): A numpy array of shape (3, N) representing the coordinates
                                       of N points in 3D space. The first row contains
                                       the X coordinates, the second row the Y coordinates,
                                       and the third row the Z coordinates of the points.
        """
        self.points = points_data
        # The total number of points is defined by the number of columns in the array (points_data).
        self.total_of_points = self.points.shape[1]

        # Compute the dimensions of the space occupied by the points
        self.length = max(points_data[0]) - min(points_data[0])  # Difference between max and min values along the X axis
        self.width = max(points_data[1]) - min(points_data[1])   # Difference between max and min values along the Y axis
        self.height = max(points_data[2]) - min(points_data[2])  # Difference between max and min values along the Z axis

    def get_point_coordinates(self, index):
        """
        Retrieve the coordinates of a point given its index.

        Parameters:
            index (int): Index of the point whose coordinates must be returned.

        Returns:
            np.ndarray: A 1D array containing the coordinates (X, Y, Z) of the specified point.
        """
        return self.points[:, index]


class Triangles:
    """
        Class representing a set of triangles in a 3D mesh.

        This class stores triangles, computes their geometric properties
        (areas, centers), and detects common edges between triangles.

        Attributes:
            * triangles (n-d-array): A numpy array of shape (3, N) representing the vertex indices
                                     of each triangle in 3D space.
            * total_of_triangles (int): The total number of triangles in the set.
            * triangles_area (n-d-array): An array containing the areas of each triangle.
            * triangles_center (n-d-array): An array containing the coordinates of each triangle’s center.
            * triangles_plus (n-d-array): An array containing the indices of triangles that share a common edge.
            * triangles_minus (n-d-array): An array containing the indices of triangles that share a common edge.
    """

    def __init__(self, triangles_data):
        """
            Initialize a Triangles object with triangle data.

            Parameter:
            triangles_data (n-d-array): A numpy array of shape (3, N) representing the vertex indices
                                        of each triangle. Each column represents one triangle
                                        and contains three indices for its vertices.
        """
        self.triangles = triangles_data
        self.triangles = self.triangles.astype(int)          # Explicit conversion to avoid errors
        self.total_of_triangles = triangles_data.shape[1]
        self.triangles_area = None
        self.triangles_center = None
        self.triangles_plus = None
        self.triangles_minus = None

    def filter_triangles(self):
        """
        Filter triangles where the fourth row is > 1.
        """
        if self.triangles.shape[0] < 4:
            raise ValueError("Triangle data must have at least 4 rows.")
        # Filter valid triangles based on the fourth row
        valid_indices = np.where(self.triangles[3, :] <= 1)[0]
        self.triangles = self.triangles[:, valid_indices].astype(int)
        self.total_of_triangles = self.triangles.shape[1]

    def calculate_triangles_area_and_center(self, points_data):
        """
            Compute the areas and centers of all triangles.

            This method uses the mesh point coordinates to compute the area and the center of each triangle.
            The area is calculated using the cross product of the vectors formed by the triangle’s vertices.
            The center is computed as the average of the coordinates of the three vertices.

            Parameter:
            points_data (Points): An object of the Points class containing the mesh point coordinates.
        """
        if self.triangles_area is None and self.triangles_center is None:
            points = points_data.points
            # Initialize arrays for triangle areas and centers
            self.triangles_area = np.zeros(self.total_of_triangles)
            self.triangles_center = np.zeros((3, self.total_of_triangles))
            for index_triangle in range(self.total_of_triangles):
                triangle = self.triangles[:3, index_triangle]               # Indices of the three vertices of the triangle
                # Vectors to compute the area using the cross product
                vecteur_1 = points[:, triangle[0]] - points[:, triangle[1]]
                vecteur_2 = points[:, triangle[2]] - points[:, triangle[1]]
                # Triangle area (cross product norm divided by 2)
                self.triangles_area[index_triangle] = np.linalg.norm(np.cross(vecteur_1, vecteur_2)) / 2
                # Triangle center (mean of the vertex coordinates)
                self.triangles_center[:, index_triangle] = np.sum(points[:, triangle], axis=1) / 3

    def set_triangles_area_and_center(self, triangles_area, triangles_center):
        """
            Manually set the areas and centers of triangles.

            Parameters:
                * triangles_area (n-d-array): Array of triangle areas.
                * triangles_center (n-d-array): Array of triangle centers.
        """
        self.triangles_area = triangles_area
        self.triangles_center = triangles_center

    def get_edges(self):
        """
            Detect common edges between triangles and determine the "plus" and "minus" triangle relations.
            
            This method analyzes triangles to find common edges and classify them into pairs
            of triangles sharing an edge. The indices of the triangles with common edges are
            stored in the arrays triangles_plus and triangles_minus. 

            Note: records edges shared between more than two triangles.

            Return:
            Edges: An object of the Edges class representing the common edges between triangles.
        """
        triangles = self.triangles[:3].T  # (n_triangles, 3)
        edge_dict = defaultdict(list)

        # Generate edges for each triangle
        tri_indices = np.arange(triangles.shape[0])
        edges = np.stack([
            np.sort(triangles[:, [0, 1]], axis=1),
            np.sort(triangles[:, [1, 2]], axis=1),
            np.sort(triangles[:, [2, 0]], axis=1)
        ], axis=1)  # shape (n_triangles, 3, 2)

        # Add each edge to a dictionary of the form (n1, n2) -> [tri1, tri2, ...]
        for tri_idx, tri_edges in zip(tri_indices, edges):
            for edge in tri_edges:
                edge_key = tuple(edge)
                edge_dict[edge_key].append(tri_idx)

        edge_points = []
        triangles_plus = []
        triangles_minus = []

        for edge, tris in edge_dict.items():
            # For each unique pair of triangles sharing the edge
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
            Manually set the triangles that share common edges.

            Parameters:
            * triangles_plus (n-d-array): Indices of the triangles sharing common edges in the "plus" order.
            * triangles_minus (n-d-array): Indices of the triangles sharing common edges in the "minus" order.
        """
        self.triangles_plus = triangles_plus
        self.triangles_minus = triangles_minus


class Edges:
    """
        Class representing a set of edges in a 3D mesh.

        This class stores the edges of the mesh defined by pairs of points
        and computes the length of each edge.

        Attributes:
            * first_points (n-d-array): A numpy array containing the indices of the first points of each edge.
            * second_points (n-d-array): A numpy array containing the indices of the second points of each edge.
            * edges_length (n-d-array): An array containing the length of each edge.
            * total_number_of_edges (int): The total number of edges in the set.
    """

    def __init__(self, first_points, second_points):
        """
            Initialize an Edges object with the indices of the points defining the edges.

            Parameters:
                * first_points (n-d-array): Array containing the indices of the first points of each edge.
                * second_points (n-d-array): Array containing the indices of the second points of each edge.
        """
        self.first_points = first_points
        self.second_points = second_points
        self.edges_length = None
        self.total_number_of_edges = first_points.shape[0]

    def compute_edges_length(self, point_data):
        """
            Compute the lengths of all edges.

            This method uses the coordinates of the mesh points to calculate the length
            of each edge using the Euclidean norm between the two points defining the edge.

            Parameter:
            point_data (Points): A Points object containing the coordinates of the mesh points.
        """
        points = point_data.points
        edges_length = []
        for edge in range(self.total_number_of_edges):
            # Compute the edge length using the Euclidean norm between the two points
            edge_length = np.linalg.norm(points[:, self.first_points[edge]] - points[:, self.second_points[edge]])
            edges_length.append(edge_length)
        self.edges_length = np.array(edges_length)       

    def set_edges(self, first_points, second_points):
        """
            Manually set the points defining the edges.

            Parameters:
                * first_points (n-d-array): Array of indices of the first points of each edge.
                * second_points (n-d-array): Array of indices of the second points of each edge.
        """
        self.first_points = first_points
        self.second_points = second_points
        self.total_number_of_edges = first_points.shape[0]

    def set_edge_length(self, edge_length):
        """
            Manually set the lengths of the edges.

            Parameter:
            edge_length (n-d-array): Array of edge lengths.
        """
        self.edges_length = edge_length


def load_mesh_file(filename, load_from_matlab=True):
    """
        Load a MAT file containing a mesh and return the mesh points and triangles.

        This function loads a MATLAB MAT file and extracts the point and triangle data.
        If the file comes from MATLAB, it adjusts the triangle indices (which often start at '1' in MATLAB)
        by converting them to a zero-based format.

        Parameters:
            * filename (str): The name of the .mat file to load.
            * load_from_matlab (bool): If True, the triangle indices will be adjusted to start at 0
                                       (MATLAB indices start at 1, but Python uses 0-based indexing).

        Returns:
            * points (n-d-array): A numpy array of shape (3, N) containing the mesh point coordinates.
            * triangles (n-d-array): A numpy array of shape (4, M) containing the indices of the triangle vertices.

        Exceptions raised:
            FileNotFoundError: If the file does not exist.
            RuntimeError: If an error occurs while loading the file.
            ValueError: If the file does not contain the variables 'p' (points) and 't' (triangles).
    """
    try:
        mesh = loadmat(filename)  # Load the MAT file
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{filename}' was not found.")  # Error if file does not exist
    except Exception as error:
        raise RuntimeError(f"Error while loading the file: {error}")  # General error during loading

    # Validate required variables in the MAT file
    if 'p' not in mesh or 't' not in mesh:
        raise ValueError("The file must contain the variables 'p' (points) and 't' (triangles).")  # Data check

    points = mesh['p']  # Extract mesh points (3 x N)
    triangles = mesh['t']  # Extract mesh triangles (4 x M)

    # If the file comes from MATLAB, adjust indices to start from 0 (instead of 1).
    if load_from_matlab:
        triangles[:3] = triangles[:3] - 1  # MATLAB indices start at '1', so we convert them to '0'

    return points, triangles  # Return extracted data

def filter_complexes_jonctions(point_data, triangle_data, edge_data):
    """
    Remove edges from T-junctions: when three triangles share the same edge.
    This corresponds to inconsistencies in a mesh (non-manifold geometry or topological artifacts).

    Parameters:
        point_data : point data (used here to recompute edge lengths after removal)
        triangle_data : contains triangles and their triangle_plus / triangle_minus relations
        edge_data : contains edges (first_points, second_points)

    Behavior:
        Removes identical edges (regardless of orientation) when they appear three times.
    """
    triangles = triangle_data.triangles
    triangles_plus = triangle_data.triangles_plus
    triangles_minus = triangle_data.triangles_minus

    # Edge representation
    edges = np.vstack((edge_data.first_points, edge_data.second_points))  # shape (2, n_edges)
    edges_inv = edges[::-1, :]  # reversed edges

    remove = []

    for i in range(edge_data.total_number_of_edges):
        current_edge = edges[:, i].reshape(2, 1)
        # Repeat current_edge as many times as there are edges
        repeated_edge = np.tile(current_edge, (1, edges.shape[1]))

        # Find edges identical to current_edge (same or reversed orientation)
        is_same_forward = np.all(edges == repeated_edge, axis=0)
        is_same_backward = np.all(edges_inv == repeated_edge, axis=0)
        same_edges_indices = np.where(is_same_forward | is_same_backward)[0]

        # If this edge is identified three times: T-junction
        if len(same_edges_indices) == 3:
            # Check if the labels of associated triangles are identical
            t_plus_labels = triangles[3, triangles_plus[same_edges_indices]]
            t_minus_labels = triangles[3, triangles_minus[same_edges_indices]]
            same_label = t_plus_labels == t_minus_labels

            # Remove only edges whose two triangles have the same label
            to_remove = same_edges_indices[np.where(same_label)[0]]
            remove.extend(to_remove)

    if remove:
        edges = np.delete(edges, remove, axis=1)
        triangles_plus = np.delete(triangles_plus, remove)
        triangles_minus = np.delete(triangles_minus, remove)

        edge_data.set_edges(edges[0], edges[1])
        triangle_data.set_triangles_plus_minus(triangles_plus, triangles_minus)
        edge_data.compute_edges_length(point_data)

        # print(f"Removed {len(remove)} T-junctions.")
    """ else:
        print("No complex junctions found.") """


class DataManager_rwg1:
    @staticmethod
    def save_data(filename, save_folder_name, points_data, triangles_data, edges_data):
        """
        Saves mesh data (points, triangles, edges, etc.) into a MAT file.

        This method takes the mesh data, organizes it into a dictionary,
        and saves it as a MAT file in the specified folder.

        Parameters:
            * filename (str): The original file name (used to generate the save file name).
            * save_folder_name (str): The folder where the save file will be stored.
            * points_data (Points): A Points object containing the mesh point data.
            * triangles_data (Triangles): A Triangles object containing the mesh triangle data.
            * edges_data (Edges): An Edges object containing the mesh edge data.

        Returns:
            str: The name of the saved file.
        """
        mesh = loadmat(filename)  # Load the MAT file

        # Create a dictionary containing all data to save
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

        # Generate the save file name based on the original file name
        base_name = os.path.splitext(os.path.basename(filename))[0]  # Remove original file extension
        save_file_name = base_name + '_mesh1.mat'
        full_save_path = os.path.join(save_folder_name, save_file_name)  # Full path to save

        # Check if the folder exists, otherwise create it
        if not os.path.exists(save_folder_name):
            os.makedirs(save_folder_name)

        # Save data to MAT file
        savemat(full_save_path, data)

        # Return the name of the saved file
        return save_file_name

    @staticmethod
    def load_data(filename):
        """
        Loads mesh data from a MAT file and returns corresponding Points, Triangles, and Edges objects.

        This method loads data from the MAT file, decompresses it, creates objects
        for points, triangles, and edges, and initializes them with the loaded data.

        Parameters:
            filename (str): The file name to load.

        Returns:
            tuple: A tuple containing Points, Triangles, and Edges objects.
        """
        try:
            # Check if the file exists before loading
            if not os.path.isfile(filename):
                raise FileNotFoundError(f"File '{filename}' does not exist.")

            # Load the .mat file
            data = loadmat(filename)

            # Squeeze dimensions if needed and create objects
            points = Points(points_data=data['points'].squeeze())
            triangles = Triangles(triangles_data=data['triangles'].squeeze())
            edges = Edges(first_points=data['edge_first_points'].squeeze(),
                          second_points=data['edge_second_points'].squeeze())
            triangles.set_triangles_plus_minus(triangles_plus=data['triangles_plus'].squeeze(),
                                               triangles_minus=data['triangles_minus'].squeeze())
            triangles.set_triangles_area_and_center(triangles_area=data['triangles_area'].squeeze(),
                                                    triangles_center=data['triangles_center'].squeeze())
            edges.set_edge_length(edge_length=data['edges_length'].squeeze())

            # Return the created objects
            return points, triangles, edges

        except FileNotFoundError as e:
            print(f"Error: {e}")
        except KeyError as e:
            print(f"Key Error: {e}")
        except ValueError as e:
            print(f"Value Error (likely malformed data): {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")