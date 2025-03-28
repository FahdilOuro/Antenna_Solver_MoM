import os
import numpy as np


def get_selected_edges(edge_obj, selected_indices):
    """
    Récupère les arêtes correspondant aux indices sélectionnés, sous forme de couples (nœud1, nœud2).
    """
    # Extraire les arêtes sous forme de tableau 2xN
    edges = np.vstack((edge_obj.first_points, edge_obj.second_points))

    # Garder les arêtes sélectionnées en couple sans suppression des doublons
    selected_edges = edges[:, selected_indices].T  # On transpose pour avoir une liste de paires (nœud1, nœud2)

    return selected_edges

def get_edge_midpoints(point_obj, selected_edges):
    """
    Calcule le point central de chaque arête sélectionnée en prenant la moyenne des coordonnées des deux nœuds.
    """
    # Récupérer les coordonnées des nœuds
    points = point_obj.points  # Matrice 3xN contenant les coordonnées des points

    # Extraire les coordonnées des deux nœuds de chaque arête
    node1_coords = points[:, selected_edges[:, 0]]  # Coordonnées des premiers nœuds des arêtes
    node2_coords = points[:, selected_edges[:, 1]]  # Coordonnées des seconds nœuds des arêtes

    # Calcul du point central (moyenne des coordonnées des deux nœuds)
    midpoints = (node1_coords + node2_coords) / 2

    return midpoints.T  # On transpose pour avoir un tableau de Nx3 (chaque ligne = un point 3D)

def load_high_current_points_from_file(filename):
    if not os.path.exists(filename):
        return np.array([])  # Retourner un tableau vide si le fichier n'existe pas

    points = []
    with open(filename, 'r') as file:
        lines = file.readlines()[1:]  # Sauter l'en-tête
        for line in lines:
            x, y, z = map(float, line.split())
            points.append([x, y, z])
    return np.array(points)

def save_high_current_points_to_file(points, filename):
    with open(filename, 'a') as file:
        for point in points:
            file.write(f"{point[0]} {point[1]} {point[2]}\n")