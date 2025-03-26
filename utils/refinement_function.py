import numpy as np


def get_nodes_from_selected_edges(edge_obj, selected_indices):

    # Extraire les arêtes sous forme de tableau 2xN
    edge = np.vstack((edge_obj.first_points, edge_obj.second_points))

    # Extraire les nœuds des arêtes sélectionnées
    selected_nodes = edge[:, selected_indices].flatten()  # Convertir en liste 1D

    # Supprimer les doublons pour ne garder que les nœuds uniques
    unique_nodes = np.unique(selected_nodes)

    return unique_nodes

def get_high_current_points_list(point_obj, selected_nodes):
    return point_obj.points[:, selected_nodes].T
