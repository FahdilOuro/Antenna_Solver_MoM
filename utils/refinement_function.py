import os
import numpy as np

# A modifier :
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

def create_pos_file(mesh_name):
    pos_folder = 'data/pos/'

    # Vérifier si le dossier existe, sinon le créer
    if not os.path.exists(pos_folder):
        os.makedirs(pos_folder)

    # Nom du fichier de points du maillage adaptatif
    filename = f'data/pos/{os.path.splitext(os.path.basename(mesh_name))[0]}.pos'

    return filename

# New part of the code :

def compute_size_from_current(nodes, triangles, current_values, mesh_size, feed_point, mesh_dividend, r_threshold=0.1):
    print("\nshapes nodes =", nodes.shape)
    print("shapes triangles =", triangles.shape)
    pts = nodes[triangles]  # forme (T, 3, 3) -> T triangles, 3 sommets, 3 coords

    # Calculer le centre de chaque triangle : moyenne des trois sommets
    centers = pts.mean(axis=1)  # forme (T, 3)

    distances = np.hypot(centers[:, 0] - feed_point[0], centers[:, 1] - feed_point[1], centers[:, 2] - feed_point[2])
    print("\nshapes distances =", distances.shape)
    print("shapes current =", current_values.shape)
    # Appliquer la condition du rayon `r_threshold` : les points dans ce rayon gardent la taille de maillage par défaut
    size_field = np.full_like(current_values, mesh_size)

    # print("Taille de size_field", size_field.shape)
    
    # Pour les points à l'extérieur du rayon, on applique la logique de taille en fonction du champ de courant
    outside_threshold = distances > r_threshold

    new_mesh_field = size_field[outside_threshold]

    count = 0
    for current in current_values[outside_threshold]:
        if current > 0.7 * np.max(current_values):
            new_mesh_field[count] = size_field[count] / mesh_dividend
        count += 1

    size_field[outside_threshold] = new_mesh_field

    '''print("Taille de size_field", size_field.shape)'''

    print("size_field = ", size_field)
    
    # Retourner le champ de taille
    return size_field