import os
import sys
import gmsh
import numpy as np
import scipy.io as sio
import math
import json

def open_mesh(file_msh_path):
    gmsh.initialize()
    gmsh.open(file_msh_path)
    if '-nopopup' not in sys.argv:
        gmsh.fltk.run()
    gmsh.finalize()

def sort_points(point):
    """
    Trie les points dans l'ordre croissant selon toutes leurs dimensions.
    """
    sorted_indices = np.lexsort(point[::-1])  # Trie en commençant par la dernière coordonnée
    return point[:, sorted_indices]

def extract_receiving_msh_to_mat(file_msh_path, save_mat_path):
    # Initialiser Gmsh
    gmsh.initialize()

    # Charger le fichier maillé
    gmsh.open(file_msh_path)

    # Récupérer tous les nœuds (points)
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    N = len(node_tags)  # Nombre de points

    # Restructurer les coordonnées en un tableau 3xN
    p = np.array(node_coords).reshape(-1, 3).T  # (3xN)

    # Extraire les éléments (triangles)
    dim = 2  # Maillage 2D
    entities = gmsh.model.getEntities(dim)

    triangles = []
    surface_indices = None

    for entity in entities:
        entity_dim, entity_tag = entity  # entity_tag est l'index de la surface

        element_types, element_tags, node_tags = gmsh.model.mesh.getElements(entity_dim, entity_tag)

        for etype, nodes in zip(element_types, node_tags):
            if etype == 2:  # Type 2 = Triangles
                num_triangles = len(nodes) // 3
                surface_indices = np.full((1, num_triangles), entity_tag)  # Créer une ligne avec le tag de surface
                triangles.append(np.vstack((np.array(nodes).reshape(-1, 3).T, surface_indices)))  # Ajouter la 4e ligne

    # Convertir la liste en un tableau numpy (4xT)
    t = np.hstack(triangles) if triangles else np.array([])

    # Sauvegarder les données dans un fichier .mat
    sio.savemat(save_mat_path, {"p": p, "t": t})

    # Fermer Gmsh
    gmsh.finalize()

    print(f"matlab file stored in {save_mat_path} successfully")

def extract_radiation_msh_to_mat(file_msh_path, mesh_name, save_mat_path):
    # Initialiser Gmsh
    gmsh.initialize()

    # Charger le fichier maillé
    gmsh.open(file_msh_path)

    # Définir le chemin du fichier JSON
    json_path = f'data/json/feed_edge_info_{os.path.splitext(mesh_name)[0]}.json'

    # Charger l'information sur l'arête insérée
    with open(json_path, "r") as f:
        edge_data = json.load(f)

    new_edge_tag = edge_data["edge_tag"]  # Tag de l'arête interne

    # Récupérer tous les nœuds (points)
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_tags = node_tags.astype(int)  # Convertir en tableau d'entiers
    N = len(node_tags)  # Nombre total de points

    # Restructurer les coordonnées en un tableau 3xN
    p = np.array(node_coords).reshape(-1, 3).T  # (3xN)

    # Extraire les triangles
    dim = 2  # Maillage 2D
    entities = gmsh.model.getEntities(dim)

    triangles = []

    for entity in entities:
        entity_dim, entity_tag = entity  # entity_tag est l'index de la surface

        element_types, element_tags, node_tags_elements = gmsh.model.mesh.getElements(entity_dim, entity_tag)

        for etype, nodes in zip(element_types, node_tags_elements):
            if etype == 2:  # Type 2 = Triangles
                num_triangles = len(nodes) // 3
                surface_indices = np.full((1, num_triangles), entity_tag)  # Ajout du tag de surface
                triangles.append(np.vstack((np.array(nodes).reshape(-1, 3).T, surface_indices)))

    # Convertir en numpy array (4xT)
    t = np.hstack(triangles) if triangles else np.array([[], [], [], []])

    # Identifier les nœuds de l'arête interne
    p_feed = np.array([])

    element_types, element_tags, edge_nodes = gmsh.model.mesh.getElements(1, new_edge_tag)  # Récupérer les nœuds de l'arête

    if edge_nodes and len(edge_nodes) > 0:
        edge_nodes_flat = edge_nodes[0].astype(int).tolist()  # Conversion en liste d'entiers
        edge_nodes_set = set(edge_nodes_flat)  # Convertir en set propre

        p_feed_indices = [i for i, tag in enumerate(node_tags.tolist()) if tag in edge_nodes_set]

        if p_feed_indices:  # Vérifier qu'on a bien trouvé des indices
            p_feed = p[:, p_feed_indices]  # Extraire les coordonnées des points appartenant à l'arête
    
    p_feed = sort_points(p_feed)

    # Détection des triangles appartenant à l’arête interne (t_feed)**
    p_feed_set = {tuple(p_feed[:, i]) for i in range(p_feed.shape[1])}  # Création d’un set pour une recherche rapide
    t_feed_list = []

    for i in range(t.shape[1]):  # Parcourir chaque triangle
        triangle_indices = t[:3, i]  # Indices des 3 sommets
        surface_index = t[3, i]  # Récupérer l'indice de la surface
        triangle_points = [tuple(p[:, int(idx) - 1]) for idx in triangle_indices]  # Coords des sommets

        # Vérifier combien de sommets sont dans p_feed
        count_feed_points = sum(1 for point in triangle_points if point in p_feed_set)

        if count_feed_points == 2:  # Seulement si 2 sommets sont dans p_feed
            t_feed_list.append(np.append(triangle_indices, surface_index))  # Ajout de l’indice de la surface

    # Convertir en tableau numpy (4xT_feed)
    t_feed = np.array(t_feed_list).T if t_feed_list else np.array([[], [], [], []])

    # Sauvegarde dans le fichier .mat
    sio.savemat(save_mat_path, {"p": p, "t": t, "p_feed": p_feed, "t_feed": t_feed})

    #  Fermeture de Gmsh
    gmsh.finalize()

    print(f"matlab file stored in {save_mat_path} successfully")

def feed_edge(surface_tag, feed_point, length_feed_edge, mesh_name, angle=0, plane="xy"):
    length_feed_edge -= 0.000001

    # Point central de l'arête
    if not (isinstance(feed_point, (list, tuple)) and len(feed_point) == 3):
        raise ValueError("feed_point doit être une liste ou un tuple de trois éléments [x, y, z].")
    
    x0, y0, z0 = feed_point
    half_length = length_feed_edge / 2

    if plane == "xy":
        dx = half_length * math.cos(angle)
        dy = half_length * math.sin(angle)
        x1 = x0 - dx
        y1 = y0 - dy
        x2 = x0 + dx
        y2 = y0 + dy
        p1 = gmsh.model.occ.addPoint(x1, y1, 0)
        p2 = gmsh.model.occ.addPoint(x2, y2, 0)

    elif plane == "yz":
        dy = half_length * math.cos(angle)
        dz = half_length * math.sin(angle)
        y1 = y0 - dy
        z1 = z0 - dz
        y2 = y0 + dy
        z2 = z0 + dz
        p1 = gmsh.model.occ.addPoint(0, y1, z1)
        p2 = gmsh.model.occ.addPoint(0, y2, z2)

    elif plane == "xz":
        dx = half_length * math.cos(angle)
        dz = half_length * math.sin(angle)
        x1 = x0 - dx
        z1 = z0 - dz
        x2 = x0 + dx
        z2 = z0 + dz
        p1 = gmsh.model.occ.addPoint(x1, 0, z1)
        p2 = gmsh.model.occ.addPoint(x2, 0, z2)

    else:
        raise ValueError("Le paramètre 'plane' doit être 'xy', 'yz' ou 'xz'.")

    # Création de l'arête interne
    feed_edge_tag = gmsh.model.occ.addLine(p1, p2)

    # Fragmentation de la surface avec l'arête
    gmsh.model.occ.fragment([(2, surface_tag)], [(1, feed_edge_tag)])
    gmsh.model.occ.synchronize()

    # Définir le chemin du fichier JSON
    json_path = f'data/json/feed_edge_info_{os.path.splitext(mesh_name)[0]}.json'

    # Vérifier si le dossier "json" existe, sinon le créer
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    # Sauvegarder l'ID de l'arête dans un fichier JSON pour éviter toute confusion
    with open(json_path, "w") as f:
        json.dump({"edge_tag": feed_edge_tag}, f, indent=4)
    
    print(f"Json File saved to the path : {json_path}")

    print("Ajout de feed_edge reussie ...!")

def save_gmsh_log(mesh_name, output_path):
    """Enregistre les logs de GMSH dans un fichier texte avec un format clair et structuré."""

    model_name = os.path.splitext(os.path.basename(mesh_name))[0]

    # Récupérer les logs
    logs = gmsh.logger.get()

    # Assurer l'existence du dossier de logs
    log_dir = "data/gmsh_log/"
    os.makedirs(log_dir, exist_ok=True)

    # Déterminer le chemin du fichier log
    log_file = os.path.join(log_dir, f"mesh_log_{model_name}.txt")

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"========== MESHING SUMMARY ==========\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Mesh file location: {os.path.abspath(output_path)}\n")
        f.write(f"-------------------------------------\n\n")

        # Écriture des logs de Gmsh
        for log in logs:
            f.write(log + "\n")

    print(f"Log saved in: {log_file}")  # Confirmation en console

def point_on_segment(px, py, pz, x1, y1, z1, x2, y2, z2, tol=1e-6):
    """
    Vérifie si le point (px, py, pz) appartient au segment [(x1, y1, z1), (x2, y2, z2)]
    avec une tolérance 'tol' pour compenser les erreurs numériques.
    """
    v = np.array([x2 - x1, y2 - y1, z2 - z1])
    w = np.array([px - x1, py - y1, pz - z1])
    cross_prod = np.linalg.norm(np.cross(v, w))

    v_norm = np.linalg.norm(v)
    if v_norm < tol:
        return False  # Segment dégénéré

    t = np.dot(w, v) / (v_norm ** 2)

    return (0 <= t <= 1) and (cross_prod < tol)

def is_point_on_boundary(surface_tag, point, tol=1e-5):
    """
    Vérifie si un point appartient au bord d'une surface quelconque.

    Arguments :
        - surface_tag : tag de la surface dans Gmsh.
        - point : tuple (x, y, z) du point à tester.
        - tol : tolérance numérique.

    Retourne :
        - True si le point est sur la frontière de la surface, False sinon.
    """
    # Récupérer tous les nœuds du maillage
    all_node_tags, all_node_coords, _ = gmsh.model.mesh.getNodes()
    all_node_coords = np.array(all_node_coords).reshape(-1, 3)

    # Dictionnaire associant chaque tag de nœud à ses coordonnées
    node_dict = {tag: coord for tag, coord in zip(all_node_tags, all_node_coords)}

    # Récupérer les entités de bord (les arêtes du contour de la surface)
    boundary = gmsh.model.getBoundary([(2, surface_tag)], oriented=False)

    segments = []
    for edge in boundary:
        edge_dim, edge_tag = edge

        # Récupérer les connectivités des arêtes
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(edge_dim, edge_tag)

        if elem_types:  # Vérifier qu'on a bien des éléments (arêtes)
            for elem_nodes in elem_node_tags[0].reshape(-1, 2):  # Segments = paires de nœuds
                id1, id2 = elem_nodes  # Indices globaux

                if id1 in node_dict and id2 in node_dict:
                    p1, p2 = node_dict[id1], node_dict[id2]
                    segments.append((p1, p2))

    # Vérifier si le point appartient à un segment du bord
    for p1, p2 in segments:
        if point_on_segment(*point, *p1, *p2, tol):
            return True

    return False
