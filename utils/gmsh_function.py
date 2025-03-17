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

def extract_msh_to_msh(file_msh_path, save_mat_path):
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

def is_inside(x, y, z, surface_tag):
    # Définir le point à tester (X, Y, Z)
    point_coords = [x, y, z]  # Remplace par tes coordonnées

    # Trouver le point projeté sur la surface
    closest_points = gmsh.model.getClosestPoint(2, surface_tag, point_coords)

    # Vérifier si le point est proche
    is_on_surface = False
    if closest_points:
        x_proj, y_proj, z_proj = closest_points[0]
        distance = ((x_proj - point_coords[0])**2 + (y_proj - point_coords[1])**2 + (z_proj - point_coords[2])**2)**0.5
        is_on_surface = distance < 1e-6  # Seuil de tolérance
    
    return is_on_surface

def feed_edge(surface_tag, feed_point, length_feed_edge, mesh_name, angle=0, plane="xy"):
    length_feed_edge -= 0.000001
    # Identifier les arêtes avant l'ajout de l'arête interne
    edges_before = set(gmsh.model.getEntities(1))

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
        z1 = z0
        z2 = z0

    elif plane == "yz":
        dy = half_length * math.cos(angle)
        dz = half_length * math.sin(angle)
        y1 = y0 - dy
        z1 = z0 - dz
        y2 = y0 + dy
        z2 = z0 + dz
        x1 = x0
        x2 = x0

    elif plane == "xz":
        dx = half_length * math.cos(angle)
        dz = half_length * math.sin(angle)
        x1 = x0 - dx
        z1 = z0 - dz
        x2 = x0 + dx
        z2 = z0 + dz
        y1 = y0
        y2 = y0

    else:
        raise ValueError("Le paramètre 'plane' doit être 'xy', 'yz' ou 'xz'.")

    if is_inside(x1, y1, z1, surface_tag) and is_inside(x2, y2, z2, surface_tag):
        if plane == "xy":
            p1 = gmsh.model.occ.addPoint(x1, y1, 0)
            p2 = gmsh.model.occ.addPoint(x2, y2, 0)
        elif plane == "yz":
            p1 = gmsh.model.occ.addPoint(0, y1, z1)
            p2 = gmsh.model.occ.addPoint(0, y2, z2)
        elif plane == "xz":
            p1 = gmsh.model.occ.addPoint(x1, 0, z1)
            p2 = gmsh.model.occ.addPoint(x2, 0, z2)

        # Création de l'arête interne
        feed_edge_tag = gmsh.model.occ.addLine(p1, p2)

        # Fragmentation de la surface avec l'arête
        gmsh.model.occ.fragment([(2, surface_tag)], [(1, feed_edge_tag)])
        gmsh.model.occ.synchronize()

        # Identifier la nouvelle arête créée
        edges_after = set(gmsh.model.getEntities(1))
        new_edge = list(edges_after - edges_before)  # Trouver l'arête ajoutée

        if new_edge:
            new_edge_tag = new_edge[0][1]  # Extraire le tag de la nouvelle arête

            # Définir le chemin du fichier JSON
            json_path = f'data/json/feed_edge_info_{os.path.splitext(mesh_name)[0]}.json'

            # Vérifier si le dossier "json" existe, sinon le créer
            os.makedirs(os.path.dirname(json_path), exist_ok=True)

            # Sauvegarder l'ID de l'arête dans un fichier JSON pour éviter toute confusion
            with open(json_path, "w") as f:
                json.dump({"edge_tag": new_edge_tag}, f, indent=4)
    else:
        raise ValueError("Les points de l'arête sortent de la surface. Aucune arête n'a été créée.")

    print("Ajout de feed_edge reussie ...!")

    return feed_edge_tag