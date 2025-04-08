import os
import sys
import gmsh
import numpy as np
import scipy.io as sio
import math
import json

from utils.refinement_function import load_high_current_points_from_file, save_high_current_points_to_file

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
        data = json.load(f)

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

    # Extraire les coordonnées depuis le fichier JSON
    coordinates = data["coordinates"]
    
    # Convertir les coordonnées en tableau Numpy
    p_feed = np.array([[point["x"], point["y"], point["z"]] for point in coordinates]).T
    
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

# -------------------------------Big_modifications--------------------
def mesh_size(mesh_size):
    # Synchronisation du modèle
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)

    # Génération du maillage
    meshing_option = 2
    gmsh.option.setNumber("Mesh.Algorithm", meshing_option)

def create_feed_edge(surface, feed_point, feed_lenght, angle, meshSize, json_path):
    dx = feed_lenght / 2 * math.cos(angle)
    dy = feed_lenght / 2 * math.sin(angle)

    # Nouveaux points à ajouter (chaque point est un tableau de forme (3,) pour un point en 3D)
    x1 = feed_point[0] - dx
    y1 = feed_point[1] - dy
    x2 = feed_point[0] + dx
    y2 = feed_point[1] + dy

    new_point1 = np.array([x1, y1, 0])  # Coordonnée du premier point
    new_point2 = np.array([x2, y2, 0])  # Coordonnée du second point

    # Ajouter la ligne d'alimentation qui coupe le carré en deux verticalement
    feed_line_start = gmsh.model.occ.addPoint(new_point1[0], new_point1[1], new_point1[2])
    feed_line_end = gmsh.model.occ.addPoint(new_point2[0], new_point2[1], new_point2[2])

    # Créer la ligne d'alimentation
    feed_line = gmsh.model.occ.addLine(feed_line_start, feed_line_end)

    # Fragmenter le carré avec la ligne d'alimentation
    gmsh.model.occ.fragment([(2, surface)], [(1, feed_line)])

    # Synchronisation du modèle
    gmsh.model.occ.synchronize()

    mesh_size(meshSize)

    gmsh.model.mesh.generate(2)

    # Extraire les tags des nœuds associés à la ligne feed_line
    nodeTags, coords, _ = gmsh.model.mesh.getNodes(dim=1, tag=feed_line)

    # On filtre uniquement les nœuds de la ligne feed_line
    points_feed_line = []
    for i in range(len(nodeTags)):
        # Les coordonnées de chaque point sont stockées dans le tableau "coords"
        points_feed_line.append(coords[3 * i: 3 * i + 3])  # x, y, z pour chaque point

    points_feed_line.append([new_point1[0], new_point1[1], new_point1[2]])
    points_feed_line.append([new_point2[0], new_point2[1], new_point2[2]])

    # Convertir la liste de points en tableau numpy
    points_feed_line = np.array(points_feed_line).T  # Transpose pour avoir les coordonnées X, Y, Z dans les bonnes dimensions

    coordinates = [{"x": point[0], "y": point[1], "z": point[2]} for point in points_feed_line.T]

    # Sauvegarder les coordonnées dans un fichier JSON
    with open(json_path, "w") as f:
        json.dump({"coordinates": coordinates}, f, indent=4)

    print(f"Json File saved to the path : {json_path}")

def apply_feed_edge(create_surface, json_path, meshSize):
    surface = create_surface()

    # Charger l'information sur l'arête insérée
    with open(json_path, "r") as f:
        data = json.load(f)

    # Extraire les coordonnées depuis le fichier JSON
    coordinates = data["coordinates"]

    # Convertir les coordonnées en tableau Numpy
    points_feed = np.array([[point["x"], point["y"], point["z"]] for point in coordinates]).T

    # Ajouter des minuscules segments verticaux aux positions souhaitées
    tiny_lines = []
    dz = 1e-6  # hauteur très petite pour ne pas affecter la surface

    for i in range(points_feed.shape[1]):
        x, y, z = points_feed[:, i]
        pt1 = gmsh.model.occ.addPoint(x, y, z)
        pt2 = gmsh.model.occ.addPoint(x, y, z + dz)
        line = gmsh.model.occ.addLine(pt1, pt2)
        tiny_lines.append((1, line))

    # Fragmenter la surface avec les petites lignes (ceci les intègre à la géométrie)
    gmsh.model.occ.fragment([(2, surface)], tiny_lines)

    # Synchronisation du modèle
    gmsh.model.occ.synchronize()

    mesh_size(meshSize)
    gmsh.model.mesh.generate(2)

def create_antenna_surface(creation_surface_func, feed_point, feed_lenght, angle, meshSize, mesh_name, save_mesh_folder, high_current_points_list=np.array([3, 0]), iteration=0):
    
    # Définir le chemin du fichier JSON
    json_path = f'data/json/feed_edge_info_{os.path.splitext(mesh_name)[0]}.json'
    # Créer les dossiers s'ils n'existent pas
    
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    # Créer le fichier JSON s'il n'existe pas
    if not os.path.exists(json_path):
    # Données initiales à mettre dans le fichier (peut être vide ou avec un contenu par défaut)
        initial_data = {
            "edge_tag": None,
            "coordinates": []
        }
        with open(json_path, "w") as f:
            json.dump(initial_data, f, indent=4)
        print(f"Fichier créé : {json_path}")
    else:
        print(f"Fichier déjà existant : {json_path}")

    create_feed_edge(creation_surface_func(), feed_point, feed_lenght, angle, meshSize, json_path)

    apply_feed_edge(creation_surface_func, json_path, meshSize)

    # Vérifier si le dossier existe, sinon le créer
    if not os.path.exists(save_mesh_folder):
        os.makedirs(save_mesh_folder)
    
    # Définir le chemin où enregistrer le fichier
    output_path = os.path.join(save_mesh_folder, mesh_name)

    gmsh.write(output_path)
    print(f"{mesh_name} saved in {output_path} successfully")

    # Sauvegarde des logs 
    save_gmsh_log(mesh_name, output_path)

    gmsh.fltk.run()
    gmsh.finalize()

    return output_path

# -------------------------------Big_modifications--------------------

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

def box_field(box_tag, x, y, rayon = 1 / 10, density = 0.5, vin = 0.1/2, vout = 0.1*2):
    gmsh.model.mesh.field.add("Box", box_tag)
    gmsh.model.mesh.field.setNumber(box_tag, "VIn", vin)
    gmsh.model.mesh.field.setNumber(box_tag, "VOut", vout)
    gmsh.model.mesh.field.setNumber(box_tag, "XMin", x - rayon)
    gmsh.model.mesh.field.setNumber(box_tag, "XMax", x + rayon)
    gmsh.model.mesh.field.setNumber(box_tag, "YMin", y - rayon)
    gmsh.model.mesh.field.setNumber(box_tag, "YMax", y + rayon)
    gmsh.model.mesh.field.setNumber(box_tag, "Thickness", density)

def box_refinement(Positions):
    count_box_tag = []
    for i in range(0, Positions.shape[0]):
        box_tag = i+1
        box_field(box_tag, Positions[i, 0], Positions[i, 1])
        count_box_tag.append(box_tag)

    gmsh.model.mesh.field.add("Min", len(count_box_tag) + 1)
    gmsh.model.mesh.field.setNumbers(len(count_box_tag) + 1, "FieldsList", count_box_tag)

    gmsh.model.mesh.field.setAsBackgroundMesh(len(count_box_tag) + 1)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

def refinement_process(filename, high_current_points_list, iteration):
    if iteration < 0:
        raise ValueError("iteration value should start with '0'")
    elif iteration == 0:
        # Vérifier si le fichier existe et le réinitialiser
        if os.path.exists(filename):
            os.remove(filename)  # Supprimer le fichier existant s'il existe

        # Créer un nouveau fichier vierge avec l'en-tête
        with open(filename, 'w') as file:
            file.write("x y z\n")
        
        # on incremente l'iteraton
        iteration = 1
    else:
        # Sauver les nouveaux points de rafinage dans le fichier
        save_high_current_points_to_file(high_current_points_list, filename)

        # Charger les points actuels à chaque itération
        all_points = load_high_current_points_from_file(filename)

        box_refinement(high_current_points_list)

def finalize_antenna_model_gmsh(filename, high_current_points_list, save_mesh_folder, mesh_name, iteration):
    Automatic = 1
    gmsh.option.setNumber("Mesh.Algorithm", Automatic)   # To set The "Automatic" algorithm / Change if necessary

    refinement_process(filename, high_current_points_list, iteration)

    # Génération du maillage
    gmsh.model.mesh.generate(2)

    # Vérifier si le dossier existe, sinon le créer
    if not os.path.exists(save_mesh_folder):
        os.makedirs(save_mesh_folder)
    
    # Définir le chemin où enregistrer le fichier
    output_path = os.path.join(save_mesh_folder, mesh_name)

    gmsh.write(output_path)
    print(f"{mesh_name} saved in {output_path} successfully")

    # Sauvegarde des logs 
    save_gmsh_log(mesh_name, output_path)

    # Fermeture de Gmsh
    gmsh.finalize()

    return output_path