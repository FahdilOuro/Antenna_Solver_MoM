import sys
import gmsh
import numpy as np
import scipy.io as sio

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