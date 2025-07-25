import os
import sys
import gmsh
import numpy as np
import scipy.io as sio

from src.radiation_algorithm.radiation_algorithm import radiation_algorithm
from utils.refinement_function import *

class Mesh:
    def __init__(self):
        self.vtags, vxyz, _ = gmsh.model.mesh.getNodes()
        self.vxyz = vxyz.reshape((-1, 3))
        vmap = dict({j: i for i, j in enumerate(self.vtags)})
        self.triangles_tags, evtags = gmsh.model.mesh.getElementsByType(2)
        evid = np.array([vmap[j] for j in evtags])
        self.triangles = evid.reshape((self.triangles_tags.shape[-1], -1))

def open_mesh(file_msh_path):
    gmsh.initialize()
    gmsh.merge(file_msh_path)
    if '-nopopup' not in sys.argv:
        gmsh.fltk.run()
    gmsh.finalize()

def run():
    gui = True
    argv = sys.argv
    if '-nopopup' in sys.argv:
        gui = False
        argv.remove('-nopopup')

    gmsh.fltk.run()

def write(save_folder_path, file_name="mesh.msh"):
    # Assure que save_folder_path est un dossier, pas un fichier
    if not os.path.isdir(save_folder_path):
        print(f"The folder '{save_folder_path}' does not exist.")
        os.makedirs(save_folder_path)
        print(f"Folder '{save_folder_path}' was created successfully.")

    # Construction du chemin complet du fichier
    save_path = os.path.join(save_folder_path, file_name)

    # Écriture du fichier
    gmsh.write(save_path)
    print(f"The .msh file was successfully saved to: '{save_path}'")

def apply_mesh_size(mesh_size):
    # Synchronisation du modèle
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), size=mesh_size)
    gmsh.option.setNumber('Mesh.Algorithm', 1)  # 1: MeshAdapt, 2: Automatic, 3: Initial mesh only, 
    # 5: Delaunay, 6: Frontal-Delaunay (Default value: 6), 7: BAMG, 8: Frontal-Delaunay for Quads, 
    # 9: Packing of Parallelograms, 11: Quasi-structured Quad

def generate_surface_mesh():
    NumberofTreads = 5
    gmsh.option.setNumber('General.NumThreads', NumberofTreads)
    gmsh.model.mesh.generate(2)  # Générer le maillage en 2D
    
def write(save_folder_path, file_name="new_mesh.msh"):
    # Assure que save_folder_path est un dossier, pas un fichier
    if not os.path.isdir(save_folder_path):
        print(f"The folder '{save_folder_path}' does not exist.")
        os.makedirs(save_folder_path)
        print(f"Folder '{save_folder_path}' was created successfully.")

    # Construction du chemin complet du fichier
    save_path = os.path.join(save_folder_path, file_name)

    # Écriture du fichier
    gmsh.write(save_path)
    # print(f"The .msh file was successfully saved to: '{save_path}'")

def rectangle_surface(x_rect, y_rect):
    point_tags = []
    for x_ti, y_ti in zip(x_rect, y_rect):
        tag = gmsh.model.occ.addPoint(x_ti, y_ti, 0)
        point_tags.append(tag)

    line_tags_terminal = []
    for i in range(len(point_tags) - 1):
        line = gmsh.model.occ.addLine(point_tags[i], point_tags[i + 1])
        line_tags_terminal.append(line)

    line_tags_terminal.append(gmsh.model.occ.addLine(point_tags[-1], point_tags[0]))
    loop_terminal = gmsh.model.occ.addCurveLoop(line_tags_terminal)
    rect_surface = gmsh.model.occ.addPlaneSurface([loop_terminal])

    return rect_surface

# -------------------------------A MODIFIER POUR LE REFIERMENT --------------------

def read_mesh_msh(fichier_msh):
    gmsh.initialize()
    # gmsh.option.setNumber("General.Terminal", 0)
    gmsh.open(str(fichier_msh))

    # Récupérer les nœuds
    vtags, vxyz, _ = gmsh.model.mesh.getNodes()
    vxyz = vxyz.reshape((-1, 3))  # (N, 3)

    # Créer un mapping tag → index
    vmap = {tag: idx for idx, tag in enumerate(vtags)}

    # Récupérer les éléments de type triangle (type 2)
    triangles_tags, evtags = gmsh.model.mesh.getElementsByType(2)
    evtags = np.array([vmap[tag] for tag in evtags])
    triangles = evtags.reshape((-1, 3))  # (T, 3)

    gmsh.finalize()
    return vxyz, triangles, triangles_tags

def save_mesh():
    mesh = {}
    for entite in gmsh.model.getEntities():
        dim, tag = entite
        frontieres = gmsh.model.getBoundary([entite])
        noeuds = gmsh.model.mesh.getNodes(dim, tag)
        elements = gmsh.model.mesh.getElements(dim, tag)
        mesh[entite] = (frontieres, noeuds, elements)
    return mesh

def copy_mesh(mesh, copy_model_name):
    gmsh.model.add(copy_model_name)
    # create discrete entities in the new model and copy the mesh
    for entite in sorted(mesh):
        dim, tag = entite
        frontieres, noeuds, elements = mesh[entite]
        gmsh.model.addDiscreteEntity(dim, tag, [b[1] for b in frontieres])
        gmsh.model.mesh.addNodes(dim, tag, noeuds[0], noeuds[1])
        gmsh.model.mesh.addElements(dim, tag, elements[0], elements[1], elements[2])

def remeshing_model(mesh, currents, mesh_size, feed_point, mesh_dividend):
    save_bowtie = save_mesh()
    new_model = "bowtie_discrete"
    copy_mesh(save_bowtie, new_model)
    print(f"creation of new model {new_model}")

    # Calculer la taille de maillage basée sur le champ de courant
    # sf_ele = compute_size_field_based_on_current(mesh_bowtie.vxyz, mesh_bowtie.triangles, currents_bowtie, lenght_feed_high, feed_point, r_threshold=lenght_feed_high, N=100)
    # mesh[0], mesh[1], mesh[2], mesh.vxyz, mesh.triangles, mesh.triangles_tags
    sf_ele = compute_size_from_current(mesh.vxyz, mesh.triangles, currents, mesh_size, feed_point, mesh_dividend, r_threshold=mesh_size*2)
    # Afficher le champ de taille
    sf_view = gmsh.view.add("mesh size field")
    gmsh.view.addModelData(sf_view, 0, new_model, "ElementData", mesh.triangles_tags, sf_ele[:, None])
    gmsh.plugin.setNumber("Smooth", "View", gmsh.view.getIndex(sf_view))
    gmsh.plugin.run("Smooth")
    return sf_view

def post_processing_meshing(model_name, sf_view):
    gmsh.model.setCurrent(model_name)
    field = gmsh.model.mesh.field.add("PostView")
    gmsh.model.mesh.field.setNumber(field, "ViewTag", sf_view)
    gmsh.model.mesh.field.setAsBackgroundMesh(field)
    gmsh.option.setNumber('Mesh.Algorithm', 1) # 1: MeshAdapt, 2: Automatic, 3: Initial mesh only, 
    # 5: Delaunay, 6: Frontal-Delaunay (Default value: 6), 7: BAMG, 8: Frontal-Delaunay for Quads, 
    # 9: Packing of Parallelograms, 11: Quasi-structured Quad
    gmsh.model.mesh.clear()
    generate_surface_mesh()
    

# merge everything here    ------------- new step ---------

def adapt_meshing():
    # code ici
    print("")

# -------------------------------REFIERMENT CODE --------------------

def extract_msh_to_mat(file_msh_path, save_mat_path):
    import gmsh
    import numpy as np
    import os
    import scipy.io as sio

    gmsh.initialize()
    gmsh.open(str(file_msh_path))

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    p = np.array(node_coords).reshape(-1, 3).T  # (3 x N)

    dim = 2
    entities = gmsh.model.getEntities(dim)

    # Récupérer tous les tags de surface et créer un mapping vers des indices 0..n
    surface_tags = sorted([tag for (d, tag) in entities])
    tag_to_index = {tag: i for i, tag in enumerate(surface_tags)}  # par exemple {3: 0, 7: 1, 12: 2}

    triangles = []

    for entity in entities:
        entity_dim, entity_tag = entity
        element_types, element_tags, node_tags = gmsh.model.mesh.getElements(entity_dim, entity_tag)

        for etype, nodes in zip(element_types, node_tags):
            if etype == 2:  # Triangles
                num_triangles = len(nodes) // 3
                surface_idx = tag_to_index[entity_tag]  # index consécutif commençant à 0
                surface_indices = np.full((1, num_triangles), surface_idx)
                triangles.append(np.vstack((np.array(nodes).reshape(-1, 3).T, surface_indices)))

    t = np.hstack(triangles) if triangles else np.array([])

    if not os.path.exists(os.path.dirname(save_mat_path)):
        os.makedirs(os.path.dirname(save_mat_path))
        print(f"Folder '{os.path.dirname(save_mat_path)}' was created successfully.")

    sio.savemat(save_mat_path, {"p": p, "t": t})

    gmsh.finalize()

def extract_ModelMsh_to_mat(model_name, save_mat_path):
    gmsh.model.setCurrent(model_name)
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

    print(f"matlab file stored in {save_mat_path} successfully")

# -------------------------------A modifier eventuellement--------------------

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

def refine_antenna(model_name, frequency, mesh_name, feed_point, mesh_size, file_name_msh, file_name_mat, save_mesh_folder, max_iterations=10):
    tolerance = 1e-3  # tolérance sur la variation d'impédance
    prev_impedance = None

    for iteration in range(max_iterations):
        write(save_mesh_folder, mesh_name)
        extract_ModelMsh_to_mat(model_name, file_name_mat)

        impedance, current_bowtie_antenna = radiation_algorithm(file_name_mat, frequency, feed_point)

        mesh_bowtie = Mesh()

        mesh_dividend = 5

        sf_view = remeshing_model(mesh_bowtie, current_bowtie_antenna, mesh_size, feed_point, mesh_dividend)

        post_processing_meshing(model_name, sf_view)

        run()

        if prev_impedance is not None:
            # Calcul de la variation relative ou absolue
            variation = np.abs(impedance - prev_impedance)
            print(f"Iteration {iteration}: impedance = {impedance}, variation = {variation}")

            if variation < tolerance:
                print("Convergence atteinte")
                break
            
        prev_impedance = impedance