import os
import sys
import gmsh
import numpy as np
import scipy.io as sio

from backend.src.radiation_algorithm.radiation_algorithm import radiation_algorithm
from backend.utils.refinement_function import *

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
    # be sure save_folder_path is a folder, not a file
    if not os.path.isdir(save_folder_path):
        print(f"The folder '{save_folder_path}' does not exist.")
        os.makedirs(save_folder_path)
        print(f"Folder '{save_folder_path}' was created successfully.")

    # construct the full file path
    save_path = os.path.join(save_folder_path, file_name)

    # write the file
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
    # Ensure save_folder_path is a folder, not a file
    if not os.path.isdir(save_folder_path):
        print(f"The folder '{save_folder_path}' does not exist.")
        os.makedirs(save_folder_path)
        print(f"Folder '{save_folder_path}' was created successfully.")

    # Construct the full file path
    save_path = os.path.join(save_folder_path, file_name)

    # Write the file
    gmsh.write(save_path)
    # print(f"The .msh file was successfully saved to: '{save_path}'")

def write_scaled_geometry(save_folder: str, geometry_name: str, scale_factor: float = 1000.0):
    """
    Save the current Gmsh geometry as a STEP file, scaling it by a given factor
    (typically 1000 to convert from meters to millimeters).

    The mesh (.msh) remains unscaled.

    Parameters
    ----------
    save_folder : str
        Path to the folder where the STEP file will be saved.
    geometry_name : str
        Name of the STEP file (e.g., 'bowtie.step').
    scale_factor : float
        Factor by which to scale the geometry before export.
    """

    # Get all existing entities in the CAD model (dim=0..3)
    entities = []
    for dim in range(4):
        for tag in gmsh.model.getEntities(dim):
            entities.append(tag)

    if not entities:
        raise RuntimeError("No geometry found to export. Ensure the model is built before calling this function.")

    # Duplicate and scale geometry temporarily
    gmsh.model.occ.dilate(entities, 0, 0, 0, scale_factor, scale_factor, scale_factor)
    gmsh.model.occ.synchronize()

    # Build output path
    output_path = os.path.join(save_folder, geometry_name)

    # Write STEP file
    gmsh.write(output_path)
    print(f"[OK] Geometry exported (scaled by ×{scale_factor}) → {output_path}")

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

# ------------------------------- TO BE MODIFIED FOR RE-CLOSING --------------------

def read_mesh_msh(fichier_msh):
    gmsh.initialize()
    # gmsh.option.setNumber("General.Terminal", 0)
    gmsh.open(str(fichier_msh))

    # Retrieve the nodes
    vtags, vxyz, _ = gmsh.model.mesh.getNodes()
    vxyz = vxyz.reshape((-1, 3))  # (N, 3)

    # Create a mapping tag → index
    vmap = {tag: idx for idx, tag in enumerate(vtags)}

    # Retrieve elements of type triangle (type 2)
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

    # Compute mesh size based on the current field
    # sf_ele = compute_size_field_based_on_current(mesh_bowtie.vxyz, mesh_bowtie.triangles, currents_bowtie, lenght_feed_high, feed_point, r_threshold=lenght_feed_high, N=100)
    # mesh[0], mesh[1], mesh[2], mesh.vxyz, mesh.triangles, mesh.triangles_tags
    sf_ele = compute_size_from_current(mesh.vxyz, mesh.triangles, currents, mesh_size, feed_point, mesh_dividend, r_threshold=mesh_size*2)
    
    # Display the size field
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

    # Retrieve all surface tags and create a mapping to indices 0..n
    surface_tags = sorted([tag for (d, tag) in entities])
    tag_to_index = {tag: i for i, tag in enumerate(surface_tags)}

    triangles = []

    for entity in entities:
        entity_dim, entity_tag = entity
        element_types, element_tags, node_tags = gmsh.model.mesh.getElements(entity_dim, entity_tag)

        for etype, nodes in zip(element_types, node_tags):
            if etype == 2:  # Triangles
                num_triangles = len(nodes) // 3
                surface_idx = tag_to_index[entity_tag]
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
    # Retrieve all nodes (points)
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    N = len(node_tags)  # Number of points

    # Reshape coordinates into a 3xN array
    p = np.array(node_coords).reshape(-1, 3).T  # (3xN)

    # Extract elements (triangles)
    dim = 2  # 2D mesh
    entities = gmsh.model.getEntities(dim)

    triangles = []
    surface_indices = None

    for entity in entities:
        entity_dim, entity_tag = entity  # entity_tag is the surface index

        element_types, element_tags, node_tags = gmsh.model.mesh.getElements(entity_dim, entity_tag)

        for etype, nodes in zip(element_types, node_tags):
            if etype == 2:  # Type 2 = Triangles
                num_triangles = len(nodes) // 3
                surface_indices = np.full((1, num_triangles), entity_tag)  # Create a row with the surface tag
                triangles.append(np.vstack((np.array(nodes).reshape(-1, 3).T, surface_indices)))  # Add 4th row

    # Convert the list into a numpy array (4xT)
    t = np.hstack(triangles) if triangles else np.array([])

    # Save data to a .mat file
    sio.savemat(save_mat_path, {"p": p, "t": t})

    print(f"MATLAB file stored in {save_mat_path} successfully")

# -------------------------------To be modified if necessary--------------------

def save_gmsh_log(mesh_name, output_path):
    """
        Saves GMSH logs to a text file in a clear and structured format.
    """

    model_name = os.path.splitext(os.path.basename(mesh_name))[0]

    # Retrieve logs
    logs = gmsh.logger.get()

    # Ensure the log folder exists
    log_dir = "data/gmsh_log/"
    os.makedirs(log_dir, exist_ok=True)

    # Determine the log file path
    log_file = os.path.join(log_dir, f"mesh_log_{model_name}.txt")

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"========== MESHING SUMMARY ==========\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Mesh file location: {os.path.abspath(output_path)}\n")
        f.write(f"-------------------------------------\n\n")

        # Write Gmsh logs
        for log in logs:
            f.write(log + "\n")

    print(f"Log saved in: {log_file}")

def refine_antenna(model_name, frequency, mesh_name, feed_point, mesh_size, file_name_msh, file_name_mat, save_mesh_folder, max_iterations=10):
    tolerance = 1e-3  # Tolerance for convergence of impedance
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
            # Calcul of variation of impedance
            variation = np.abs(impedance - prev_impedance)
            print(f"Iteration {iteration}: impedance = {impedance}, variation = {variation}")

            if variation < tolerance:
                print("convergence reached")
                break
            
        prev_impedance = impedance