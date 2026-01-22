import os
import sys
import gmsh
import numpy as np
import scipy.io as sio

from typing import Sequence


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
    gmsh.option.setNumber('Mesh.Algorithm', 6)  # 1: MeshAdapt, 2: Automatic, 3: Initial mesh only, 
    # 5: Delaunay, 6: Frontal-Delaunay (Default value: 6), 7: BAMG, 8: Frontal-Delaunay for Quads, 
    # 9: Packing of Parallelograms, 11: Quasi-structured Quad

def setup_performance_config(n_threads=None):
    """
    Optimizes Gmsh performance for a 16-thread processor.
    """
    if n_threads is None:
        # Use the provided number or fallback to system detection
        threads = os.cpu_count()
    else:
        threads = n_threads
    
    # 1. Global threads for mesh generation and internal tasks
    gmsh.option.setNumber("General.NumThreads", threads)
    
    # 2. Select a parallel-friendly 2D algorithm (optional)
    # Algorithm 8 (Frontal-Delaunay) often scales better with multiple threads
    # than the default Delaunay (Algorithm 5).
    gmsh.option.setNumber("Mesh.Algorithm", 6)

    # 3. Enable optimization to further smooth the mesh
    # "Mesh.Smoothing" sets the number of smoothing steps (Netgen algorithm)
    gmsh.option.setNumber("Mesh.Smoothing", 10)

    # 4. Enable the HXT algorithm if you ever switch to 3D
    # HXT is specifically designed for high-thread-count CPUs
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)

    # 5. Remove points very close to each other
    gmsh.option.setNumber("Geometry.Tolerance", 1e-6)

    print(f"[PERFORMANCE] Gmsh configured to utilize {threads} threads.")

def generate_surface_mesh():
    NumberofTreads = 5
    gmsh.option.setNumber('General.NumThreads', NumberofTreads)
    gmsh.model.mesh.generate(2)  # Générer le maillage en 2D

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

def rectangle_surface(x_rect: Sequence[float], y_rect: Sequence[float], z_pos: float = 0.0) -> int:
    """
    Create a planar rectangular surface in the XY plane using given X and Y coordinates.

    Parameters
    ----------
    x_rect : Sequence[float]
        X coordinates of the rectangle vertices (ordered).
    y_rect : Sequence[float]
        Y coordinates of the rectangle vertices (ordered).
    z_pos : float, optional
        Z coordinate for all points (default = 0.0).

    Returns
    -------
    int
        The tag of the created Gmsh surface.

    Raises
    ------
    AssertionError
        If x_rect and y_rect have different lengths or less than 3 points.
    """

    # --- Safety check ---
    assert len(x_rect) == len(y_rect) >= 3, \
        "x_rect and y_rect must define at least 3 points with matching lengths."

    # --- Create points ---
    point_tags = []
    for x_ti, y_ti in zip(x_rect, y_rect):
        tag = gmsh.model.occ.addPoint(float(x_ti), float(y_ti), float(z_pos))
        point_tags.append(tag)

    # --- Create connecting lines ---
    line_tags = []
    for i in range(len(point_tags) - 1):
        line_tags.append(gmsh.model.occ.addLine(point_tags[i], point_tags[i + 1]))

    # Close the loop
    line_tags.append(gmsh.model.occ.addLine(point_tags[-1], point_tags[0]))

    # --- Create surface ---
    loop = gmsh.model.occ.addCurveLoop(line_tags)
    surface = gmsh.model.occ.addPlaneSurface([loop])

    return surface

def extract_msh_to_mat(file_msh_path, save_mat_path):
    gmsh.initialize()
    gmsh.open(str(file_msh_path))

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_tags = np.array(node_tags, dtype=int)
    p = np.array(node_coords).reshape(-1, 3).T   # (3 x N)

    # Build mapping GMsh tag -> 1..N index
    tag_to_idx = {tag: i+1 for i, tag in enumerate(node_tags)}

    triangles = []

    for dim, tag in gmsh.model.getEntities(2):   # only 2D surfaces
        elem_types, elem_tags, elem_nodes = gmsh.model.mesh.getElements(dim, tag)
        for etype, nodes in zip(elem_types, elem_nodes):
            if etype == 2:   # triangle
                nodes = np.array(nodes, dtype=int).reshape(-1,3)   # Nx3
                mapped = np.vectorize(tag_to_idx.get)(nodes)       # remap tags
                surface_index = np.full((mapped.shape[0],1), tag)  # store physical tag
                tri = np.hstack((mapped, surface_index)).T         # 4 x NT
                triangles.append(tri)

    t = np.hstack(triangles) if triangles else np.array([])

    sio.savemat(save_mat_path, {"p": p, "t": t})
    gmsh.finalize()

def extract_ModelMsh_to_mat(save_mat_path):
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

def optimize_mesh(dim=2, iterations=3):
    """
    Optimizes a mesh while preserving local refinement gradients.
    This function is suitable for adaptive mesh refinement projects.
    
    Parameters:
    - dim: Dimension of the mesh to optimize (2 for surfaces, 3 for volumes)
    - iterations: Number of optimization passes
    """
    print(f"--- Starting Mesh Optimization (Dim: {dim}) ---")
    
    # 1. Relocate nodes: improves element quality (angles/aspect ratio)
    # without trying to equalize the size of adjacent elements.
    # This preserves your adaptive refinement density.
    method_relocate = "Relocate2D" if dim == 2 else "Relocate3D"
    gmsh.model.mesh.optimize(method_relocate)
    
    # 2. General optimization: Uses Gmsh's default heuristic to fix
    # any remaining topological issues or poor quality elements.
    for i in range(iterations):
        gmsh.model.mesh.optimize("")

    print("--- Optimization Complete ---")

def generate_and_save_mesh(geo_filename, msh_filename, initial_mesh_size):
    gmsh.model.occ.synchronize()
    gmsh.write(geo_filename)
    print(f"Geometry file saved in {geo_filename} successfully")

    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), initial_mesh_size)

    gmsh.model.mesh.generate(2)

    optimize_mesh()
    
    # gmsh.fltk.run()            # Uncomment to have the gmsh view
    gmsh.write(msh_filename)
    print(f"Mesh file saved in {msh_filename} successfully")