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

def mesh_and_gap_parameters(frequency, resolution=20, gap_fraction=0.15, n_edges_min=3, c=3e8):
    """
    Compute mesh_size and gap_width for MoM simulation with stability guardrails.

    Physics of the gap_fraction parameter
    --------------------------------------
    gap_width is set as a FIXED fraction of λ, independent of mesh refinement.
    This is critical: if gap_width ∝ mesh_size (both → 0 together), the gap source
    approaches a delta function on a 2D conducting surface, whose Green's function has
    a logarithmic singularity → Z_in ∝ 1/mesh_size → ∞.

    With gap_width = gap_fraction × λ fixed:
      - Z_in converges to a finite physical value as mesh_size → 0
      - The mesh only controls accuracy, not the excitation model

    Stability constraint
    --------------------
    The mesh must be fine enough that the gap contains ≥ n_edges_min elements:
        mesh_size ≤ gap_width / n_edges_min  ↔  resolution ≥ n_edges_min / gap_fraction

    Default: gap_fraction=0.15, n_edges_min=3  →  resolution_min = 3/0.15 = 20
    (So the default resolution=20 satisfies the constraint exactly.)

    Empirical validation (strip dipole, 75 MHz, λ=4 m, from Analyse.md)
    ----------------------------------------------------------------------
    gap_fraction=0.15 (gap=0.6 m), resolution=20: Z_in = 87 + 42j Ω  ✓
    gap_fraction=0.075 (gap=0.3 m), resolution=40: Z_in = 85 + 45j Ω  ✓
    Both are consistent with the half-wave dipole reference (~73–85 Ω).

    Parameters
    ----------
    frequency   : float — operating frequency [Hz]
    resolution  : float — desired λ/resolution (may be auto-increased if too coarse)
    gap_fraction: float — W = gap_fraction × λ; empirical range [0.05, 0.20]
                          Default 0.15 allows resolution=20 with 3 edges in gap.
    n_edges_min : int   — minimum number of mesh edges inside gap (numerical stability)
    c           : float — speed of light [m/s]

    Returns
    -------
    mesh_size  : float — element size [m]  (may differ from λ/resolution if adjusted)
    gap_width  : float — gap source width [m]
    wavelength : float — free-space wavelength [m]
    """
    wavelength = c / frequency
    gap_width  = gap_fraction * wavelength          # fixed physical gap

    desired_mesh     = wavelength / resolution
    mesh_size_max    = gap_width / n_edges_min      # stability upper bound on mesh_size

    if desired_mesh > mesh_size_max:
        # Requested resolution is too coarse: auto-tighten
        mesh_size       = mesh_size_max
        eff_res         = wavelength / mesh_size
        res_min_needed  = n_edges_min / gap_fraction
        print(f"[mesh_and_gap] WARNING: resolution=λ/{resolution:.0f} too coarse "
              f"(minimum needed: λ/{res_min_needed:.0f} for gap_fraction={gap_fraction}).")
        print(f"  → Auto-adjusted to λ/{eff_res:.0f} (mesh_size = {mesh_size:.4f} m)")
    else:
        mesh_size = desired_mesh

    # Guardrail: mesh not finer than λ/100 (risk of Z-matrix ill-conditioning)
    if mesh_size < wavelength / 100:
        print(f"[mesh_and_gap] WARNING: mesh_size = λ/{wavelength/mesh_size:.0f} < λ/100 "
              f"— Z-matrix may be ill-conditioned. Consider reducing resolution.")

    # Guardrail: gap not too large (non-local excitation distorts the current pattern)
    if gap_fraction > 0.20:
        print(f"[mesh_and_gap] WARNING: gap_fraction={gap_fraction:.2f} → "
              f"gap = {gap_width:.3f} m = {gap_fraction:.2f}λ > 0.20λ — physically non-local.")

    n_edges = gap_width / mesh_size
    print(f"λ = {wavelength:.4f} m  |  mesh_size = λ/{wavelength/mesh_size:.0f} = {mesh_size:.4f} m  "
          f"|  gap = {gap_fraction:.2f}λ = {gap_width:.4f} m  |  N_edges_in_gap ≈ {n_edges:.1f}")

    return mesh_size, gap_width, wavelength


def check_simulation_stability(Z_in, frequency, mesh_size, gap_width, c=3e8,
                                R_min=5.0, R_max=500.0):
    """
    Post-hoc sanity check on MoM radiation simulation results.

    Tests:
      1. Z_in is finite and in a physically plausible range [R_min, R_max]
      2. gap_width / mesh_size ≥ 3 (enough edges in gap)
      3. mesh_size is within the recommended MoM range [λ/100, λ/10]

    Parameters
    ----------
    Z_in      : complex scalar or 1-D array — input impedance per port [Ω]
    frequency : float — operating frequency [Hz]
    mesh_size : float — mesh element size used in the simulation [m]
    gap_width : float — gap width used in the simulation [m]
    c         : float — speed of light [m/s]
    R_min     : float — minimum plausible R_in [Ω]  (default 5)
    R_max     : float — maximum plausible R_in [Ω]  (default 500)

    Returns
    -------
    bool — True if all checks pass
    """
    wavelength = c / frequency
    Z_in = np.atleast_1d(np.asarray(Z_in, dtype=complex))
    ok   = True

    print("\n[Stability Check]")

    # --- Per-port impedance checks ---
    for idx, z in enumerate(Z_in):
        tag = f"Port {idx}"
        R, X = np.real(z), np.imag(z)

        if not (np.isfinite(R) and np.isfinite(X)):
            print(f"  {tag}: Z_in = {z} — INFINITE/NaN → simulation failed"); ok = False
            continue

        status = "✓ OK"
        if R < R_min:
            status = f"✗ R_in < {R_min} Ω — negative/near-zero resistance"; ok = False
        elif R > R_max:
            status = f"✗ R_in > {R_max} Ω — likely divergence (gap too large or wrong excitation)"; ok = False

        sign = "+" if X >= 0 else ""
        print(f"  {tag}: Z_in = {R:.2f}{sign}{X:.2f}j Ω  — {status}")

    # --- Gap geometry check ---
    ratio = gap_width / mesh_size
    if ratio < 0.67:
        print(f"  Gap check: gap/mesh = {ratio:.2f} < 0.67 — NO EDGES in gap. P_in = 0. Results INVALID."); ok = False
    elif ratio < 3.0:
        print(f"  Gap check: gap/mesh = {ratio:.2f} — borderline (< 3). Results may be unstable.")
    else:
        print(f"  Gap check: gap/mesh = {ratio:.1f} ≥ 3 — ✓ OK")

    # --- Mesh resolution check ---
    lam_over_mesh = wavelength / mesh_size
    if mesh_size > wavelength / 10:
        print(f"  Mesh check: λ/{lam_over_mesh:.0f} > λ/10 — too coarse for accurate MoM."); ok = False
    elif mesh_size < wavelength / 100:
        print(f"  Mesh check: λ/{lam_over_mesh:.0f} < λ/100 — ill-conditioning risk.")
    else:
        print(f"  Mesh check: λ/{lam_over_mesh:.0f} — ✓ in recommended range [λ/10, λ/100]")

    print(f"  Overall: {'✓ PASS' if ok else '✗ FAIL — review warnings above'}")
    return ok


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

def generate_and_save_mesh(path=None, initial_mesh_size=1e10, geo_filename=None, msh_filename=None):
    """
    Generates a 2D mesh using Gmsh and saves the geometry and mesh files.

    This function checks if the required filenames are provided directly
    or via a path object. If they are missing, it raises a ValueError.

    Args:
        geo_filename (str, optional): The name of the geometry file.
        msh_filename (str, optional): The name of the mesh file.
        path (object, optional): An object containing .geo and .msh attributes.
        initial_mesh_size (float, optional): The initial element size for the mesh. Defaults to 1e10.

    Raises:
        ValueError: If the required filenames are not provided.

    Returns:
        bool: True if successful, False otherwise.
    """
    
    # Check if a path object is NOT provided
    if path is None:
        # Check if either geo_filename or msh_filename is missing
        if geo_filename is None or msh_filename is None:
            # Raise an error directly to indicate missing arguments
            raise ValueError("Both 'geo_filename' and 'msh_filename' must be provided when 'path' is None.")
    else:
        # Extract the filenames from the path object's attributes
        geo_filename = path.geo
        msh_filename = path.msh

    try:
        # Synchronize the internal CAD representation with the Gmsh model
        gmsh.model.occ.synchronize()

        # Write the geometry data to the specified .geo file
        gmsh.write(geo_filename)
        print(f"Geometry file saved in {geo_filename} successfully")

        # Retrieve all points (dimension 0) in the model to set their mesh size
        points = gmsh.model.getEntities(0)

        # Apply the specified initial mesh size to all retrieved points
        gmsh.model.mesh.setSize(points, initial_mesh_size)

        # Generate a 2D mesh based on the geometry
        gmsh.model.mesh.generate(2)

        # Call the external function to optimize the generated mesh
        optimize_mesh()
        
        # gmsh.fltk.run()            # Uncomment to have the gmsh view

        # Write the generated mesh data to the specified .msh file
        gmsh.write(msh_filename)
        print(f"Mesh file saved in {msh_filename} successfully")

        # Return True to indicate successful execution
        return True

    except Exception as e:
        # Catch any exceptions raised by Gmsh and print the error message
        print(f"An error occurred during mesh generation: {e}")
        # Return False to indicate failure
        return False

def create_hollow_sphere(radius=1.0, mesh_size=0.055, save_msh_file='data/gmsh_files/hollow_sphere.msh'):
    """
    Creates the geometry of a hollow sphere (surface only) using the Gmsh OpenCASCADE kernel.
    
    Args:
        radius (float): The radius of the sphere.
        mesh_size (float): The target size of the mesh elements.
    """
    # Initialize gmsh
    gmsh.initialize()
    gmsh.model.add("HollowSphere")

    # Create a sphere. By default, this creates a volume.
    # The function returns the tag of the entity.
    sphere_tag = gmsh.model.occ.addSphere(0, 0, 0, radius)
    
    # Synchronize to propagate changes to the model
    gmsh.model.occ.synchronize()

    # To keep ONLY the surface, we can remove the volume but keep the boundary.
    # In Gmsh, a sphere volume is usually composed of one or more surfaces.
    # For a simple shell, we retrieve the boundaries of the volume (3D).
    # Dim 3 = Volume, Dim 2 = Surface.
    sphere_volume = [(3, sphere_tag)]
    boundaries = gmsh.model.getBoundary(sphere_volume, combined=False)
    
    # We remove the volume (dim 3) to ensure we only mesh the shell (dim 2)
    gmsh.model.removeEntities(sphere_volume)

    # Set mesh size field (optional but recommended for control)
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)

    # Generate 2D mesh (surface)
    gmsh.model.mesh.generate(2)

    # Save the mesh
    gmsh.write(save_msh_file)

    # Launch the GUI to visualize
    # gmsh.fltk.run()

    gmsh.finalize()