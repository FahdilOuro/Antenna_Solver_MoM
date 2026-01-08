from types import SimpleNamespace
import gmsh
import numpy as np
from scipy.spatial import KDTree

def get_surface_orientation(surface_tag):
    """
    Extracts the reference point and builds a local orthonormal basis (u, v) 
    based on the surface normal.
    """
    # Get a reference point at the parametric center
    p0 = np.array(gmsh.model.getValue(2, surface_tag, [0.5, 0.5])).reshape(3)
  
    # Get the normal vector
    normal = np.array(gmsh.model.getNormal(surface_tag, [0.5, 0.5])).reshape(3)
    normal /= np.linalg.norm(normal)

    # Build local basis vectors (u_axis, v_axis)
    if abs(normal[0]) < 0.9:
        u_axis = np.cross(normal, [1, 0, 0])
    else:
        u_axis = np.cross(normal, [0, 1, 0])
    u_axis /= np.linalg.norm(u_axis)
    v_axis = np.cross(normal, u_axis)
    
    return p0, u_axis, v_axis

def calculate_grid_spacing(density, spacing_min=0.05, spacing_max=1.0):
    """
    Computes the step distance between points based on the density parameter.
    """
    density = max(0.01, min(1.0, density)) # Clamp density between 0.01 and 1
    return spacing_max * (spacing_min / spacing_max) ** density

def generate_and_filter_points(surface_tag, p0, u_axis, v_axis, spacing):
    """
    Scans the plane using vectorized NumPy operations and filters points 
    located inside the surface.
    """
    # 1. Determine search range using the Bounding Box
    bbox = gmsh.model.getBoundingBox(2, surface_tag)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox
    diag = np.sqrt((xmax-xmin)**2 + (ymax-ymin)**2 + (zmax-zmin)**2)

    # 2. Create 1D grids for local coordinates i and j
    i_range = np.arange(-diag, diag, spacing)
    j_range = np.arange(-diag, diag, spacing)

    # 3. Vectorization: Create a 2D meshgrid of (i, j) coordinates
    # I and J will be 2D matrices
    I, J = np.meshgrid(i_range, j_range)

    # 4. Transform all points to World Coordinates (x, y, z) at once
    # Resulting shape will be (N, 3) where N is the total number of candidates
    # Formula: P = P0 + I*U + J*V
    all_world_pts = (p0 + 
                     I.reshape(-1, 1) * u_axis + 
                     J.reshape(-1, 1) * v_axis)

    point_tags = []
    points_coords = []

    # 5. Filter points and add them to Gmsh
    # We still need a loop here because gmsh.model.isInside is not vectorized
    for world_pt in all_world_pts:
        # Pass the point as a list to gmsh.model.isInside
        if gmsh.model.isInside(2, surface_tag, world_pt.tolist()):
            # Add point to OpenCASCADE
            pt_tag = gmsh.model.occ.addPoint(world_pt[0], world_pt[1], world_pt[2])
            point_tags.append(pt_tag)
            points_coords.append(world_pt)
                
    return point_tags, np.array(points_coords)

def Grid(file_path, density=0.5):
    """
    Main orchestration function to load geometry, generate a grid, and save the result.
    """
    if not file_path.lower().endswith(".brep"):
        raise ValueError("Input file must be a .brep file")
    
    # Initialization
    gmsh.initialize()
    gmsh.merge(file_path)
    gmsh.model.occ.synchronize()

    # 1. Target the surface
    entities = gmsh.model.getEntities(2)
    if not entities:
        gmsh.finalize()
        return np.array([])
    surface_tag = entities[0][1]

    # 2. Get Geometry orientation
    p0, u_axis, v_axis = get_surface_orientation(surface_tag)

    # 3. Get Spacing
    spacing = calculate_grid_spacing(density)

    # 4. Generate points
    point_tags, points_coords = generate_and_filter_points(surface_tag, p0, u_axis, v_axis, spacing)

    # 5. Finalize Geometry and Mesh embedding
    gmsh.model.occ.synchronize()
    if point_tags:
        gmsh.model.mesh.embed(0, point_tags, 2, surface_tag)

    """# 6. Export and Clean up
    base, ext = os.path.splitext(file_path)
    gmsh.write(base + "_grid" + ext)"""
    
    # gmsh.fltk.run() # Uncomment to visualize
    gmsh.finalize()
    
    return points_coords

def compute_vicinity_radius(grid_points):
    """
    Computes the vicinity radius r_vicinity for a set of grid points.
    
    Algorithm:
    1. For each point, find the distance to its nearest neighbor.
    2. r_vicinity is the maximum of these minimum distances.
    
    Args:
        grid_points (np.ndarray): Array of shape (N, 3) containing point coordinates.
        
    Returns:
        float: The calculated vicinity radius.
    """
    if len(grid_points) < 2:
        return 0.0

    # Step 1: Build a KD-Tree for efficient spatial queries
    tree = KDTree(grid_points)

    # Step 2: Query the two nearest neighbors for each point
    # k=2 because the 1st neighbor is the point itself (distance 0)
    distances, indices = tree.query(grid_points, k=2)

    # Step 3: Extract the distance to the 2nd neighbor (the actual nearest neighbor)
    # distances[:, 1] contains r_n for each point
    nearest_neighbor_distances = distances[:, 1]

    # Step 4: Find the maximum of all these nearest-neighbor distances
    r_vicinity = np.max(nearest_neighbor_distances)

    print(f"\nComputed r_vicinity: {r_vicinity}")
    return r_vicinity

def generate_embedded_grid(geo_path, spacing):
    """
    Generates a grid of points embedded within the geometry and 
    computes the vicinity radius for those points.
    
    Args:
        geo_path (str): Path to the geometry file (.brep or .geo).
        spacing (float): The step or resolution for the grid generation.
        
    Returns:
        tuple: (grid_points, r_vicinity)
    """
    
    # 1. Initialize the Grid object using the geometry file and spacing
    # This step maps the points inside the defined plate geometry
    grid_points = Grid(geo_path, spacing)
    
    # 2. Compute the vicinity radius
    # This identifies the influence area or connectivity for the grid points
    r_vicinity = compute_vicinity_radius(grid_points)

    # 3. Debugging output to verify grid density
    print(f"Grid generation complete. Points shape: {grid_points.shape}")
    
    return grid_points, r_vicinity

# --- 3. Helper Functions ---
def setup_geometry(geo_file_path, grid_coords):
    """
    General function: Loads ANY geometry and syncs grid points.
    Works for any shape (not just rectangles).
    """
    # 1. Import the geometry (could be any BREP file)
    tags = gmsh.model.occ.importShapes(geo_file_path)
    gmsh.model.occ.synchronize()

    # 2. Get the surface(s) automatically
    # We take the first surface found in the file
    surfaces = gmsh.model.getEntities(2)
    if not surfaces:
        raise ValueError("No surface found in the BREP file!")
    surface_tag = surfaces[0][1]

    # 3. Add the grid points to the CAD factory
    point_tags = []
    for coord in grid_coords:
        pt = gmsh.model.occ.addPoint(coord[0], coord[1], coord[2])
        point_tags.append(pt)
    
    gmsh.model.occ.synchronize()
    
    # 4. Return the surface tag and the new point tags
    return surface_tag, point_tags

def define_mesh_by_grid(geo_file_path, grid_coords, size_to_apply):
    surface_tag, point_tags = setup_geometry(geo_file_path, grid_coords)
    # Set sizes on all grid points
    for pt_tag in point_tags: gmsh.model.mesh.setSize([(0, pt_tag)], size_to_apply)

    # Embed all grid points into the surface
    gmsh.model.mesh.embed(0, point_tags, 2, surface_tag)

    # Set meshing options
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 1)  # Frontal-Delaunay

    # Generate mesh
    gmsh.model.mesh.generate(2)
    gmsh.fltk.run()

def define_mesh_by_grid_refined(geo_file_path, grid_coords, size_to_apply, background_size):
    surface_tag, point_tags = setup_geometry(geo_file_path, grid_coords)

    # 1. Create a Distance field
    # This calculates the distance to the specified points
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "PointsList", point_tags)

    # 2. Create a Threshold field
    # This maps the distance value to a specific mesh size
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", size_to_apply)    # Size inside the influence zone
    gmsh.model.mesh.field.setNumber(2, "SizeMax", background_size) # Size outside the influence zone
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.5)             # Radius where size is strictly SizeMin
    gmsh.model.mesh.field.setNumber(2, "DistMax", 2.0)             # Distance where size reaches SizeMax

    # 3. Set the Threshold field as the background mesh size
    gmsh.model.mesh.field.setAsBackgroundMesh(2)

    # Disable the global extension from points/boundaries to let the Field take control
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    
    # Standard meshing options
    gmsh.model.mesh.embed(0, point_tags, 2, surface_tag)
    gmsh.option.setNumber("Mesh.Algorithm", 1)
    gmsh.model.mesh.generate(2)
    gmsh.fltk.run()

def define_mesh_by_grid_refined_2(geo_file_path, grid_coords, size_to_apply, background_size):
    surface_tag, point_tags = setup_geometry(geo_file_path, grid_coords)

    # 1. Create a Distance field to measure distance from our grid points
    field_dist = 1
    gmsh.model.mesh.field.add("Distance", field_dist)
    gmsh.model.mesh.field.setNumbers(field_dist, "PointsList", point_tags)

    # 2. Create a Threshold field to define the local refinement zone
    field_thresh = 2
    gmsh.model.mesh.field.add("Threshold", field_thresh)
    gmsh.model.mesh.field.setNumber(field_thresh, "InField", field_dist)
    
    # Mesh size inside the DistMin radius
    gmsh.model.mesh.field.setNumber(field_thresh, "SizeMin", size_to_apply)
    # Mesh size outside the DistMax radius
    gmsh.model.mesh.field.setNumber(field_thresh, "SizeMax", background_size)
    
    # Radius of maximum refinement (influence stays concentrated here)
    gmsh.model.mesh.field.setNumber(field_thresh, "DistMin", 0.2) 
    # Distance at which we reach the background size
    gmsh.model.mesh.field.setNumber(field_thresh, "DistMax", 1.0)
    
    # CRITICAL: Stop the field influence beyond DistMax 
    # This prevents the small size from leaking further than intended
    gmsh.model.mesh.field.setNumber(field_thresh, "StopAtDistMax", 1)

    # 3. Use the Min field to ensure we respect the background size everywhere
    # This acts as a safety layer
    field_min = 3
    gmsh.model.mesh.field.add("Min", field_min)
    gmsh.model.mesh.field.setNumbers(field_min, "FieldsList", [field_thresh])
    
    # Set the final field as the background mesh
    gmsh.model.mesh.field.setAsBackgroundMesh(field_min)

    # 4. Disable default Gmsh behaviors as suggested by the documentation
    # This prevents points and boundaries from forcing their own sizes
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    # Set hard limits for the mesh size to avoid extreme values
    gmsh.option.setNumber("Mesh.MeshSizeMin", size_to_apply)
    gmsh.option.setNumber("Mesh.MeshSizeMax", background_size)

    # Embed points and generate
    gmsh.model.mesh.embed(0, point_tags, 2, surface_tag)
    gmsh.model.mesh.generate(2)
    gmsh.fltk.run()              # Uncomment to show

# Define the callback OUTSIDE to ensure it persists in memory
# and is easily accessible by the Gmsh C++ core.
def mesh_size_manager(dim, tag, x, y, z, lc):
    """
    Global callback function. 
    Gmsh calls this for every potential mesh node.
    """
    # ACCESS GLOBAL PARAMETERS 
    # (In a production script, you might use a class or global dictionary)
    global_target_size = gmsh.option.getNumber("Mesh.MeshSizeMin")
    global_max_size = gmsh.option.getNumber("Mesh.MeshSizeMax")
    
    # RADICAL LOGIC:
    # If the suggested size (lc) is starting to grow (meaning we are leaving
    # the DistMin zone), we immediately force it to a larger size to 
    # prevent the 'bleed' or propagation of small elements.
    if lc > global_target_size * 1.05:
        return global_max_size
    
    return lc

def define_mesh_by_grid_refined_3(geo_file_path, grid_coords, size_to_apply, background_size):
    surface_tag, point_tags = setup_geometry(geo_file_path, grid_coords)

    # --- 1. Field Setup ---
    # We still use fields because they provide the initial 'lc' value to the callback
    field_dist = 1
    gmsh.model.mesh.field.add("Distance", field_dist)
    gmsh.model.mesh.field.setNumbers(field_dist, "PointsList", point_tags)

    field_thresh = 2
    gmsh.model.mesh.field.add("Threshold", field_thresh)
    gmsh.model.mesh.field.setNumber(field_thresh, "InField", field_dist)
    gmsh.model.mesh.field.setNumber(field_thresh, "SizeMin", size_to_apply)
    gmsh.model.mesh.field.setNumber(field_thresh, "SizeMax", background_size)
    
    # Keep DistMin small to localize the size_to_apply
    gmsh.model.mesh.field.setNumber(field_thresh, "DistMin", 0.1)
    # DistMax is where lc reaches background_size in the field calculation
    gmsh.model.mesh.field.setNumber(field_thresh, "DistMax", 0.5)
    gmsh.model.mesh.field.setNumber(field_thresh, "StopAtDistMax", 1)

    gmsh.model.mesh.field.setAsBackgroundMesh(field_thresh)

    # --- 2. Options & Constraints ---
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    # These are used by the callback via getNumber
    gmsh.option.setNumber("Mesh.MeshSizeMin", size_to_apply)
    gmsh.option.setNumber("Mesh.MeshSizeMax", background_size)

    # --- 3. Register Callback ---
    # Ensure the function name passed is reachable
    gmsh.model.mesh.setSizeCallback(mesh_size_manager)

    # --- 4. Generation ---
    gmsh.model.mesh.embed(0, point_tags, 2, surface_tag)
    gmsh.option.setNumber("Mesh.Algorithm", 6) # Frontal-Delaunay
    
    gmsh.model.mesh.generate(2)
    gmsh.fltk.run()

def get_mesh_centroids(mesh_file_path):
    """
    Opens a specific mesh file and extracts only the centroids of the 2D elements.
    """
    # Initialize the Gmsh API
    gmsh.initialize()
    
    # Load the mesh file provided in the path
    gmsh.open(mesh_file_path)

    all_centroids = []

    # Get all 2D entities (surfaces) from the model
    entities = gmsh.model.getEntities(2)

    for _, tag in entities:
        # Retrieve the barycenters (centroids) of the elements in this entity
        # Parameters: dim=2, tag=tag, fast=False, primary=True
        centroids = gmsh.model.mesh.getBarycenters(2, tag, False, True)
        
        # If centroids are found, reshape the flat array into a (N, 3) matrix
        if len(centroids) > 0:
            formatted_centroids = np.array(centroids).reshape(-1, 3)
            all_centroids.extend(formatted_centroids)

    # Convert the list to a NumPy array for easier processing
    all_centroids = np.array(all_centroids)

    # Finalize the Gmsh API to free resources
    gmsh.finalize()

    return all_centroids

def initialize_refinement_config(grid_points, initial_mesh_size, iterations=3, threshold=0.8, gamma=0.44):
    """
    Initializes hyperparameters as an object and the mesh size distribution array.
    Using SimpleNamespace allows accessing parameters with dot notation (e.g., config.max_iterations).
    """
    
    # 1. Create a configuration object using SimpleNamespace
    # This enables dot notation access: config.parameter_name
    config = SimpleNamespace(
        max_iterations=iterations,
        threshold_percentage=threshold,
        refinement_factor_gamma=gamma,
        initial_mesh_size=initial_mesh_size
    )
    
    # 2. Initialize the mesh size array based on the number of grid points
    current_sizes = np.ones(len(grid_points)) * initial_mesh_size
    
    # 3. Log initialization details
    print(f"Configuration initialized. Max iterations: {config.max_iterations}")
    print(f"Mesh size array initialized with uniform size: {initial_mesh_size}")

    return config, current_sizes