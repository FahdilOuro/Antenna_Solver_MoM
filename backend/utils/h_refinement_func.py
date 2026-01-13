from types import SimpleNamespace
import gmsh
import numpy as np
from scipy.spatial import KDTree

from backend.utils.gmsh_function import optimize_mesh, setup_performance_config

def Grid(geo_path):
    """
    Loads a BREP file and generates centroids using a spacing 
    fixed at 10% of the object's diagonal size.
    
    Args:
        file_path (str): Path to the .brep file.
        
    Returns:
        np.array: A numpy array of shape (N, 3) containing the grid points.
    """
    
    # 1. Initialize Gmsh
    gmsh.initialize()
    setup_performance_config()

    # 2. Load geometry
    gmsh.merge(geo_path)
    gmsh.model.occ.synchronize()

    # 3. Calculate Automatic Spacing (10% of Bounding Box Diagonal)
    # getBoundingBox(-1, -1) gets the box for the whole model
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
    diagonal = np.sqrt((xmax - xmin)**2 + (ymax - ymin)**2 + (zmax - zmin)**2)

    # Set target_spacing to exactly 10%
    target_spacing = 0.07 * diagonal
    print(f"Object diagonal: {diagonal:.4f}")
    print(f"Fixed spacing (10%): {target_spacing:.4f}")

    # 4. Configure Mesh Options
    gmsh.option.setNumber("Mesh.MeshSizeMin", target_spacing)
    gmsh.option.setNumber("Mesh.MeshSizeMax", target_spacing)
    gmsh.option.setNumber("Mesh.Algorithm", 6) # Frontal-Delaunay

    # 5. Generate 2D Mesh
    gmsh.model.mesh.generate(2)

    # 6. Mesh Optimization
    # Smoothes the distribution for better centroid regularity
    gmsh.model.mesh.optimize("Laplace2D", niter=3)
    gmsh.model.mesh.optimize("Relocate2D")

    # 7. Extract Elements and Nodes
    # Retrieve all 2D elements (Surfaces)
    elem_types, elem_tags, node_tags = gmsh.model.mesh.getElements(2)

    # Get all nodes to build a coordinate lookup map
    all_node_tags, all_coords, _ = gmsh.model.mesh.getNodes()
    node_map = {tag: all_coords[i*3 : i*3+3] for i, tag in enumerate(all_node_tags)}

    centroids = []

    # 8. Compute Centroids
    for i, e_type in enumerate(elem_types):
        # Get number of nodes per element (3 for triangles, 4 for quads)
        _, _, _, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(e_type)
        current_elem_node_tags = node_tags[i]
        
        # Loop through elements of this type
        for j in range(0, len(current_elem_node_tags), num_nodes):
            element_nodes = current_elem_node_tags[j : j + num_nodes]
            
            # Fetch node coordinates and calculate the mean (barycenter)
            coords = [node_map[tag] for tag in element_nodes]
            centroid = np.mean(coords, axis=0)
            centroids.append(centroid)

    centroids_array = np.array(centroids)
    print(f"Generated {len(centroids_array)} points.")

    gmsh.finalize()

    return centroids_array

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

def generate_embedded_grid(geo_path):
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
    grid_points = Grid(geo_path)
    
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

def define_mesh_by_grid_refined(geo_file_path, grid_coords, size_to_apply, background_size):
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

    # Optimise the mesh
    optimize_mesh()

    gmsh.model.mesh.generate(2)
    gmsh.fltk.run()              # Uncomment to show

def get_mesh_centroids(mesh_file_path):
    """
    Opens a specific mesh file and extracts only the centroids of the 2D elements.
    """
    # Initialize the Gmsh API
    gmsh.initialize()
    setup_performance_config()
    
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

def initialize_refinement_config(grid_points, initial_mesh_size, iterations=3, threshold=0.75, gamma=0.3):
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