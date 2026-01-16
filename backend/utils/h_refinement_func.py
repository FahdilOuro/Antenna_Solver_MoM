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

    # Set target_spacing
    ratio = 0.05
    target_spacing = ratio * diagonal
    print(f"Object diagonal: {diagonal:.4f}")
    print(f"Fixed spacing ({ratio*100}%): {target_spacing:.4f}")

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

def view_grid_point(file_path, points_grid_coords):
    """
    Loads a geometry and embeds specific points into the surface topology.
    This allows visualizing the points as geometric vertices without generating a full mesh.
    
    Args:
        file_path (str): Path to the .brep file.
        points_grid_coords (np.array): Numpy array of shape (N, 3) containing coordinates.
    """
    
    # 1. Initialize Gmsh
    if not gmsh.isInitialized():
        gmsh.initialize()
    
    # 2. Load the Geometry
    gmsh.merge(file_path)
    gmsh.model.occ.synchronize()

    # 3. Get the Target Surface
    # We retrieve all 2D entities (surfaces).
    entities = gmsh.model.getEntities(2)
    if not entities:
        print("No surface found in the file.")
        gmsh.finalize()
        return
    
    # Assuming we target the first surface found in the file
    surface_tag = entities[0][1]

    # 4. Create Geometric Points
    # We iterate through the coordinates and add them as points in the CAD kernel (OpenCASCADE).
    point_tags = []
    for pt in points_grid_coords:
        # addPoint(x, y, z, meshSize, tag)
        # We perform the check to ensure data is valid floats
        x, y, z = pt[0], pt[1], pt[2]
        tag = gmsh.model.occ.addPoint(x, y, z)
        point_tags.append(tag)
    
    # 5. Synchronize
    # This is crucial: it pushes the new points from the CAD kernel to the Gmsh model.
    # Without this, the 'point_tags' exist in OCC but not in the Gmsh model for embedding.
    gmsh.model.occ.synchronize()

    # 6. Embed the Points
    # This constrains the points to belong to the surface.
    # 0 = dimension of the entity to embed (0 for points)
    # point_tags = list of point tags
    # 2 = dimension of the target entity (2 for surface)
    # surface_tag = tag of the surface
    print(f"Embedding {len(point_tags)} points into Surface {surface_tag}...")
    gmsh.model.mesh.embed(0, point_tags, 2, surface_tag)

    # Note: We do NOT run gmsh.model.mesh.generate(2) here as requested.
    
    # 7. Visualization
    # To see the points clearly, we ensure Geometry Points are visible in the GUI options.
    gmsh.option.setNumber("Geometry.Points", 1)      # Show geometric points
    gmsh.option.setNumber("Geometry.PointSize", 5)   # Increase point size for visibility
    gmsh.option.setColor("Geometry.Points", 255, 0, 0) # Make points Red

    # Run the GUI
    gmsh.fltk.run()
    
    # 8. Clean up
    gmsh.finalize()

def compute_vicinity_radius(grid_points):
    """
    Computes the vicinity radius r_vicinity for a set of grid points.
    
    This implementation follows Algorithm 1:
    1. Find the distance to the nearest neighbor (r_n) for each point.
    2. r_vicinity is the maximum of these distances.
    
    Args:
        grid_points (np.ndarray): Array of shape (N, 3) containing point coordinates.
        
    Returns:
        float: The calculated vicinity radius.
    """
    # Safety check: if there are fewer than 2 points, distance cannot be calculated
    if len(grid_points) < 2:
        return 0.0

    # Step 1: Build a KD-Tree for efficient spatial neighbor search
    # This optimizes the O(N^2) complexity of a naive search to O(N log N)
    tree = KDTree(grid_points)

    # Step 2: Query the two nearest neighbors for each point
    # k=2 because the first result (index 0) is the point itself (distance = 0)
    # the second result (index 1) is the actual nearest distinct neighbor
    distances, _ = tree.query(grid_points, k=2)

    # Step 3: Extract the distances to the nearest neighbor (r_n)
    # We take the second column which corresponds to the distance to the 2nd neighbor
    nearest_neighbor_distances = distances[:, 1]

    # Step 4: Compute the final radius as the maximum of all nearest-neighbor distances
    # Mathematical equivalent: r_vicinity = max(r_n)
    r_vicinity = np.max(nearest_neighbor_distances)

    print(f"Computed r_vicinity: {r_vicinity:.6f}")
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

def define_mesh_by_grid_refined(geo_file_path, grid_coords, sizes_array, background_size):
    """
    Refines the mesh ONLY where needed. Points with background_size are ignored
    to prevent over-constraining the Gmsh Delaunay algorithm.
    """
    # 1. Filter: Only keep points that are actually refined
    # We use a small epsilon (1e-6) to avoid floating point issues
    refined_mask = sizes_array < (background_size - 1e-6)
    
    if not np.any(refined_mask):
        # If no points are refined, just mesh with the default size
        setup_geometry(geo_file_path, []) # No points to embed
        gmsh.option.setNumber("Mesh.MeshSizeMin", background_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", background_size)
        gmsh.model.mesh.generate(2)
        return

    # Only these points will be sent to Gmsh
    active_coords = grid_coords[refined_mask]
    active_sizes = sizes_array[refined_mask]

    # 2. Setup geometry with ONLY active points
    surface_tag, point_tags = setup_geometry(geo_file_path, active_coords)

    # 3. Group active points by their unique refinement levels
    unique_sizes = np.unique(np.round(active_sizes, 8))
    field_ids = []
    
    for i, s_target in enumerate(unique_sizes):
        indices = np.where(np.round(active_sizes, 8) == s_target)[0]
        subset_tags = [point_tags[idx] for idx in indices]
        
        f_dist = 100 + (i * 2)
        f_thresh = 101 + (i * 2)
        
        gmsh.model.mesh.field.add("Distance", f_dist)
        gmsh.model.mesh.field.setNumbers(f_dist, "PointsList", subset_tags)
        
        gmsh.model.mesh.field.add("Threshold", f_thresh)
        gmsh.model.mesh.field.setNumber(f_thresh, "InField", f_dist)
        gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", s_target)
        gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", background_size)
        
        # Adaptive transition distance to keep triangles beautiful
        gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", s_target * 2)
        gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", s_target * 10)
        gmsh.model.mesh.field.setNumber(f_thresh, "StopAtDistMax", 1)
        
        field_ids.append(f_thresh)

    # 4. Finalizing Mesh Fields
    f_min = 1000
    gmsh.model.mesh.field.add("Min", f_min)
    gmsh.model.mesh.field.setNumbers(f_min, "FieldsList", field_ids)
    gmsh.model.mesh.field.setAsBackgroundMesh(f_min)

    # 5. Global Options
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    
    # Crucial: This only embeds the points that actually need a specific size
    gmsh.model.mesh.embed(0, point_tags, 2, surface_tag)
    
    gmsh.model.mesh.generate(2)

    # Optimise the mesh
    optimize_mesh()

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

def initialize_refinement_config(grid_points, initial_mesh_size, iterations=4, threshold=0.75, gamma=0.3):
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