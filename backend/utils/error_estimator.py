import numpy as np

from backend.rwg.rwg1 import load_mesh_file, Points, Triangles


def simple_estimation(surface_current_density):
    """
    Normalizes the surface current density to a range between 0 and 1.
    Values close to the minimum will be near 0, and values significantly
    higher than the average (towards the maximum) will be near 1.
    """
    
    # Step 1: Get the magnitude (handle complex numbers if present)
    j_mag = np.abs(surface_current_density)
    
    # Step 2: Identify the range boundaries
    min_val = np.min(j_mag)
    max_val = np.max(j_mag)
    
    # Step 3: Linear remapping (Min-Max Scaling)
    # Formula: (x - min) / (max - min)
    range_width = max_val - min_val
    
    if range_width == 0:
        # Avoid division by zero if all values are identical
        return np.zeros_like(j_mag)
    
    # Values near min become 0, values near max become 1
    normalized_error = (j_mag - min_val) / range_width

    return normalized_error

def get_simplified_mesh_properties(points_obj, triangles_obj):
    """
    Simplifies property extraction by using attributes already present 
    in the Triangles object from rwg1.py.
    """
    # triangles_data is already 0-indexed and of shape (3, M) from load_mesh_file
    triangles = triangles_obj.triangles
    points = points_obj.points
    M = triangles.shape[1]

    # Neighbor identification logic
    # Note: If your rwg1.py already links triangles_plus and triangles_minus, 
    # you could optimize this even further.
    neighbors = np.full((M, 3), -1, dtype=int)
    edge_map = {}

    for tri_idx in range(M):
        # Unique edge representation (sorted vertex indices)
        edges_list = [
            tuple(sorted((triangles[0, tri_idx], triangles[1, tri_idx]))),
            tuple(sorted((triangles[1, tri_idx], triangles[2, tri_idx]))),
            tuple(sorted((triangles[2, tri_idx], triangles[0, tri_idx])))
        ]
        for edge_slot, edge in enumerate(edges_list):
            if edge in edge_map:
                neighbor_idx, n_slot = edge_map[edge]
                neighbors[tri_idx, edge_slot] = neighbor_idx
                neighbors[neighbor_idx, n_slot] = tri_idx
            else:
                edge_map[edge] = (tri_idx, edge_slot)

    return neighbors

def compute_curl_estimation(surface_current_density, neighbors):
    """
    Corrected Fast Vectorized Curl Estimation (CE) for scalar input (M,).
    
    Parameters:
        * surface_current_density: np.ndarray of shape (M,) representing magnitude or coefficients.
        * neighbors: np.ndarray of shape (M, 3) containing neighbor indices.
        * tangents: Tuple of 3 arrays (3, M) - Not strictly used if input is scalar, 
                    but kept for signature compatibility.
    """
    # Fix 1: Access the first dimension for shape (M,)
    M = surface_current_density.shape[0]
    ce_error = np.zeros(M)
    
    # Fix 2: Use np.abs for scalar normalization instead of linalg.norm
    max_J = np.max(np.abs(surface_current_density))
    if max_J == 0: max_J = 1.0

    # Iterate over the 3 possible edges of each triangle
    for i in range(3):
        neighbor_indices = neighbors[:, i]
        mask = neighbor_indices != -1 # Filter out boundary edges (no neighbor)
        
        # Fix 3: Use 1D indexing [mask] instead of 2D [:, mask]
        # Calculate scalar difference between current in triangle n and its neighbor m
        diff_J = surface_current_density[mask] - surface_current_density[neighbor_indices[mask]]
        
        # NOTE: Since input is scalar (M,), we cannot perform a vector dot product (einsum).
        # We assume the "tangential jump" is the absolute difference between these scalars.
        # If you need the true CE method, you must provide J as (3, M) vectors.
        ce_error[mask] += np.abs(diff_J)

    # Final normalization as per Haipl's formula
    return ce_error / (3 * max_J)

def compute_error_estimation(mesh_file, current_density_vector, alpha=0.5):
    """
    Main execution flow using rwg1.py objects.
    """
    # 1. Use the load_mesh_file from rwg1.py to get structured objects
    p, t = load_mesh_file(mesh_file)

    # 2. Define points and triangles from the mesh
    points_obj = Points(p)
    triangles_obj = Triangles(t)
    
    # 3. Get connectivity and tangents
    neighbors = get_simplified_mesh_properties(points_obj, triangles_obj)

    # print(f"neighbord = \n{neighbors}")
    
    # 4. Compute Curl Estimation (CE)
    # We pass the raw current density and pre-computed properties
    ce_vals = compute_curl_estimation(current_density_vector, neighbors)

    return ce_vals
    
    # 5. Compute Simple Estimation (SE) 
    # Directly use triangles_obj.triangles_area instead of recalculating
    se_vals = compute_simple_estimation(current_density_vector, triangles_obj.triangles_area)
    
    # 6. Hybrid Result
    hybrid_error = (alpha * ce_vals) + ((1 - alpha) * se_vals)
    
    return hybrid_error