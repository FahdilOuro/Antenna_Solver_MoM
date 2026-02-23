import numpy as np


def single_gap_source(edges, index_feeding_edges, voltage_amplitude):
    edges_length = edges.edges_length
    total_number_of_edges = edges.total_number_of_edges
    voltage = np.zeros(total_number_of_edges, dtype=complex)

    voltage[index_feeding_edges] = voltage_amplitude * edges_length[index_feeding_edges]

    return voltage

def multiple_gap_sources(triangles, edges, vecteurs_rho, voltage_amplitude, feed_point, excitation_unit_vector, gap_width=0.08):
    """
    Implements the Enhanced Gap Source model centered at a specific feed_point.
    
    Parameters:
    - feed_point: np.array([x, y, z]) location of the excitation.
    - gap_width: The width W of the gap source.
    - excitation_unit_vector: Direction of the E-field ('x', 'y', or 'z').
    """
    
    # 1. Map the excitation direction to the correct coordinate index
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if excitation_unit_vector not in axis_map:
        raise ValueError("excitation_unit_vector must be 'x', 'y', or 'z'.")
    ax_idx = axis_map[excitation_unit_vector]
    # print(f"ax_idx = {ax_idx}")

    # 2. Determine the center of the gap on the relevant axis
    # If we excite along 'z', we only care about the z-coordinate of the feed_point
    gap_center = feed_point[ax_idx]
    # print(f"Gap_center = {gap_center}")

    # 3. Extract geometry data
    l_m = edges.edges_length
    tri_plus_idx = triangles.triangles_plus
    tri_minus_idx = triangles.triangles_minus  
    centroids = triangles.triangles_center 
    
    rho_p = vecteurs_rho.vecteur_rho_plus   
    rho_n = vecteurs_rho.vecteur_rho_minus  

    # 4. Define the Centered Window function
    # Math: [u(z - z_feed + W/2) - u(z - z_feed - W/2)]
    # This checks if the centroid coordinate is within distance W/2 of the gap_center
    def window_function(coord):
        # Calculate absolute distance from the feed point coordinate
        distance = np.abs(coord - gap_center)
        # Return 1.0 if inside the gap width, 0.0 otherwise
        return np.where(distance <= gap_width / 2, 1.0, 0.0)
    
    # print(f"gap_width : {gap_width}")

    # 5. Evaluate the window function at the centroids of T+ and T-
    # We only look at the coordinates along the excitation axis (ax_idx)
    coord_cp = centroids[ax_idx, tri_plus_idx]
    # print(f"coord_cp : {coord_cp}")

    coord_cn = centroids[ax_idx, tri_minus_idx]
    # print(f"coord_cn : {coord_cn}")

    # Check the minimum distance found in your mesh
    min_dist_p = np.min(np.abs(coord_cp - gap_center))
    min_dist_n = np.min(np.abs(coord_cn - gap_center))
    threshold = gap_width / 2

    # print(f"--- Debug Gap ---")
    # print(f"Threshold (gap_width/2): {threshold}")
    # print(f"Closest T+ centroid distance: {min_dist_p}")
    # print(f"Closest T- centroid distance: {min_dist_n}")

    if min_dist_p > threshold:
        raise ValueError("Warning: No T+ centroids are within the gap! The mesh is too coarse.")
    
    window_p = window_function(coord_cp)
    # print(f"window_p : {window_p}")
    window_n = window_function(coord_cn)
    # print(f"window_n : {window_p}")

    # 6. Apply Equation (7) components
    # Extract the component of rho vectors along the excitation axis
    rho_p_component = rho_p[ax_idx, :]
    rho_n_component = rho_n[ax_idx, :]

    # Calculate the common factor: (lm * V) / (2 * W)
    common_factor = (l_m * voltage_amplitude) / (2 * gap_width)
    # print(f"Common factor : {common_factor}")

    # Term 1 for T+ and Term 2 for T-
    term_plus = common_factor * window_p * rho_p_component
    # print(f"term_plus : {term_plus}")
    
    term_minus = common_factor * window_n * rho_n_component

    # Final excitation vector Vm
    voltage_vector = term_plus + term_minus

    return voltage_vector