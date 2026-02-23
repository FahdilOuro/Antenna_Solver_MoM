import numpy as np


def multiple_gap_sources(triangles, edges, vecteurs_rho, voltage_amplitude, feed_point, excitation_unit_vector, gap_width):
    """
    Implements the Enhanced Gap Source model based on Equation (7).
    
    This function computes the excitation vector for RWG basis functions
    by evaluating the field distribution within a finite gap width.
    """
    
    # 1. Map excitation axis to coordinate index
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if excitation_unit_vector not in axis_map:
        raise ValueError("excitation_unit_vector must be 'x', 'y', or 'z'.")
    ax_idx = axis_map[excitation_unit_vector]
    gap_center = feed_point[ax_idx]

    # 2. Extract necessary geometric properties
    l_m = edges.edges_length
    tri_plus_idx = triangles.triangles_plus
    tri_minus_idx = triangles.triangles_minus  
    centroids = triangles.triangles_center 
    
    rho_p = vecteurs_rho.vecteur_rho_plus   
    rho_n = vecteurs_rho.vecteur_rho_minus  

    # 3. Extract coordinates along the excitation axis for centroids
    coord_cp = centroids[ax_idx, tri_plus_idx]
    coord_cn = centroids[ax_idx, tri_minus_idx]

    # 4. Mesh density validation
    # Ensure at least one triangle centroid falls within the gap threshold
    min_dist = min(np.min(np.abs(coord_cp - gap_center)), np.min(np.abs(coord_cn - gap_center)))
    threshold = gap_width / 2

    if min_dist > threshold:
        raise ValueError(f"Coarse mesh error: No centroids found within {gap_width} units of the feed point.")

    # 5. Compute the Window Function [u(z + W/2) - u(z - W/2)]
    # Evaluates to 1.0 if the centroid is inside the gap, 0.0 otherwise
    window_p = np.where(np.abs(coord_cp - gap_center) <= threshold, 1.0, 0.0)
    window_n = np.where(np.abs(coord_cn - gap_center) <= threshold, 1.0, 0.0)

    # 6. Apply the Enhanced Gap Source formula (Eq. 7)
    # The term involves the dot product of rho and the unit excitation vector
    rho_p_ax_idx = rho_p[ax_idx, :]
    rho_n_ax_idx = rho_n[ax_idx, :]

    # Calculation of the voltage contribution for each edge
    common_factor = (l_m * voltage_amplitude) / (2 * gap_width)
    term_plus = common_factor * window_p * rho_p_ax_idx
    term_minus = common_factor * window_n * rho_n_ax_idx

    return term_plus + term_minus