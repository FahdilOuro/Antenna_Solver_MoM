import numpy as np


def multiple_gap_sources(triangles, edges, vecteurs_rho, voltage_amplitude, feed_points, 
                         excitation_unit_vector, gap_width, phases=None):
    """
    Implements the Enhanced Gap Source model with independent location, amplitude, 
    phase, and excitation axis for each feed point.
    """
    
    # 1. Standardize feed_points to (N, 3)
    feed_points = np.atleast_2d(feed_points)
    num_feeds = feed_points.shape[0]

    # 2. Handle vector/scalar for voltage_amplitude
    if np.isscalar(voltage_amplitude):
        amplitudes = np.full(num_feeds, voltage_amplitude)
    else:
        amplitudes = np.asarray(voltage_amplitude)

    # 3. Handle vector/scalar for phases
    if phases is None:
        phases = np.zeros(num_feeds)
    elif np.isscalar(phases):
        phases = np.full(num_feeds, phases)
    else:
        phases = np.asarray(phases)

    # 4. Handle vector/scalar for excitation_unit_vector
    # If a single string is provided, use it for all feeds
    if isinstance(excitation_unit_vector, str):
        unit_vectors = [excitation_unit_vector] * num_feeds
    else:
        unit_vectors = excitation_unit_vector

    if len(unit_vectors) != num_feeds:
        raise ValueError("The number of excitation vectors must match the number of feed points.")

    # Pre-calculate complex coefficients
    complex_amplitudes = amplitudes * np.exp(1j * phases)

    # 5. Extract fixed geometry data
    num_edges = edges.total_number_of_edges
    l_m = edges.edges_length
    tri_plus_idx = triangles.triangles_plus
    tri_minus_idx = triangles.triangles_minus  
    centroids = triangles.triangles_center 
    
    total_voltage_vector = np.zeros(num_edges, dtype=complex)
    threshold = gap_width / 2
    axis_map = {'x': 0, 'y': 1, 'z': 2}

    # 6. Accumulate contributions
    for i in range(num_feeds):
        # Determine the axis for this specific feed
        axis_str = unit_vectors[i]
        ax_idx = axis_map[axis_str]
        
        gap_center = feed_points[i, ax_idx]
        v_complex = complex_amplitudes[i]

        # Extract components specific to the current excitation axis
        rho_p_ax = vecteurs_rho.vecteur_rho_plus[ax_idx, :]   
        rho_n_ax = vecteurs_rho.vecteur_rho_minus[ax_idx, :]  
        coord_cp = centroids[ax_idx, tri_plus_idx]
        coord_cn = centroids[ax_idx, tri_minus_idx]

        # Calculate distances
        dist_p = np.abs(coord_cp - gap_center)
        dist_n = np.abs(coord_cn - gap_center)

        # Safety check for mesh density at this port
        if np.min(dist_p) > threshold and np.min(dist_n) > threshold:
            print(f"Warning: Feed point {i} at {feed_points[i]} has no nearby centroids. Check gap_width or mesh.")
            continue

        # Windowing and scaling (Equation 7)
        window_p = np.where(dist_p <= threshold, 1.0, 0.0)
        window_n = np.where(dist_n <= threshold, 1.0, 0.0)
        common_factor = (l_m * v_complex) / (2 * gap_width)
        
        # Cumulative sum
        total_voltage_vector += common_factor * (window_p * rho_p_ax + window_n * rho_n_ax)

    return total_voltage_vector