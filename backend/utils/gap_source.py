import numpy as np

def multiple_gap_sources(triangles, edges, vecteurs_rho, voltage_amplitude, feed_points, 
                         excitation_unit_vector, gap_width, phases=None):
    """
    Implements the Enhanced Gap Source model for multiple feed points.
    Returns the global voltage vector and a list of indices for each port.
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
    if isinstance(excitation_unit_vector, str):
        unit_vectors = [excitation_unit_vector] * num_feeds
    else:
        unit_vectors = excitation_unit_vector

    if len(unit_vectors) != num_feeds:
        raise ValueError("The number of excitation vectors must match the number of feed points.")

    complex_amplitudes = amplitudes * np.exp(1j * phases)

    # 5. Extract fixed geometry data
    num_edges = edges.total_number_of_edges
    l_m = edges.edges_length
    tri_plus_idx = triangles.triangles_plus
    tri_minus_idx = triangles.triangles_minus  
    centroids = triangles.triangles_center
    
    total_voltage_vector = np.zeros(num_edges, dtype=complex)
    
    # This list will store the indices of feeding edges for each port
    all_feeding_indices = []
    
    threshold = gap_width / 2
    axis_map = {'x': 0, 'y': 1, 'z': 2}

    # 6. Accumulate contributions
    for i in range(num_feeds):
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

        # Mesh density check
        if np.min(dist_p) > threshold and np.min(dist_n) > threshold:
            print(f"Warning: Port {i} has no nearby centroids. Storing empty indices.")
            all_feeding_indices.append(np.array([], dtype=int))
            continue

        # Windowing (Equation 7)
        # Note: An edge is part of the gap if either triangle T+ or T- is within the window
        window_p = np.where(dist_p <= threshold, 1.0, 0.0)
        window_n = np.where(dist_n <= threshold, 1.0, 0.0)
        
        # Save indices for this specific port
        # We find indices where at least one window function is active (non-zero)
        port_indices = np.where((window_p > 0) | (window_n > 0))[0]
        all_feeding_indices.append(port_indices)

        # Apply Equation (7) and add to global vector
        common_factor = (l_m * v_complex) / (2 * gap_width)
        total_voltage_vector += common_factor * (window_p * rho_p_ax + window_n * rho_n_ax)

    return total_voltage_vector, all_feeding_indices