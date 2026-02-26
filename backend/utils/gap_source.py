import numpy as np


def multiple_gap_sources(triangles, edges, vecteurs_rho, voltage_amplitude, feed_points, 
                         excitation_unit_vector, gap_width, phases=None):
    """
    Implements the Enhanced Gap Source model for multiple feed points with phase shifts.
    
    Parameters:
    - feed_points: np.array of shape (N, 3) representing feed locations.
    - voltage_amplitude: Scalar or array of magnitudes for each feed.
    - phases: np.array of phase shifts (in radians) for each feed. Default is 0.
    """
    
    # 1. Prepare feed points and amplitudes
    feed_points = np.atleast_2d(feed_points)
    num_feeds = feed_points.shape[0]

    # Handle voltage_amplitude (scalar or array)
    if np.isscalar(voltage_amplitude):
        amplitudes = np.full(num_feeds, voltage_amplitude)
    else:
        amplitudes = voltage_amplitude

    # 2. Handle phases
    if phases is None:
        phases = np.zeros(num_feeds)
    elif np.isscalar(phases):
        phases = np.full(num_feeds, phases)
    
    # Pre-calculate complex coefficients: V * exp(j * phase)
    complex_amplitudes = amplitudes * np.exp(1j * phases)

    # 3. Map excitation axis
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    ax_idx = axis_map[excitation_unit_vector]

    # 4. Extract geometry and initialize global vector
    num_edges = edges.total_number_of_edges
    l_m = edges.edges_length
    tri_plus_idx = triangles.triangles_plus
    tri_minus_idx = triangles.triangles_minus  
    centroids = triangles.triangles_center 
    
    rho_p_ax = vecteurs_rho.vecteur_rho_plus[ax_idx, :]   
    rho_n_ax = vecteurs_rho.vecteur_rho_minus[ax_idx, :]  
    coord_cp = centroids[ax_idx, tri_plus_idx]
    coord_cn = centroids[ax_idx, tri_minus_idx]

    total_voltage_vector = np.zeros(num_edges, dtype=complex)
    threshold = gap_width / 2

    # 5. Accumulate contributions with phase shifts
    for i in range(num_feeds):
        gap_center = feed_points[i, ax_idx]
        v_complex = complex_amplitudes[i]

        # Calculate distances to current feed point
        dist_p = np.abs(coord_cp - gap_center)
        dist_n = np.abs(coord_cn - gap_center)

        # Skip with warning if mesh is too coarse for this specific point
        if np.min(dist_p) > threshold and np.min(dist_n) > threshold:
            print(f"Warning: No edges found within gap for feed point {feed_points[i]}. Check mesh density.")
            continue

        # Generate window masks (Automatic indexing)
        window_p = np.where(dist_p <= threshold, 1.0, 0.0)
        window_n = np.where(dist_n <= threshold, 1.0, 0.0)

        # Apply Equation (7) with the complex amplitude (magnitude + phase)
        common_factor = (l_m * v_complex) / (2 * gap_width)
        
        # Superposition of this port's contribution
        total_voltage_vector += common_factor * (window_p * rho_p_ax + window_n * rho_n_ax)

    return total_voltage_vector