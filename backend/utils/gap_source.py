import numpy as np

def multiple_gap_sources(triangles, edges, vecteurs_rho, voltage_amplitude,
                         feed_points, excitation_unit_vector, gap_width, phases=None):
    """
    Implements the Enhanced Gap Source model (Ding et al., 2013, eq. 7).
    
    Parameters
    ----------
    triangles : object
        Must expose: triangles_plus (int array, shape (num_edges,)),
                     triangles_minus (int array, shape (num_edges,)),
                     triangles_center (float array, shape (3, num_triangles))
    edges : object
        Must expose: total_number_of_edges (int),
                     edges_length (float array, shape (num_edges,))
    vecteurs_rho : object
        Must expose: vecteur_rho_plus  (float array, shape (3, num_edges)),
                     vecteur_rho_minus (float array, shape (3, num_edges))
    voltage_amplitude : float or array-like
    feed_points : array-like, shape (num_feeds, 3)
    excitation_unit_vector : str or list of str  ('x', 'y', or 'z')
    gap_width : float
    phases : float or array-like, optional (radians)
    
    Returns
    -------
    total_voltage_vector : complex ndarray, shape (num_edges,)
    all_feeding_edges : list of int ndarray
    """
    feed_points = np.atleast_2d(feed_points)
    num_feeds = feed_points.shape[0]

    # --- Broadcast scalar inputs ---
    if np.isscalar(voltage_amplitude):
        amplitudes = np.full(num_feeds, float(voltage_amplitude))
    else:
        amplitudes = np.asarray(voltage_amplitude, dtype=float)

    if phases is None:
        phases = np.zeros(num_feeds)
    elif np.isscalar(phases):
        phases = np.full(num_feeds, float(phases))
    else:
        phases = np.asarray(phases, dtype=float)

    if isinstance(excitation_unit_vector, str):
        unit_vectors = [excitation_unit_vector] * num_feeds
    else:
        unit_vectors = list(excitation_unit_vector)

    if len(unit_vectors) != num_feeds:
        raise ValueError("excitation_unit_vector length must match number of feed points.")

    complex_amplitudes = amplitudes * np.exp(1j * phases)

    # --- Fixed geometry ---
    num_edges    = edges.total_number_of_edges
    l_m          = edges.edges_length                  # (num_edges,)
    tri_plus_idx = triangles.triangles_plus            # (num_edges,)
    tri_minus_idx= triangles.triangles_minus           # (num_edges,)
    centroids    = triangles.triangles_center          # (3, num_triangles)

    rho_plus  = vecteurs_rho.vecteur_rho_plus          # (3, num_edges)
    rho_minus = vecteurs_rho.vecteur_rho_minus         # (3, num_edges)

    total_voltage_vector = np.zeros(num_edges, dtype=complex)
    all_feeding_edges    = []
    axis_map = {'x': 0, 'y': 1, 'z': 2}

    for i in range(num_feeds):
        ax_idx     = axis_map[unit_vectors[i]]
        gap_center = feed_points[i, ax_idx]
        v_complex  = complex_amplitudes[i]

        # Centroid coordinate along excitation axis for T+ and T-
        coord_cp = centroids[ax_idx, tri_plus_idx]    # (num_edges,)
        coord_cn = centroids[ax_idx, tri_minus_idx]   # (num_edges,)

        # Step function window: 1 if centroid is inside the gap, 0 otherwise
        # Implements u(z + W/2) - u(z - W/2) from eq. (7)
        half_W   = gap_width / 2.0
        window_p = np.where(np.abs(coord_cp - gap_center) <= half_W, 1.0, 0.0)
        window_n = np.where(np.abs(coord_cn - gap_center) <= half_W, 1.0, 0.0)

        # Warn if no edge is selected (mesh too coarse relative to gap width)
        port_indices = np.where((window_p > 0) | (window_n > 0))[0]
        if port_indices.size == 0:
            print(f"Warning: Port {i} — no edges found within gap. "
                  f"Gap width ({gap_width:.4g}) may be smaller than mesh size.")
            all_feeding_edges.append(port_indices)
            continue

        all_feeding_edges.append(port_indices)

        # dot product rho . e_hat  (eq. 7: rho_m^+ . e_z, rho_m^- . e_z)
        rho_p_dot = rho_plus[ax_idx, :]    # (num_edges,)  scalar proj. on excitation axis
        rho_n_dot = rho_minus[ax_idx, :]   # (num_edges,)

        # Assemble eq. (7)
        common_factor = (l_m * v_complex) / (2.0 * gap_width)
        total_voltage_vector += common_factor * (
            window_p * rho_p_dot + window_n * rho_n_dot
        )

    return total_voltage_vector, all_feeding_edges