import numpy as np


class GapSourceError(ValueError):
    """Raised when a gap-source excitation finds no RWG edge to feed."""


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

    Notes on gap localisation
    -------------------------
    The gap window is a cube of side `gap_width` centred on each feed point,
    tested independently on all three coordinates -- not just the excitation
    axis. For a simple 1D wire/strip (the case eq. 7 was derived for), the
    structure's cross-section is already far smaller than gap_width, so this
    is equivalent to testing the excitation axis alone.

    It stops being equivalent at a junction between differently-oriented
    surfaces sharing the feed point (e.g. a monopole blade standing on a
    ground plane): testing only the excitation-axis coordinate would then
    also catch far-away triangles that merely happen to share that one
    coordinate with the feed point (e.g. a whole band of the ground plane at
    the same x as a monopole fed along x, regardless of how far it is along
    y) -- spuriously exciting geometry that has nothing to do with the actual
    feed location. The 3D box keeps the excitation localised to a small
    neighbourhood of the feed point regardless of how many differently
    oriented surfaces meet there.

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
        v_complex  = complex_amplitudes[i]
        feed_pt    = feed_points[i, :]                  # (3,) full 3D feed point

        # Centroid position (all 3 coordinates) for T+ and T-
        centroid_p = centroids[:, tri_plus_idx]         # (3, num_edges)
        centroid_n = centroids[:, tri_minus_idx]        # (3, num_edges)

        # 3D box window: 1 if the centroid is within gap_width/2 of the feed
        # point on EVERY coordinate, 0 otherwise (see "Notes on gap
        # localisation" above). Reduces to eq. (7)'s 1D step function
        # u(z+W/2)-u(z-W/2) whenever the structure's cross-section already
        # fits inside the box, as for a simple wire/strip.
        half_W   = gap_width / 2.0
        window_p = np.all(np.abs(centroid_p - feed_pt[:, None]) <= half_W, axis=0).astype(float)
        window_n = np.all(np.abs(centroid_n - feed_pt[:, None]) <= half_W, axis=0).astype(float)

        # Fail loudly if no edge is selected, rather than silently solving
        # with a zero excitation for this port: a mismatch between
        # gap_width, excitation_unit_vector, and feed_points here means the
        # requested port is not actually feeding anything, which would
        # otherwise show up only indirectly (e.g. a "missing" structure in
        # the results) rather than at the point the mistake was made.
        port_indices = np.where((window_p > 0) | (window_n > 0))[0]
        if port_indices.size == 0:
            raise GapSourceError(
                f"Port {i} (feed_point={feed_pt}, excitation_unit_vector="
                f"'{unit_vectors[i]}', gap_width={gap_width:.4g}): no RWG "
                f"edge found within the gap. Check that feed_points lies on "
                f"(or very near) the mesh, that gap_width is large enough to "
                f"reach at least one triangle on every side of the feed you "
                f"intend to excite, and that excitation_unit_vector matches "
                f"the actual direction current should flow at this feed "
                f"(e.g. the antenna's own axis at a junction, not just the "
                f"axis you happen to be thinking of)."
            )

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