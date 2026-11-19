import warnings

import numpy as np


class GapSourceError(ValueError):
    """Raised when a gap-source excitation finds no RWG edge to feed."""


class GapSourceWarning(UserWarning):
    """Emitted when the conductor crosses the gap outside the gap window (part of it is not fed)."""


def _clip_polygon_to_box(polygon, lower, upper):
    """
    Sutherland-Hodgman clipping of a planar polygon (list of (3,) points) by the
    axis-aligned box lower <= r <= upper. Returns the clipped polygon (possibly empty).
    """
    poly = list(polygon)
    for axis in range(3):
        for bound, keep_above in ((lower[axis], True), (upper[axis], False)):
            if not poly:
                return poly
            out = []
            n = len(poly)
            for k in range(n):
                a, b = poly[k], poly[(k + 1) % n]
                da = (a[axis] - bound) if keep_above else (bound - a[axis])
                db = (b[axis] - bound) if keep_above else (bound - b[axis])
                if da >= 0:
                    out.append(a)
                if (da >= 0) != (db >= 0):
                    t = da / (da - db)
                    out.append(a + t * (b - a))
            poly = out
    return poly


def _polygon_area_centroid(polygon):
    """Area and centroid of a planar polygon in 3D (fan triangulation from its first vertex)."""
    if len(polygon) < 3:
        return 0.0, None
    p0 = polygon[0]
    area = 0.0
    moment = np.zeros(3)
    for k in range(1, len(polygon) - 1):
        a = 0.5 * np.linalg.norm(np.cross(polygon[k] - p0, polygon[k + 1] - p0))
        area += a
        moment += a * (p0 + polygon[k] + polygon[k + 1]) / 3.0
    if area <= 0.0:
        return 0.0, None
    return area, moment / area


def _gap_fractions(tri_vertices, tri_area, centroids, feed_pt, half_W):
    """
    For every triangle: the fraction of its area inside the gap cube, and the
    centroid of that part (equal to the triangle centroid when it is entirely
    inside). Only triangles whose bounding box meets the cube are clipped.

    tri_vertices : (N_tri, 3, 3) vertex coordinates, [triangle, vertex, xyz]
    """
    lower, upper = feed_pt - half_W, feed_pt + half_W
    vmin = tri_vertices.min(axis=1)                       # (N_tri, 3)
    vmax = tri_vertices.max(axis=1)
    fraction = np.zeros(tri_vertices.shape[0])
    part_centroid = centroids.T.copy()                    # (N_tri, 3)

    inside = np.all((vmin >= lower) & (vmax <= upper), axis=1)
    fraction[inside] = 1.0
    touching = np.all((vmax >= lower) & (vmin <= upper), axis=1) & ~inside
    for t in np.flatnonzero(touching):
        area, centroid = _polygon_area_centroid(
            _clip_polygon_to_box([tri_vertices[t, k] for k in range(3)], lower, upper))
        if area > 0.0:
            fraction[t] = min(area / tri_area[t], 1.0)
            part_centroid[t] = centroid
    return fraction, part_centroid


def _warn_if_conductor_crosses_gap_outside_window(i, tri_vertices, fraction, ax_idx, feed_pt, half_W,
                                                  triangles_plus, triangles_minus):
    """
    Warn when the conductor runs ACROSS the gap outside the cube: a triangle that
    lies in the gap slab along the excitation axis, reaches beyond the cube on a
    transverse coordinate, belongs to a surface that contains the excitation
    direction, and shares an RWG edge with a fed triangle. The part of the gap
    cross-section it covers is not fed, which the input impedance does not show.
    Surfaces perpendicular to the excitation direction (a ground plane under a
    monopole fed along its axis) carry no current across the gap and are ignored.
    """
    tol = 1e-9 * max(half_W, 1e-30)
    in_slab = ((tri_vertices[:, :, ax_idx].max(axis=1) > feed_pt[ax_idx] - half_W + tol)
               & (tri_vertices[:, :, ax_idx].min(axis=1) < feed_pt[ax_idx] + half_W - tol))
    transverse = [k for k in range(3) if k != ax_idx]
    beyond = np.zeros(tri_vertices.shape[0], dtype=bool)
    for k in transverse:
        beyond |= np.abs(tri_vertices[:, :, k] - feed_pt[k]).max(axis=1) > half_W + tol
    normals = np.cross(tri_vertices[:, 1] - tri_vertices[:, 0], tri_vertices[:, 2] - tri_vertices[:, 0])
    norm = np.linalg.norm(normals, axis=1)
    along = np.ones_like(norm, dtype=bool)
    ok = norm > 0
    along[ok] = np.abs(normals[ok, ax_idx] / norm[ok]) < 0.9  # the surface contains the excitation axis
    fed = fraction > 0
    neighbour_of_fed = np.zeros(tri_vertices.shape[0], dtype=bool)
    pair_fed = fed[triangles_plus] | fed[triangles_minus]
    neighbour_of_fed[triangles_plus[pair_fed]] = True
    neighbour_of_fed[triangles_minus[pair_fed]] = True
    suspicious = in_slab & beyond & along & neighbour_of_fed
    if np.any(suspicious):
        warnings.warn(
            f"Port {i}: the conductor crosses the gap outside the gap window "
            f"(gap_width={2 * half_W:.4g} m, a cube centred on feed_point={feed_pt}): "
            f"{int(np.count_nonzero(suspicious))} triangle(s) in the gap reach beyond it "
            f"across the conductor and are not fed, so only part of the gap cross-section "
            f"is excited and the input impedance is wrong. Make gap_width at least as wide "
            f"as the conductor at the feed.",
            GapSourceWarning, stacklevel=3)


def multiple_gap_sources(triangles, edges, vecteurs_rho, voltage_amplitude,
                         feed_points, excitation_unit_vector, gap_width, phases=None, points=None):
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
    points : Points object or (3, N_points) array, optional
        Vertex coordinates. When given (as radiation_algorithm and
        frequency_sweep do), eq. (7) is integrated exactly over the part of
        each triangle that lies in the gap window (see "Notes on partial
        triangles"), and a GapSourceWarning is emitted when the conductor
        crosses the gap outside the window. Without it, the former one-point
        rule is used: a triangle is fed entirely if its centroid is in the
        window, not at all otherwise. `triangles` must then also expose
        `.triangles` (vertex indices, 0-based, shape (3 or 4, num_triangles))
        and `.triangles_area`.

    Notes on partial triangles
    --------------------------
    E^i = (V/W) e_hat inside the window, so eq. (7) is
        V_m = (V/W) e_hat . integral over (T_m^+- inside the window) of f_m dS.
    f_m is linear on each triangle, so the integral over the clipped part of a
    triangle (area A_c, centroid r_cc) is exact with one point:
        integral = (l_m / 2A) A_c (r_cc - v^+)      on T^+,
                   (l_m / 2A) A_c (v^- - r_cc)      on T^-.
    For a triangle entirely inside, A_c = A and r_cc is its centroid: this is
    exactly the one-point rule. The two differ only on triangles cut by the
    edge of the window. With the one-point rule such a triangle is fed wholly
    or not at all depending on where its centroid falls, so the applied
    voltage jumps from one mesh to the next: on a gmsh dipole whose gap edges
    were not mesh lines, the input resistance ranged from 63 to 182 Ohm
    instead of 72 Ohm (documentation, user guide "Choosing the mesh size").

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

    tri_vertices = None
    if points is not None:
        pts = np.asarray(points.points if hasattr(points, "points") else points, dtype=float)
        tri_idx = np.asarray(triangles.triangles)[:3].astype(int)          # (3, num_triangles)
        tri_vertices = np.transpose(pts[:, tri_idx], (2, 1, 0))            # (num_triangles, 3 vertices, 3 xyz)
        tri_area = np.asarray(triangles.triangles_area, dtype=float)

    for i in range(num_feeds):
        ax_idx     = axis_map[unit_vectors[i]]
        v_complex  = complex_amplitudes[i]
        feed_pt    = np.asarray(feed_points[i, :], dtype=float)   # (3,) full 3D feed point
        half_W     = gap_width / 2.0

        if tri_vertices is None:
            # One-point rule. 3D box window: 1 if the centroid is within
            # gap_width/2 of the feed point on EVERY coordinate, 0 otherwise
            # (see "Notes on gap localisation" above). Reduces to eq. (7)'s 1D
            # step function u(z+W/2)-u(z-W/2) whenever the structure's
            # cross-section already fits inside the box, as for a simple wire/strip.
            centroid_p = centroids[:, tri_plus_idx]         # (3, num_edges)
            centroid_n = centroids[:, tri_minus_idx]        # (3, num_edges)
            window_p = np.all(np.abs(centroid_p - feed_pt[:, None]) <= half_W, axis=0).astype(float)
            window_n = np.all(np.abs(centroid_n - feed_pt[:, None]) <= half_W, axis=0).astype(float)
            shift_p = shift_n = 0.0
        else:
            # Exact integration over the part of each triangle inside the window
            # (see "Notes on partial triangles"): weight = area fraction, and the
            # rho vectors are taken at the centroid of that part.
            fraction, part_centroid = _gap_fractions(tri_vertices, tri_area, centroids, feed_pt, half_W)
            window_p = fraction[tri_plus_idx]
            window_n = fraction[tri_minus_idx]
            offset = part_centroid.T - centroids                       # (3, num_triangles)
            shift_p = offset[ax_idx, tri_plus_idx]                     # r_cc - r_c on T+
            shift_n = -offset[ax_idx, tri_minus_idx]                   # -(r_cc - r_c) on T-
            _warn_if_conductor_crosses_gap_outside_window(
                i, tri_vertices, fraction, ax_idx, feed_pt, half_W,
                np.asarray(tri_plus_idx).astype(int), np.asarray(tri_minus_idx).astype(int))

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
            window_p * (rho_p_dot + shift_p) + window_n * (rho_n_dot + shift_n)
        )

    return total_voltage_vector, all_feeding_edges