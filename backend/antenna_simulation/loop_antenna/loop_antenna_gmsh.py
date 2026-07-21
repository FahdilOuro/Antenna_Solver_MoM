import gmsh
import numpy as np


def _helix_ribbon_points(a, w, pitch, theta_start, theta_end, segments_per_turn):
    """Sample a helical ribbon centerline and offset it radially by +/- w.

    The centerline is a standard helix of radius a, gaining `pitch` in height
    per full turn (2*pi rad). The ribbon width lies in the local radial
    direction (in-plane with the helix axis and the sampled point), which is
    the simplest well-defined "wire width" direction to build with straight
    gmsh primitives while still following the coil.
    """
    n_turns = (theta_end - theta_start) / (2 * np.pi)
    n_segments = max(1, round(segments_per_turn * n_turns))
    thetas = np.linspace(theta_start, theta_end, n_segments + 1)

    centers = np.stack([
        a * np.cos(thetas),
        a * np.sin(thetas),
        pitch / (2 * np.pi) * (thetas - theta_start),
    ], axis=1)
    radial = np.stack([np.cos(thetas), np.sin(thetas), np.zeros_like(thetas)], axis=1)

    inner = centers - w * radial
    outer = centers + w * radial
    return inner, outer


def create_helix_ribbon(a, w, pitch, theta_start, theta_end, segments_per_turn, z_offset=0.0):
    """Build one continuous helical ribbon as a chain of ruled quad surfaces.

    Returns (surface_tags, start_point, end_point) where start/end_point are
    the centerline coordinates at theta_start/theta_end (after z_offset).
    """
    inner, outer = _helix_ribbon_points(a, w, pitch, theta_start, theta_end, segments_per_turn)
    inner[:, 2] += z_offset
    outer[:, 2] += z_offset

    n_segments = inner.shape[0] - 1
    surfaces = []
    for i in range(n_segments):
        pA = gmsh.model.occ.addPoint(*inner[i])
        pB = gmsh.model.occ.addPoint(*outer[i])
        pC = gmsh.model.occ.addPoint(*outer[i + 1])
        pD = gmsh.model.occ.addPoint(*inner[i + 1])

        lAB = gmsh.model.occ.addLine(pA, pB)
        lBC = gmsh.model.occ.addLine(pB, pC)
        lCD = gmsh.model.occ.addLine(pC, pD)
        lDA = gmsh.model.occ.addLine(pD, pA)

        loop = gmsh.model.occ.addCurveLoop([lAB, lBC, lCD, lDA])
        surf = gmsh.model.occ.addSurfaceFilling(loop)
        surfaces.append(surf)

    start_point = 0.5 * (inner[0] + outer[0])
    end_point = 0.5 * (inner[-1] + outer[-1])
    return surfaces, start_point, end_point


def create_loop_antenna_geometry(a, w, pitch, n_turns_lower, n_turns_upper, segments_per_turn=12):
    """
    Build a center-fed normal-mode helical antenna: two helical ribbons
    stacked along z with a small feed gap between them, both starting at the
    same angular position (theta=0) so the gap is a simple vertical break --
    matching the helix / straight-wire-equivalent sketch (coil of radius a,
    strip width 2w, pitch R (here `pitch`), fed through a small gap d (here
    `gap`) at the middle).

    Parameters
    ----------
    a : float, helix radius (2a = coil diameter)
    w : float, ribbon half-width (2w = wire/strip width)
    pitch : float, height gained per full turn (R in the sketch)
    gap : float, feed gap height at the centre (d in the sketch)
    n_turns_lower, n_turns_upper : float, number of turns below/above the gap
    segments_per_turn : int, ribbon discretisation resolution

    Returns
    -------
    feed_point : (3,) ndarray, midpoint of the feed gap
    total_height : float, overall antenna height (including the feed gap)
    """
    all_surfaces = []

    lower_surfaces, _, lower_top = create_helix_ribbon(
        a, w, pitch, 0.0, n_turns_lower * 2 * np.pi, segments_per_turn, z_offset=0.0)
    all_surfaces += lower_surfaces

    upper_z_offset = lower_top[2]
    upper_surfaces, upper_bottom, upper_top = create_helix_ribbon(
        a, w, pitch, 0.0, n_turns_upper * 2 * np.pi, segments_per_turn, z_offset=upper_z_offset)
    all_surfaces += upper_surfaces

    gmsh.model.occ.fragment([(2, s) for s in all_surfaces], [])
    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()

    surfaces_after = gmsh.model.getEntities(2)
    gmsh.model.addPhysicalGroup(2, [s[1] for s in surfaces_after], name="Loop_Antenna_Shell")
    total_height = upper_top[2]
    return total_height
