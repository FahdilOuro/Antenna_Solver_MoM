"""
Exact integration of the enhanced gap source (eq. 7 of Ding et al., 2013) over
triangles cut by the gap window, and the warning for a conductor wider than it.

With the one-point (centroid) rule a triangle is fed wholly or not at all, so
when the gap edges are not mesh lines the applied voltage jumps from one mesh
to the next. Given the vertex coordinates (`points`), multiple_gap_sources
integrates f_m . e over the part of each triangle inside the window instead.
These tests use structured strips (solver.optim.geometries.planar), whose grid
lines can be placed on or off the gap edges at will.
"""

import warnings

import numpy as np
import pytest

from solver.optim.geometries.planar import tensor_grid_mesh
from solver.utils.gap_source import GapSourceError, GapSourceWarning, multiple_gap_sources


def strip(xs, width, n_width=1):
    """Strip along x in z = 0, grid lines xs along x, n_width cells across."""
    return tensor_grid_mesh(np.asarray(xs, dtype=float), np.linspace(-width / 2, width / 2, n_width + 1))


def gap_voltage(mesh, gap_width, exact=True, feed=(0.0, 0.0, 0.0)):
    return multiple_gap_sources(mesh.triangles, mesh.edges, mesh.vecteurs_rho, 1.0, [list(feed)], "x",
                                gap_width, points=mesh.points if exact else None)


def monte_carlo_voltage(mesh, gap_width, n_samples=400_000, seed=0):
    """Independent estimate of eq. (7): sample each triangle uniformly, keep the points in the window."""
    rng = np.random.default_rng(seed)
    pts = mesh.points.points
    tri = mesh.triangles.triangles[:3].astype(int)
    area = mesh.triangles.triangles_area
    centre = mesh.triangles.triangles_center
    plus, minus = mesh.triangles.triangles_plus, mesh.triangles.triangles_minus
    rho_p, rho_m = mesh.vecteurs_rho.vecteur_rho_plus, mesh.vecteurs_rho.vecteur_rho_minus
    length = mesh.edges.edges_length
    half = gap_width / 2

    def integral_x(t, free_vertex, sign):
        """integral over (T inside the window) of sign * (r - v)_x dS."""
        a, b, c = pts[:, tri[0, t]], pts[:, tri[1, t]], pts[:, tri[2, t]]
        u, v = rng.random(n_samples), rng.random(n_samples)
        flip = u + v > 1
        u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        r = a[:, None] + u * (b - a)[:, None] + v * (c - a)[:, None]
        inside = np.all(np.abs(r) <= half, axis=0)
        return area[t] * np.mean(inside * sign * (r[0] - free_vertex[0]))

    V = np.zeros(mesh.n_edges)
    for m in range(mesh.n_edges):
        v_plus = centre[:, plus[m]] - rho_p[:, m]
        v_minus = centre[:, minus[m]] + rho_m[:, m]
        total = (integral_x(plus[m], v_plus, +1.0) / area[plus[m]]
                 + integral_x(minus[m], v_minus, -1.0) / area[minus[m]])
        V[m] = length[m] / (2 * gap_width) * total
    return V


def test_aligned_gap_gives_the_one_point_rule():
    """Gap edges on grid lines: every triangle is wholly in or out, both rules agree."""
    xs = np.concatenate([np.linspace(-0.25, -0.015, 8), [-0.005, 0.005], np.linspace(0.015, 0.25, 8)])
    mesh = strip(xs, 0.01)
    V_exact, fed_exact = gap_voltage(mesh, 0.030)
    V_point, fed_point = gap_voltage(mesh, 0.030, exact=False)
    np.testing.assert_allclose(V_exact, V_point, rtol=0, atol=1e-15)
    assert set(fed_exact[0].tolist()) == set(fed_point[0].tolist())


def test_cut_triangles_match_an_independent_integration():
    """Grid lines off the gap edges: the clipped integral equals a Monte-Carlo estimate of eq. (7)."""
    xs = np.linspace(-0.25, 0.25, 22)          # cells of 23.8 mm, gap edges at +-17.5 mm cut them
    mesh = strip(xs, 0.01, n_width=2)
    V_exact, _ = gap_voltage(mesh, 0.035)
    V_mc = monte_carlo_voltage(mesh, 0.035)
    scale = np.max(np.abs(V_exact))
    assert scale > 0
    np.testing.assert_allclose(V_exact.real, V_mc, rtol=0, atol=5e-3 * scale)
    # The test discriminates: on this mesh the one-point rule is far from the integral.
    V_point, _ = gap_voltage(mesh, 0.035, exact=False)
    assert np.max(np.abs(V_point.real - V_mc)) > 0.1 * scale


def test_voltage_varies_continuously_with_the_gap_width():
    """No jumps when a gap edge crosses a centroid, unlike the one-point rule."""
    mesh = strip(np.linspace(-0.25, 0.25, 22), 0.01)
    widths = np.linspace(0.030, 0.060, 301)
    jumps_exact, jumps_point = [], []
    previous_exact = previous_point = None
    for w in widths:
        # Compare the voltage times W (eq. 7 divides by W, which varies smoothly anyway)
        v_exact = gap_voltage(mesh, w)[0] * w
        v_point = gap_voltage(mesh, w, exact=False)[0] * w
        if previous_exact is not None:
            jumps_exact.append(np.max(np.abs(v_exact - previous_exact)))
            jumps_point.append(np.max(np.abs(v_point - previous_point)))
        previous_exact, previous_point = v_exact, v_point
    scale = np.max(np.abs(previous_exact))
    assert max(jumps_exact) < 0.02 * scale
    assert max(jumps_point) > 0.2 * scale


def test_warns_when_the_strip_is_wider_than_the_gap():
    mesh = strip(np.linspace(-0.25, 0.25, 61), 0.040, n_width=4)
    with pytest.warns(GapSourceWarning, match="crosses the gap outside the gap window"):
        gap_voltage(mesh, 0.030)


def test_no_warning_when_the_gap_covers_the_strip():
    for width, gap, n_width in ((0.040, 0.050, 4), (0.010, 0.030, 1)):
        mesh = strip(np.linspace(-0.25, 0.25, 61), width, n_width=n_width)
        with warnings.catch_warnings():
            warnings.simplefilter("error", GapSourceWarning)
            gap_voltage(mesh, gap)


def test_still_raises_when_nothing_is_in_the_window():
    mesh = strip(np.linspace(-0.25, 0.25, 22), 0.01)
    with pytest.raises(GapSourceError, match="no RWG edge found within the gap"):
        gap_voltage(mesh, 0.01, feed=(1.0, 0.0, 0.0))
