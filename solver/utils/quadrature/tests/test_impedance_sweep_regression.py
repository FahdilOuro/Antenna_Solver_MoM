"""
Tests for impedance_matrix_z_sweep (solver/utils/impmet.py).

Context
-------
A frequency sweep that calls impedance_matrix_z once per frequency point
recomputes every triangle-pair geometric quantity (quadrature points, rho
vectors, pairwise distances R, rho_t.rho_s dot products, Duffy nodes) from
scratch at each frequency, even though none of it depends on frequency.
impedance_matrix_z_sweep computes that geometry once per pair and vectorises
only the genuinely frequency-dependent kernel (G = exp(-jkR)/R and the
physical prefactors) over a leading frequency axis. See workflow.md,
"frequency_sweep one-time precompute".

The correctness bar here is that impedance_matrix_z_sweep(frequencies)[i]
must equal impedance_matrix_z(frequencies[i]) to machine precision for every
i -- this is meant to be a pure caching/vectorisation change, not a change
in what is computed.
"""

import numpy as np
import pytest

from solver.rwg.rwg1 import Points, Triangles
from solver.utils.impmet import impedance_matrix_z, impedance_matrix_z_sweep

EPSILON = 8.854e-12
MU      = 1.257e-6
C       = 1.0 / np.sqrt(EPSILON * MU)


def _build_rwg_objects(raw_points, raw_triangles):
    points_obj = Points(raw_points)
    tri_obj = Triangles(raw_triangles)
    tri_obj.calculate_triangles_area_and_center(points_obj)
    edges_obj = tri_obj.get_edges()
    edges_obj.compute_edges_length(points_obj)
    return points_obj, tri_obj, edges_obj


def _four_triangle_mesh():
    points = np.array([
        [0., 1., 2., 0., 1., 2.],
        [0., 0., 0., 1., 1., 1.],
        [0., 0., 0., 0., 0., 0.],
    ])
    triangles = np.array([
        [0, 1, 1, 2],
        [1, 4, 2, 5],
        [3, 3, 4, 4],
        [0, 0, 0, 0],
    ], dtype=int)
    return points, triangles


def _six_triangle_mesh():
    points = np.array([
        [0., 1., 2., 3., 0., 1., 2., 3.],
        [0., 0., 0., 0., 1., 1., 1., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
    ])
    triangles = np.array([
        [0, 1, 1, 2, 2, 3],
        [1, 5, 2, 6, 3, 7],
        [4, 4, 5, 5, 6, 6],
        [0, 0, 0, 0, 0, 0],
    ], dtype=int)
    return points, triangles


FREQS = np.array([1e8, 3e8, 6.5e8])   # spans quasi-static to resonant-ish k*cell


@pytest.mark.parametrize("builder", [_four_triangle_mesh, _six_triangle_mesh])
def test_sweep_matches_single_frequency_calls(builder):
    raw_points, raw_triangles = builder()
    points_obj, tri_obj, edges_obj = _build_rwg_objects(raw_points, raw_triangles)

    Z_sweep = impedance_matrix_z_sweep(edges_obj, tri_obj, points_obj, FREQS, MU, EPSILON,
                                        quad_order=6, duffy_order=7)
    assert Z_sweep.shape == (len(FREQS), edges_obj.total_number_of_edges,
                              edges_obj.total_number_of_edges)

    for i, freq in enumerate(FREQS):
        omega = 2.0 * np.pi * freq
        k_complex = 1j * omega / C
        Z_single = impedance_matrix_z(edges_obj, tri_obj, points_obj, k_complex, omega,
                                       MU, EPSILON, quad_order=6, duffy_order=7)
        np.testing.assert_allclose(
            Z_sweep[i], Z_single, rtol=1e-10, atol=1e-12,
            err_msg=f"impedance_matrix_z_sweep disagrees with impedance_matrix_z at f={freq:.3e} Hz",
        )


def test_sweep_single_frequency_matches_scalar_call():
    """A one-point 'sweep' must reduce to the ordinary single call."""
    raw_points, raw_triangles = _four_triangle_mesh()
    points_obj, tri_obj, edges_obj = _build_rwg_objects(raw_points, raw_triangles)

    freq = 3e8
    Z_sweep = impedance_matrix_z_sweep(edges_obj, tri_obj, points_obj, [freq], MU, EPSILON)
    omega = 2.0 * np.pi * freq
    k_complex = 1j * omega / C
    Z_single = impedance_matrix_z(edges_obj, tri_obj, points_obj, k_complex, omega, MU, EPSILON)

    np.testing.assert_allclose(Z_sweep[0], Z_single, rtol=1e-10, atol=1e-12)
