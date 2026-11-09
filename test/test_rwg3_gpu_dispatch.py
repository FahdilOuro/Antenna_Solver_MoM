"""
Tests for the size-based GPU/CPU dispatch in calculate_z_matrice and
calculate_z_matrice_sweep (solver/rwg/rwg3.py).

Context
-------
impedance_matrix_z_sweep_gpu's batched assembly only pays off once there
are enough triangle pairs to amortise its fixed overhead -- measured on the
reference NVIDIA T600 (workflow.md, "GPU acceleration (CuPy)") to lose
below ~E=57 RWG edges and win comfortably above it. calculate_z_matrice and
calculate_z_matrice_sweep therefore pick their implementation via
solver.utils.compute_backend.should_use_gpu(E, use_gpu), not
unconditionally: small meshes keep using the original per-pair
implementation even when a GPU is present, since it was measured faster
there.

These tests use small synthetic meshes (well below the threshold) so they
run in both code paths deterministically and check both paths produce the
same physics -- not that one path is "GPU" and the other "CPU" (that's
already covered by test_impedance_sweep_gpu_regression.py).
"""

import numpy as np
import pytest

from solver.rwg.rwg1 import Points, Triangles
from solver.rwg.rwg3 import calculate_z_matrice, calculate_z_matrice_sweep, calculate_z_matrice_lumped_elements
from solver.utils.impmet import impedance_matrix_z, impedance_matrix_z_sweep

EPSILON = 8.854e-12
MU      = 1.257e-6


def _build_rwg_objects(raw_points, raw_triangles):
    points_obj = Points(raw_points)
    tri_obj = Triangles(raw_triangles)
    tri_obj.calculate_triangles_area_and_center(points_obj)
    edges_obj = tri_obj.get_edges()
    edges_obj.compute_edges_length(points_obj)
    return points_obj, tri_obj, edges_obj


def _six_triangle_mesh():
    """1x3 strip, 5 interior edges -- far below DEFAULT_GPU_EDGE_THRESHOLD."""
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


FREQ = 3e8


def test_calculate_z_matrice_below_threshold_matches_impedance_matrix_z():
    """A small mesh (E=5) must take the plain per-pair path by default and
    reproduce impedance_matrix_z exactly."""
    raw_points, raw_triangles = _six_triangle_mesh()
    points_obj, tri_obj, edges_obj = _build_rwg_objects(raw_points, raw_triangles)
    assert edges_obj.total_number_of_edges < 100  # sanity: below the default threshold

    omega, mu, epsilon, c, eta, Z_dispatch = calculate_z_matrice(points_obj, tri_obj, edges_obj, FREQ)

    k_complex = 1j * omega / c
    Z_ref = impedance_matrix_z(edges_obj, tri_obj, points_obj, k_complex, omega, mu, epsilon)

    np.testing.assert_allclose(Z_dispatch, Z_ref, rtol=1e-12, atol=1e-14)


def test_calculate_z_matrice_use_gpu_true_forces_batched_path():
    """use_gpu=True must go through the batched implementation regardless
    of mesh size, and still agree physically with the reference."""
    raw_points, raw_triangles = _six_triangle_mesh()
    points_obj, tri_obj, edges_obj = _build_rwg_objects(raw_points, raw_triangles)

    from solver.utils.compute_backend import GPU_AVAILABLE
    if not GPU_AVAILABLE:
        with pytest.raises(RuntimeError, match="no usable CUDA GPU"):
            calculate_z_matrice(points_obj, tri_obj, edges_obj, FREQ, use_gpu=True)
        return

    omega, mu, epsilon, c, eta, Z_gpu = calculate_z_matrice(points_obj, tri_obj, edges_obj, FREQ, use_gpu=True)
    k_complex = 1j * omega / c
    Z_ref = impedance_matrix_z(edges_obj, tri_obj, points_obj, k_complex, omega, mu, epsilon)

    np.testing.assert_allclose(Z_gpu, Z_ref, rtol=1e-8, atol=1e-10)


def test_calculate_z_matrice_sweep_below_threshold_matches_reference():
    raw_points, raw_triangles = _six_triangle_mesh()
    points_obj, tri_obj, edges_obj = _build_rwg_objects(raw_points, raw_triangles)
    freqs = np.array([1e8, 3e8, 6.5e8])

    omega, mu, epsilon, c, eta, Z_dispatch = calculate_z_matrice_sweep(points_obj, tri_obj, edges_obj, freqs)
    Z_ref = impedance_matrix_z_sweep(edges_obj, tri_obj, points_obj, freqs, mu, epsilon)

    np.testing.assert_allclose(Z_dispatch, Z_ref, rtol=1e-12, atol=1e-14)


def test_calculate_z_matrice_lumped_elements_dispatch_consistent():
    """calculate_z_matrice_lumped_elements applies its lumped-element patch
    on top of whichever base Z the size/use_gpu dispatch produced; the
    default (below-threshold, per-pair) path and a forced GPU path must
    give the same physics, patch included."""
    raw_points, raw_triangles = _six_triangle_mesh()
    points_obj, tri_obj, edges_obj = _build_rwg_objects(raw_points, raw_triangles)

    LoadPoint = np.array([[0.5, 0.5, 0.0]])          # (LNumber, 3)
    LoadValue = np.array([[0.0, None, 50.0]], dtype=object)   # L=0, C=None (->Z_C~0), R=50 Ohm
    LoadDir   = np.array([[0.0, 1.0, 0.0]])          # (LNumber, 3)

    omega, mu, epsilon, c, eta, Z_default, imp_default = calculate_z_matrice_lumped_elements(
        points_obj, tri_obj, edges_obj, FREQ, LoadPoint, LoadValue, LoadDir)

    from solver.utils.compute_backend import GPU_AVAILABLE
    if not GPU_AVAILABLE:
        pytest.skip("no usable CUDA GPU on this machine to cross-check the forced-GPU path")

    _, _, _, _, _, Z_gpu, imp_gpu = calculate_z_matrice_lumped_elements(
        points_obj, tri_obj, edges_obj, FREQ, LoadPoint, LoadValue, LoadDir, use_gpu=True)

    np.testing.assert_array_equal(imp_default, imp_gpu)
    np.testing.assert_allclose(Z_gpu, Z_default, rtol=1e-8, atol=1e-10)
