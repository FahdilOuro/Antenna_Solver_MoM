"""
Tests for impedance_matrix_z_sweep_gpu (backend/utils/impmet.py).

Context
-------
impedance_matrix_z_sweep (Session 4) already reuses all frequency-
independent geometry across a sweep, but still fills the impedance matrix
with a Python-level loop over every (m,n) pair. That loop does tiny
(~12x12 or 49x49-element) work per iteration, so it cannot benefit from a
GPU: replacing numpy with cupy INSIDE the loop would be slower than CPU,
since GPU kernel-launch/PCIe overhead would dominate over up to 4*E^2
sub-pair iterations (see workflow.md, "GPU acceleration (CuPy)").

impedance_matrix_z_sweep_gpu restructures the fill so the FAR and
EDGE/VERTEX categories -- which dominate the pair COUNT for any mesh of
practical size -- are each assembled as a handful of large batched tensor
operations (chunked to bound memory) instead of a per-pair loop, on
whichever array backend (backend.utils.compute_backend) this machine
actually has: CuPy on a usable CUDA GPU, NumPy otherwise. SELF pairs keep
the original per-pair Duffy quadrature on CPU (irregular per-pair work,
small O(N_tri) count -- not worth the batching complexity).

These tests check that impedance_matrix_z_sweep_gpu(frequencies)[i] matches
impedance_matrix_z_sweep(frequencies[i]) (the already-validated CPU
reference) to numerical precision, for both use_gpu=False (isolates the
bucket-by-category restructuring itself, on CPU, from any GPU floating-
point reordering) and use_gpu=True when a GPU is actually available on the
machine running the tests (skipped otherwise -- this is a real machine
check, not mocked, matching this project's practice of validating against
real hardware / real reference values throughout).
"""

import numpy as np
import pytest

from backend.rwg.rwg1 import Points, Triangles
from backend.utils.impmet import impedance_matrix_z_sweep, impedance_matrix_z_sweep_gpu
from backend.utils.compute_backend import GPU_AVAILABLE

EPSILON = 8.854e-12
MU      = 1.257e-6

FREQS = np.array([1e8, 3e8, 6.5e8])   # spans quasi-static to resonant-ish k*cell


def _build_rwg_objects(raw_points, raw_triangles):
    points_obj = Points(raw_points)
    tri_obj = Triangles(raw_triangles)
    tri_obj.calculate_triangles_area_and_center(points_obj)
    edges_obj = tri_obj.get_edges()
    edges_obj.compute_edges_length(points_obj)
    return points_obj, tri_obj, edges_obj


def _four_triangle_mesh():
    """2x1 strip, 3 interior edges: SELF + EDGE/VERTEX categories, no FAR."""
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
    """1x3 strip, 5 interior edges: exercises SELF, EDGE/VERTEX, AND FAR (e0/e4)."""
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


@pytest.mark.parametrize("builder", [_four_triangle_mesh, _six_triangle_mesh])
def test_batched_cpu_matches_reference(builder):
    """Isolates the bucket-by-category restructuring from GPU floating-point
    reordering: use_gpu=False must match the per-pair reference on CPU."""
    raw_points, raw_triangles = builder()
    points_obj, tri_obj, edges_obj = _build_rwg_objects(raw_points, raw_triangles)

    Z_ref = impedance_matrix_z_sweep(edges_obj, tri_obj, points_obj, FREQS, MU, EPSILON,
                                      quad_order=6, duffy_order=7)
    Z_cpu = impedance_matrix_z_sweep_gpu(edges_obj, tri_obj, points_obj, FREQS, MU, EPSILON,
                                          quad_order=6, duffy_order=7, use_gpu=False)

    np.testing.assert_allclose(
        Z_cpu, Z_ref, rtol=1e-10, atol=1e-12,
        err_msg="batched CPU assembly disagrees with the per-pair reference",
    )


@pytest.mark.skipif(not GPU_AVAILABLE, reason="no usable CUDA GPU on this machine")
@pytest.mark.parametrize("builder", [_four_triangle_mesh, _six_triangle_mesh])
def test_batched_gpu_matches_reference(builder):
    """Runs on the real GPU (no mocking) when one is available; skipped otherwise."""
    raw_points, raw_triangles = builder()
    points_obj, tri_obj, edges_obj = _build_rwg_objects(raw_points, raw_triangles)

    Z_ref = impedance_matrix_z_sweep(edges_obj, tri_obj, points_obj, FREQS, MU, EPSILON,
                                      quad_order=6, duffy_order=7)
    Z_gpu = impedance_matrix_z_sweep_gpu(edges_obj, tri_obj, points_obj, FREQS, MU, EPSILON,
                                          quad_order=6, duffy_order=7, use_gpu=True)

    assert isinstance(Z_gpu, np.ndarray), "result must always be a host (NumPy) array"
    np.testing.assert_allclose(
        Z_gpu, Z_ref, rtol=1e-8, atol=1e-10,
        err_msg="GPU-batched assembly disagrees with the per-pair CPU reference",
    )


def test_use_gpu_true_raises_without_gpu(monkeypatch):
    """use_gpu=True must fail loudly, not silently fall back to CPU."""
    if GPU_AVAILABLE:
        pytest.skip("a real GPU is available on this machine; cannot test the no-GPU error path")
    raw_points, raw_triangles = _four_triangle_mesh()
    points_obj, tri_obj, edges_obj = _build_rwg_objects(raw_points, raw_triangles)
    with pytest.raises(RuntimeError, match="no usable CUDA GPU"):
        impedance_matrix_z_sweep_gpu(edges_obj, tri_obj, points_obj, FREQS, MU, EPSILON, use_gpu=True)


def test_small_chunk_size_still_matches_reference():
    """chunk_size is purely a memory/performance knob; must not change results."""
    raw_points, raw_triangles = _six_triangle_mesh()
    points_obj, tri_obj, edges_obj = _build_rwg_objects(raw_points, raw_triangles)

    Z_ref = impedance_matrix_z_sweep(edges_obj, tri_obj, points_obj, FREQS, MU, EPSILON,
                                      quad_order=6, duffy_order=7)
    Z_small_chunk = impedance_matrix_z_sweep_gpu(edges_obj, tri_obj, points_obj, FREQS, MU, EPSILON,
                                                  quad_order=6, duffy_order=7, use_gpu=False,
                                                  chunk_size=1)

    np.testing.assert_allclose(Z_small_chunk, Z_ref, rtol=1e-10, atol=1e-12)
