"""
Tests for the Gauss-Dunavant-based rwg_gram_matrix (solver/utils/impmet.py).

Context
-------
rwg_gram_matrix assembles the RWG mass matrix used for the surface-impedance
boundary condition (SIBC) term of a lossy conductor:
    Z_lossy = matrice_z + impedance_zs * rwg_gram_matrix(...)

It replaces a legacy implementation that approximated the integral
INT_T f_m.f_n dS with an unweighted average over 9 barycentric sub-triangle
centres (a rule only exact for degree <= 1). Since f_m.f_n is an exact
degree-2 polynomial on the shared triangle, that approximation carried real
(not just numerical-noise-level) discretisation error -- see workflow.md,
"Gram matrix Gauss-Dunavant refactor" for a worked example: the legacy rule
gave ~0.0087 for an edge pair whose true overlap integral is exactly 0, and
was off by ~3.6% (Frobenius) on the diagonal self terms.

Validation strategy
--------------------
1. Hand-derived analytical reference (edge-midpoint quadrature, exact for
   degree <= 2) for a small mesh, computed independently of the module under
   test -- the real correctness anchor.
2. Quadrature-order invariance: quad_order in {3,4,5,6,7,11} must all agree
   to machine precision (the polynomial-exactness claim).
3. Structural properties: G must be exactly symmetric and positive
   semi-definite (it is literally a mass matrix: c^T G c = INT_S |sum_m
   c_m f_m|^2 dS >= 0 for any real coefficient vector c).
4. End-to-end wiring: Vecteurs_Rho.calculate_vecteurs_rho -> DataManager_rwg2
   save/load round trip -> rwg_gram_matrix, matching the real production
   call path in rwg4.calculate_current_radiation(lossy_conductor=True).
"""

import os
import tempfile

import numpy as np
import pytest

from solver.rwg.rwg1 import Points, Triangles
from solver.rwg.rwg2 import Vecteurs_Rho, DataManager_rwg2
from solver.utils.impmet import rwg_gram_matrix


def _build_rwg_objects(raw_points, raw_triangles):
    points_obj = Points(raw_points)
    tri_obj = Triangles(raw_triangles)
    tri_obj.calculate_triangles_area_and_center(points_obj)
    edges_obj = tri_obj.get_edges()
    edges_obj.compute_edges_length(points_obj)
    return points_obj, tri_obj, edges_obj


def _four_triangle_mesh():
    """2x1 strip, 3 interior edges. Same mesh used in FINDINGS.md Phase 6."""
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
    """1x3 strip, 5 interior edges."""
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


def _compute_gram(raw_points, raw_triangles, quad_order=4):
    points_obj, tri_obj, edges_obj = _build_rwg_objects(raw_points, raw_triangles)
    return rwg_gram_matrix(edges_obj, tri_obj, points_obj, quad_order=quad_order)


# ---------------------------------------------------------------------------
# 1. Hand-derived analytical reference (four-triangle mesh)
# ---------------------------------------------------------------------------
#
# T1 = (v1, v4, v3) with v1=(1,0,0), v4=(1,1,0), v3=(0,1,0) is shared by
# edge 0 ("-" side, free vertex v4) and edge 1 ("+" side, free vertex v3).
# Using the edge-midpoint rule (exact for degree <= 2 polynomials):
#   G[0,0] self term on T0 and T1: 1/3 + 1/3 = 2/3 exactly.
#   G[0,1] cross term on T1: INT_T1 (r-v4).(r-v3) dS = 0 exactly
#     (edge-midpoint values 0.25, -0.25, 0.0 sum to zero).
# Full derivation in workflow.md.

def test_four_triangle_matches_hand_derived_values():
    raw_points, raw_triangles = _four_triangle_mesh()
    G = _compute_gram(raw_points, raw_triangles)

    np.testing.assert_allclose(np.diag(G), [2.0 / 3.0] * 3, atol=1e-12,
        err_msg="diagonal self terms must equal the hand-derived 2/3")
    np.testing.assert_allclose(G[0, 1], 0.0, atol=1e-12,
        err_msg="edge0-edge1 overlap on the shared triangle must be exactly 0")
    np.testing.assert_allclose(G[0, 2], 0.0, atol=1e-12,
        err_msg="edge0 and edge2 share no triangle -> must be exactly 0")


# ---------------------------------------------------------------------------
# 2. Quadrature-order invariance (polynomial exactness)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("builder", [_four_triangle_mesh, _six_triangle_mesh])
@pytest.mark.parametrize("order", [3, 5, 6, 7, 11])
def test_gram_matrix_independent_of_quad_order(builder, order):
    """f_m.f_n is an exact degree-2 polynomial: any order >= 2 must agree."""
    raw_points, raw_triangles = builder()
    G_ref   = _compute_gram(raw_points, raw_triangles, quad_order=4)
    G_other = _compute_gram(raw_points, raw_triangles, quad_order=order)
    np.testing.assert_allclose(G_other, G_ref, atol=1e-12, rtol=1e-12,
        err_msg=f"quad_order={order} disagrees with quad_order=4 -- "
                f"integrand should be exact for any order >= 2")


# ---------------------------------------------------------------------------
# 3. Structural properties: exact symmetry, positive semi-definiteness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("builder", [_four_triangle_mesh, _six_triangle_mesh])
def test_gram_matrix_is_symmetric(builder):
    G = _compute_gram(*builder())
    np.testing.assert_allclose(G, G.T, atol=1e-13,
        err_msg="Gram matrix must be exactly symmetric by construction")


@pytest.mark.parametrize("builder", [_four_triangle_mesh, _six_triangle_mesh])
def test_gram_matrix_is_positive_semidefinite(builder):
    """G is a mass matrix: c^T G c = INT_S |sum_m c_m f_m|^2 dS >= 0."""
    G = _compute_gram(*builder())
    eigvals = np.linalg.eigvalsh(G)
    assert eigvals.min() >= -1e-10 * eigvals.max(), \
        f"min eigenvalue {eigvals.min():.3e} is negative (non-physical)"


# ---------------------------------------------------------------------------
# 4. End-to-end wiring: Vecteurs_Rho -> DataManager_rwg2 round trip -> Gram
# ---------------------------------------------------------------------------

class _FakePath:
    """Minimal stand-in for utils.file_path.FilePath, mesh1/mesh2 only."""
    def __init__(self, directory):
        self.mat_mesh1 = os.path.join(directory, "test_mesh1.mat")
        self.mat_mesh2 = os.path.join(directory, "test_mesh2.mat")


def test_end_to_end_vecteurs_rho_datamanager_gram_matrix():
    """Matches the real production call path in
    rwg4.calculate_current_radiation(lossy_conductor=True): rwg1 -> rwg2
    (Vecteurs_Rho + DataManager_rwg2 save/load) -> rwg_gram_matrix.
    """
    from scipy.io import savemat
    from solver.rwg.rwg1 import DataManager_rwg1

    raw_points, raw_triangles = _four_triangle_mesh()
    points_obj, tri_obj, edges_obj = _build_rwg_objects(raw_points, raw_triangles)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = _FakePath(tmpdir)
        DataManager_rwg1.save_data(path, points_obj, tri_obj, edges_obj)

        vecteurs_rho = Vecteurs_Rho()
        vecteurs_rho.calculate_vecteurs_rho(points_obj, tri_obj, edges_obj)
        DataManager_rwg2.save_data(path, vecteurs_rho)

        loaded = DataManager_rwg2.load_data(path.mat_mesh2)
        assert loaded is not None, "round-trip load_data returned None"
        points_ld, tri_ld, edges_ld, vecteurs_rho_ld = loaded

        assert not hasattr(vecteurs_rho_ld, "vecteur_rho_barycentric_plus")

        G_direct  = rwg_gram_matrix(edges_obj, tri_obj, points_obj)
        G_roundtrip = rwg_gram_matrix(edges_ld, tri_ld, points_ld)

        np.testing.assert_allclose(G_roundtrip, G_direct, atol=1e-10,
            err_msg="Gram matrix computed after a save/load round trip must "
                    "match the matrix computed directly")
