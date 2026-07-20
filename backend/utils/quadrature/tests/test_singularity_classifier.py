"""
Tests for singularity_classifier.py.

Uses the same 2-, 4-, and 6-triangle meshes from brute_force_reference.py
so that the classifications can be verified by inspection.
"""

import numpy as np
import pytest

from backend.utils.quadrature.singularity_classifier import (
    FAR, VERTEX, EDGE, SELF,
    classify_triangle_pair,
    classify_all_triangle_pairs,
    classify_edge_pair_max,
    classify_all_edge_pairs,
    get_pairs_by_category,
)


# ---------------------------------------------------------------------------
# classify_triangle_pair
# ---------------------------------------------------------------------------

class TestClassifyTrianglePair:
    def test_self(self):
        assert classify_triangle_pair([0, 1, 2], [0, 1, 2]) == SELF

    def test_self_permuted_vertices(self):
        # Same vertex set, different order → still SELF
        assert classify_triangle_pair([0, 1, 2], [2, 0, 1]) == SELF

    def test_edge_adjacent(self):
        # Two triangles sharing edge (1,2)
        assert classify_triangle_pair([0, 1, 2], [3, 1, 2]) == EDGE

    def test_vertex_adjacent(self):
        assert classify_triangle_pair([0, 1, 2], [3, 4, 2]) == VERTEX

    def test_far(self):
        assert classify_triangle_pair([0, 1, 2], [3, 4, 5]) == FAR

    def test_numpy_arrays(self):
        # Should work with numpy integer arrays, not just Python lists
        v1 = np.array([0, 1, 2], dtype=np.int32)
        v2 = np.array([1, 2, 5], dtype=np.int32)
        assert classify_triangle_pair(v1, v2) == EDGE


# ---------------------------------------------------------------------------
# classify_all_triangle_pairs — 4-triangle mesh
#
# Mesh layout (two sharing a long edge, plus two more outer triangles):
#
#    tri 0: vertices 0,1,3   (shares edge 1-3 with tri 1)
#    tri 1: vertices 1,2,3   (shares vertex 2 with tri 2 via edge 2-3? No...)
#
# We build a concrete mesh and verify by inspection.
# ---------------------------------------------------------------------------

def _four_triangle_mesh_vertices():
    """
    4 triangles that form a diamond (2 on each side of a central edge).
    Vertex layout:
        0 - top
        1 - left
        2 - right
        3 - bottom-left
        4 - bottom-right

    Triangles:
        T0 = (0, 1, 2)   ← top diamond
        T1 = (1, 3, 2)   ← shares edge (1,2) with T0
        T2 = (2, 3, 4)   ← shares vertex 3 with T1 and vertex 2 with T0/T1
        T3 = (5, 6, 7)   ← far from all
    """
    tri = np.array([
        [0, 1, 2, 5],   # first vertex
        [1, 3, 3, 6],   # second vertex
        [2, 2, 4, 7],   # third vertex
    ], dtype=np.int64)
    return tri


class TestClassifyAllTrianglePairs:
    def setup_method(self):
        self.tri = _four_triangle_mesh_vertices()
        self.cat = classify_all_triangle_pairs(self.tri)

    def test_diagonal_is_self(self):
        assert np.all(np.diag(self.cat) == SELF)

    def test_symmetry(self):
        np.testing.assert_array_equal(self.cat, self.cat.T)

    def test_shape(self):
        assert self.cat.shape == (4, 4)

    def test_t0_t1_edge_adjacent(self):
        # T0=(0,1,2), T1=(1,3,2): share vertices {1,2} → EDGE
        assert self.cat[0, 1] == EDGE

    def test_t0_t2_vertex_adjacent(self):
        # T0=(0,1,2), T2=(2,3,4): share vertex {2} → VERTEX
        assert self.cat[0, 2] == VERTEX

    def test_t0_t3_far(self):
        # T0=(0,1,2), T3=(5,6,7): no shared vertices → FAR
        assert self.cat[0, 3] == FAR

    def test_t1_t2_vertex_adjacent(self):
        # T1=(1,3,2), T2=(2,3,4): share vertices {2,3} → EDGE
        assert self.cat[1, 2] == EDGE

    def test_t2_t3_far(self):
        assert self.cat[2, 3] == FAR

    def test_t1_t3_far(self):
        assert self.cat[1, 3] == FAR


# ---------------------------------------------------------------------------
# get_pairs_by_category
# ---------------------------------------------------------------------------

class TestGetPairsByCategory:
    def setup_method(self):
        tri = _four_triangle_mesh_vertices()
        self.cat = classify_all_triangle_pairs(tri)

    def test_far_pairs_are_correct(self):
        far = set(get_pairs_by_category(self.cat, FAR))
        assert (0, 3) in far
        assert (3, 0) in far
        assert (1, 3) in far
        assert (3, 1) in far
        assert (2, 3) in far
        assert (3, 2) in far

    def test_self_pairs_are_diagonal(self):
        self_pairs = get_pairs_by_category(self.cat, SELF)
        assert sorted(self_pairs) == [(0, 0), (1, 1), (2, 2), (3, 3)]

    def test_no_wrong_category_in_far(self):
        far = get_pairs_by_category(self.cat, FAR)
        # None of the diagonal entries should appear in far
        for i, j in far:
            assert i != j


# ---------------------------------------------------------------------------
# classify_edge_pair_max — 2-triangle mesh (single RWG edge)
#
# Mesh: 2 triangles sharing edge (1,2):
#   T0 = (0, 1, 2)
#   T1 = (3, 1, 2)
# Edge 0: T+= T0, T-= T1
# Edge pair (0,0) → SELF
# ---------------------------------------------------------------------------

def _two_triangle_mesh():
    """Returns (tri_vertices, tri_plus, tri_minus)."""
    tri = np.array([[0, 3], [1, 1], [2, 2]], dtype=np.int64)  # (3, 2)
    plus  = np.array([0], dtype=np.int64)
    minus = np.array([1], dtype=np.int64)
    return tri, plus, minus


def _six_triangle_mesh():
    """
    6 triangles, 5 RWG edges.  Mirroring brute_force_reference.build_six_triangle_mesh().

    Vertices:
        0: (0, 0, 0)
        1: (1, 0, 0)
        2: (0, 1, 0)
        3: (2, 0, 0)
        4: (0, 2, 0)
        5: (3, 0, 0)
        6: (0, 3, 0)

    Triangles (each row = one triangle):
        T0 = (0, 1, 2)
        T1 = (1, 3, 2)
        T2 = (0, 2, 4)   ← T2 and T0 share edge (0,2)
        T3 = (1, 2, 3)   ← note: same as T1 vertices but different order for test coverage
        T4 = (3, 5, 2)   ← T4 shares vertex 3 with T1
        T5 = (4, 2, 6)   ← T5 shares vertex 4 with T2

    Edges (+ / - triangles):
        e0: T+ = T0, T- = T1  (shared edge 1-2)
        e1: T+ = T0, T- = T2  (shared edge 0-2)
        e2: T+ = T1, T- = T4  (shared edge 3-2... shared vertex 3 between T1 and T4)
             Actually T1=(1,3,2), T4=(3,5,2) share edge 3-2 → EDGE
        e3: T+ = T2, T- = T5  (shared edge 4-2... T2=(0,2,4), T5=(4,2,6) share edge 2-4 → EDGE)
        e4: T+ = T4, T- = T5  (T4=(3,5,2), T5=(4,2,6) share vertex 2 only → VERTEX,
             but we make them non-adjacent for a TRUE far pair)

    For a clean far pair between e0 and e4, we need the vertex sets of
    {T0, T1} and {T4, T5} to be disjoint.

    Adjusted vertex assignment:
        T0 = (0, 1, 2)   vertices {0,1,2}
        T1 = (3, 1, 2)   vertices {3,1,2}  → e0: union {0,1,2,3}
        T4 = (5, 6, 7)   vertices {5,6,7}
        T5 = (8, 6, 7)   vertices {8,6,7}  → e4: union {5,6,7,8}
    So e0 and e4 share no vertices → FAR.
    """
    tri = np.array([
        # T0   T1   T2   T3   T4   T5
        [  0,   3,   0,   0,   5,   8],   # v0
        [  1,   1,   2,   1,   6,   6],   # v1
        [  2,   2,   4,   3,   7,   7],   # v2
    ], dtype=np.int64)

    plus  = np.array([0, 0, 1, 2, 4], dtype=np.int64)
    minus = np.array([1, 2, 3, 3, 5], dtype=np.int64)
    # e0: {T0,T1} = {0,1,2} ∪ {3,1,2} = {0,1,2,3}
    # e4: {T4,T5} = {5,6,7} ∪ {8,6,7} = {5,6,7,8}
    # → no overlap → FAR
    return tri, plus, minus


class TestClassifyEdgePairMax:
    def test_self_pair(self):
        tri, plus, minus = _two_triangle_mesh()
        assert classify_edge_pair_max(0, 0, plus, minus, tri) == SELF

    def test_far_pair_six_tri(self):
        tri, plus, minus = _six_triangle_mesh()
        # e0 and e4 should be FAR
        assert classify_edge_pair_max(0, 4, plus, minus, tri) == FAR
        assert classify_edge_pair_max(4, 0, plus, minus, tri) == FAR


class TestClassifyAllEdgePairs:
    def setup_method(self):
        self.tri, self.plus, self.minus = _six_triangle_mesh()
        self.cat = classify_all_edge_pairs(self.plus, self.minus, self.tri)

    def test_shape(self):
        assert self.cat.shape == (5, 5)

    def test_diagonal_is_self(self):
        assert np.all(np.diag(self.cat) == SELF)

    def test_symmetry(self):
        np.testing.assert_array_equal(self.cat, self.cat.T)

    def test_e0_e4_far(self):
        assert self.cat[0, 4] == FAR
        assert self.cat[4, 0] == FAR

    def test_e0_e1_not_far(self):
        # e0 and e1 both involve T0 → at minimum vertex-adjacent, likely EDGE
        assert self.cat[0, 1] >= VERTEX
