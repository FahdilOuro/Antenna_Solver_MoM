"""
Tests for multiple_gap_sources (backend/utils/gap_source.py).

Context
-------
The gap window used to test only the excitation-axis coordinate of each
triangle's centroid against the feed point (eq. 7's 1D step function,
u(z+W/2)-u(z-W/2), correct for a simple wire/strip where nothing else in
the mesh shares that axis coordinate far from the feed). At a junction
between differently-oriented surfaces sharing a feed point -- e.g. a
vertical monopole blade standing on a horizontal ground plate, fed along
the blade's axis -- this spuriously excites any triangle elsewhere in the
mesh that happens to share that one coordinate, regardless of how far away
it is on the other two axes. Confirmed on a real monopole mesh: exciting
along x with gap_width=0.6 m lit up 198 of 296 edges, spanning y from -0.91
to +0.92 m on a plate 2 m wide -- essentially the whole ground plane, not a
localised feed at the monopole's base.

The fix replaces the 1D test with a 3D box (gap_width cube centred on the
feed point, tested independently on all three coordinates). These tests use
lightweight fake mesh objects (SimpleNamespace) exposing exactly the
attributes multiple_gap_sources documents needing, so the junction geometry
is small and exact rather than relying on a full gmsh-generated mesh.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from backend.utils.gap_source import multiple_gap_sources, GapSourceError


def _make_mesh(tri_centers, edge_pairs, rho_plus, rho_minus, edge_lengths=None):
    """
    tri_centers : (3, N_tri) triangle centroids
    edge_pairs  : list of (T_plus, T_minus) triangle-index pairs, one per RWG edge
    rho_plus/rho_minus : (3, N_edges)
    """
    n_edges = len(edge_pairs)
    triangles_plus  = np.array([p for p, _ in edge_pairs])
    triangles_minus = np.array([m for _, m in edge_pairs])
    triangles = SimpleNamespace(
        triangles_plus=triangles_plus,
        triangles_minus=triangles_minus,
        triangles_center=np.asarray(tri_centers, dtype=float),
    )
    edges = SimpleNamespace(
        total_number_of_edges=n_edges,
        edges_length=np.ones(n_edges) if edge_lengths is None else np.asarray(edge_lengths, dtype=float),
    )
    vecteurs_rho = SimpleNamespace(
        vecteur_rho_plus=np.asarray(rho_plus, dtype=float),
        vecteur_rho_minus=np.asarray(rho_minus, dtype=float),
    )
    return triangles, edges, vecteurs_rho


# ---------------------------------------------------------------------------
# Core fix: a junction where a far-away triangle shares the excitation-axis
# coordinate with the feed point, but is nowhere near it in the other axes.
# ---------------------------------------------------------------------------

def test_excludes_far_geometry_sharing_only_the_excitation_axis_coordinate():
    """
    Mimics a monopole-on-ground-plane fed along x: a "near" edge sits right
    at the feed point, a "far" edge sits at the same x but y=5 m away
    (e.g. the far side of a wide ground plate). Only the near edge should be
    excited.
    """
    # T0,T1 near the feed point (0,0,0); T2,T3 far away in y but same x.
    tri_centers = np.array([
        [0.0,  0.05, 0.0,  0.05],   # x
        [0.0,  0.0,  5.0,  5.0 ],   # y
        [0.0,  0.0,  0.0,  0.0 ],   # z
    ])
    edge_pairs = [(0, 1), (2, 3)]   # edge 0 = near, edge 1 = far
    rho_plus  = np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    rho_minus = np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])

    triangles, edges, vecteurs_rho = _make_mesh(tri_centers, edge_pairs, rho_plus, rho_minus)

    V, feeding_edges = multiple_gap_sources(
        triangles, edges, vecteurs_rho,
        voltage_amplitude=1.0, feed_points=np.array([0.0, 0.0, 0.0]),
        excitation_unit_vector='x', gap_width=1.0,
    )

    assert set(feeding_edges[0].tolist()) == {0}, \
        "the far edge (same x, y=5m away) must not be excited"
    assert V[0] != 0.0
    assert V[1] == 0.0, "far edge must receive zero excitation voltage"


def test_includes_far_geometry_when_gap_width_actually_spans_it():
    """Sanity check: the far edge DOES get excited if gap_width is made large
    enough to genuinely reach it in all 3 dimensions (box, not a bug)."""
    tri_centers = np.array([
        [0.0,  0.05, 0.0,  0.05],
        [0.0,  0.0,  5.0,  5.0 ],
        [0.0,  0.0,  0.0,  0.0 ],
    ])
    edge_pairs = [(0, 1), (2, 3)]
    rho_plus  = np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    rho_minus = np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    triangles, edges, vecteurs_rho = _make_mesh(tri_centers, edge_pairs, rho_plus, rho_minus)

    V, feeding_edges = multiple_gap_sources(
        triangles, edges, vecteurs_rho,
        voltage_amplitude=1.0, feed_points=np.array([0.0, 0.0, 0.0]),
        excitation_unit_vector='x', gap_width=12.0,   # half-width 6m > 5m
    )

    assert set(feeding_edges[0].tolist()) == {0, 1}
    assert V[1] != 0.0


# ---------------------------------------------------------------------------
# Backward compatibility: a simple in-line structure (cross-section already
# well inside the box) must behave exactly as the old axial-only test did.
# ---------------------------------------------------------------------------

def test_inline_structure_matches_axial_only_window():
    """For a thin wire/strip along the excitation axis, restricting the other
    two axes too must not change anything, since the cross-section already
    fits inside gap_width."""
    gap_width = 1.0   # half_W = 0.5
    # A strip along y (excitation axis), cross-section +-1mm in x and z --
    # tiny compared to gap_width=1.0. Triangle centroids at y=0.0 (edge 0),
    # y=0.3 (edge 1): both inside |y|<=0.5. y=0.6 (edge 2): outside.
    tri_centers = np.array([
        [0.0,   0.001, -0.001],   # x (all << half_W=0.5)
        [0.0,   0.3,    0.6  ],   # y
        [0.0,   0.0,   -0.001],   # z (all << half_W=0.5)
    ])
    edge_pairs = [(0, 0), (1, 1), (2, 2)]   # degenerate T+=T- just to test windowing per-triangle
    rho_plus  = np.ones((3, 3))
    rho_minus = np.ones((3, 3))
    triangles, edges, vecteurs_rho = _make_mesh(tri_centers, edge_pairs, rho_plus, rho_minus)

    V, feeding_edges = multiple_gap_sources(
        triangles, edges, vecteurs_rho,
        voltage_amplitude=1.0, feed_points=np.array([0.0, 0.0, 0.0]),
        excitation_unit_vector='y', gap_width=gap_width,
    )

    # Old axial-only test would give the same answer here: |y| <= 0.5 for
    # edges 0 (y=0.0) and 1 (y=0.3), and NOT edge 2 (y=0.6).
    assert set(feeding_edges[0].tolist()) == {0, 1}


# ---------------------------------------------------------------------------
# Existing behaviour preserved: no-edges-in-gap warning, multi-port handling
# ---------------------------------------------------------------------------

def test_raises_when_no_edge_in_gap():
    """A port that feeds nothing must fail loudly, not silently solve with
    a zero excitation for that port (see workflow.md, "fail fast on empty
    gap")."""
    tri_centers = np.array([[10.0], [10.0], [10.0]])
    edge_pairs = [(0, 0)]
    rho_plus  = np.ones((3, 1))
    rho_minus = np.ones((3, 1))
    triangles, edges, vecteurs_rho = _make_mesh(tri_centers, edge_pairs, rho_plus, rho_minus)

    with pytest.raises(GapSourceError, match="no RWG edge found within the gap"):
        multiple_gap_sources(
            triangles, edges, vecteurs_rho,
            voltage_amplitude=1.0, feed_points=np.array([0.0, 0.0, 0.0]),
            excitation_unit_vector='x', gap_width=0.1,
        )


def test_multi_port_independent_localisation():
    """Two ports at different locations must each excite only their own
    local edge, not each other's."""
    tri_centers = np.array([
        [0.0, 0.0, 5.0, 5.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    edge_pairs = [(0, 0), (2, 2)]   # edge0 near port A, edge1 near port B
    rho_plus  = np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    rho_minus = np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    triangles, edges, vecteurs_rho = _make_mesh(tri_centers, edge_pairs, rho_plus, rho_minus)

    feed_points = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    V, feeding_edges = multiple_gap_sources(
        triangles, edges, vecteurs_rho,
        voltage_amplitude=[1.0, 1.0], feed_points=feed_points,
        excitation_unit_vector=['x', 'x'], gap_width=1.0,
    )
    assert set(feeding_edges[0].tolist()) == {0}
    assert set(feeding_edges[1].tolist()) == {1}
    assert V[0] != 0.0 and V[1] != 0.0
