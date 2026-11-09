"""
Tests for solver.optim.ports: port geometry, orientation and error handling.
No Z assembly here -- these run in milliseconds.
"""

import numpy as np
import pytest

from solver.optim import (
    Port, PortDefinitionError, incidence_matrix, port_from_cut, port_from_edge,
)
from conftest import X_HAT


def test_cut_port_geometry(wide_strip, narrow_strip):
    m = wide_strip
    port = port_from_cut(m.points, m.triangles, m.edges, *m.cut_at(0.0), direction=X_HAT,
                         name="feed")

    lengths = m.edges.edges_length[port.edge_indices]
    assert port.n_edges == 3                                   # ny = 3 cells across
    assert abs(lengths.sum() - m.width) < 1e-12                # the cut spans the strip
    assert abs(port.width(m.edges) - m.width) < 1e-12
    assert np.all(port.signs == port.signs[0])                 # straight transverse cut
    assert np.all(port.signs == 1.0)                           # T+ is on the -x side here

    a = port.incidence_vector(m.edges)
    assert a.shape == (m.edges.total_number_of_edges,)
    np.testing.assert_array_equal(np.flatnonzero(a), np.sort(port.edge_indices))
    np.testing.assert_allclose(a[port.edge_indices], port.signs * lengths, rtol=0, atol=0)

    # Every selected edge really is transverse: both endpoints at x = 0.
    pts = m.points.points
    for e in port.edge_indices:
        assert abs(pts[0, m.edges.first_points[e]]) < 1e-12
        assert abs(pts[0, m.edges.second_points[e]]) < 1e-12

    # Reversing the positive direction flips every sign, nothing else.
    rev = port_from_cut(m.points, m.triangles, m.edges, *m.cut_at(0.0), direction=-X_HAT,
                        name="feed_rev")
    np.testing.assert_array_equal(rev.edge_indices, port.edge_indices)
    np.testing.assert_array_equal(rev.signs, -port.signs)

    # Single-edge-wide strip: the cut is one edge, of length = width.
    n = narrow_strip
    p1 = port_from_cut(n.points, n.triangles, n.edges, *n.cut_at(0.0), direction=X_HAT,
                       name="feed")
    assert p1.n_edges == 1
    assert abs(n.edges.edges_length[p1.edge_indices[0]] - n.width) < 1e-12


def test_cut_port_matches_single_edge_port(narrow_strip):
    """On a one-edge-wide strip, the cut and the rwg3-style edge selection agree."""
    n = narrow_strip
    for x in (0.0, 0.1, -0.17):
        cut = port_from_cut(n.points, n.triangles, n.edges, *n.cut_at(x), direction=X_HAT,
                            name="cut")
        edge = port_from_edge(n.points, n.triangles, n.edges, [x, 0.0, 0.0], X_HAT, name="edge")
        np.testing.assert_array_equal(cut.edge_indices, edge.edge_indices)
        np.testing.assert_array_equal(cut.signs, edge.signs)


def test_cut_port_errors(wide_strip):
    m = wide_strip
    args = (m.points, m.triangles, m.edges)

    # Cut entirely off the mesh.
    with pytest.raises(PortDefinitionError, match="no RWG edge"):
        port_from_cut(*args, [1.0, -0.01, 0.0], [1.0, 0.01, 0.0], X_HAT, "off_mesh")

    # Cut between grid lines: no edge lies on it.
    x_mid = 0.5 * (m.xs[10] + m.xs[11])
    with pytest.raises(PortDefinitionError, match="no RWG edge"):
        port_from_cut(*args, *m.cut_at(x_mid), X_HAT, "between_lines")

    # Cut parallel to the current: along x (on an interior horizontal grid line),
    # with the current also along x.
    y1 = m.ys[1]
    with pytest.raises(PortDefinitionError, match="parallel"):
        port_from_cut(*args, [-0.1, y1, 0.0], [0.1, y1, 0.0], X_HAT, "parallel")

    # Direction normal to the strip: no in-surface component across any edge.
    with pytest.raises(PortDefinitionError, match="does not cross"):
        port_from_cut(*args, *m.cut_at(0.0), [0.0, 0.0, 1.0], "normal")

    # Cut longer than the strip is wide: edges found but they do not cover it.
    with pytest.raises(PortDefinitionError, match="total length"):
        port_from_cut(*args, [0.0, -m.width, 0.0], [0.0, m.width, 0.0], X_HAT, "too_long")

    # Overlapping ports.
    p_a = port_from_cut(*args, *m.cut_at(0.0), X_HAT, "a")
    p_b = port_from_cut(*args, *m.cut_at(0.0), X_HAT, "b")
    with pytest.raises(PortDefinitionError, match="share edge"):
        incidence_matrix([p_a, p_b], m.edges)

    # Partially overlapping (only one shared edge) is caught too.
    p_c = Port("c", edge_indices=p_a.edge_indices[:1], signs=[1.0], direction=X_HAT)
    with pytest.raises(PortDefinitionError, match="share edge"):
        incidence_matrix([p_c, port_from_cut(*args, *m.cut_at(0.0), X_HAT, "d")], m.edges)

    # Duplicate names.
    p_e = port_from_cut(*args, *m.cut_at(m.xs[4]), X_HAT, "a")
    with pytest.raises(PortDefinitionError, match="unique"):
        incidence_matrix([p_a, p_e], m.edges)

    # Single-edge port with no edge perpendicular to the direction.
    with pytest.raises(PortDefinitionError, match="perpendicular"):
        port_from_edge(*args, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], "skew")

    # Edge index beyond the mesh.
    bad = Port("bad", edge_indices=[m.edges.total_number_of_edges], signs=[1.0], direction=X_HAT)
    with pytest.raises(PortDefinitionError, match="only"):
        bad.incidence_vector(m.edges)


def test_port_is_immutable(wide_strip):
    m = wide_strip
    port = port_from_cut(m.points, m.triangles, m.edges, *m.cut_at(0.0), X_HAT, "feed")
    with pytest.raises(ValueError):
        port.signs[0] = -1.0
    with pytest.raises(AttributeError):
        port.name = "other"
