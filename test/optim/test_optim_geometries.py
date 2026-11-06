"""
Tests for solver.optim.geometries.planar: port cuts, conformity, closed loop.
No Z assembly -- fast.
"""

from collections import Counter

import numpy as np
import pytest

from solver.optim.geometries import (
    grading_nodes, parallel_strips, square_loop, strip_dipole,
)


def _edge_multiplicity(mesh):
    """Counter {(v_a, v_b): number of triangles having that side}."""
    tri = mesh.triangles.triangles[:3].T
    c = Counter()
    for a, b, d in tri:
        for e in ((a, b), (b, d), (d, a)):
            c[tuple(sorted((int(e[0]), int(e[1]))))] += 1
    return c


def _check_conformal(mesh):
    mult = _edge_multiplicity(mesh)
    assert max(mult.values()) <= 2                              # no edge with > 2 triangles
    interior = {e for e, n in mult.items() if n == 2}
    rwg = {tuple(sorted((int(a), int(b))))
           for a, b in zip(mesh.edges.first_points, mesh.edges.second_points)}
    assert rwg == interior                                       # RWG edges = interior edges
    assert mesh.n_edges == len(interior)                         # each listed once
    return mult


def _boundary_edges(mesh, mult):
    pts = mesh.points.points
    return [(pts[:, a], pts[:, b]) for (a, b), n in mult.items() if n == 1]


@pytest.mark.parametrize("grading", [None, "cosine"])
def test_strip_dipole(grading):
    m = strip_dipole(length=0.15, width=0.004, n_len=40, n_width=4, edge_grading=grading)
    mult = _check_conformal(m)
    port = m.port()
    assert port.n_edges == 4
    assert abs(port.width(m.edges) - 0.004) < 1e-12
    assert np.all(port.signs == 1.0)
    v = m.points.total_of_points
    assert v - len(mult) + m.triangles.total_of_triangles == 1     # Euler: a disc
    if grading == "cosine":
        lens = np.sort(m.edges.edges_length[port.edge_indices])
        assert lens[0] < lens[-1]                                    # graded: unequal cells


@pytest.mark.parametrize("w", [1e-3, 2e-3, 4e-3])
@pytest.mark.parametrize("grading", [None, "cosine"])
def test_square_loop(w, grading):
    s = 0.02
    m = square_loop(side_centerline=s, width=w, n_side=16, n_width=4, edge_grading=grading)
    mult = _check_conformal(m)

    port = m.port()
    assert port.n_edges == 4
    assert abs(port.width(m.edges) - w) < 1e-12                     # cut length = width
    assert np.all(port.signs == port.signs[0])
    load = m.port("load")                                            # opposite side (V5b')
    assert load.n_edges == 4 and abs(load.width(m.edges) - w) < 1e-12
    assert not set(load.edge_indices) & set(port.edge_indices)

    # Closed loop: Euler characteristic of an annulus is 0 ...
    v = m.points.total_of_points
    assert v - len(mult) + m.triangles.total_of_triangles == 0
    # ... and every free edge lies on the outer or inner boundary square,
    a_out, a_in = s / 2 + w / 2, s / 2 - w / 2
    boundary = _boundary_edges(m, mult)
    for p, q in boundary:
        on = False
        for a in (a_out, a_in):
            for ax in (0, 1):
                if abs(abs(p[ax]) - a) < 1e-12 and abs(abs(q[ax]) - a) < 1e-12 and p[ax] == q[ax]:
                    other = 1 - ax
                    on |= max(abs(p[other]), abs(q[other])) <= a + 1e-12
        assert on, f"free edge {p} -> {q} is not on the loop's boundary"
    # ... which together they cover exactly: 4 * 2a_out + 4 * 2a_in.
    perimeter = sum(np.linalg.norm(q - p) for p, q in boundary)
    assert abs(perimeter - 8 * (a_out + a_in)) < 1e-12


def test_parallel_strips():
    m = parallel_strips(length=0.45, width=0.01, spacing=0.05, n_len=30, n_width=2)
    mult = _check_conformal(m)
    feed, parasite = m.ports()
    assert feed.name == "feed" and parasite.name == "parasite"
    for p in (feed, parasite):
        assert p.n_edges == 2 and abs(p.width(m.edges) - 0.01) < 1e-12
    v = m.points.total_of_points
    assert v - len(mult) + m.triangles.total_of_triangles == 2       # two discs


def test_grading_nodes():
    t = grading_nodes(8, "cosine")
    assert t[0] == 0.0 and t[-1] == 1.0
    np.testing.assert_allclose(t + t[::-1], 1.0, atol=1e-15)          # symmetric
    h = np.diff(t)
    assert h[0] < h[len(h) // 2]                                       # denser at the edges
    np.testing.assert_allclose(grading_nodes(4), [0, 0.25, 0.5, 0.75, 1.0])
    with pytest.raises(ValueError):
        grading_nodes(4, "bogus")
    with pytest.raises(ValueError):
        strip_dipole(0.1, 0.01, n_len=5, n_width=1)                    # odd: no x = 0 line
