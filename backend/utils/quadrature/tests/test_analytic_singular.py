"""
Tests for the closed-form 1/R surface integral (analytic_singular.py) and for
the accuracy it buys the EDGE/VERTEX branch of the impedance assembly.

The reference used here is deliberately NOT another quadrature rule from the
same family: it is an independent polar-coordinate derivation. Splitting the
triangle into three sub-triangles from the observation point and integrating
in polar coordinates about that point, the 1/rho of the kernel cancels the
rho of the area element, so

    int_T dS'/R = sum over sub-triangles of  int rho_max(theta) d(theta)

and for a straight opposite edge at signed distance d, rho_max = d/cos(phi),
whose integral is d*[ln|sec(phi) + tan(phi)|]. That gives an exact value with
no quadrature involved at all -- but only for an observation point INSIDE the
triangle, which is why it cannot simply replace the Wilton form the module
implements (near-singular EDGE/VERTEX pairs need exterior points too).
"""

import numpy as np
import pytest

from backend.utils.quadrature.analytic_singular import (
    inv_R_over_triangle, inv_R_over_triangle_batch,
)
from backend.utils.quadrature.gauss_dunavant import dunavant_rule, map_to_triangle

H = 0.04                       # a representative mesh cell edge, in metres
V1 = np.array([0.0, 0.0, 0.0])
V2 = np.array([H, 0.0, 0.0])
V3 = np.array([0.0, H, 0.0])


def _polar_reference(r0, p1, p2, p3):
    """Independent exact value, valid for r0 strictly inside the triangle."""
    normal = np.cross(p2 - p1, p3 - p1)
    normal = normal / np.linalg.norm(normal)
    total = 0.0
    for A, B in ((p1, p2), (p2, p3), (p3, p1)):
        e_hat = (B - A) / np.linalg.norm(B - A)
        foot = A + np.dot(r0 - A, e_hat) * e_hat
        d_vec = foot - r0
        d = np.linalg.norm(d_vec)
        if d < 1e-14:
            continue
        n_hat = d_vec / d
        t_hat = np.cross(normal, n_hat)
        angle = lambda u: np.arctan2(np.dot(u, t_hat), np.dot(u, n_hat))
        prim = lambda ph: np.log(abs(1.0 / np.cos(ph) + np.tan(ph)))
        uA = (A - r0) / np.linalg.norm(A - r0)
        uB = (B - r0) / np.linalg.norm(B - r0)
        total += d * abs(prim(angle(uB)) - prim(angle(uA)))
    return total


@pytest.mark.parametrize("bary", [
    (1 / 3, 1 / 3, 1 / 3),        # centroid
    (0.90, 0.05, 0.05),           # close to a vertex
    (0.48, 0.48, 0.04),           # close to an edge
    (0.499, 0.499, 0.002),        # very close to an edge
])
def test_matches_independent_polar_reference(bary):
    r0 = bary[0] * V1 + bary[1] * V2 + bary[2] * V3
    got = inv_R_over_triangle(r0, V1, V2, V3)
    expected = _polar_reference(r0, V1, V2, V3)
    np.testing.assert_allclose(got, expected, rtol=1e-12)


@pytest.mark.parametrize("r0", [
    V1,                                  # exactly on a vertex
    0.5 * (V1 + V2),                     # exactly on an edge
    np.array([2 * H, 0.0, 0.0]),         # on an edge's line, past the endpoint
    np.array([2 * H / 3, 2 * H / 3, 0]), # outside the triangle
    np.array([H, H, 0.0]),               # opposite corner of the cell
])
def test_stays_finite_on_the_degenerate_configurations(r0):
    """
    The (R + l) factors collapse to zero when r0 lies on an edge's line beyond
    an endpoint, and P0 vanishes when it lies on the line itself. Both must be
    handled rather than producing inf/nan, because near-singular EDGE/VERTEX
    pairs put observation points exactly there.
    """
    value = inv_R_over_triangle(r0, V1, V2, V3)
    assert np.isfinite(value)
    assert value >= 0.0


def test_batch_matches_scalar():
    rng = np.random.default_rng(0)
    n = 200
    a, b = rng.random(n), rng.random(n)
    outside = a + b > 1
    a[outside], b[outside] = 1 - a[outside], 1 - b[outside]
    pts = np.zeros((3, n))
    pts[0], pts[1] = a * H, b * H

    batched = inv_R_over_triangle_batch(
        pts,
        np.repeat(V1[:, None], n, axis=1),
        np.repeat(V2[:, None], n, axis=1),
        np.repeat(V3[:, None], n, axis=1),
    )
    scalar = np.array([inv_R_over_triangle(pts[:, i], V1, V2, V3) for i in range(n)])
    np.testing.assert_allclose(batched, scalar, rtol=0, atol=0)


def _double_integral_via_extraction(T_test, T_source, order):
    """int_Tt int_Ts dS dS'/R, analytic inner + Gauss outer (the production scheme)."""
    xi, w = dunavant_rule(order)
    pts = map_to_triangle(xi, *T_test)
    area_t = 0.5 * np.linalg.norm(np.cross(T_test[1] - T_test[0], T_test[2] - T_test[0]))
    inner = np.array([inv_R_over_triangle(pts[:, q], *T_source) for q in range(len(w))])
    return 2.0 * area_t * float(np.dot(w, inner))


def _double_integral_via_plain_gauss(T_test, T_source, order):
    """The same integral left entirely to a Gauss rule -- what the branch used to do."""
    xi, w = dunavant_rule(order)
    r_t = map_to_triangle(xi, *T_test)
    r_s = map_to_triangle(xi, *T_source)
    area_t = 0.5 * np.linalg.norm(np.cross(T_test[1] - T_test[0], T_test[2] - T_test[0]))
    area_s = 0.5 * np.linalg.norm(np.cross(T_source[1] - T_source[0], T_source[2] - T_source[0]))
    R = np.linalg.norm(r_t[:, :, None] - r_s[:, None, :], axis=0)
    return np.einsum('p,q,pq->', w, w, 1.0 / R) * (2 * area_t) * (2 * area_s)


def test_extraction_beats_plain_gauss_on_an_edge_adjacent_pair():
    """
    The whole point of the EDGE/VERTEX branch change: on a shared-edge pair the
    1/R peak sits on the boundary, where a Gauss rule cannot resolve it. The
    reference is built by subdividing the test triangle (so the outer rule sees
    a smooth integrand on each piece) with the exact inner formula.
    """
    T_test = (V1, V2, V3)
    T_source = (np.array([H, 0.0, 0.0]), np.array([H, H, 0.0]), np.array([0.0, H, 0.0]))

    # Reference: same exact inner, but the outer integral resolved by subdivision.
    tris = [T_test]
    for _ in range(4):
        refined = []
        for a, b, c in tris:
            ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
            refined += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
        tris = refined
    reference = sum(_double_integral_via_extraction(t, T_source, 6) for t in tris)

    extracted = _double_integral_via_extraction(T_test, T_source, 11)
    plain = _double_integral_via_plain_gauss(T_test, T_source, 11)

    err_extracted = abs(extracted - reference) / reference
    err_plain = abs(plain - reference) / reference

    assert err_extracted < 1e-3, f"extraction error {err_extracted:.2e} unexpectedly large"
    assert err_plain > 20 * err_extracted, (
        f"plain Gauss ({err_plain:.2e}) should be far worse than extraction "
        f"({err_extracted:.2e}); if it is not, the branch no longer needs the closed form"
    )
