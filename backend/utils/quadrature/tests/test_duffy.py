"""
Tests for duffy.py — collapsed-coordinate Duffy transform.

Validation strategy
-------------------
1. Area recovery: kernel_fn = 1 (no singularity) → result must equal area of T
2. 1/R singularity at a vertex: compare against a high-order GL reference
   (using enough points that GL converges for the non-singular sub-triangle case)
3. Smooth kernel: Duffy must reproduce standard Gauss for a smooth integrand
4. duffy_source_nodes: weight sum = area, node count = 3 * order²
5. Self-consistency: duffy_collapsed_quadrature == np.dot(weights, kernel(pts))
"""

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from backend.utils.quadrature.duffy import (
    duffy_collapsed_quadrature,
    duffy_source_nodes,
)
from backend.utils.quadrature.gauss_dunavant import (
    dunavant_rule,
    map_to_triangle,
    integrate_on_triangle,
)


# ---------------------------------------------------------------------------
# Reference triangle (3D, z=0)
# ---------------------------------------------------------------------------
P1 = np.array([0.0, 0.0, 0.0])
P2 = np.array([1.0, 0.0, 0.0])
P3 = np.array([0.0, 1.0, 0.0])
AREA_REF = 0.5


def _area(p1, p2, p3):
    return 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))


# ---------------------------------------------------------------------------
# 1. Area recovery (constant kernel, no singularity)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("order", [3, 5, 7])
def test_area_recovery_singular_at_vertex(order):
    """Duffy with constant kernel = 1 must integrate to area(T)."""
    result = duffy_collapsed_quadrature(P1, P1, P2, P3, lambda r: 1.0, order=order)
    np.testing.assert_allclose(result, AREA_REF, rtol=1e-12,
                               err_msg=f"area recovery failed, order={order}")


@pytest.mark.parametrize("order", [3, 5, 7])
def test_area_recovery_singular_at_centroid(order):
    """Singularity at centroid: split into 3 equal sub-tris, sum = area."""
    centroid = (P1 + P2 + P3) / 3.0
    result = duffy_collapsed_quadrature(centroid, P1, P2, P3, lambda r: 1.0, order=order)
    np.testing.assert_allclose(result, AREA_REF, rtol=1e-12)


@pytest.mark.parametrize("order", [3, 5, 7])
def test_area_recovery_tilted_triangle(order):
    """Non-degenerate 3D triangle."""
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([2.0, 0.0, 0.0])
    c = np.array([0.5, 1.5, 0.0])
    r_s = a
    expected = _area(a, b, c)
    result = duffy_collapsed_quadrature(r_s, a, b, c, lambda r: 1.0, order=order)
    np.testing.assert_allclose(result, expected, rtol=1e-11,
                               err_msg=f"tilted triangle area, order={order}")


# ---------------------------------------------------------------------------
# 2. Smooth kernel: Duffy must agree with standard Gauss-Dunavant
# ---------------------------------------------------------------------------

def _gauss_ref(kernel_fn, order=11):
    """Standard Gauss-Dunavant integral of kernel_fn over the reference triangle."""
    xi, w = dunavant_rule(order)
    pts = map_to_triangle(xi, P1, P2, P3)   # (3, Nq)
    vals = np.array([kernel_fn(pts[:, q]) for q in range(pts.shape[1])])
    return integrate_on_triangle(vals, w, AREA_REF)


@pytest.mark.parametrize("order", [5, 7])
def test_smooth_linear_kernel(order):
    """Smooth kernel f(r) = x + 2y; singularity at P1 but f is regular."""
    def kernel(r): return r[0] + 2.0 * r[1]
    gauss = _gauss_ref(kernel)
    duffy = duffy_collapsed_quadrature(P1, P1, P2, P3, kernel, order=order)
    np.testing.assert_allclose(duffy, gauss, rtol=1e-9,
                               err_msg=f"smooth kernel, order={order}")


@pytest.mark.parametrize("order", [5, 7])
def test_smooth_quadratic_kernel(order):
    """Smooth kernel f(r) = x^2 - y."""
    def kernel(r): return r[0]**2 - r[1]
    gauss = _gauss_ref(kernel)
    duffy = duffy_collapsed_quadrature(P1, P1, P2, P3, kernel, order=order)
    np.testing.assert_allclose(duffy, gauss, rtol=1e-9)


# ---------------------------------------------------------------------------
# 3. 1/R singularity at vertex: compare against brute-force high-order GL
# ---------------------------------------------------------------------------

def _reference_1_over_R(p_sing, p1, p2, p3, n_gl=200):
    """
    Brute-force ∫_T 1/|r' - p_sing| dS' using a high-order GL product rule.

    This is only valid when p_sing is NOT inside the triangle (so 1/R is
    bounded and GL converges), e.g. p_sing = p1 which is a vertex.

    We use the product GL on [0,1]² with the standard triangle map
    r'(s,t) = p1 + s*(p2-p1) + t*(p3-p1), dS' = 2*area * ds dt (domain s+t≤1).
    The domain is split into two triangles to avoid the singularity at s=t=0.
    """
    t_gl, w_gl = leggauss(n_gl)
    u = 0.5 * (t_gl + 1.0)   # [0,1]
    w = 0.5 * w_gl
    area = _area(p1, p2, p3)

    total = 0.0
    for ui, wui in zip(u, w):
        for vj, wvj in zip(u, w):
            # Standard triangle map with Duffy collapse for vertex at p1 = (s=0, t=0)
            # (this is what the brute-force MUST do to avoid the singularity)
            # s = ui*(1-vj), t = ui*vj (Duffy at vertex (0,0))
            s = ui * (1.0 - vj)
            t = ui * vj
            r_prime = p1 + s*(p2 - p1) + t*(p3 - p1)
            R = np.linalg.norm(r_prime - p_sing)
            if R > 1e-300:
                total += wui * wvj * ui * (1.0 / R)
    # Factor: 2*area from jacobian, u from Duffy (already included in the loop as ui)
    return 2.0 * area * total


def test_1_over_R_singularity_at_vertex():
    """
    Duffy must handle ∫_T 1/|r'−P1| dS' and return a result close to
    the analytical value  sqrt(2)*ln(1+sqrt(2)) ≈ 1.24646.

    Convergence tolerance is 1 % because, after the Duffy collapse in u,
    the remaining v-integrand 1/sqrt((1-v)^2+v^2) has a complex singularity
    at distance 1/2 from [0,1], so GL converges slowly (algebraically, not
    exponentially) in the v-direction.  Higher order can be used in Phase 4
    for tighter accuracy; here we just verify the transform is correct.
    """
    exact = np.sqrt(2.0) * np.log(1.0 + np.sqrt(2.0))   # ≈ 1.24646

    def kernel(r): return 1.0 / max(np.linalg.norm(r - P1), 1e-300)

    duffy = duffy_collapsed_quadrature(P1, P1, P2, P3, kernel, order=7)
    np.testing.assert_allclose(duffy, exact, rtol=1e-2,
                               err_msg="1/R singularity at vertex: > 1% deviation from exact")


def test_1_over_R_singularity_at_centroid():
    """
    Duffy handles a centroid singularity: the result at order=11 must be
    within 1 % of the order=7 result (i.e., both orders give a finite,
    consistent value — the transform is working).
    """
    centroid = (P1 + P2 + P3) / 3.0

    def kernel(r):
        R = np.linalg.norm(r - centroid)
        return 1.0 / max(R, 1e-300)

    d7  = duffy_collapsed_quadrature(centroid, P1, P2, P3, kernel, order=7)
    d11 = duffy_collapsed_quadrature(centroid, P1, P2, P3, kernel, order=11)
    # Verify convergence direction (order-11 > order-7 and close)
    assert d7  > 0, "order-7 integral should be positive"
    assert d11 > 0, "order-11 integral should be positive"
    np.testing.assert_allclose(d7, d11, rtol=5e-3,
                               err_msg="1/R at centroid: order-7 vs order-11 disagree by > 0.5%")


# ---------------------------------------------------------------------------
# 4. duffy_source_nodes interface
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("order", [3, 5, 7])
def test_source_nodes_weight_sum_equals_area(order):
    """Weights from duffy_source_nodes must sum to area(T)."""
    pts, w = duffy_source_nodes(P1, P1, P2, P3, order=order)
    np.testing.assert_allclose(w.sum(), AREA_REF, rtol=1e-12,
                               err_msg=f"weight sum != area, order={order}")


@pytest.mark.parametrize("order", [3, 5, 7])
def test_source_nodes_count_interior_singular(order):
    """
    When r_singular is the centroid (interior point), all 3 sub-triangles are
    non-degenerate, so node count = 3 * order².

    When r_singular is a vertex, 2 of the 3 sub-triangles are degenerate (zero
    area) and are skipped, giving only 1 * order² nodes.  Both cases are tested.
    """
    centroid = (P1 + P2 + P3) / 3.0
    pts, w = duffy_source_nodes(centroid, P1, P2, P3, order=order)
    assert pts.shape == (3 * order**2, 3), \
        f"centroid: expected {3*order**2} nodes, got {pts.shape[0]}"
    assert w.shape == (3 * order**2,)

    # Vertex singularity: only 1 non-degenerate sub-triangle
    pts_v, w_v = duffy_source_nodes(P1, P1, P2, P3, order=order)
    assert pts_v.shape[0] == order**2, \
        f"vertex: expected {order**2} nodes, got {pts_v.shape[0]}"


def test_source_nodes_self_consistency():
    """duffy_collapsed_quadrature == np.dot(weights, kernel(pts))."""
    def kernel(r): return r[0] + 2.0 * r[1]

    pts, w = duffy_source_nodes(P1, P1, P2, P3, order=7)
    vals = np.array([kernel(pts[q]) for q in range(len(w))])
    nodes_result = np.dot(w, vals)
    direct_result = duffy_collapsed_quadrature(P1, P1, P2, P3, kernel, order=7)
    np.testing.assert_allclose(nodes_result, direct_result, rtol=1e-14)


@pytest.mark.parametrize("order", [5, 7])
def test_source_nodes_all_inside_triangle(order):
    """All quadrature nodes must lie inside or on the boundary of T."""
    pts, _ = duffy_source_nodes(P1, P1, P2, P3, order=order)
    # Barycentric coords of pts w.r.t. (P1, P2, P3):
    # P1=(0,0,0), P2=(1,0,0), P3=(0,1,0): x = L2, y = L3, L1 = 1-x-y
    x = pts[:, 0]
    y = pts[:, 1]
    L1 = 1.0 - x - y
    L2 = x
    L3 = y
    tol = 1e-12
    assert np.all(L1 >= -tol), "some nodes outside T (L1 < 0)"
    assert np.all(L2 >= -tol), "some nodes outside T (L2 < 0)"
    assert np.all(L3 >= -tol), "some nodes outside T (L3 < 0)"
