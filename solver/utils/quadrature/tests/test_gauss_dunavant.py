"""
Unit tests for gauss_dunavant.py.

For each implemented rule of degree d, every monomial x^a y^b with a+b <= d
is integrated over the reference triangle and compared to the closed-form
exact value:

    integral_{ref tri} x^a y^b dx dy = a! * b! / (a+b+2)!

This simultaneously verifies (a) the barycentric coordinates, (b) the weights,
and (c) the degree claim.

Run with:
    pytest solver/utils/quadrature/tests/test_gauss_dunavant.py -v
"""

import math
import pytest
import numpy as np

from solver.utils.quadrature.gauss_dunavant import (
    dunavant_rule, map_to_triangle, triangle_jacobian, integrate_on_triangle
)


# ---------------------------------------------------------------------------
# Exact monomial integral on the reference triangle (0,0)-(1,0)-(0,1)
# ---------------------------------------------------------------------------

def _exact_monomial_ref(a: int, b: int) -> float:
    """Integral of x^a y^b over the reference triangle (area 0.5)."""
    return math.factorial(a) * math.factorial(b) / math.factorial(a + b + 2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Reference triangle vertices (in 3D, z=0 for simplicity)
_P1 = np.array([0.0, 0.0, 0.0])
_P2 = np.array([1.0, 0.0, 0.0])
_P3 = np.array([0.0, 1.0, 0.0])


def _integrate_monomial(order, a, b):
    """Integrate x^a * y^b over the reference triangle using Dunavant rule."""
    xi, w = dunavant_rule(order)
    pts = map_to_triangle(xi, _P1, _P2, _P3)  # (3, Nq)
    x = pts[0]
    y = pts[1]
    f = x**a * y**b
    # integral_T f dS = 2*area * sum_q w_q * f_q  (area of ref tri = 0.5)
    return 2.0 * 0.5 * np.dot(w, f)   # = sum_q w_q * f_q


# ---------------------------------------------------------------------------
# Parametrised tests: one test per (order, monomial)
# ---------------------------------------------------------------------------

def _monomial_cases(order):
    """All monomials x^a y^b with a+b <= order."""
    return [(a, b) for a in range(order + 1) for b in range(order + 1 - a)]


@pytest.mark.parametrize("a,b", _monomial_cases(1))
def test_degree1_monomials(a, b):
    got   = _integrate_monomial(1, a, b)
    exact = _exact_monomial_ref(a, b)
    np.testing.assert_allclose(got, exact, atol=1e-12,
        err_msg=f"order=1, x^{a}*y^{b}: got {got}, exact {exact}")


@pytest.mark.parametrize("a,b", _monomial_cases(3))
def test_degree3_monomials(a, b):
    got   = _integrate_monomial(3, a, b)
    exact = _exact_monomial_ref(a, b)
    np.testing.assert_allclose(got, exact, atol=1e-12,
        err_msg=f"order=3, x^{a}*y^{b}: got {got}, exact {exact}")


@pytest.mark.parametrize("a,b", _monomial_cases(4))
def test_degree4_monomials(a, b):
    got   = _integrate_monomial(4, a, b)
    exact = _exact_monomial_ref(a, b)
    np.testing.assert_allclose(got, exact, atol=1e-12,
        err_msg=f"order=4, x^{a}*y^{b}: got {got}, exact {exact}")


@pytest.mark.parametrize("a,b", _monomial_cases(5))
def test_degree5_monomials(a, b):
    got   = _integrate_monomial(5, a, b)
    exact = _exact_monomial_ref(a, b)
    np.testing.assert_allclose(got, exact, atol=1e-12,
        err_msg=f"order=5, x^{a}*y^{b}: got {got}, exact {exact}")


@pytest.mark.parametrize("a,b", _monomial_cases(6))
def test_degree6_monomials(a, b):
    got   = _integrate_monomial(6, a, b)
    exact = _exact_monomial_ref(a, b)
    np.testing.assert_allclose(got, exact, atol=1e-12,
        err_msg=f"order=6, x^{a}*y^{b}: got {got}, exact {exact}")


@pytest.mark.parametrize("a,b", _monomial_cases(7))
def test_degree7_monomials(a, b):
    got   = _integrate_monomial(7, a, b)
    exact = _exact_monomial_ref(a, b)
    np.testing.assert_allclose(got, exact, atol=1e-12,
        err_msg=f"order=7, x^{a}*y^{b}: got {got}, exact {exact}")


@pytest.mark.parametrize("a,b", _monomial_cases(11))
def test_degree11_monomials(a, b):
    got   = _integrate_monomial(11, a, b)
    exact = _exact_monomial_ref(a, b)
    np.testing.assert_allclose(got, exact, atol=1e-12,
        err_msg=f"order=11, x^{a}*y^{b}: got {got}, exact {exact}")


# ---------------------------------------------------------------------------
# Weights sum to 0.5 (area of reference triangle)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("order", [1, 3, 4, 5, 6, 7, 11])
def test_weights_sum_to_half(order):
    _, w = dunavant_rule(order)
    np.testing.assert_allclose(w.sum(), 0.5, atol=1e-14,
        err_msg=f"order={order}: weights sum to {w.sum()}, expected 0.5")


# ---------------------------------------------------------------------------
# Barycentric rows sum to 1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("order", [1, 3, 4, 5, 6, 7, 11])
def test_barycentric_rows_sum_to_one(order):
    xi, _ = dunavant_rule(order)
    np.testing.assert_allclose(xi.sum(axis=1), 1.0, atol=1e-14,
        err_msg=f"order={order}: barycentric rows don't sum to 1")


# ---------------------------------------------------------------------------
# Barycentric coordinates are non-negative (points inside the triangle)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("order", [1, 4, 5, 6, 7, 11])
def test_barycentric_nonnegative(order):
    xi, _ = dunavant_rule(order)
    assert np.all(xi >= -1e-14), \
        f"order={order}: some barycentric coords < 0 (outside triangle)"


# ---------------------------------------------------------------------------
# map_to_triangle: integrate constant 1 gives physical triangle area
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("order", [1, 5, 7])
def test_area_recovery_physical_triangle(order):
    xi, w = dunavant_rule(order)

    triangles = [
        # (p1, p2, p3, expected_area)
        (np.array([0.,0.,0.]), np.array([1.,0.,0.]), np.array([0.,1.,0.]), 0.5),
        (np.array([0.,0.,0.]), np.array([2.,0.,0.]), np.array([0.,3.,0.]), 3.0),
        (np.array([1.,1.,1.]), np.array([2.,1.,1.]), np.array([1.,2.,1.]), 0.5),
        (np.array([0.,0.,0.]), np.array([1.,1.,0.]), np.array([0.,1.,1.]), np.sqrt(3)/4*2),
    ]

    for p1, p2, p3, expected_area in triangles:
        pts = map_to_triangle(xi, p1, p2, p3)   # (3, Nq)
        jac = triangle_jacobian(p1, p2, p3)       # = 2 * area
        computed_area = jac * np.sum(w)            # 2*area * sum(w) = 2*area*0.5 = area
        np.testing.assert_allclose(
            computed_area, expected_area, rtol=1e-13,
            err_msg=f"order={order}: area of triangle mismatch"
        )


# ---------------------------------------------------------------------------
# integrate_on_triangle convenience wrapper
# ---------------------------------------------------------------------------

def test_integrate_on_triangle_constant():
    xi, w = dunavant_rule(5)
    p1, p2, p3 = _P1, _P2, _P3
    pts = map_to_triangle(xi, p1, p2, p3)
    f = np.ones(len(w))
    result = integrate_on_triangle(f, w, 0.5)
    np.testing.assert_allclose(result, 0.5, rtol=1e-13)


def test_integrate_on_triangle_linear():
    xi, w = dunavant_rule(5)
    p1 = np.array([0., 0., 0.])
    p2 = np.array([2., 0., 0.])
    p3 = np.array([0., 2., 0.])
    area = 2.0
    pts = map_to_triangle(xi, p1, p2, p3)
    f = pts[0]   # f(x,y) = x; exact integral = integral_T x dS = area * x_centroid = 2*(2/3) = 4/3
    result = integrate_on_triangle(f, w, area)
    np.testing.assert_allclose(result, 4.0 / 3.0, rtol=1e-12)


# ---------------------------------------------------------------------------
# Unsupported order raises ValueError
# ---------------------------------------------------------------------------

def test_unsupported_order_raises():
    with pytest.raises(ValueError, match="Unsupported Dunavant order"):
        dunavant_rule(2)
