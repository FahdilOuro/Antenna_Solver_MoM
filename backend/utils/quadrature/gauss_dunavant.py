"""
Symmetric Gauss-Dunavant quadrature rules on the reference triangle.

Reference
---------
Dunavant, D.A. (1985). "High degree efficient symmetrical Gaussian quadrature
rules for the triangle." International Journal for Numerical Methods in
Engineering, 21(6):1129-1148.

All rules are given in *barycentric coordinates* (L1, L2, L3) with L1+L2+L3=1.
Weights are normalized so that sum(w) = 0.5 (the area of the reference
triangle with vertices (0,0), (1,0), (0,1)).

That is, for a function f defined on the *reference* triangle T_ref:
    integral_T_ref f dA  ≈  sum_q w_q * f(r_q)

For a *physical* triangle T with area A:
    integral_T f dS  ≈  2*A * sum_q w_q * f(r_q)     [see map_to_triangle]

or equivalently: sum_q (w_q * 2 * A) * f(r_q).

Supported polynomial degrees: 1, 3, 4, 5, 6, 7, 11, 16, 20, 24.
The degree-d rule integrates every polynomial of total degree <= d exactly.

Degrees 7 and above are Gauss-Legendre PRODUCT rules built through the Duffy
map (see _rule_product_gl), not symmetric Dunavant orbits: published symmetric
coefficients above degree 6 cannot be transcribed to full double precision, and
a rule that is only 5e-7-exact is useless for a convergence study. The product
rules are generated from numpy.polynomial.legendre.leggauss, so their only
error is roundoff. The price is that their nodes are not invariant under
permutation of the triangle's vertices; that costs nothing here, because every
use integrates a function that has no such symmetry either.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Internal helper: expand symmetry orbits into (Nq, 3) barycentric arrays
# ---------------------------------------------------------------------------

def _s3_orbit(a):
    """Single centroid-like point: (a,a,a) with a=1/3."""
    return np.array([[a, a, a]])


def _s21_orbit(a):
    """Orbit with one repeated coordinate: 3 permutations of (a, a, b=1-2a)."""
    b = 1.0 - 2.0 * a
    return np.array([
        [a, a, b],
        [a, b, a],
        [b, a, a],
    ])


def _s111_orbit(a, b):
    """Fully asymmetric orbit: 6 permutations of (a, b, c=1-a-b)."""
    c = 1.0 - a - b
    return np.array([
        [a, b, c], [a, c, b],
        [b, a, c], [b, c, a],
        [c, a, b], [c, b, a],
    ])


# ---------------------------------------------------------------------------
# Rule tables
# ---------------------------------------------------------------------------

def _rule_degree1():
    """1-point, degree-1-exact (centroid rule). Exact for linears."""
    bary = _s3_orbit(1.0 / 3.0)
    w    = np.array([0.5])
    return bary, w


def _rule_degree3():
    """4-point, degree-3-exact rule.
    Dunavant (1985), Table III, degree 3.
    """
    a1 = 0.6 / 2.0   # = 0.3
    # Actually using the standard 4-point degree-3 rule:
    a1 = 1.0 / 3.0
    a2 = 0.2
    b2 = 0.6
    # Orbits:
    #  orbit 1 (S3 centroid): weight w1
    #  orbit 2 (S21 with a=0.2): weight w2
    w1 = -0.5625 / 2.0      # = -0.28125  (negative weight is correct)
    w2 =  0.520833333333333 / 2.0

    bary = np.vstack([
        _s3_orbit(a1),
        _s21_orbit(a2),
    ])
    w = np.concatenate([
        [w1],
        [w2, w2, w2],
    ])
    return bary, w


def _rule_degree4():
    """6-point, degree-4-exact rule.
    Dunavant (1985), Table IV.  Two S21 orbits.
    Weights are per-point and sum to 1 for unit-area triangle;
    dividing by 2 adapts them to our reference triangle (area = 0.5).

    Orbit 1: a = 0.445948490915965, w/point = 0.223381589678011
    Orbit 2: a = 0.091576213509771, w/point = 0.109951743655322
    Check:  3*0.22338 + 3*0.10995 = 0.67014 + 0.32986 = 1.000 ✓
    """
    a1 = 0.445948490915965
    w1 = 0.223381589678011 / 2.0
    a2 = 0.091576213509771
    w2 = 0.109951743655322 / 2.0

    bary = np.vstack([
        _s21_orbit(a1),
        _s21_orbit(a2),
    ])
    w = np.concatenate([
        [w1, w1, w1],
        [w2, w2, w2],
    ])
    return bary, w


def _rule_degree6():
    """12-point, degree-6-exact rule.
    Dunavant (1985), Table VI.
    """
    # Orbit 1 (S21): a = 0.063089014491502
    a1 = 0.063089014491502
    w1 = 0.050844906370207 / 2.0
    # Orbit 2 (S21): a = 0.249286745170910
    a2 = 0.249286745170910
    w2 = 0.116786275726379 / 2.0
    # Orbit 3 (S111): a = 0.310352451033785, b = 0.053145049844816
    a3 = 0.310352451033785
    b3 = 0.053145049844816
    w3 = 0.082851075618374 / 2.0

    bary = np.vstack([
        _s21_orbit(a1),
        _s21_orbit(a2),
        _s111_orbit(a3, b3),
    ])
    w = np.concatenate([
        [w1]*3, [w2]*3, [w3]*6,
    ])
    return bary, w


def _rule_degree5():
    """7-point, degree-5-exact rule (Dunavant 1985, Table V).

    This is the workhorse rule used in Phase 0 diagnostics and as the
    default for non-singular far-pair quadrature.
    """
    a1 = 0.47014206410511508
    a2 = 0.10128650732345633

    w1 = 0.22500000000000000 / 2.0
    w2 = 0.13239415278850616 / 2.0
    w3 = 0.12593918054482717 / 2.0

    bary = np.vstack([
        _s3_orbit(1.0 / 3.0),
        _s21_orbit(a1),
        _s21_orbit(a2),
    ])
    w = np.concatenate([
        [w1],
        [w2, w2, w2],
        [w3, w3, w3],
    ])
    return bary, w


def _rule_product_gl(n):
    """
    n²-point Gauss-Legendre product rule on the reference triangle,
    computed via the Duffy transformation.

    The Duffy map (u,v) ∈ [0,1]² → (x,y) = (u, v*(1-u)) maps the unit
    square onto the reference triangle with Jacobian (1-u).  GL with n
    points on each factor is exact for all monomials x^a y^b with
    a+b ≤ 2n-2 on the triangle.

    Weights are computed analytically from numpy.polynomial.legendre.leggauss
    and are exact to machine precision — no coefficient transcription needed.

    Parameters
    ----------
    n : int
        Number of GL points per dimension.  Degree of exactness = 2n-2.

    Returns
    -------
    bary : (n^2, 3) float64   barycentric coordinates, all in (0,1), rows sum to 1
    w    : (n^2,)  float64    weights, sum to 0.5
    """
    from numpy.polynomial.legendre import leggauss
    t, w1d = leggauss(n)   # n-point GL on [-1,1]
    u = 0.5 * (t + 1.0)    # shift to (0,1)
    wu = 0.5 * w1d          # weights on [0,1]

    UU, VV = np.meshgrid(u, u, indexing='ij')    # (n, n)
    WU, WV = np.meshgrid(wu, wu, indexing='ij')

    # Duffy map: x = u,  y = v*(1-u)
    x  = UU
    y  = VV * (1.0 - UU)
    jac = (1.0 - UU)   # Duffy Jacobian

    # Barycentric: L1 = (1-u)(1-v), L2 = u, L3 = v(1-u)
    L2 = x
    L3 = y
    L1 = 1.0 - L2 - L3

    bary  = np.stack([L1.ravel(), L2.ravel(), L3.ravel()], axis=1)
    w_out = (WU * WV * jac).ravel()
    return bary, w_out


def _rule_degree7():
    """25-point, degree-7-exact rule via Duffy-GL product (n=5).

    Symmetric Dunavant coefficient tables for degree 7 suffer from
    limited-precision published coefficients (~5e-7 monomial error).
    This product rule is computed from numpy.polynomial.legendre.leggauss,
    which is exact to machine precision, so all degree-7 monomials are
    integrated with error < 1e-15.
    Points: 25 (= 5²).  Degree of exactness = 2*5-2 = 8 ≥ 7.
    """
    return _rule_product_gl(5)


def _rule_degree11():
    """49-point, degree-11-exact rule via Duffy-GL product (n=7).

    Degree-11 Dunavant symmetric rules (28 points) require coefficients
    that are difficult to transcribe to full machine precision. The 7×7
    product GL rule is exact for degree ≤ 12 and has no transcription risk.
    Points: 49 (= 7²).  Degree of exactness = 2*7-2 = 12 ≥ 11.
    """
    return _rule_product_gl(7)


# --- Degrees above 11 -------------------------------------------------------
#
# Degree 11 used to be the top of this table, and impedance_matrix_z's near
# (EDGE/VERTEX) rule is min(quad_order + 5, 11) -- i.e. already pinned at the
# ceiling for every default call. Any convergence study of the OUTER order for
# the singular branches therefore had nowhere to go, which makes a truncation
# error indistinguishable from a stagnating one. These three rules exist to
# lift that ceiling so the distinction can actually be measured. See
# backend/utils/quadrature/outer_order_study.py.

def _rule_degree16():
    """81-point, degree-16-exact rule via Duffy-GL product (n=9)."""
    return _rule_product_gl(9)


def _rule_degree20():
    """121-point, degree-20-exact rule via Duffy-GL product (n=11)."""
    return _rule_product_gl(11)


def _rule_degree24():
    """169-point, degree-24-exact rule via Duffy-GL product (n=13)."""
    return _rule_product_gl(13)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_RULES = {
    1:  _rule_degree1,
    3:  _rule_degree3,
    4:  _rule_degree4,
    5:  _rule_degree5,
    6:  _rule_degree6,
    7:  _rule_degree7,
    11: _rule_degree11,
    16: _rule_degree16,
    20: _rule_degree20,
    24: _rule_degree24,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def dunavant_rule(order: int) -> tuple:
    """
    Return a degree-`order`-exact symmetric quadrature rule on the
    unit reference triangle.

    Parameters
    ----------
    order : int
        Polynomial degree that the rule integrates exactly.
        Supported: 1, 3, 4, 5, 6, 7, 11, 16, 20, 24
        (point counts 1, 4, 6, 7, 12, 25, 49, 81, 121, 169).

    Returns
    -------
    xi : ndarray, shape (Nq, 3)
        Barycentric coordinates; each row sums to 1.
    w  : ndarray, shape (Nq,)
        Quadrature weights; sum to 0.5 (area of reference triangle).
    """
    if order not in _RULES:
        raise ValueError(
            f"Unsupported Dunavant order {order}. Available: {sorted(_RULES)}"
        )
    xi, w = _RULES[order]()
    xi = np.asarray(xi, dtype=np.float64)
    w  = np.asarray(w,  dtype=np.float64)
    return xi, w


def map_to_triangle(xi: np.ndarray, p1, p2, p3) -> np.ndarray:
    """
    Map barycentric quadrature points to physical coordinates.

    Parameters
    ----------
    xi : ndarray, shape (Nq, 3)
        Barycentric coordinates (rows sum to 1).
    p1, p2, p3 : array-like, shape (3,)
        Vertices of the physical triangle in 3D.

    Returns
    -------
    r : ndarray, shape (3, Nq)
        Physical coordinates of each quadrature point.
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)
    xi = np.asarray(xi, dtype=np.float64)
    # p_i[:,None] is (3,1), xi[:,i] is (Nq,) → product is (3,Nq)
    return p1[:, None] * xi[:, 0] + p2[:, None] * xi[:, 1] + p3[:, None] * xi[:, 2]


def triangle_jacobian(p1, p2, p3) -> float:
    """
    Return 2 * area(T) = |det J| of the barycentric-to-physical map.

    The physical integration weight for quadrature point q is:
        w_phys_q = w_q * triangle_jacobian(p1, p2, p3)

    so that  sum_q w_phys_q * f(r_q)  ≈  integral_T f dS.

    Parameters
    ----------
    p1, p2, p3 : array-like, shape (3,)
        Triangle vertices.

    Returns
    -------
    jac : float
        Twice the triangle area.
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)
    return float(np.linalg.norm(np.cross(p2 - p1, p3 - p1)))


def integrate_on_triangle(f_vals: np.ndarray, w: np.ndarray, area: float) -> float:
    """
    Compute integral_T f dS ≈ 2*area * sum_q w_q * f(r_q).

    Parameters
    ----------
    f_vals : ndarray, shape (Nq,) or (Nq, ...)
        Function values at quadrature points.
    w : ndarray, shape (Nq,)
        Quadrature weights (must sum to 0.5).
    area : float
        Physical triangle area.

    Returns
    -------
    Approximated integral.
    """
    return 2.0 * area * np.einsum('q,...q->...', w, f_vals.T if f_vals.ndim > 1 else f_vals)
