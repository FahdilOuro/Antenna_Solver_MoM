
"""
Collapsed-coordinate Duffy transform for near-singular triangle integrals.

Background
----------
The RWG EFIE impedance matrix requires double surface integrals of the form

    Z_mn = ∫∫ f_m(r) · G(r,r') · f_n(r') dS' dS

where G(r,r') = exp(-jk|r-r'|)/|r-r'|.  When the test and source triangles
share geometry (self/edge/vertex pairs), |r-r'| → 0 and the naive Gauss rule
blows up.  Duffy's collapsed-coordinate substitution in the SOURCE triangle
removes the singularity analytically so that the transformed integrand is
regular and standard GL quadrature converges.

Algorithm (vertex Duffy)
------------------------
For a sub-triangle (q0, q1, q2) with the singularity at vertex q0:

  Parametrization:  r'(s,t) = q0 + s(q1-q0) + t(q2-q0),  0≤s, 0≤t, s+t≤1
  Duffy collapse:   u = s+t,  v = t/(s+t)
                    → s = u(1-v),  t = uv,  Jacobian = u

  Area element:  dS' = 2·area(q0,q1,q2)·u du dv  (u∈[0,1], v∈[0,1])

  Since |r' - q0| = u·|( 1-v)(q1-q0) + v(q2-q0)| = O(u), the u in the
  Jacobian exactly cancels the 1/R singularity and the integrand is regular.

General case
------------
When r_singular is not a vertex but lies inside (or on an edge of) the
triangle, split T into three sub-triangles sharing the singular point as a
common vertex, apply the vertex Duffy to each, and sum.

Public API
----------
duffy_collapsed_quadrature(r_singular, p1, p2, p3, kernel_fn, order=7)
    → float : ∫_T kernel_fn(r') dS', singularity handled at r_singular

duffy_source_nodes(r_singular, p1, p2, p3, order=7)
    → (pts, weights) : Duffy-weighted nodes for the outer-loop use case in
      Phase 4, where the caller evaluates kernel_fn and applies the sum manually.
"""

import numpy as np
from numpy.polynomial.legendre import leggauss


# ---------------------------------------------------------------------------
# Internal GL cache
# ---------------------------------------------------------------------------

_gl_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _gl_on_01(n: int) -> tuple[np.ndarray, np.ndarray]:
    """n-point Gauss-Legendre nodes and weights on [0, 1]."""
    if n not in _gl_cache:
        t, w = leggauss(n)
        _gl_cache[n] = (0.5 * (t + 1.0), 0.5 * w)
    return _gl_cache[n]


# ---------------------------------------------------------------------------
# Core: one sub-triangle with singularity at vertex 0
# ---------------------------------------------------------------------------

def _duffy_vertex_subtriangle(
    q0: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    kernel_fn,
    u: np.ndarray,
    wu: np.ndarray,
    v: np.ndarray,
    wv: np.ndarray,
) -> float:
    """
    Evaluate ∫_{T(q0,q1,q2)} kernel_fn(r') dS' with the singularity at q0.

    The collapsed-coordinate map is:
        r'(u,v) = q0 + u*(q1-q0) + u*v*(q2-q1)
    with Jacobian u and dS' = 2*area * u du dv.

    Parameters
    ----------
    q0, q1, q2 : (3,) arrays — vertices; singularity is at q0
    kernel_fn  : callable (3,) → complex or float
    u, wu      : GL nodes and weights on [0,1] (outer; carries the Jacobian factor)
    v, wv      : GL nodes and weights on [0,1] (inner)

    Returns
    -------
    float or complex : contribution from this sub-triangle
    """
    e1 = q1 - q0          # (3,)
    e2 = q2 - q1          # (3,)
    sub_area = 0.5 * np.linalg.norm(np.cross(q1 - q0, q2 - q0))

    total = type(kernel_fn(q0))(0)   # scalar with matching dtype (float or complex)

    for ui, wui in zip(u, wu):
        for vj, wvj in zip(v, wv):
            r_prime = q0 + ui * e1 + ui * vj * e2
            # Jacobian ui cancels 1/R; weight is wu*wv*u (area factor added below)
            total += wui * wvj * ui * kernel_fn(r_prime)

    return 2.0 * sub_area * total


# ---------------------------------------------------------------------------
# Public: evaluate the integral (evaluate kernel_fn internally)
# ---------------------------------------------------------------------------

def duffy_collapsed_quadrature(
    r_singular: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    kernel_fn,
    order: int = 7,
) -> float:
    """
    Evaluate  ∫_{T} kernel_fn(r') dS'  with the collpased Duffy transform.

    The singularity of kernel_fn at r_singular (inside or on the triangle T)
    is removed analytically.  Standard Gauss-Legendre of order `order` is
    applied to the regularised integrand on [0,1]² for each of the three
    sub-triangles (r_singular, p_i, p_{i+1}).

    Parameters
    ----------
    r_singular : (3,) float  physical coordinates of the singular point;
                 must lie inside T or on its boundary.
    p1, p2, p3 : (3,) float  triangle vertices
    kernel_fn  : callable (3,) → complex or float;
                 may have a 1/|r' - r_singular| singularity.
    order      : int  Gauss-Legendre order per dimension (default 7, exact for
                 regular integrands of degree ≤ 2*order-1).

    Returns
    -------
    complex or float : approximation of ∫_T kernel_fn(r') dS'
    """
    r_s = np.asarray(r_singular, dtype=np.float64)
    q1  = np.asarray(p1, dtype=np.float64)
    q2  = np.asarray(p2, dtype=np.float64)
    q3  = np.asarray(p3, dtype=np.float64)

    u, wu = _gl_on_01(order)
    v, wv = _gl_on_01(order)

    # Three sub-triangles, each with the singular vertex at r_s
    total = _duffy_vertex_subtriangle(r_s, q1, q2, kernel_fn, u, wu, v, wv)
    total += _duffy_vertex_subtriangle(r_s, q2, q3, kernel_fn, u, wu, v, wv)
    total += _duffy_vertex_subtriangle(r_s, q3, q1, kernel_fn, u, wu, v, wv)
    return total


# ---------------------------------------------------------------------------
# Public: provide nodes + weights for the outer-loop use case (Phase 4)
# ---------------------------------------------------------------------------

def duffy_source_nodes(
    r_singular: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    order: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return physical quadrature nodes and combined Duffy weights for the source
    triangle, with the singularity at r_singular.

    The caller evaluates the (possibly singular) kernel at the returned nodes
    and sums  result = np.dot(weights, kernel_values).

    Parameters
    ----------
    r_singular : (3,) singular point (vertex of each sub-triangle)
    p1, p2, p3 : (3,) triangle vertices
    order      : int  GL order per dimension

    Returns
    -------
    pts     : (Nq, 3) float64   physical coordinates of quadrature points
    weights : (Nq,)  float64    combined weights (area factor absorbed);
              sum over weights = ∫_T 1 dS' = area(p1,p2,p3)
    """
    r_s = np.asarray(r_singular, dtype=np.float64)
    q1  = np.asarray(p1, dtype=np.float64)
    q2  = np.asarray(p2, dtype=np.float64)
    q3  = np.asarray(p3, dtype=np.float64)

    u, wu = _gl_on_01(order)
    v, wv = _gl_on_01(order)

    # Precompute outer product of u-weights × v-weights × u-nodes
    # combined_w[i,j] = wu[i] * wv[j] * u[i]
    UU, VV = np.meshgrid(u, v, indexing='ij')    # (n, n)
    WU, WV = np.meshgrid(wu, wv, indexing='ij')
    combined_w = WU * WV * UU                      # (n, n)

    sub_tris = [
        (r_s, q1, q2),
        (r_s, q2, q3),
        (r_s, q3, q1),
    ]

    all_pts = []
    all_w   = []

    for (s0, s1, s2) in sub_tris:
        sub_area = 0.5 * np.linalg.norm(np.cross(s1 - s0, s2 - s0))
        if sub_area < 1e-300:    # degenerate sub-triangle (r_s on an edge)
            continue
        e1 = s1 - s0
        e2 = s2 - s1

        # r'(u,v) = s0 + u*e1 + u*v*e2  shape: (3, n, n)
        pts_sub = (s0[:, None, None]
                   + UU[None, :, :] * e1[:, None, None]
                   + UU[None, :, :] * VV[None, :, :] * e2[:, None, None])  # (3, n, n)

        pts_sub = pts_sub.reshape(3, -1).T    # (n^2, 3)
        w_sub   = (2.0 * sub_area * combined_w).ravel()   # (n^2,)

        all_pts.append(pts_sub)
        all_w.append(w_sub)

    if not all_pts:
        return np.zeros((0, 3)), np.zeros(0)

    return np.vstack(all_pts), np.concatenate(all_w)
