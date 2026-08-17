"""
Closed-form surface integral of the static kernel 1/R over a planar triangle.

Used to remove the 1/R singularity from the EFIE assembly analytically instead
of asking a Gauss rule to resolve it. For a triangle T and an observation point
r0 lying in T's plane (which is the case for every planar-mesh pair, and for
the in-plane component in general),

    int_T dS' / |r0 - r'|  =  sum over the three edges of
                              P0_i * ln( (R_i^+ + l_i^+) / (R_i^- + l_i^-) )

with, for an edge running A -> B (triangle traversed consistently):

    e     = (B - A) / |B - A|          unit vector along the edge
    u     = e x n                      in-plane normal to that edge
    P0    = (A - r0) . u               SIGNED distance from r0 to the edge line
    l^-   = (A - r0) . e               signed position of A along the edge
    l^+   = (B - r0) . e               signed position of B along the edge
    R^+-  = |B - r0| , |A - r0|

Because P0 is signed, the same expression is valid whether r0 lies inside the
triangle or outside it -- which is exactly what near-singular EDGE/VERTEX pairs
need, and what a Duffy-style subdivision from r0 cannot give (that subdivision
only tiles the triangle when r0 is inside it).

Reference: Wilton et al., "Potential integrals for uniform and linear source
distributions on polygonal and polyhedral domains", IEEE TAP-32, 1984.
"""

import numpy as np

# Below this, the observation point sits on the edge's line: P0 multiplies the
# logarithm, so the whole term vanishes and evaluating the log is pointless
# (and would divide by zero in the stabilised form).
_P0_FLOOR = 1e-14


def inv_R_over_triangle(r0, v1, v2, v3):
    """
    Evaluate int_T dS' / |r0 - r'| in closed form.

    Parameters
    ----------
    r0         : (3,) observation point, coplanar with the triangle
    v1, v2, v3 : (3,) triangle vertices

    Returns
    -------
    float -- the integral, exact up to floating-point roundoff
    """
    r0 = np.asarray(r0, dtype=np.float64)
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)
    v3 = np.asarray(v3, dtype=np.float64)

    normal = np.cross(v2 - v1, v3 - v1)
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-300:          # degenerate triangle contributes nothing
        return 0.0
    normal = normal / norm_len

    total = 0.0
    for A, B in ((v1, v2), (v2, v3), (v3, v1)):
        edge = B - A
        edge_len = np.linalg.norm(edge)
        if edge_len < 1e-300:
            continue
        e_hat = edge / edge_len
        u_hat = np.cross(e_hat, normal)

        P0 = float(np.dot(A - r0, u_hat))
        if abs(P0) < _P0_FLOOR:
            continue

        l_minus = float(np.dot(A - r0, e_hat))
        l_plus = float(np.dot(B - r0, e_hat))
        R_minus = float(np.linalg.norm(A - r0))
        R_plus = float(np.linalg.norm(B - r0))

        d_plus = R_plus + l_plus
        d_minus = R_minus + l_minus

        # (R + l) collapses to ~0 when r0 sits on the edge's line beyond that
        # endpoint, so the direct ratio loses all precision there. Since
        # (R + l)(R - l) = R^2 - l^2 = P0^2, the vanishing factor can always be
        # rewritten through P0^2. Pick the form whose denominator is the larger.
        if d_minus > d_plus:
            total += P0 * np.log(d_plus * (R_minus - l_minus) / (P0 * P0))
        else:
            total -= P0 * np.log(d_minus * (R_plus - l_plus) / (P0 * P0))

    return total


def inv_R_over_triangle_batch(r0, v1, v2, v3):
    """
    Vectorised `inv_R_over_triangle` over many (point, triangle) pairs.

    Parameters
    ----------
    r0         : (3, N) observation points
    v1, v2, v3 : (3, N) triangle vertices, one triangle per point

    Returns
    -------
    (N,) float64
    """
    r0 = np.asarray(r0, dtype=np.float64)
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)
    v3 = np.asarray(v3, dtype=np.float64)

    normal = np.cross(v2 - v1, v3 - v1, axis=0)
    norm_len = np.linalg.norm(normal, axis=0)
    safe_len = np.where(norm_len > 1e-300, norm_len, 1.0)
    normal = normal / safe_len

    total = np.zeros(r0.shape[1], dtype=np.float64)
    for A, B in ((v1, v2), (v2, v3), (v3, v1)):
        edge = B - A
        edge_len = np.linalg.norm(edge, axis=0)
        safe_edge = np.where(edge_len > 1e-300, edge_len, 1.0)
        e_hat = edge / safe_edge
        u_hat = np.cross(e_hat, normal, axis=0)

        P0 = np.einsum('ij,ij->j', A - r0, u_hat)
        l_minus = np.einsum('ij,ij->j', A - r0, e_hat)
        l_plus = np.einsum('ij,ij->j', B - r0, e_hat)
        R_minus = np.linalg.norm(A - r0, axis=0)
        R_plus = np.linalg.norm(B - r0, axis=0)

        d_plus = R_plus + l_plus
        d_minus = R_minus + l_minus
        P0_sq = np.where(np.abs(P0) < _P0_FLOOR, 1.0, P0 * P0)

        use_plus_form = d_minus > d_plus
        term = np.where(
            use_plus_form,
            P0 * np.log(np.maximum(d_plus * (R_minus - l_minus), 1e-300) / P0_sq),
            -P0 * np.log(np.maximum(d_minus * (R_plus - l_plus), 1e-300) / P0_sq),
        )
        valid = (np.abs(P0) >= _P0_FLOOR) & (edge_len > 1e-300) & (norm_len > 1e-300)
        total += np.where(valid, term, 0.0)

    return total
