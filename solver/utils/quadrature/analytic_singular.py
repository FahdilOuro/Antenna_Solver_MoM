"""
Closed-form surface integral of the static kernel 1/R over a planar triangle.

Used to remove the 1/R singularity from the EFIE assembly analytically instead
of asking a Gauss rule to resolve it. For a triangle T and any observation point
r0 (in the triangle's plane or above it),

    int_T dS' / |r0 - r'| = sum over the three edges of
          P0_i * ln( (R_i^+ + l_i^+) / (R_i^- + l_i^-) )
        - |h| * [ atan( P0_i*l_i^+ / (R0_i^2 + |h|*R_i^+) )
                 -atan( P0_i*l_i^- / (R0_i^2 + |h|*R_i^-) ) ]

where h is the signed height of r0 above the plane and everything else is
measured on the PROJECTION of r0 into that plane:

    n     = unit normal of T
    h     = (r0 - v1) . n              signed height
    rho   = r0 - h*n                   projection of r0 into the plane
    e     = (B - A) / |B - A|          unit vector along the edge
    u     = e x n                      in-plane normal to that edge
    P0    = (A - rho) . u              SIGNED distance from rho to the edge line
    l^-   = (A - rho) . e              signed position of A along the edge
    l^+   = (B - rho) . e              signed position of B along the edge
    R0^2  = P0^2 + h^2
    R^+-  = |B - r0| , |A - r0|

Because P0 is signed, the same expression holds whether the projection falls
inside the triangle or outside it -- which is what near-singular EDGE/VERTEX
pairs need, and what a Duffy subdivision from the singular point cannot give
(that subdivision only tiles the triangle when the point is inside it).

The |h| arctangent term is not optional. Dropping it leaves a formula that is
exact for a COPLANAR observation point and badly wrong otherwise: measured on a
40 mm cell it under-predicts by 64% at half a cell above the plane, by 34% at a
90-degree junction, and goes NEGATIVE -- for a strictly positive integral -- a
full cell above. Planar meshes never notice; a chassis-plus-panels structure is
made of exactly the non-coplanar pairs that do.

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

    h = float(np.dot(r0 - v1, normal))
    abs_h = abs(h)
    rho = r0 - h * normal          # projection of r0 into the triangle's plane

    total = 0.0
    for A, B in ((v1, v2), (v2, v3), (v3, v1)):
        edge = B - A
        edge_len = np.linalg.norm(edge)
        if edge_len < 1e-300:
            continue
        e_hat = edge / edge_len
        u_hat = np.cross(e_hat, normal)

        P0 = float(np.dot(A - rho, u_hat))
        l_minus = float(np.dot(A - rho, e_hat))
        l_plus = float(np.dot(B - rho, e_hat))
        R_minus = float(np.linalg.norm(A - r0))
        R_plus = float(np.linalg.norm(B - r0))
        R0_sq = P0 * P0 + h * h

        if abs(P0) >= _P0_FLOOR:
            d_plus = R_plus + l_plus
            d_minus = R_minus + l_minus
            # (R + l)(R - l) = R^2 - l^2 = R0^2, so the vanishing factor can
            # always be routed through R0^2. Use whichever denominator is larger.
            if d_minus > d_plus:
                total += P0 * np.log(d_plus * (R_minus - l_minus) / R0_sq) if R0_sq > 0 else 0.0
            else:
                total -= P0 * np.log(d_minus * (R_plus - l_plus) / R0_sq) if R0_sq > 0 else 0.0

        # Height term: zero for a coplanar point, dominant well above the plane.
        if abs_h > _P0_FLOOR and R0_sq > 0:
            total -= abs_h * (np.arctan2(P0 * l_plus, R0_sq + abs_h * R_plus)
                              - np.arctan2(P0 * l_minus, R0_sq + abs_h * R_minus))

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

    h = np.einsum('ij,ij->j', r0 - v1, normal)
    abs_h = np.abs(h)
    rho = r0 - h[None, :] * normal

    total = np.zeros(r0.shape[1], dtype=np.float64)
    for A, B in ((v1, v2), (v2, v3), (v3, v1)):
        edge = B - A
        edge_len = np.linalg.norm(edge, axis=0)
        safe_edge = np.where(edge_len > 1e-300, edge_len, 1.0)
        e_hat = edge / safe_edge
        u_hat = np.cross(e_hat, normal, axis=0)

        P0 = np.einsum('ij,ij->j', A - rho, u_hat)
        l_minus = np.einsum('ij,ij->j', A - rho, e_hat)
        l_plus = np.einsum('ij,ij->j', B - rho, e_hat)
        R_minus = np.linalg.norm(A - r0, axis=0)
        R_plus = np.linalg.norm(B - r0, axis=0)
        R0_sq = P0 * P0 + h * h
        R0_safe = np.where(R0_sq > 0, R0_sq, 1.0)

        d_plus = R_plus + l_plus
        d_minus = R_minus + l_minus
        use_plus_form = d_minus > d_plus
        log_term = np.where(
            use_plus_form,
            P0 * np.log(np.maximum(d_plus * (R_minus - l_minus), 1e-300) / R0_safe),
            -P0 * np.log(np.maximum(d_minus * (R_plus - l_plus), 1e-300) / R0_safe),
        )
        valid = (np.abs(P0) >= _P0_FLOOR) & (edge_len > 1e-300) & (norm_len > 1e-300) & (R0_sq > 0)
        total += np.where(valid, log_term, 0.0)

        atan_term = abs_h * (np.arctan2(P0 * l_plus, R0_safe + abs_h * R_plus)
                             - np.arctan2(P0 * l_minus, R0_safe + abs_h * R_minus))
        height_valid = (abs_h > _P0_FLOOR) & (edge_len > 1e-300) & (norm_len > 1e-300) & (R0_sq > 0)
        total -= np.where(height_valid, atan_term, 0.0)

    return total
