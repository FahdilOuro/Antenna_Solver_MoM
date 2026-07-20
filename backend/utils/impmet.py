"""
Impedance matrix assembly for the RWG MoM EFIE solver.

Two implementations are provided:

impedance_matrix_z (new — Phase 4)
    Correct double Gauss-Dunavant product quadrature on both testing and source
    triangles.  Self pairs use the Duffy collapsed-coordinate transform.
    Adjacent pairs use higher-order Gauss.  Returns a symmetric Z with physical
    units and no hidden 1/9 averaging.

impedance_matrix_z_makarov_legacy (renamed from `impedance_matrix_z`)
    Original hybrid single-point/9-barycentric-centre rule from Makarov's
    MATLAB book.  Kept for comparison.  Known defects: Z ≠ Zᵀ (~13–31 %
    symmetry error), negative eigenvalues of Re(Z), ~30× error on far pairs.
    See `_diagnostics/FINDINGS.md`.

rwg_gram_matrix_legacy (renamed from `rwg_gram_matrix`)
    Original Gram matrix for surface-impedance boundary conditions.
"""

import numpy as np

from backend.utils.quadrature.gauss_dunavant import (
    dunavant_rule, map_to_triangle,
)
from backend.utils.quadrature.duffy import duffy_source_nodes
from backend.utils.quadrature.singularity_classifier import (
    classify_triangle_pair, SELF, EDGE, VERTEX, FAR,
)


# ---------------------------------------------------------------------------
# New correct impedance matrix (Phase 4)
# ---------------------------------------------------------------------------

def impedance_matrix_z(edges_data, triangles_data, points_data, k_complex, omega, mu, epsilon, quad_order: int = 6, duffy_order: int = 7) -> np.ndarray:
    """
    Assemble the RWG EFIE impedance matrix using proper double surface quadrature.

    The formula implemented is (Rao, Wilton & Glisson 1982):

        Z_mn = jωμ/(4π) * ∑_{T_t,T_s} s_ts * l_m * l_n
                          * ∑_{p,q} w_p w_q ρ_t(r_p)·ρ_s(r_q) G(r_p,r_q)
             + 1/(jωε·4π) * 4 * ∑_{T_t,T_s} s_ts * l_m * l_n
                             * ∑_{p,q} w_p w_q G(r_p,r_q)

    where the sum over (T_t,T_s) runs over the four combinations
    (T^+_m × T^+_n), (T^+_m × T^-_n), (T^-_m × T^+_n), (T^-_m × T^-_n)
    and s_ts = s_t * s_s ∈ {+1, −1} is the product of the edge signs.

    Quadrature strategy per sub-triangle-pair category:
        FAR            → Gauss-Dunavant(quad_order) on both T_t and T_s
        SELF           → Gauss-Dunavant(quad_order) on T_t;
                         Duffy(duffy_order) on T_s with r_sing = test point
        EDGE / VERTEX  → Gauss-Dunavant(quad_order+5) on both (higher-order
                         Gauss compensates for near-singularity; integrand is
                         bounded since Gauss nodes lie strictly in the interior)

    Parameters
    ----------
    edges_data      : Edges object — .total_number_of_edges, .edges_length,
                      .first_points, .second_points
    triangles_data  : Triangles object — .triangles (3+, N_tri) vertex indices,
                      .triangles_plus/minus (E,), .triangles_area (N_tri,),
                      .total_of_triangles
    points_data     : Points object — .points (3, N_pts) physical coordinates
    k_complex       : complex float  = 1j * k  (complex wavenumber; k real)
    omega           : float  angular frequency (rad/s)
    mu              : float  free-space permeability (H/m)
    epsilon         : float  free-space permittivity (F/m)
    quad_order      : int    Gauss-Dunavant order for FAR pairs (default 6)
    duffy_order     : int    GL order for Duffy inner integral (default 7)

    Returns
    -------
    matrice_z : complex (E, E)  impedance matrix
    """
    E      = edges_data.total_number_of_edges
    N_tri  = triangles_data.total_of_triangles
    tri    = triangles_data.triangles[:3, :]   # (3, N_tri) vertex indices (int)
    pts    = points_data.points                 # (3, N_pts)
    plus   = triangles_data.triangles_plus      # (E,)
    minus  = triangles_data.triangles_minus     # (E,)
    lengths = edges_data.edges_length           # (E,)
    first_pts  = edges_data.first_points        # (E,) vertex index of edge start
    second_pts = edges_data.second_points       # (E,) vertex index of edge end

    # Physical prefactors
    pref_A   = 1j * omega * mu / (4.0 * np.pi)
    pref_Phi = 1.0 / (1j * omega * epsilon * 4.0 * np.pi)

    # Precompute triangle vertex arrays: shape (N_tri, 3, 3)
    # tri_verts[t, :, i] = physical coords of i-th vertex of triangle t
    tri_verts = np.zeros((N_tri, 3, 3), dtype=np.float64)
    for t in range(N_tri):
        for i in range(3):
            tri_verts[t, :, i] = pts[:, tri[i, t]]

    # Precompute free vertex for each edge in each ± triangle
    # free_plus[:, m] = vertex of T^+_m NOT on edge m
    free_plus  = np.empty((3, E), dtype=np.float64)
    free_minus = np.empty((3, E), dtype=np.float64)
    for m in range(E):
        ev = {int(first_pts[m]), int(second_pts[m])}
        tp = int(plus[m]);  tm = int(minus[m])
        for vi in range(3):
            if int(tri[vi, tp]) not in ev:
                free_plus[:, m] = pts[:, tri[vi, tp]];  break
        for vi in range(3):
            if int(tri[vi, tm]) not in ev:
                free_minus[:, m] = pts[:, tri[vi, tm]]; break

    # Quadrature rules
    xi_far,  w_far  = dunavant_rule(quad_order)       # far pairs
    xi_near, w_near = dunavant_rule(min(quad_order + 5, 11))  # adjacent pairs

    # Result matrix
    matrice_z = np.zeros((E, E), dtype=complex)

    for m in range(E):
        lm = lengths[m]
        tp_m = int(plus[m]);  tm_m = int(minus[m])

        # Testing triangles with their sign and free vertex
        test_sides = [
            (tp_m, +1.0, free_plus[:, m]),
            (tm_m, -1.0, free_minus[:, m]),
        ]

        for n in range(E):
            ln = lengths[n]
            tp_n = int(plus[n]);  tm_n = int(minus[n])

            source_sides = [
                (tp_n, +1.0, free_plus[:, n]),
                (tm_n, -1.0, free_minus[:, n]),
            ]

            Z_mn = 0.0 + 0.0j

            for (T_t, s_t, fv_t) in test_sides:
                v_t = tri_verts[T_t]            # (3, 3)
                tri_t_verts = tri[:, T_t]       # vertex indices of T_t

                for (T_s, s_s, fv_s) in source_sides:
                    s_ts  = s_t * s_s
                    cat   = classify_triangle_pair(tri_t_verts, tri[:, T_s])
                    v_s   = tri_verts[T_s]      # (3, 3)

                    if cat == FAR:
                        # ---- standard double Gauss ----
                        r_t = map_to_triangle(xi_far, v_t[:, 0], v_t[:, 1], v_t[:, 2])   # (3, Nq)
                        r_s = map_to_triangle(xi_far, v_s[:, 0], v_s[:, 1], v_s[:, 2])   # (3, Nq)
                        rho_t = r_t - fv_t[:, None]   # (3, Nq_t)
                        rho_s = r_s - fv_s[:, None]   # (3, Nq_s)

                        # Pairwise distances (Nq_t, Nq_s); k_complex=1j*k_real
                        diff = r_t[:, :, None] - r_s[:, None, :]   # (3, Nq_t, Nq_s)
                        R    = np.sqrt(np.einsum('ijk,ijk->jk', diff, diff))
                        G    = np.exp(-k_complex * R) / R           # exp(-jk_real*R)/R

                        rho_dot = np.einsum('ip,iq->pq', rho_t, rho_s)   # (Nq_t, Nq_s)
                        integrand = (pref_A * rho_dot + 4.0 * pref_Phi) * G

                        sub = np.einsum('p,q,pq->', w_far, w_far, integrand)
                        Z_mn += s_ts * lm * ln * sub

                    elif cat == SELF:
                        # ---- outer Gauss, inner Duffy ----
                        r_t   = map_to_triangle(xi_far, v_t[:, 0], v_t[:, 1], v_t[:, 2])  # (3, Nq_t)
                        rho_t = r_t - fv_t[:, None]

                        sub = 0.0 + 0.0j
                        for p in range(len(w_far)):
                            r_sing = r_t[:, p]
                            rho_tp = rho_t[:, p]   # (3,)

                            d_pts, d_w = duffy_source_nodes(
                                r_sing, v_s[:, 0], v_s[:, 1], v_s[:, 2],
                                order=duffy_order,
                            )                               # (Nq_d, 3), (Nq_d,)
                            rho_sp = d_pts.T - fv_s[:, None]   # (3, Nq_d)
                            diff_p = r_sing[:, None] - d_pts.T  # (3, Nq_d)
                            R_p    = np.sqrt(np.einsum('id,id->d', diff_p, diff_p))  # (Nq_d,)
                            G_p    = np.where(R_p > 1e-300, np.exp(-k_complex * R_p) / R_p, 0.0)
                            rho_dot_p = np.einsum('i,id->d', rho_tp, rho_sp)  # (Nq_d,)

                            inner = np.dot(d_w, (pref_A * rho_dot_p + 4.0 * pref_Phi) * G_p)
                            sub  += w_far[p] * inner

                        Z_mn += s_ts * lm * ln * sub

                    else:   # EDGE or VERTEX — bounded near-singularity, higher-order Gauss
                        r_t = map_to_triangle(xi_near, v_t[:, 0], v_t[:, 1], v_t[:, 2])
                        r_s = map_to_triangle(xi_near, v_s[:, 0], v_s[:, 1], v_s[:, 2])
                        rho_t = r_t - fv_t[:, None]
                        rho_s = r_s - fv_s[:, None]

                        diff = r_t[:, :, None] - r_s[:, None, :]
                        R    = np.sqrt(np.einsum('ijk,ijk->jk', diff, diff))
                        G    = np.exp(-k_complex * R) / R

                        rho_dot   = np.einsum('ip,iq->pq', rho_t, rho_s)
                        integrand = (pref_A * rho_dot + 4.0 * pref_Phi) * G

                        sub = np.einsum('p,q,pq->', w_near, w_near, integrand)
                        Z_mn += s_ts * lm * ln * sub

            matrice_z[m, n] = Z_mn

    return matrice_z


# ---------------------------------------------------------------------------
# Legacy (renamed) — Makarov 2002 hybrid-quadrature implementation
# ---------------------------------------------------------------------------

def impedance_matrix_z_makarov_legacy(edges_data, triangles_data, barycentric_triangles_data,
                                      vecteurs_rho_data, parameter_k, factor_a, factor_fi):
    """
    LEGACY — Makarov (2002) hybrid-quadrature Z matrix.

    Known defects documented in `_diagnostics/FINDINGS.md`:
        symmetry error ‖R−Rᵀ‖/‖R‖ ≈ 10–13 %
        negative eigenvalues of Re(Z)
        ~30× error on far pairs vs. brute-force reference

    Original name was `impedance_matrix_z`; renamed so the correct
    implementation can take that name.
    """
    total_number_of_edges = edges_data.total_number_of_edges
    total_of_triangles = triangles_data.total_of_triangles
    triangles_plus = triangles_data.triangles_plus
    triangles_minus = triangles_data.triangles_minus
    triangles_center = triangles_data.triangles_center
    edges_length = edges_data.edges_length
    barycentric_triangle_center = barycentric_triangles_data.barycentric_triangle_center
    vecteur_rho_plus = vecteurs_rho_data.vecteur_rho_plus
    vecteur_rho_minus = vecteurs_rho_data.vecteur_rho_minus
    vecteur_rho_barycentric_plus = vecteurs_rho_data.vecteur_rho_barycentric_plus
    vecteur_rho_barycentric_minus = vecteurs_rho_data.vecteur_rho_barycentric_minus

    vecteur_rho_plus_tiled  = np.tile(vecteur_rho_plus[:, None, :],  (1, 9, 1))
    vecteur_rho_minus_tiled = np.tile(vecteur_rho_minus[:, None, :], (1, 9, 1))

    matrice_z = np.zeros((total_number_of_edges, total_number_of_edges), dtype=complex)

    for triangle in range(total_of_triangles):
        positions_plus  = triangles_plus  == triangle
        positions_minus = triangles_minus == triangle

        distances        = barycentric_triangle_center - triangles_center[:, triangle][:, None, None]
        norm_of_distances = np.sqrt(np.sum(distances**2, axis=0, keepdims=True))
        g_function        = np.exp(-parameter_k * norm_of_distances) / norm_of_distances

        g_function_plus  = g_function[:, :, triangles_plus]
        g_function_minus = g_function[:, :, triangles_minus]

        fi = (np.sum(g_function_plus,  axis=1, keepdims=True)
              - np.sum(g_function_minus, axis=1, keepdims=True))

        impedance_coupling_zf = factor_fi.reshape(-1, 1) * fi.squeeze().reshape(-1, 1)

        def update(the_position, vecteur_rho_barycentric_p_m, sign):
            vecteur_rho_barycentric = np.tile(
                vecteur_rho_barycentric_p_m[:, :, the_position][:, :, None],
                (1, 1, total_number_of_edges),
            )
            a_contribution = (
                np.sum(g_function_plus  * np.sum(vecteur_rho_barycentric * vecteur_rho_plus_tiled,  axis=0), axis=0)
                + np.sum(g_function_minus * np.sum(vecteur_rho_barycentric * vecteur_rho_minus_tiled, axis=0), axis=0)
            )
            z1 = factor_a * a_contribution[:, None]
            z1_reshaped = z1.squeeze(axis=1).sum(axis=0)
            matrice_z[:, the_position] += edges_length[the_position] * (
                z1_reshaped + sign * impedance_coupling_zf.squeeze()
            )

        for position in np.where(positions_plus)[0]:
            update(position, vecteur_rho_barycentric_plus,  +1)
        for position in np.where(positions_minus)[0]:
            update(position, vecteur_rho_barycentric_minus, -1)

    return matrice_z


# ---------------------------------------------------------------------------
# Legacy Gram matrix (renamed)
# ---------------------------------------------------------------------------

def rwg_gram_matrix_legacy(edges_data, triangles_data, vecteurs_rho_data):
    """
    LEGACY — RWG Gram matrix for surface-impedance boundary conditions.
    Original name was `rwg_gram_matrix`.
    """
    num_edges = edges_data.total_number_of_edges
    num_tri   = triangles_data.total_of_triangles
    G_global  = np.zeros((num_edges, num_edges), dtype=np.float64)

    tri_to_edges  = [[] for _ in range(num_tri)]
    tri_edge_sides = [[] for _ in range(num_tri)]

    for e_idx in range(num_edges):
        tp = triangles_data.triangles_plus[e_idx]
        tm = triangles_data.triangles_minus[e_idx]
        if tp != -1:
            tri_to_edges[tp].append(e_idx);   tri_edge_sides[tp].append('plus')
        if tm != -1:
            tri_to_edges[tm].append(e_idx);   tri_edge_sides[tm].append('minus')

    print(f"Computing Gram matrix ({num_tri} triangles)...")

    for tri_idx in range(num_tri):
        area         = triangles_data.triangles_area[tri_idx]
        edges_in_tri = tri_to_edges[tri_idx]
        sides_in_tri = tri_edge_sides[tri_idx]
        n_local      = len(edges_in_tri)

        if not edges_in_tri:
            continue

        l_local   = edges_data.edges_length[edges_in_tri]
        rho_local = np.zeros((n_local, 9, 3), dtype=np.float64)

        for i, (e_idx, side) in enumerate(zip(edges_in_tri, sides_in_tri)):
            if side == 'plus':
                rho_local[i] = vecteurs_rho_data.vecteur_rho_barycentric_plus[:, :, e_idx].T
            else:
                rho_local[i] = vecteurs_rho_data.vecteur_rho_barycentric_minus[:, :, e_idx].T

        g_block = np.zeros((n_local, n_local), dtype=np.float64)

        for i in range(n_local):
            for j in range(i, n_local):
                dot_ij = np.einsum('kd,kd->', rho_local[i], rho_local[j])
                val    = (l_local[i] * l_local[j] / (36.0 * area)) * dot_ij
                g_block[i, j] = val
                g_block[j, i] = val

        G_global[np.ix_(edges_in_tri, edges_in_tri)] += g_block

    return G_global


# Backward-compat aliases (so existing code using `from impmet import *` still works)
rwg_gram_matrix = rwg_gram_matrix_legacy
