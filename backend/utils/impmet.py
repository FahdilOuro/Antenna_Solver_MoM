"""
Impedance matrix assembly for the RWG MoM EFIE solver.

impedance_matrix_z
    EFIE double-surface integral (Green's function kernel, singular for
    self/adjacent triangle pairs).  Double Gauss-Dunavant product quadrature
    on both testing and source triangles; self pairs use the Duffy
    collapsed-coordinate transform; adjacent pairs use higher-order Gauss.
    Returns a symmetric Z with physical units and no hidden 1/9 averaging.

rwg_gram_matrix
    Single-surface RWG mass ("Gram") matrix for the surface-impedance
    boundary condition term of a lossy conductor (Z_lossy = Z + Zs * G).
    No Green's function and no singularity: f_m . f_n is an exact degree-2
    polynomial on the shared triangle, so a low-order Gauss-Dunavant rule
    integrates it to machine precision.  Replaces the previous 9-point
    barycentric-average approximation (see workflow.md, "Gram matrix
    Gauss-Dunavant refactor").
"""

import numpy as np

from backend.utils.quadrature.gauss_dunavant import (
    dunavant_rule, map_to_triangle,
)
from backend.utils.quadrature.duffy import duffy_source_nodes
from backend.utils.quadrature.singularity_classifier import (
    classify_triangle_pair, SELF, EDGE, VERTEX, FAR,
)


def _duffy_weight_scale(area_source: float) -> float:
    """
    Rescale Duffy weights into the same convention as the Dunavant weights.

    The two quadrature helpers used by the SELF branch normalise differently:

      * dunavant_rule(...) returns weights summing to 1/2 -- the area of the
        REFERENCE triangle. Turning those into a physical surface integral
        needs the mapping Jacobian 2*A:  int_T g dS = 2*A * sum_p w_p g(r_p).
      * duffy_source_nodes(...) returns weights summing to A, the PHYSICAL
        triangle area, i.e. sum_d w_d g(r_d) is already int_T g dS.

    That difference is invisible in the FAR/EDGE/VERTEX branches (Dunavant on
    both triangles), where the 2*A_t * 2*A_s Jacobians cancel exactly against
    the RWG basis normalisation l/(2A) on each side -- which is precisely why
    those branches carry no explicit area factor and why the assembled term
    reads (pref_A * rho_t.rho_s + 4 * pref_Phi).

    The SELF branch mixes the two (Dunavant outside, Duffy inside), so the
    inner integral arrives carrying a physical 1/(2*A_s) too many. Left
    uncorrected, SELF contributions come out scaled by 2*A_s -- for a
    millimetre-scale mesh cell that is ~1e-3, i.e. effectively deleting them.

    That is far worse than it sounds for R = Re(Z): on the diagonal, the four
    (T+-,T+-) sub-pairs must cancel (sum of s_ts = 0) for the O(1/k) scalar-
    potential term to vanish and leave the true O(k) radiated power. Dropping
    the two SELF sub-pairs breaks that cancellation, so R keeps a large
    spurious near-diagonal residue: it becomes numerically full-rank and
    loses positive semi-definiteness, instead of the low-rank PSD radiation
    operator it must be. CMA (which solves X*J = lambda*R*J) is then junk.
    """
    return 1.0 / (2.0 * area_source)


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
    tri_verts = np.transpose(pts[:, tri], (2, 0, 1))

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

    # ---- Per-triangle quadrature points, computed ONCE per triangle ----
    # (previously map_to_triangle was re-evaluated for every (m,n) pair that
    # touched a given triangle, i.e. up to O(E) times instead of once)
    Nq_far, Nq_near = len(w_far), len(w_near)
    r_tri_far  = np.empty((N_tri, 3, Nq_far),  dtype=np.float64)
    r_tri_near = np.empty((N_tri, 3, Nq_near), dtype=np.float64)
    for t in range(N_tri):
        v = tri_verts[t]
        r_tri_far[t]  = map_to_triangle(xi_far,  v[:, 0], v[:, 1], v[:, 2])
        r_tri_near[t] = map_to_triangle(xi_near, v[:, 0], v[:, 1], v[:, 2])

    # ---- Per-edge rho vectors at those points, computed ONCE per edge ----
    # (previously recomputed for every (m,n) pair; fv_t/fv_s only depend on
    # the edge+side, not on the pairing partner)
    rho_plus_far   = r_tri_far[plus]   - free_plus.T[:, :, None]
    rho_minus_far  = r_tri_far[minus]  - free_minus.T[:, :, None]
    rho_plus_near  = r_tri_near[plus]  - free_plus.T[:, :, None]
    rho_minus_near = r_tri_near[minus] - free_minus.T[:, :, None]

    # Result matrix
    matrice_z = np.zeros((E, E), dtype=complex)

    for m in range(E):
        lm = lengths[m]
        tp_m = int(plus[m]);  tm_m = int(minus[m])

        # Testing triangles with their sign, free vertex, and precomputed rho
        test_sides = [
            (tp_m, +1.0, free_plus[:, m],  rho_plus_far[m],  rho_plus_near[m]),
            (tm_m, -1.0, free_minus[:, m], rho_minus_far[m], rho_minus_near[m]),
        ]

        for n in range(E):
            ln = lengths[n]
            tp_n = int(plus[n]);  tm_n = int(minus[n])

            source_sides = [
                (tp_n, +1.0, free_plus[:, n],  rho_plus_far[n],  rho_plus_near[n]),
                (tm_n, -1.0, free_minus[:, n], rho_minus_far[n], rho_minus_near[n]),
            ]

            Z_mn = 0.0 + 0.0j

            for (T_t, s_t, fv_t, rho_t_far, rho_t_near) in test_sides:
                tri_t_verts = tri[:, T_t]       # vertex indices of T_t

                for (T_s, s_s, fv_s, rho_s_far, rho_s_near) in source_sides:
                    s_ts  = s_t * s_s
                    cat   = classify_triangle_pair(tri_t_verts, tri[:, T_s])

                    if cat == FAR:
                        # ---- standard double Gauss ----
                        r_t, r_s = r_tri_far[T_t], r_tri_far[T_s]
                        rho_t, rho_s = rho_t_far, rho_s_far

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
                        r_t, rho_t = r_tri_far[T_t], rho_t_far
                        v_s = tri_verts[T_s]      # (3, 3) — Duffy needs raw vertices
                        duffy_scale = _duffy_weight_scale(triangles_data.triangles_area[T_s])

                        sub = 0.0 + 0.0j
                        for p in range(Nq_far):
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
                            sub  += w_far[p] * inner * duffy_scale

                        Z_mn += s_ts * lm * ln * sub

                    else:   # EDGE or VERTEX — bounded near-singularity, higher-order Gauss
                        r_t, r_s = r_tri_near[T_t], r_tri_near[T_s]
                        rho_t, rho_s = rho_t_near, rho_s_near

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
# Multi-frequency sweep: assemble Z at every frequency in a single pass
# ---------------------------------------------------------------------------

def impedance_matrix_z_sweep(edges_data, triangles_data, points_data, frequencies, mu, epsilon,
                              quad_order: int = 6, duffy_order: int = 7) -> np.ndarray:
    """
    Assemble the RWG EFIE impedance matrix at MULTIPLE frequencies in one pass.

    A frequency sweep that calls `impedance_matrix_z` once per frequency point
    recomputes, from scratch, every triangle-pair quantity that does not
    actually depend on frequency: quadrature points, rho vectors, pairwise
    distances R = |r_t - r_s|, rho_t.rho_s dot products, and (for SELF pairs)
    the Duffy collapsed-coordinate nodes. Only the Green's-function kernel
    G(f) = exp(-jk(f)R)/R and the physical prefactors actually change with f.

    This function has the exact same (m,n) fill loop and per-pair
    categorisation as `impedance_matrix_z`, but computes each pair's
    frequency-INDEPENDENT geometry once, then evaluates the frequency-
    DEPENDENT kernel for every sweep point via a broadcast over a leading
    frequency axis -- so the expensive geometric work (R, rho_dot, Duffy
    nodes) is paid once per pair instead of once per pair per frequency.

    Every Z_all[i] is identical (to full float64 precision) to what
    `impedance_matrix_z` would return for frequencies[i] with the same
    quad_order/duffy_order -- this is purely a caching/vectorisation change,
    not a change in what is computed. See workflow.md, "frequency_sweep
    one-time precompute" for the profiled speedup on a real mesh.

    Parameters
    ----------
    edges_data, triangles_data, points_data : as in impedance_matrix_z
    frequencies : array-like (N_freq,)  sweep frequencies in Hz
    mu, epsilon : float  free-space permeability / permittivity
    quad_order, duffy_order : int  as in impedance_matrix_z

    Returns
    -------
    matrice_z : complex (N_freq, E, E)
    """
    frequencies = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    N_freq = frequencies.shape[0]

    light_speed_c = 1.0 / np.sqrt(mu * epsilon)
    omega_arr     = 2.0 * np.pi * frequencies                          # (Nf,)
    k_complex_arr = 1j * (omega_arr / light_speed_c)                   # (Nf,)
    pref_A_arr    = 1j * omega_arr * mu / (4.0 * np.pi)                # (Nf,)
    pref_Phi_arr  = 1.0 / (1j * omega_arr * epsilon * 4.0 * np.pi)     # (Nf,)

    E      = edges_data.total_number_of_edges
    N_tri  = triangles_data.total_of_triangles
    tri    = triangles_data.triangles[:3, :]
    pts    = points_data.points
    plus   = triangles_data.triangles_plus
    minus  = triangles_data.triangles_minus
    lengths = edges_data.edges_length
    first_pts  = edges_data.first_points
    second_pts = edges_data.second_points

    # --- Frequency-independent geometry: identical to impedance_matrix_z ---
    tri_verts = np.transpose(pts[:, tri], (2, 0, 1))

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

    xi_far,  w_far  = dunavant_rule(quad_order)
    xi_near, w_near = dunavant_rule(min(quad_order + 5, 11))
    Nq_far, Nq_near = len(w_far), len(w_near)

    r_tri_far  = np.empty((N_tri, 3, Nq_far),  dtype=np.float64)
    r_tri_near = np.empty((N_tri, 3, Nq_near), dtype=np.float64)
    for t in range(N_tri):
        v = tri_verts[t]
        r_tri_far[t]  = map_to_triangle(xi_far,  v[:, 0], v[:, 1], v[:, 2])
        r_tri_near[t] = map_to_triangle(xi_near, v[:, 0], v[:, 1], v[:, 2])

    rho_plus_far   = r_tri_far[plus]   - free_plus.T[:, :, None]
    rho_minus_far  = r_tri_far[minus]  - free_minus.T[:, :, None]
    rho_plus_near  = r_tri_near[plus]  - free_plus.T[:, :, None]
    rho_minus_near = r_tri_near[minus] - free_minus.T[:, :, None]

    matrice_z = np.zeros((N_freq, E, E), dtype=complex)

    for m in range(E):
        lm = lengths[m]
        tp_m = int(plus[m]);  tm_m = int(minus[m])

        test_sides = [
            (tp_m, +1.0, free_plus[:, m],  rho_plus_far[m],  rho_plus_near[m]),
            (tm_m, -1.0, free_minus[:, m], rho_minus_far[m], rho_minus_near[m]),
        ]

        for n in range(E):
            ln = lengths[n]
            tp_n = int(plus[n]);  tm_n = int(minus[n])

            source_sides = [
                (tp_n, +1.0, free_plus[:, n],  rho_plus_far[n],  rho_plus_near[n]),
                (tm_n, -1.0, free_minus[:, n], rho_minus_far[n], rho_minus_near[n]),
            ]

            Z_mn = np.zeros(N_freq, dtype=complex)

            for (T_t, s_t, fv_t, rho_t_far, rho_t_near) in test_sides:
                tri_t_verts = tri[:, T_t]

                for (T_s, s_s, fv_s, rho_s_far, rho_s_near) in source_sides:
                    s_ts = s_t * s_s
                    cat  = classify_triangle_pair(tri_t_verts, tri[:, T_s])

                    if cat == FAR:
                        r_t, r_s = r_tri_far[T_t], r_tri_far[T_s]
                        rho_t, rho_s = rho_t_far, rho_s_far

                        # Geometric part -- computed ONCE, reused for every frequency
                        diff    = r_t[:, :, None] - r_s[:, None, :]           # (3,Nq,Nq)
                        R       = np.sqrt(np.einsum('ijk,ijk->jk', diff, diff))  # (Nq,Nq)
                        rho_dot = np.einsum('ip,iq->pq', rho_t, rho_s)        # (Nq,Nq)

                        # Frequency-dependent kernel -- vectorised over the sweep
                        G = np.exp(-k_complex_arr[:, None, None] * R[None, :, :]) / R[None, :, :]
                        integrand = (pref_A_arr[:, None, None] * rho_dot[None, :, :]
                                     + 4.0 * pref_Phi_arr[:, None, None]) * G
                        sub = np.einsum('p,q,fpq->f', w_far, w_far, integrand)   # (Nf,)
                        Z_mn += s_ts * lm * ln * sub

                    elif cat == SELF:
                        r_t, rho_t = r_tri_far[T_t], rho_t_far
                        v_s = tri_verts[T_s]
                        duffy_scale = _duffy_weight_scale(triangles_data.triangles_area[T_s])

                        sub = np.zeros(N_freq, dtype=complex)
                        for p in range(Nq_far):
                            r_sing = r_t[:, p]
                            rho_tp = rho_t[:, p]

                            # Duffy nodes/weights are pure geometry -- computed
                            # ONCE per (m,n,p), reused for every frequency
                            d_pts, d_w = duffy_source_nodes(
                                r_sing, v_s[:, 0], v_s[:, 1], v_s[:, 2],
                                order=duffy_order,
                            )
                            rho_sp    = d_pts.T - fv_s[:, None]
                            diff_p    = r_sing[:, None] - d_pts.T
                            R_p       = np.sqrt(np.einsum('id,id->d', diff_p, diff_p))
                            rho_dot_p = np.einsum('i,id->d', rho_tp, rho_sp)

                            R_p_safe = np.where(R_p > 1e-300, R_p, 1.0)
                            G_p = np.exp(-k_complex_arr[:, None] * R_p_safe[None, :]) / R_p_safe[None, :]
                            G_p = np.where((R_p > 1e-300)[None, :], G_p, 0.0)   # (Nf,Nq_d)
                            integrand_p = (pref_A_arr[:, None] * rho_dot_p[None, :]
                                           + 4.0 * pref_Phi_arr[:, None]) * G_p
                            inner = integrand_p @ d_w                            # (Nf,)
                            sub  += w_far[p] * inner * duffy_scale

                        Z_mn += s_ts * lm * ln * sub

                    else:   # EDGE or VERTEX
                        r_t, r_s = r_tri_near[T_t], r_tri_near[T_s]
                        rho_t, rho_s = rho_t_near, rho_s_near

                        diff    = r_t[:, :, None] - r_s[:, None, :]
                        R       = np.sqrt(np.einsum('ijk,ijk->jk', diff, diff))
                        rho_dot = np.einsum('ip,iq->pq', rho_t, rho_s)

                        G = np.exp(-k_complex_arr[:, None, None] * R[None, :, :]) / R[None, :, :]
                        integrand = (pref_A_arr[:, None, None] * rho_dot[None, :, :]
                                     + 4.0 * pref_Phi_arr[:, None, None]) * G
                        sub = np.einsum('p,q,fpq->f', w_near, w_near, integrand)
                        Z_mn += s_ts * lm * ln * sub

            matrice_z[:, m, n] = Z_mn

    return matrice_z


# ---------------------------------------------------------------------------
# Batched, GPU-capable multi-frequency sweep (bucket pairs by category)
# ---------------------------------------------------------------------------

def impedance_matrix_z_sweep_gpu(edges_data, triangles_data, points_data, frequencies, mu, epsilon,
                                  quad_order: int = 6, duffy_order: int = 7,
                                  use_gpu=None, chunk_size: int = 20000) -> np.ndarray:
    """
    Same result as impedance_matrix_z_sweep, restructured so the FAR and
    EDGE/VERTEX categories -- which dominate the pair COUNT for any mesh of
    practical size -- are assembled as a handful of large batched tensor
    operations instead of a Python-level loop over every (m,n) pair. This is
    the prerequisite identified in workflow.md ("frequency_sweep one-time
    precompute" / GPU audit) before GPU acceleration can help at all: the
    original per-pair loop does tiny (~12x12 or 49x49 element) work per
    iteration, so replacing numpy with cupy INSIDE that loop would be slower
    than the CPU version -- GPU kernel-launch/PCIe overhead would dominate
    over up to 4*E^2 sub-pair iterations. Batching first, then running the
    batches on whichever array backend is active, is what actually pays off.

    Categorisation (SELF/EDGE/VERTEX/FAR) is computed for all 4*E^2 sub-pairs
    at once via vertex-index comparison (vectorised equivalent of
    classify_triangle_pair), then:
      - FAR and EDGE+VERTEX pairs (identical maths, EDGE/VERTEX only using a
        higher-order quadrature rule -- see impedance_matrix_z) are each
        assembled in chunks of `chunk_size` pairs as batched tensor
        contractions, on `xp` = cupy if a usable GPU was detected
        (backend.utils.compute_backend), else numpy.
      - SELF pairs keep the original per-pair Duffy quadrature on CPU: they
        number O(N_tri), a small fraction of the O(E^2) total, and Duffy's
        node count varies per pair (irregular work), which does not batch as
        cleanly as the regular categories. Batching the 90%+ of pairs that
        ARE regular already captures most of the available speedup.

    Every Z_all[i] matches impedance_matrix_z_sweep(frequencies)[i] to
    numerical precision (float64/complex128 arithmetic reordering only --
    see workflow.md for the measured CPU-restructure and GPU benchmarks).

    Parameters
    ----------
    edges_data, triangles_data, points_data : as in impedance_matrix_z
    frequencies : array-like (N_freq,)  sweep frequencies in Hz
    mu, epsilon : float  free-space permeability / permittivity
    quad_order, duffy_order : int  as in impedance_matrix_z
    use_gpu : bool or None  None=auto-detect (GPU if available, else CPU),
              True=force GPU (raises if unavailable), False=force CPU.
    chunk_size : int  number of FAR-category pairs processed per batch (the
              EDGE/VERTEX bucket scales this down automatically since its
              quadrature rule has more points per pair, to keep per-chunk
              memory comparable). Lower this if you hit an out-of-memory
              error on a large mesh / long sweep on a small GPU.

    Returns
    -------
    matrice_z : complex (N_freq, E, E)  (always a host/NumPy array)
    """
    from backend.utils.compute_backend import get_array_module, to_numpy
    xp = get_array_module(use_gpu)

    frequencies = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    N_freq = frequencies.shape[0]

    light_speed_c = 1.0 / np.sqrt(mu * epsilon)
    omega_arr    = 2.0 * np.pi * frequencies
    k_complex_np = 1j * (omega_arr / light_speed_c)
    pref_A_np    = 1j * omega_arr * mu / (4.0 * np.pi)
    pref_Phi_np  = 1.0 / (1j * omega_arr * epsilon * 4.0 * np.pi)

    E      = edges_data.total_number_of_edges
    N_tri  = triangles_data.total_of_triangles
    tri    = triangles_data.triangles[:3, :]
    pts    = points_data.points
    plus   = triangles_data.triangles_plus
    minus  = triangles_data.triangles_minus
    lengths_np = edges_data.edges_length
    first_pts  = edges_data.first_points
    second_pts = edges_data.second_points

    # --- Frequency-independent geometry: identical to impedance_matrix_z_sweep ---
    tri_verts = np.transpose(pts[:, tri], (2, 0, 1))

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

    xi_far,  w_far_np  = dunavant_rule(quad_order)
    xi_near, w_near_np = dunavant_rule(min(quad_order + 5, 11))
    Nq_far, Nq_near = len(w_far_np), len(w_near_np)

    r_tri_far  = np.empty((N_tri, 3, Nq_far),  dtype=np.float64)
    r_tri_near = np.empty((N_tri, 3, Nq_near), dtype=np.float64)
    for t in range(N_tri):
        v = tri_verts[t]
        r_tri_far[t]  = map_to_triangle(xi_far,  v[:, 0], v[:, 1], v[:, 2])
        r_tri_near[t] = map_to_triangle(xi_near, v[:, 0], v[:, 1], v[:, 2])

    rho_plus_far_np   = r_tri_far[plus]   - free_plus.T[:, :, None]
    rho_minus_far_np  = r_tri_far[minus]  - free_minus.T[:, :, None]
    rho_plus_near_np  = r_tri_near[plus]  - free_plus.T[:, :, None]
    rho_minus_near_np = r_tri_near[minus] - free_minus.T[:, :, None]

    # --- Move frequency-dependent scalars and the shared regular-pair tables
    #     onto the active backend once ---
    k_complex = xp.asarray(k_complex_np)
    pref_A    = xp.asarray(pref_A_np)
    pref_Phi  = xp.asarray(pref_Phi_np)
    w_far     = xp.asarray(w_far_np)
    w_near    = xp.asarray(w_near_np)
    r_tri_far_x  = xp.asarray(r_tri_far)
    r_tri_near_x = xp.asarray(r_tri_near)
    rho_far_x  = {'+': xp.asarray(rho_plus_far_np),  '-': xp.asarray(rho_minus_far_np)}
    rho_near_x = {'+': xp.asarray(rho_plus_near_np), '-': xp.asarray(rho_minus_near_np)}
    lengths_x  = xp.asarray(lengths_np)

    Z_batched = xp.zeros((N_freq, E, E), dtype=complex)
    Z_self_np = np.zeros((N_freq, E, E), dtype=complex)

    # Keep per-chunk memory roughly comparable between FAR (Nq_far^2 points
    # per pair) and EDGE/VERTEX (Nq_near^2, typically several times larger).
    chunk_size_near = max(1, int(chunk_size * (Nq_far / Nq_near) ** 2))

    def _process_regular_bucket(m_idx, n_idx, T_t_idx, T_s_idx, sign_scalar,
                                 r_tri_x, rho_t_table, rho_s_table, w, bucket_chunk):
        """Batched assembly for one (test_side, source_side, category) bucket."""
        P = m_idx.shape[0]
        if P == 0:
            return
        m_idx_x = xp.asarray(m_idx); n_idx_x = xp.asarray(n_idx)
        T_t_x = xp.asarray(T_t_idx); T_s_x = xp.asarray(T_s_idx)
        lm_all = lengths_x[m_idx_x]
        ln_all = lengths_x[n_idx_x]

        for start in range(0, P, bucket_chunk):
            sl = slice(start, min(start + bucket_chunk, P))
            r_t = r_tri_x[T_t_x[sl]]           # (chunk, 3, Nq)
            r_s = r_tri_x[T_s_x[sl]]           # (chunk, 3, Nq)
            rho_t = rho_t_table[m_idx_x[sl]]   # (chunk, 3, Nq)
            rho_s = rho_s_table[n_idx_x[sl]]   # (chunk, 3, Nq)

            diff = r_t[:, :, :, None] - r_s[:, :, None, :]           # (chunk,3,Nq,Nq)
            R = xp.sqrt(xp.einsum('cijk,cijk->cjk', diff, diff, optimize=True))   # (chunk,Nq,Nq)
            rho_dot = xp.einsum('cip,ciq->cpq', rho_t, rho_s, optimize=True)      # (chunk,Nq,Nq)

            G = xp.exp(-k_complex[:, None, None, None] * R[None, :, :, :]) / R[None, :, :, :]
            integrand = (pref_A[:, None, None, None] * rho_dot[None, :, :, :]
                         + 4.0 * pref_Phi[:, None, None, None]) * G      # (Nf,chunk,Nq,Nq)
            # Contract the two quadrature-weight vectors sequentially (matrix-
            # vector products) rather than as one 4-operand einsum: einsum's
            # generic evaluator is much slower than BLAS-backed matmul for
            # this shape, and the difference is what makes this batched
            # CPU path actually faster than the old per-pair loop (see
            # workflow.md, "GPU acceleration (CuPy)").
            sub = (integrand @ w) @ w                                    # (Nf,chunk,Nq)@(Nq,) -> (Nf,chunk)

            contrib = (sign_scalar * lm_all[sl][None, :] * ln_all[sl][None, :]) * sub
            Z_batched[:, m_idx_x[sl], n_idx_x[sl]] += contrib

    m_grid, n_grid = np.meshgrid(np.arange(E), np.arange(E), indexing='ij')
    side_key = ['+', '-']
    side_tri = [plus, minus]
    side_sign = [1.0, -1.0]

    for ts in range(2):        # test_side:   0='+', 1='-'
        for ss in range(2):    # source_side: 0='+', 1='-'
            T_t_grid = np.broadcast_to(side_tri[ts][:, None], (E, E))
            T_s_grid = np.broadcast_to(side_tri[ss][None, :], (E, E))
            s_ts = side_sign[ts] * side_sign[ss]

            # Vectorised equivalent of classify_triangle_pair for the whole grid:
            # count shared vertex indices between T_t and T_s at every (m,n).
            tri_t_verts = tri[:, T_t_grid]   # (3, E, E)
            tri_s_verts = tri[:, T_s_grid]   # (3, E, E)
            n_shared = (tri_t_verts[:, None, :, :] == tri_s_verts[None, :, :, :]).sum(axis=(0, 1))

            is_far  = n_shared == 0
            is_near = (n_shared == 1) | (n_shared == 2)
            is_self = n_shared == 3

            _process_regular_bucket(
                m_grid[is_far], n_grid[is_far], T_t_grid[is_far], T_s_grid[is_far],
                s_ts, r_tri_far_x, rho_far_x[side_key[ts]], rho_far_x[side_key[ss]],
                w_far, chunk_size,
            )
            _process_regular_bucket(
                m_grid[is_near], n_grid[is_near], T_t_grid[is_near], T_s_grid[is_near],
                s_ts, r_tri_near_x, rho_near_x[side_key[ts]], rho_near_x[side_key[ss]],
                w_near, chunk_size_near,
            )

            # SELF pairs: CPU, per-pair Duffy quadrature (unchanged maths).
            m_self, n_self = m_grid[is_self], n_grid[is_self]
            T_t_self, T_s_self = T_t_grid[is_self], T_s_grid[is_self]
            fv_s_table = free_plus if ss == 0 else free_minus
            rho_t_table_np = rho_plus_far_np if ts == 0 else rho_minus_far_np

            for pi in range(m_self.shape[0]):
                m = int(m_self[pi]); n = int(n_self[pi])
                T_t = int(T_t_self[pi]); T_s = int(T_s_self[pi])
                lm = lengths_np[m]; ln = lengths_np[n]
                fv_s = fv_s_table[:, n]
                r_t = r_tri_far[T_t]
                rho_t = rho_t_table_np[m]
                v_s = tri_verts[T_s]
                duffy_scale = _duffy_weight_scale(triangles_data.triangles_area[T_s])

                sub = np.zeros(N_freq, dtype=complex)
                for p in range(Nq_far):
                    r_sing = r_t[:, p]
                    rho_tp = rho_t[:, p]
                    d_pts, d_w = duffy_source_nodes(
                        r_sing, v_s[:, 0], v_s[:, 1], v_s[:, 2], order=duffy_order,
                    )
                    rho_sp = d_pts.T - fv_s[:, None]
                    diff_p = r_sing[:, None] - d_pts.T
                    R_p = np.sqrt(np.einsum('id,id->d', diff_p, diff_p))
                    rho_dot_p = np.einsum('i,id->d', rho_tp, rho_sp)
                    R_p_safe = np.where(R_p > 1e-300, R_p, 1.0)
                    G_p = np.exp(-k_complex_np[:, None] * R_p_safe[None, :]) / R_p_safe[None, :]
                    G_p = np.where((R_p > 1e-300)[None, :], G_p, 0.0)
                    integrand_p = (pref_A_np[:, None] * rho_dot_p[None, :]
                                   + 4.0 * pref_Phi_np[:, None]) * G_p
                    inner = integrand_p @ d_w
                    sub += w_far_np[p] * inner * duffy_scale

                Z_self_np[:, m, n] += s_ts * lm * ln * sub

    return to_numpy(Z_batched) + Z_self_np


# ---------------------------------------------------------------------------
# RWG Gram matrix (surface-impedance boundary condition term)
# ---------------------------------------------------------------------------

def rwg_gram_matrix(edges_data, triangles_data, points_data, quad_order: int = 4) -> np.ndarray:
    """
    Assemble the RWG mass ("Gram") matrix used for the surface-impedance
    boundary condition (SIBC) term of a lossy conductor:

        Z_lossy = matrice_z + impedance_zs * rwg_gram_matrix(...)

    Formula (single surface integral, summed over the triangles shared by
    edges m and n):

        G_mn = sum_{T in {T+_m,T-_m} ∩ {T+_n,T-_n}}  s_m(T) * s_n(T)
               * (l_m * l_n / (4 * area(T)^2)) * ∫_T rho_m(r)·rho_n(r) dS

    where rho_i(r) = r - free_vertex_i(T) and s_i(T) = +1 if T is edge i's
    "+" triangle, -1 if it is the "-" triangle (same sign convention as
    `impedance_matrix_z`: rho is r-minus-free-vertex on BOTH sides, and the
    sign is applied explicitly rather than baked into the vector).

    Unlike `impedance_matrix_z`, there is no Green's function and therefore
    no singularity to regularise: rho_m·rho_n is an exact polynomial of
    total degree 2 in r, so any Gauss-Dunavant rule of degree >= 2 integrates
    it to machine precision — `quad_order=4` (6 points, all-positive
    weights) is used by default. This replaces the previous 9-point
    barycentric-average approximation, which had genuine (if small)
    discretisation error; see workflow.md for the before/after comparison.

    Parameters
    ----------
    edges_data      : Edges object
    triangles_data  : Triangles object
    points_data     : Points object
    quad_order      : int  Gauss-Dunavant degree (default 4; any value >= 2
                      in {3,4,5,6,7,11} gives the same result to machine
                      precision since the integrand is an exact quadratic)

    Returns
    -------
    G : real (E, E) Gram matrix
    """
    E      = edges_data.total_number_of_edges
    N_tri  = triangles_data.total_of_triangles
    tri    = triangles_data.triangles[:3, :]
    pts    = points_data.points
    plus   = triangles_data.triangles_plus
    minus  = triangles_data.triangles_minus
    areas  = triangles_data.triangles_area
    lengths = edges_data.edges_length
    first_pts  = edges_data.first_points
    second_pts = edges_data.second_points

    # Free vertex per edge/side (same construction as impedance_matrix_z)
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

    tri_verts = np.transpose(pts[:, tri], (2, 0, 1))   # (N_tri, 3, 3)

    xi, w = dunavant_rule(quad_order)
    Nq = len(w)
    r_tri = np.empty((N_tri, 3, Nq), dtype=np.float64)
    for t in range(N_tri):
        v = tri_verts[t]
        r_tri[t] = map_to_triangle(xi, v[:, 0], v[:, 1], v[:, 2])

    rho_plus  = r_tri[plus]  - free_plus.T[:, :, None]    # (E, 3, Nq)
    rho_minus = r_tri[minus] - free_minus.T[:, :, None]   # (E, 3, Nq)

    # Group edges by the triangle they touch, keeping the +/- sign per edge
    tri_to_edges   = [[] for _ in range(N_tri)]
    tri_edge_signs = [[] for _ in range(N_tri)]
    for e_idx in range(E):
        tri_to_edges[int(plus[e_idx])].append(e_idx);  tri_edge_signs[int(plus[e_idx])].append(1.0)
        tri_to_edges[int(minus[e_idx])].append(e_idx); tri_edge_signs[int(minus[e_idx])].append(-1.0)

    G_global = np.zeros((E, E), dtype=np.float64)

    for t in range(N_tri):
        edges_in_tri = tri_to_edges[t]
        if not edges_in_tri:
            continue
        signs_in_tri = tri_edge_signs[t]
        area = areas[t]
        l_local = lengths[edges_in_tri]

        rho_local = np.stack([
            rho_plus[e] if s > 0 else rho_minus[e]
            for e, s in zip(edges_in_tri, signs_in_tri)
        ])   # (n_local, 3, Nq)
        sign_local = np.array(signs_in_tri)   # (n_local,)

        # ∫_T rho_i . rho_j dS = 2*area * sum_q w_q rho_i(q).rho_j(q)
        dot_q  = np.einsum('ikq,jkq->ijq', rho_local, rho_local)
        dot_ij = 2.0 * area * np.einsum('ijq,q->ij', dot_q, w)

        prefactor = (np.outer(sign_local, sign_local)
                     * np.outer(l_local, l_local) / (4.0 * area ** 2))
        g_block = prefactor * dot_ij

        G_global[np.ix_(edges_in_tri, edges_in_tri)] += g_block

    return G_global
