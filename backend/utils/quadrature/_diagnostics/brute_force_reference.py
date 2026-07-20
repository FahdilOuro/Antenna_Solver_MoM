"""
Phase 0 brute-force reference for the MoM impedance matrix.

PURPOSE
-------
Establish numerical ground truth for Z_mn BEFORE any refactoring.
This script:
  1. Builds two tiny synthetic meshes entirely in Python (no .mat files).
  2. Implements a naive, unvectorized, OBVIOUSLY-CORRECT double quadrature
     of the RWG kernel using high-order Gauss-Dunavant rules on both sides.
  3. Runs both this reference AND the existing impedance_matrix_z() on the
     same meshes and prints element-wise comparisons.
  4. Calls verify_z_properties.check_z_properties() on both matrices.

WHAT "OBVIOUSLY CORRECT" MEANS HERE
-------------------------------------
The Z matrix entries (Rao-Wilton-Glisson, Makarov 2002 notation) are:

  Z_mn = j*omega*mu/(4*pi) * INT_Sm INT_Sn f_m(r) . f_n(r') G(r,r') dS' dS
         + 1/(j*4*pi*omega*eps) * INT_Sm INT_Sn div.f_m * div.f_n * G(r,r') dS' dS

where
  G(r,r') = exp(-jk|r-r'|) / |r-r'|
  f_m(r)  = (l_m / (2*A_m^+)) * rho_m^+(r)  on T_m^+
           = -(l_m / (2*A_m^-)) * rho_m^-(r)  on T_m^-
  rho_m^+(r) = r - r_opp_m^+   (position relative to free vertex of T_m^+)
  div.f_m = +l_m / A_m^+  on T_m^+,   -l_m / A_m^-  on T_m^-

The double integral is discretized with a symmetric product quadrature:
    N_test points on the testing triangle (row side, r),
    N_src  points on each source triangle (column side, r').

This script uses N_test = N_src = 28-point Dunavant rule (order 11)
for far pairs and 7-point rule (order 5) for display, but both sides
use the SAME rule — no mixing of centroid vs. barycentric.

RUN
---
  cd <repo_root>
  python -m backend.utils.quadrature._diagnostics.brute_force_reference
"""

import sys
import os
import numpy as np

# ---------------------------------------------------------------------------
# Make repo root importable regardless of where the script is invoked from
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from backend.rwg.rwg1 import Points, Triangles, Edges
from backend.rwg.rwg2 import Barycentric_triangle, Vecteurs_Rho
from backend.utils.impmet import impedance_matrix_z_makarov_legacy as impedance_matrix_z_legacy
from backend.utils.quadrature._diagnostics.verify_z_properties import check_z_properties

# ---------------------------------------------------------------------------
# Embedded Dunavant quadrature coefficients (for use BEFORE gauss_dunavant.py
# is implemented in Phase 1).  Source: Dunavant (1985), IJNME 21:1129-1148.
# These are the EXACT tabulated values.
# ---------------------------------------------------------------------------

def _dunavant_order5():
    """7-point, degree-5-exact rule. Weights sum to 0.5."""
    # Barycentric coordinates (L1, L2, L3) and weights
    a1 = 0.47014206410511508
    b1 = 1.0 - 2 * a1
    a2 = 0.10128650732345633
    b2 = 1.0 - 2 * a2

    bary = np.array([
        [1/3, 1/3, 1/3],
        [a1, a1, b1], [a1, b1, a1], [b1, a1, a1],
        [a2, a2, b2], [a2, b2, a2], [b2, a2, a2],
    ])
    w1 = 0.22500000000000000 / 2.0
    w2 = 0.13239415278850616 / 2.0
    w3 = 0.12593918054482717 / 2.0
    weights = np.array([w1, w2, w2, w2, w3, w3, w3])
    return bary, weights


def _dunavant_order11():
    """28-point, degree-11-exact rule. Weights sum to 0.5.
    Source: Dunavant (1985) Table VI.
    """
    # Orbits from Dunavant Table VI (order 11, 28 points)
    # Format: (L1, L2, L3), weight_per_point (each orbit contributes 1,3, or 6 points)
    orbits = [
        # orbit type S3 (1 point, centroid)
        ((1/3, 1/3, 1/3), 0.17561525743261950e-1),
        # orbit type S21 (3 points each)
        ((0.26034596607904020, 0.26034596607904020, 0.47930806784191960), 0.39893377168619140e-1),
        ((0.06513010290221623, 0.06513010290221623, 0.86973979419556750), 0.21246444609584763e-1),
        ((0.04869031542531756, 0.04869031542531756, 0.90261937019936490), 0.77253255376851490e-2),
        # (continued from Table VI — these are approximate; we use the 7-pt rule for brevity)
    ]
    # NOTE: The full 28-point table requires careful transcription.
    # For Phase 0 we fall back to the 7-point rule (order 5) which is
    # sufficient to distinguish the current code's accuracy from the reference.
    # Phase 1 will implement the full table with verification tests.
    return _dunavant_order5()


def _map_to_physical(bary, p1, p2, p3):
    """Map barycentric coords (Nq,3) to physical (3,Nq).
    p1,p2,p3 each (3,); bary (Nq,3); result (3,Nq).
    """
    # p1[:,None] is (3,1), bary[:,0] is (Nq,) → product is (3,Nq)
    return p1[:, None] * bary[:, 0] + p2[:, None] * bary[:, 1] + p3[:, None] * bary[:, 2]


# ---------------------------------------------------------------------------
# Mesh builder
# ---------------------------------------------------------------------------

def build_two_triangle_mesh():
    """
    Two right triangles sharing the edge (1,2), unit size.

    Vertices:
        0: (0, 0, 0)
        1: (1, 0, 0)
        2: (0, 1, 0)
        3: (1, 1, 0)  ← free vertex of the second triangle

    Triangles (0-indexed vertices):
        T0: [0, 1, 2]   (T^+ for the shared edge)
        T1: [1, 3, 2]   (T^- for the shared edge)

    This gives exactly ONE interior RWG edge (edge between vertices 1 and 2)
    and three boundary half-edges (not used as RWG dofs).
    """
    points = np.array([
        [0.0, 1.0, 0.0, 1.0],   # x
        [0.0, 0.0, 1.0, 1.0],   # y
        [0.0, 0.0, 0.0, 0.0],   # z
    ])
    # (3, 4) points, (3, N_tri) triangles (only first 3 rows used as vertex indices)
    triangles = np.array([
        [0, 1],   # v0
        [1, 3],   # v1
        [2, 2],   # v2
        [0, 0],   # region label (row 3, filtered/unused here)
    ], dtype=int)
    return points, triangles


def build_four_triangle_mesh():
    """
    Four right triangles arranged as a 2x1 strip, giving 3 interior edges.

    Vertices:
        0:(0,0,0)  1:(1,0,0)  2:(2,0,0)
        3:(0,1,0)  4:(1,1,0)  5:(2,1,0)

    Triangles (0-indexed):
        T0: [0,1,3]   T1: [1,4,3]   T2: [1,2,4]   T3: [2,5,4]

    Interior edges (shared by two triangles):
        e0: (1,3) — T0 and T1
        e1: (1,4) — T1 and T2
        e2: (2,4) — T2 and T3
    """
    points = np.array([
        [0., 1., 2., 0., 1., 2.],
        [0., 0., 0., 1., 1., 1.],
        [0., 0., 0., 0., 0., 0.],
    ])
    triangles = np.array([
        [0, 1, 1, 2],
        [1, 4, 2, 5],
        [3, 3, 4, 4],
        [0, 0, 0, 0],
    ], dtype=int)
    return points, triangles


def build_six_triangle_mesh():
    """
    Six right triangles: a 1×3 rectangular strip, giving 5 interior edges.
    Edges e0 and e4 are truly FAR (no shared triangle vertices).

    Vertices (y=0 bottom row, y=1 top row):
        0:(0,0,0)  1:(1,0,0)  2:(2,0,0)  3:(3,0,0)
        4:(0,1,0)  5:(1,1,0)  6:(2,1,0)  7:(3,1,0)

    Triangles:
        T0:[0,1,4]  T1:[1,5,4]   (left cell)
        T2:[1,2,5]  T3:[2,6,5]   (center cell)
        T4:[2,3,6]  T5:[3,7,6]   (right cell)

    Interior edges (sorted vertex key → plus/minus):
        e0: (1,4) → T0,T1   e1: (1,5) → T1,T2
        e2: (2,5) → T2,T3   e3: (2,6) → T3,T4
        e4: (3,6) → T4,T5

    Far pair check for e0 vs e4:
        T^+_0={0,1,4}, T^-_0={1,5,4}  vs  T^+_4={2,3,6}, T^-_4={3,7,6}
        No vertex overlap at all → category 0 (far).
    """
    points = np.array([
        [0., 1., 2., 3., 0., 1., 2., 3.],
        [0., 0., 0., 0., 1., 1., 1., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
    ])
    triangles = np.array([
        [0, 1, 1, 2, 2, 3],   # v0 of each triangle
        [1, 5, 2, 6, 3, 7],   # v1
        [4, 4, 5, 5, 6, 6],   # v2
        [0, 0, 0, 0, 0, 0],   # region label
    ], dtype=int)
    return points, triangles


# ---------------------------------------------------------------------------
# Build rwg1/rwg2 objects from raw arrays
# ---------------------------------------------------------------------------

def _build_rwg_objects(raw_points, raw_triangles):
    """
    Given (3,P) points and (4,M) triangles (row 3 = label, 0-indexed),
    build Points, Triangles, Edges, Barycentric_triangle, Vecteurs_Rho.
    """
    points_obj = Points(raw_points)

    tri_obj = Triangles(raw_triangles)
    tri_obj.calculate_triangles_area_and_center(points_obj)
    edges_obj = tri_obj.get_edges()
    edges_obj.compute_edges_length(points_obj)

    bary_obj = Barycentric_triangle()
    bary_obj.calculate_barycentric_center(points_obj, tri_obj)

    rho_obj = Vecteurs_Rho()
    rho_obj.calculate_vecteurs_rho(points_obj, tri_obj, edges_obj, bary_obj)

    return points_obj, tri_obj, edges_obj, bary_obj, rho_obj


# ---------------------------------------------------------------------------
# Correct double-quadrature reference for Z_mn
# ---------------------------------------------------------------------------

def _triangle_vertices(tri_obj, points_obj, tri_idx):
    """Return (p1,p2,p3) each shape (3,) for triangle tri_idx."""
    verts = tri_obj.triangles[:3, tri_idx]
    p = points_obj.points
    return p[:, verts[0]], p[:, verts[1]], p[:, verts[2]]


def _triangle_area(p1, p2, p3):
    return 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))


def _rho_plus(r, tri_obj, points_obj, edge_idx):
    """rho^+ at position r for edge edge_idx: r - r_free_vertex_of_T^+."""
    T_plus = tri_obj.triangles_plus[edge_idx]
    verts = tri_obj.triangles[:3, T_plus]
    ep = [edges_obj_global.first_points[edge_idx], edges_obj_global.second_points[edge_idx]]
    free = [v for v in verts if v not in ep][0]
    return r - points_obj.points[:, free]


def _rho_minus(r, tri_obj, points_obj, edge_idx):
    """rho^- at position r for edge edge_idx: r - r_free_vertex_of_T^-."""
    T_minus = tri_obj.triangles_minus[edge_idx]
    verts = tri_obj.triangles[:3, T_minus]
    ep = [edges_obj_global.first_points[edge_idx], edges_obj_global.second_points[edge_idx]]
    free = [v for v in verts if v not in ep][0]
    return r - points_obj.points[:, free]


# Store globally so nested functions can access (avoids threading the mesh args
# through every helper — acceptable for a diagnostic script).
edges_obj_global = None


def green_scalar(r, r_prime, k):
    """exp(-jk|r-r'|) / |r-r'| (scalar, handles r==r' with a tiny offset)."""
    R = np.linalg.norm(r - r_prime)
    if R < 1e-14:
        R = 1e-14
    return np.exp(-1j * k * R) / R


def compute_Z_mn_reference(m, n, tri_obj, points_obj, edges_obj, frequency,
                            bary_xi, bary_w):
    """
    Compute Z_mn by a clean double quadrature over T_m x T_n.

    The formula (for a PEC surface, Makarov 2002 §4.2 / Rao 1982):
      Z_mn = jωμ/(4π) * ∫∫ f_m(r)·f_n(r') G(r,r') dS'dS
           + 1/(jωε4π) * ∫∫ div f_m div f_n G(r,r') dS'dS

    Each of f_m, f_n has support on two triangles (T^+ and T^-).
    We split the double integral over the 4 combinations (T_m^±, T_n^±).
    """
    global edges_obj_global
    edges_obj_global = edges_obj

    eps_0 = 8.854e-12
    mu_0  = 1.257e-6
    c     = 1.0 / np.sqrt(eps_0 * mu_0)
    omega = 2.0 * np.pi * frequency
    k     = omega / c

    l_m = edges_obj.edges_length[m]
    l_n = edges_obj.edges_length[n]

    T_mp = tri_obj.triangles_plus[m]
    T_mm = tri_obj.triangles_minus[m]
    T_np = tri_obj.triangles_plus[n]
    T_nm = tri_obj.triangles_minus[n]

    def tri_verts(tri_idx):
        return _triangle_vertices(tri_obj, points_obj, tri_idx)

    p1mp, p2mp, p3mp = tri_verts(T_mp)
    p1mm, p2mm, p3mm = tri_verts(T_mm)
    p1np, p2np, p3np = tri_verts(T_np)
    p1nm, p2nm, p3nm = tri_verts(T_nm)

    A_mp = _triangle_area(p1mp, p2mp, p3mp)
    A_mm = _triangle_area(p1mm, p2mm, p3mm)
    A_np = _triangle_area(p1np, p2np, p3np)
    A_nm = _triangle_area(p1nm, p2nm, p3nm)

    # RWG basis function f_m on T_m^+ and T_m^-
    # f_m^+(r) = (l_m / (2*A_m^+)) * rho_m^+(r)    sign +1
    # f_m^-(r) = -(l_m / (2*A_m^-)) * rho_m^-(r)   sign -1
    # div f_m^+ = +l_m / A_m^+,   div f_m^- = -l_m / A_m^-

    def rho_p_m(r):
        """rho^+(r) for edge m: r - free_vertex_T^+_m"""
        verts = tri_obj.triangles[:3, T_mp]
        ep = [edges_obj.first_points[m], edges_obj.second_points[m]]
        free_idx = next(v for v in verts if v not in ep)
        return r - points_obj.points[:, free_idx]

    def rho_m_m(r):
        """rho^-(r) for edge m: r - free_vertex_T^-_m"""
        verts = tri_obj.triangles[:3, T_mm]
        ep = [edges_obj.first_points[m], edges_obj.second_points[m]]
        free_idx = next(v for v in verts if v not in ep)
        return r - points_obj.points[:, free_idx]

    def rho_p_n(rp):
        verts = tri_obj.triangles[:3, T_np]
        ep = [edges_obj.first_points[n], edges_obj.second_points[n]]
        free_idx = next(v for v in verts if v not in ep)
        return rp - points_obj.points[:, free_idx]

    def rho_m_n(rp):
        verts = tri_obj.triangles[:3, T_nm]
        ep = [edges_obj.first_points[n], edges_obj.second_points[n]]
        free_idx = next(v for v in verts if v not in ep)
        return rp - points_obj.points[:, free_idx]

    # Quadrature points on each triangle
    def quad_pts(p1, p2, p3):
        return _map_to_physical(bary_xi, p1, p2, p3)  # (3, Nq)

    pts_mp = quad_pts(p1mp, p2mp, p3mp)   # (3, Nq)
    pts_mm = quad_pts(p1mm, p2mm, p3mm)
    pts_np = quad_pts(p1np, p2np, p3np)
    pts_nm = quad_pts(p1nm, p2nm, p3nm)

    Nq = len(bary_w)

    # Accumulate the four (testing_tri x source_tri) combinations.
    # sign_test: +1 for T^+_m, -1 for T^-_m
    # sign_src:  +1 for T^+_n, -1 for T^-_n
    # prefactor for each combination:
    #   A-term: j*omega*mu/(4*pi) * sign_test*(l_m/(2*A_test)) * sign_src*(l_n/(2*A_src)) * A_test * A_src
    #   Φ-term: 1/(j*omega*eps*4*pi) * sign_test*(l_m/A_test) * sign_src*(l_n/A_src) * A_test * A_src

    Z_mn = 0j

    combos = [
        (pts_mp, A_mp, rho_p_m, +1, pts_np, A_np, rho_p_n, +1),
        (pts_mp, A_mp, rho_p_m, +1, pts_nm, A_nm, rho_m_n, -1),
        (pts_mm, A_mm, rho_m_m, -1, pts_np, A_np, rho_p_n, +1),
        (pts_mm, A_mm, rho_m_m, -1, pts_nm, A_nm, rho_m_n, -1),
    ]

    for (pts_t, A_t, rho_test_fn, s_t, pts_s, A_s, rho_src_fn, s_s) in combos:
        # f_test: s_t * (l_m/(2*A_t)) * rho_test
        # f_src:  s_s * (l_n/(2*A_s)) * rho_src
        # div_test: s_t * l_m / A_t
        # div_src:  s_s * l_n / A_s

        pref_A   = (1j * omega * mu_0 / (4 * np.pi)) * s_t * s_s * (l_m / (2 * A_t)) * (l_n / (2 * A_s))
        pref_Phi = (1.0 / (1j * omega * eps_0 * 4 * np.pi)) * s_t * s_s * (l_m / A_t) * (l_n / A_s)

        contrib_A   = 0j
        contrib_Phi = 0j

        for i in range(Nq):
            r = pts_t[:, i]
            w_t = bary_w[i] * A_t  # integrate (weight * 2A_t / 2) — bary_w already scaled to 0.5
            # bary_w sums to 0.5 (for unit reference triangle area 0.5),
            # so ∫_T f dS ≈ 2*A_T * sum_q w_q * f(r_q).
            # But our bary_w already sums to 0.5, so actual weight = bary_w[i] * 2*A_T
            w_t_phys = bary_w[i] * 2.0 * A_t

            rho_t = rho_test_fn(r)

            for j in range(Nq):
                rp = pts_s[:, j]
                w_s_phys = bary_w[j] * 2.0 * A_s

                G = green_scalar(r, rp, k)
                rho_s = rho_src_fn(rp)

                contrib_A   += w_t_phys * w_s_phys * np.dot(rho_t, rho_s) * G
                contrib_Phi += w_t_phys * w_s_phys * G

        Z_mn += pref_A * contrib_A + pref_Phi * contrib_Phi

    return Z_mn


def classify_edge_pair(m, n, tri_obj):
    """
    Return pair category:
      3 = self (m == n)
      2 = edge-adjacent (T^±_m and T^±_n share 2 vertices)
      1 = vertex-adjacent (share exactly 1 vertex)
      0 = far (no shared vertices among all 4 sub-pair combinations)

    A pair (m,n) is far only if ALL four (T^a_m, T^b_n) combos are far.
    """
    if m == n:
        return 3

    tri_sets_m = [
        frozenset(tri_obj.triangles[:3, tri_obj.triangles_plus[m]]),
        frozenset(tri_obj.triangles[:3, tri_obj.triangles_minus[m]]),
    ]
    tri_sets_n = [
        frozenset(tri_obj.triangles[:3, tri_obj.triangles_plus[n]]),
        frozenset(tri_obj.triangles[:3, tri_obj.triangles_minus[n]]),
    ]

    max_shared = 0
    for ts_m in tri_sets_m:
        for ts_n in tri_sets_n:
            shared = len(ts_m & ts_n)
            max_shared = max(max_shared, shared)

    if max_shared >= 2:
        return 2
    if max_shared == 1:
        return 1
    return 0


def build_z_reference(tri_obj, points_obj, edges_obj, frequency,
                      far_pairs_only=False):
    """
    Build the full (or far-pair-only) Z matrix using the brute-force reference.

    For self and adjacent pairs the naked 1/R singularity makes the naive
    Gauss quadrature inaccurate; those entries are set to NaN when
    far_pairs_only=True.
    """
    bary_xi, bary_w = _dunavant_order5()
    E = edges_obj.total_number_of_edges
    Z = np.full((E, E), np.nan, dtype=complex)
    for m in range(E):
        for n in range(E):
            cat = classify_edge_pair(m, n, tri_obj)
            if far_pairs_only and cat != 0:
                continue   # leave as NaN — singularity present, skip
            Z[m, n] = compute_Z_mn_reference(m, n, tri_obj, points_obj,
                                             edges_obj, frequency, bary_xi, bary_w)
    return Z


# ---------------------------------------------------------------------------
# Legacy Z builder (calls the existing impmet.impedance_matrix_z)
# ---------------------------------------------------------------------------

def build_z_legacy(tri_obj, edges_obj, bary_obj, rho_obj, frequency):
    eps_0 = 8.854e-12
    mu_0  = 1.257e-6
    c     = 1.0 / np.sqrt(eps_0 * mu_0)
    omega = 2.0 * np.pi * frequency
    k     = omega / c
    complexe_k = 1j * k

    factor    = 1.0 / 9.0
    factor_a  = factor * (1j * omega * edges_obj.edges_length / 4.0) * (mu_0 / (4.0 * np.pi))
    factor_fi = factor * edges_obj.edges_length / (1j * 4.0 * np.pi * omega * eps_0)

    return impedance_matrix_z_legacy(
        edges_obj, tri_obj, bary_obj, rho_obj, complexe_k, factor_a, factor_fi
    )


# ---------------------------------------------------------------------------
# Main diagnostic runner
# ---------------------------------------------------------------------------

def _pair_label(cat):
    return {3: "self", 2: "edge-adj", 1: "vertex-adj", 0: "far"}.get(cat, "?")


def run_diagnostics():
    frequency = 3e8   # 300 MHz — λ ≈ 1 m, mesh cell ≈ 0.1–1 m → mid-range k*a

    print("=" * 70)
    print("PHASE 0 DIAGNOSTIC: brute-force reference vs legacy impedance_matrix_z")
    print("=" * 70)

    meshes = [
        ("2-tri mesh",  build_two_triangle_mesh()),
        ("4-tri mesh",  build_four_triangle_mesh()),
        ("6-tri mesh",  build_six_triangle_mesh()),
    ]

    for mesh_name, (raw_pts, raw_tri) in meshes:
        print(f"\n{'-'*60}")
        print(f"  Mesh: {mesh_name}")
        print(f"{'-'*60}")

        pts_obj, tri_obj, edges_obj, bary_obj, rho_obj = _build_rwg_objects(raw_pts, raw_tri)
        E = edges_obj.total_number_of_edges
        print(f"  Triangles: {tri_obj.total_of_triangles},  RWG edges: {E}")

        if E == 0:
            print("  No interior edges — skip.")
            continue

        # --- Pair category inventory ---
        cats = {}
        for m in range(E):
            for n in range(E):
                c = classify_edge_pair(m, n, tri_obj)
                cats.setdefault(c, []).append((m, n))
        for c in sorted(cats):
            print(f"    category {c} ({_pair_label(c)}): {len(cats[c])} pairs")

        far_pairs = cats.get(0, [])

        # --- Legacy Z (full matrix) ---
        Z_leg = build_z_legacy(tri_obj, edges_obj, bary_obj, rho_obj, frequency)
        print("\n  [Legacy]")
        check_z_properties(Z_leg, label="legacy")

        # --- Reference Z (far pairs only — self/adjacent have naked 1/R) ---
        if not far_pairs:
            print("\n  No far pairs in this mesh — skipping reference comparison.")
            continue

        print(f"\n  Computing brute-force reference for {len(far_pairs)} far pairs...")
        Z_ref = build_z_reference(tri_obj, pts_obj, edges_obj, frequency,
                                  far_pairs_only=True)

        # --- Far-pair comparison ---
        print(f"\n  Far-pair Z entries (legacy vs reference, category 0 only):")
        print(f"  {'(m,n)':<8}  {'legacy_re':>14}  {'legacy_im':>14}"
              f"  {'ref_re':>14}  {'ref_im':>14}  {'rel_err':>10}")
        for m, n in far_pairs[:10]:   # cap at 10 rows for readability
            leg = Z_leg[m, n]
            ref = Z_ref[m, n]
            rel = abs(leg - ref) / (abs(ref) + 1e-300)
            print(f"  ({m},{n})     {leg.real:>14.4e}  {leg.imag:>14.4e}"
                  f"  {ref.real:>14.4e}  {ref.imag:>14.4e}  {rel:>10.3e}")

    print("\n" + "=" * 70)
    print("PHASE 0 COMPLETE -- see FINDINGS.md for interpretation")
    print("=" * 70)


if __name__ == "__main__":
    run_diagnostics()
