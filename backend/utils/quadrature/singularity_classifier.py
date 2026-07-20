"""
Classify triangle pairs and edge pairs by their integration singularity type.

For the RWG MoM impedance matrix, Z_mn involves a double surface integral
over (T^+_m ∪ T^-_m) × (T^+_n ∪ T^-_n).  Each of the four sub-triangle-pairs
(T_test, T_source) requires a different integration strategy depending on whether
T_test and T_source share geometry:

    SELF   (3)  identical triangles             → Duffy self-term
    EDGE   (2)  two shared vertices (one edge)  → Duffy edge-adjacent
    VERTEX (1)  one shared vertex               → Duffy vertex-adjacent
    FAR    (0)  no shared vertices              → standard Gauss-Dunavant

Usage for Phase 4 impedance matrix assembly::

    from backend.utils.quadrature.singularity_classifier import (
        classify_triangle_pair, classify_all_triangle_pairs,
        classify_edge_pair_max, FAR, VERTEX, EDGE, SELF
    )

    # Per sub-pair lookup (preferred in Phase 4 inner loops)
    cat = classify_triangle_pair(tri[:, tp_m], tri[:, tp_n])

    # Pre-computed full matrix for small meshes / diagnostics
    cat_matrix = classify_all_triangle_pairs(tri_vertices)
    far_pairs = get_pairs_by_category(cat_matrix, FAR)
"""

import numpy as np

# ---------------------------------------------------------------------------
# Category constants (int8 to keep large matrices compact)
# ---------------------------------------------------------------------------
FAR    = np.int8(0)
VERTEX = np.int8(1)
EDGE   = np.int8(2)
SELF   = np.int8(3)


# ---------------------------------------------------------------------------
# Per-triangle-pair classifier
# ---------------------------------------------------------------------------

def classify_triangle_pair(verts_t: np.ndarray, verts_s: np.ndarray) -> np.int8:
    """
    Classify one triangle pair by the size of their shared vertex set.

    Parameters
    ----------
    verts_t : array-like, shape (3,)  vertex indices for the testing triangle
    verts_s : array-like, shape (3,)  vertex indices for the source triangle

    Returns
    -------
    np.int8  one of FAR (0), VERTEX (1), EDGE (2), SELF (3)
    """
    st = frozenset(int(v) for v in verts_t)
    ss = frozenset(int(v) for v in verts_s)
    n_shared = len(st & ss)
    if st == ss:
        return SELF
    elif n_shared == 2:
        return EDGE
    elif n_shared == 1:
        return VERTEX
    else:
        return FAR


# ---------------------------------------------------------------------------
# Full triangle-pair matrix (for small meshes and diagnostics)
# ---------------------------------------------------------------------------

def classify_all_triangle_pairs(tri_vertices: np.ndarray) -> np.ndarray:
    """
    Build the (N_tri, N_tri) singularity-category matrix for all triangle pairs.

    Parameters
    ----------
    tri_vertices : (3, N_tri) int array
        Each column holds the three vertex indices of one triangle.

    Returns
    -------
    cat : (N_tri, N_tri) int8 array
        cat[i, j] ∈ {FAR, VERTEX, EDGE, SELF}
    """
    N = tri_vertices.shape[1]
    # Pre-build frozensets to avoid repeated construction inside the double loop
    sets = [frozenset(int(v) for v in tri_vertices[:, i]) for i in range(N)]

    cat = np.empty((N, N), dtype=np.int8)
    for i in range(N):
        si = sets[i]
        for j in range(N):
            sj = sets[j]
            n_sh = len(si & sj)
            if i == j:                # same triangle index → self
                cat[i, j] = SELF
            elif si == sj:            # same vertex set (degenerate mesh) → self
                cat[i, j] = SELF
            elif n_sh == 2:
                cat[i, j] = EDGE
            elif n_sh == 1:
                cat[i, j] = VERTEX
            else:
                cat[i, j] = FAR
    return cat


# ---------------------------------------------------------------------------
# Edge-pair convenience functions (for Z-matrix assembly in Phase 4)
# ---------------------------------------------------------------------------

def classify_edge_pair_max(
    m: int,
    n: int,
    triangles_plus: np.ndarray,
    triangles_minus: np.ndarray,
    tri_vertices: np.ndarray,
) -> np.int8:
    """
    Return the MAXIMUM (worst) singularity category among the four sub-triangle
    pairs (T^+_m, T^+_n), (T^+_m, T^-_n), (T^-_m, T^+_n), (T^-_m, T^-_n).

    This answers: "does this edge pair need any Duffy treatment?"

    Parameters
    ----------
    m, n            : edge indices
    triangles_plus  : (E,) int array  — + triangle index for each edge
    triangles_minus : (E,) int array  — - triangle index for each edge
    tri_vertices    : (3, N_tri) int array

    Returns
    -------
    np.int8  maximum category across the four sub-pairs
    """
    tp_m, tm_m = int(triangles_plus[m]), int(triangles_minus[m])
    tp_n, tm_n = int(triangles_plus[n]), int(triangles_minus[n])

    if m == n:
        return SELF

    sub_pairs = [
        (tp_m, tp_n), (tp_m, tm_n),
        (tm_m, tp_n), (tm_m, tm_n),
    ]
    worst = FAR
    for ti, sj in sub_pairs:
        c = classify_triangle_pair(tri_vertices[:, ti], tri_vertices[:, sj])
        if c > worst:
            worst = c
            if worst == SELF:   # can't get worse
                return worst
    return worst


def classify_all_edge_pairs(
    triangles_plus: np.ndarray,
    triangles_minus: np.ndarray,
    tri_vertices: np.ndarray,
) -> np.ndarray:
    """
    Build the (E, E) edge-pair category matrix using the MAX-over-sub-pairs rule.

    Suitable for small meshes and for pre-computing which (m,n) pairs need Duffy.
    For large meshes use `classify_edge_pair_max` on demand inside the fill loop.

    Parameters
    ----------
    triangles_plus  : (E,) int
    triangles_minus : (E,) int
    tri_vertices    : (3, N_tri) int

    Returns
    -------
    cat : (E, E) int8
    """
    E = len(triangles_plus)
    # Pre-build frozensets for all triangles once
    N_tri = tri_vertices.shape[1]
    sets = [frozenset(int(v) for v in tri_vertices[:, t]) for t in range(N_tri)]

    cat = np.empty((E, E), dtype=np.int8)
    for m in range(E):
        tp_m = int(triangles_plus[m])
        tm_m = int(triangles_minus[m])
        for n in range(E):
            if m == n:
                cat[m, n] = SELF
                continue
            tp_n = int(triangles_plus[n])
            tm_n = int(triangles_minus[n])
            worst = FAR
            for ti, sj in [(tp_m, tp_n), (tp_m, tm_n), (tm_m, tp_n), (tm_m, tm_n)]:
                n_sh = len(sets[ti] & sets[sj])
                if sets[ti] == sets[sj]:
                    c = SELF
                elif n_sh == 2:
                    c = EDGE
                elif n_sh == 1:
                    c = VERTEX
                else:
                    c = FAR
                if c > worst:
                    worst = c
                    if worst == SELF:
                        break
            cat[m, n] = worst
    return cat


# ---------------------------------------------------------------------------
# Query helper
# ---------------------------------------------------------------------------

def get_pairs_by_category(
    cat_matrix: np.ndarray,
    value: int,
) -> list:
    """
    Return a list of (i, j) index pairs where cat_matrix[i, j] == value.

    Parameters
    ----------
    cat_matrix : (M, M) int8 array returned by classify_all_triangle_pairs
                 or classify_all_edge_pairs
    value      : one of FAR, VERTEX, EDGE, SELF

    Returns
    -------
    list of (int, int) tuples
    """
    rows, cols = np.where(cat_matrix == value)
    return list(zip(rows.tolist(), cols.tolist()))
