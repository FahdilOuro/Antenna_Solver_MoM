"""
Regression tests for impedance_matrix_z (backend/utils/impmet.py).

These lock in the numeric output of the Z-matrix assembly against golden
values captured from the implementation BEFORE the performance refactor that
hoists per-triangle quadrature points and per-edge rho vectors out of the
O(E^2) (m,n) fill loop (see workflow.md, "impedance_matrix_z performance
refactor"). The refactor is a pure caching change -- every Z_mn entry is
still produced by the exact same formula, categorisation, and quadrature
nodes as before -- so the golden values must match to full float64
precision, not just within a loose tolerance.

Meshes mirror the 4-triangle and 6-triangle strips used during the original
Phase 0/6 validation (documented in `_diagnostics/FINDINGS.md`), rebuilt here
directly via the rwg1 pipeline so this test has no dependency on the
(now-deleted) one-off diagnostic scripts. Together the two meshes exercise
all four singularity categories: SELF (diagonal), EDGE/VERTEX (adjacent
triangles), and FAR (six-triangle mesh only, edges e0/e4).
"""

import numpy as np
import pytest

from backend.rwg.rwg1 import Points, Triangles
from backend.utils.impmet import impedance_matrix_z

EPSILON = 8.854e-12
MU      = 1.257e-6
C       = 1.0 / np.sqrt(EPSILON * MU)
FREQ    = 3e8   # 300 MHz, matches FINDINGS.md Phase 6 symmetry/PSD checks


def _build_rwg_objects(raw_points, raw_triangles):
    points_obj = Points(raw_points)
    tri_obj = Triangles(raw_triangles)
    tri_obj.calculate_triangles_area_and_center(points_obj)
    edges_obj = tri_obj.get_edges()
    edges_obj.compute_edges_length(points_obj)
    return points_obj, tri_obj, edges_obj


def _four_triangle_mesh():
    """2x1 strip, 3 interior edges (SELF + EDGE/VERTEX only, no far pairs)."""
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


def _six_triangle_mesh():
    """1x3 strip, 5 interior edges; e0 and e4 are genuinely FAR."""
    points = np.array([
        [0., 1., 2., 3., 0., 1., 2., 3.],
        [0., 0., 0., 0., 1., 1., 1., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
    ])
    triangles = np.array([
        [0, 1, 1, 2, 2, 3],
        [1, 5, 2, 6, 3, 7],
        [4, 4, 5, 5, 6, 6],
        [0, 0, 0, 0, 0, 0],
    ], dtype=int)
    return points, triangles


def _compute_z(raw_points, raw_triangles, freq=FREQ):
    points_obj, tri_obj, edges_obj = _build_rwg_objects(raw_points, raw_triangles)
    omega = 2.0 * np.pi * freq
    k_complex = 1j * omega / C
    return impedance_matrix_z(edges_obj, tri_obj, points_obj, k_complex, omega, MU, EPSILON,
                               quad_order=6, duffy_order=7)


# ---------------------------------------------------------------------------
# Golden values: impedance_matrix_z @ 300 MHz, quad_order=6, duffy_order=7,
# captured BEFORE the per-triangle/per-edge precompute refactor.
# ---------------------------------------------------------------------------
Z_FOUR_TRI_GOLDEN = np.array([
    [75.57562510757265 + 17.415815648340185j, 10.172190850095806 + 0.9895129808934797j, 3.0392857114657668 - 1.7353987179218069j],
    [10.181807330981787 + 0.9662163843527649j, 115.93990772646151 + 18.197526469951796j, 10.181807376636968 + 1.1520366317529087j],
    [3.039285711465751 - 1.7353987179218158j, 10.172190895751001 + 1.1753332282936082j, 75.57562510757262 + 17.415815648340207j],
])

Z_SIX_TRI_GOLDEN = np.array([
    [75.57562510757265 + 17.415815648340185j, 10.172190850095806 + 0.9895129808934797j, 3.0392857114657668 - 1.7353987179218069j, 0.3482414390369195 - 0.4792245382730227j, 0.33772705724887947 + 0.09352777316283323j],
    [10.181807330981787 + 0.9662163843527649j, 115.93990772646151 + 18.197526469951796j, 10.181807376636968 + 1.1520366317529087j, -2.3313016664587227 - 5.65266296138412j, 0.34824089207294895 - 0.4790767036959568j],
    [3.039285711465751 - 1.7353987179218158j, 10.172190895751001 + 1.1753332282936082j, 75.57562510757262 + 17.415815648340207j, 10.17219085009581 + 0.9895129808934735j, 3.039285711465773 - 1.7353987179218038j],
    [0.34824143903691596 - 0.47922453827301714j, -2.3313016664587156 - 5.6526629613841255j, 10.181807330981798 + 0.966216384352756j, 115.93990772646151 + 18.197526469951768j, 10.181807376636968 + 1.1520366317529032j],
    [0.33772705724888036 + 0.093527773162833j, 0.34824089207295206 - 0.4790767036959577j, 3.0392857114657583 - 1.7353987179218087j, 10.172190895750992 + 1.1753332282935913j, 75.57562510757263 + 17.415815648340157j],
])


@pytest.mark.parametrize("name,builder,golden", [
    ("four_tri", _four_triangle_mesh, Z_FOUR_TRI_GOLDEN),
    ("six_tri",  _six_triangle_mesh,  Z_SIX_TRI_GOLDEN),
])
def test_impedance_matrix_matches_golden(name, builder, golden):
    """impedance_matrix_z must not silently drift under future refactors."""
    raw_points, raw_triangles = builder()
    Z = _compute_z(raw_points, raw_triangles)
    np.testing.assert_allclose(
        Z, golden, rtol=1e-10, atol=1e-12,
        err_msg=f"{name}: impedance_matrix_z output drifted from the frozen golden reference",
    )


@pytest.mark.parametrize("builder", [_four_triangle_mesh, _six_triangle_mesh])
def test_impedance_matrix_is_finite(builder):
    Z = _compute_z(*builder())
    assert np.all(np.isfinite(Z))


@pytest.mark.parametrize("builder", [_four_triangle_mesh, _six_triangle_mesh])
def test_impedance_matrix_psd_and_symmetry(builder):
    """Acceptance targets from FINDINGS.md Phase 6: symmetric, PSD Re(Z)."""
    Z = _compute_z(*builder())
    R, X = np.real(Z), np.imag(Z)

    sym_R = np.linalg.norm(R - R.T) / np.linalg.norm(R)
    sym_X = np.linalg.norm(X - X.T) / np.linalg.norm(X)
    assert sym_R < 5e-4, f"||R-Rt||/||R|| = {sym_R:.2e} exceeds Phase 6 target"
    assert sym_X < 5e-3, f"||X-Xt||/||X|| = {sym_X:.2e} exceeds Phase 6 target"

    min_eig = np.linalg.eigvalsh((R + R.T) * 0.5).min()
    assert min_eig >= -1e-8 * np.abs(R).max(), f"min_eig(Re Z) = {min_eig:.4e} is negative (non-physical)"
