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
# Golden values: impedance_matrix_z @ 300 MHz, quad_order=6, duffy_order=7.
#
# Refreshed when the EDGE/VERTEX branch started extracting the 1/R singularity
# analytically instead of leaving it to a plain Gauss rule (measured on an
# edge-adjacent pair: +1.15% -> -0.008%). Only the IMAGINARY parts moved: the
# extracted 1/R kernel is real and its prefactor 4*pref_Phi is purely
# imaginary, so Re(Z) is untouched -- and indeed the real parts below are
# bit-identical to the pre-extraction values, which is the cheapest available
# check that the change did what it claims and nothing else.
# ---------------------------------------------------------------------------
Z_FOUR_TRI_GOLDEN = np.array([
    [75.57562510757265+16.984890775899505j, 10.172190850095804+1.153095234427167j, 3.039285711465767-1.75289681320944j],
    [10.181807330981787+1.1347610608185281j, 115.9399077264615+18.178071372146384j, 10.181807376636964+1.3157510190323407j],
    [3.0392857114657508-1.756973534450701j, 10.172190895750997+1.3440540940642134j, 75.57562510757262+16.984890775899526j],
])

Z_SIX_TRI_GOLDEN = np.array([
    [75.57562510757265+16.984890775899505j, 10.172190850095804+1.153095234427167j, 3.039285711465767-1.75289681320944j, 0.3482414390369195-0.4791051384108471j, 0.33772705724887947+0.09352777316283323j],
    [10.181807330981787+1.1347610608185281j, 115.9399077264615+18.178071372146384j, 10.181807376636964+1.3157510190323407j, -2.3313016664587205-5.75974388718469j, 0.34824089207294895-0.47908943757957545j],
    [3.0392857114657508-1.756973534450701j, 10.172190895750997+1.3440540940642134j, 75.57562510757262+16.984890775899526j, 10.17219085009581+1.1530952344271537j, 3.0392857114657703-1.752896813209456j],
    [0.34824143903691596-0.479132247743582j, -2.3313016664587134-5.761125829906885j, 10.181807330981794+1.1347610608185321j, 115.93990772646151+18.178071372146363j, 10.181807376636968+1.3157510190323276j],
    [0.33772705724888036+0.093527773162833j, 0.34824089207295206-0.47916060247139425j, 3.0392857114657597-1.7569735344507063j, 10.172190895750992+1.3440540940641927j, 75.57562510757263+16.9848907758995j],
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
