"""
Regression tests for impedance_matrix_z (solver/utils/impmet.py).

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

from solver.rwg.rwg1 import Points, Triangles
from solver.utils.impmet import impedance_matrix_z, _mirror_upper_triangle

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
# captured with full_fill=True.
#
# Refreshed twice so far, both times because the EDGE/VERTEX branch changed:
#
#   1. when it started extracting the 1/R singularity analytically instead of
#      leaving it to a plain Gauss rule (measured on an edge-adjacent pair:
#      +1.15% -> -0.008%);
#   2. when impmet.NEAR_ORDER_DEFAULT rose from 11 to 16 -- 11 being both the
#      old default AND the largest rule gauss_dunavant then offered, so the
#      branch had been sitting on the table's ceiling. Sweeping the new
#      degrees showed err_T falling cleanly from 5.2e-3 (degree 6) to 1.3e-4
#      (degree 24) on a CubeSat mesh, i.e. plain truncation error, and 16 is
#      the lowest degree under RECIPROCITY_TOL. See
#      solver/utils/quadrature/outer_order_study.py.
#
# Change 1 moved the imaginary parts ONLY, and that was worth stating because
# the extracted 1/R kernel is real while its prefactor 4*pref_Phi is purely
# imaginary, so it cannot reach Re(Z) at all. Change 2 is a different animal:
# it also refines the BOUNDED remainder (exp(-jkR)-1)/R, which does reach
# Re(Z). The real parts below therefore moved too -- but only by 7.1e-8
# relative, against 5.0e-3 for the imaginary parts, a ratio of ~7e4 that is
# itself the cleanest confirmation that the singular half still goes where the
# derivation says it goes.
#
# These arrays are captured with full_fill=True, asymmetric lower half and all,
# even though production fills only m <= n and mirrors. Freezing the symmetric
# output would throw away the very thing they exist to protect -- the record of
# what the quadrature actually produces, rather than what mirroring imposes on
# it. The test below asserts the two statements that are together stronger: the
# upper triangle is bit-identical to what is frozen here, and the production
# matrix is exactly that half mirrored. The gap between the lower halves below
# and their mirrored counterparts is the quadrature asymmetry itself, now
# 2.300e-04 on these meshes (it was 2.547e-04 at degree 11).
# ---------------------------------------------------------------------------
Z_FOUR_TRI_GOLDEN = np.array([
    [75.5756201191694+16.4035195737268j, 10.1721969875938+1.11859368684408j, 3.0392791970667-1.74218857367031j],
    [10.1818134684798+1.09721308525129j, 115.93989934812+18.2085643850392j, 10.1818134683893+1.21392669677967j],
    [3.03927919706669-1.74376803915401j, 10.1721969875033+1.23914998186851j, 75.5756201191694+16.4035195737269j],
])

Z_SIX_TRI_GOLDEN = np.array([
    [75.5756201191694+16.4035195737268j, 10.1721969875938+1.11859368684408j, 3.0392791970667-1.74218857367031j, 0.348241702875171-0.479117695005645j, 0.337727057248879+0.0935277731628332j],
    [10.1818134684798+1.09721308525129j, 115.93989934812+18.2085643850392j, 10.1818134683893+1.21392669677967j, -2.33130271727113-5.89255133814994j, 0.348241702699816-0.479113576321574j],
    [3.03927919706669-1.74376803915401j, 10.1721969875033+1.23914998186851j, 75.5756201191694+16.4035195737269j, 10.1721969875938+1.11859368684408j, 3.0392791970667-1.74218857367031j],
    [0.34824170287517-0.479123845503098j, -2.33130271727106-5.89309580973583j, 10.1818134684798+1.09721308525126j, 115.93989934812+18.2085643850392j, 10.1818134683893+1.21392669677966j],
    [0.33772705724888+0.093527773162833j, 0.348241702699813-0.479130420419322j, 3.03927919706671-1.743768039154j, 10.1721969875033+1.23914998186851j, 75.5756201191694+16.4035195737268j],
])


@pytest.mark.parametrize("name,builder,golden", [
    ("four_tri", _four_triangle_mesh, Z_FOUR_TRI_GOLDEN),
    ("six_tri",  _six_triangle_mesh,  Z_SIX_TRI_GOLDEN),
])
def test_impedance_matrix_matches_golden(name, builder, golden):
    """impedance_matrix_z must not silently drift under future refactors."""
    raw_points, raw_triangles = builder()
    Z = _compute_z(raw_points, raw_triangles)

    # The half that is actually computed must not have moved at all.
    iu = np.triu_indices(Z.shape[0])
    np.testing.assert_allclose(
        Z[iu], golden[iu], rtol=1e-10, atol=1e-12,
        err_msg=f"{name}: the computed (upper) half drifted from the frozen golden reference",
    )
    # And the full matrix must be that half, mirrored. np.triu first: the
    # golden still carries its historical lower half, which would otherwise be
    # added on top of the reflection.
    np.testing.assert_allclose(
        Z, _mirror_upper_triangle(np.triu(golden)), rtol=1e-10, atol=1e-12,
        err_msg=f"{name}: Z is not the mirror of its own upper triangle",
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

    # Phase 6 asked for 5e-4 / 5e-3 because the assembler computed both halves
    # independently and they agreed only to quadrature accuracy. It now fills
    # m <= n and mirrors, so symmetry is a property of the construction and the
    # only defensible target is exact equality -- anything else would mean the
    # mirroring is not actually being applied.
    assert np.array_equal(R, R.T), (
        f"Re(Z) not exactly symmetric: max|R-Rt| = {np.abs(R - R.T).max():.3e}")
    assert np.array_equal(X, X.T), (
        f"Im(Z) not exactly symmetric: max|X-Xt| = {np.abs(X - X.T).max():.3e}")

    min_eig = np.linalg.eigvalsh((R + R.T) * 0.5).min()
    assert min_eig >= -1e-8 * np.abs(R).max(), f"min_eig(Re Z) = {min_eig:.4e} is negative (non-physical)"
