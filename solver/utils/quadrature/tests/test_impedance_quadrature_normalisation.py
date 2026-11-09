"""
Guards on the quadrature-weight normalisation inside the impedance assembly
(solver/utils/impmet.py).

Why this file exists
--------------------
The SELF branch mixes two quadrature helpers that normalise DIFFERENTLY:
dunavant_rule weights sum to 1/2 (reference-triangle area, leaving the 2*A
mapping Jacobian to cancel against the RWG 1/(2A) factor), while
duffy_source_nodes weights sum to A (physical area, integral already
complete). Reconciling them is what impmet._duffy_weight_scale does.

A missing 1/(2*A_s) there scales every SELF contribution by 2*A_s, which
silently deletes them on a fine mesh, breaks the sum-of-signs cancellation
that makes R = Re(Z) the low-rank PSD radiation operator, and turns any
Characteristic Mode Analysis into noise.

That bug was invisible to every pre-existing test: the synthetic meshes in
test_impedance_matrix_regression.py are built on unit squares, so every
triangle has area exactly 0.5 and the missing factor 1/(2*A) equals exactly
1.0. The tests below deliberately use a mesh whose triangles do NOT have
area 1/2, and pin the physical invariant (scale invariance) rather than a
golden number, so the same class of bug cannot come back unnoticed.
"""

import numpy as np
import pytest

from solver.rwg.rwg1 import Points, Triangles
from solver.utils.impmet import impedance_matrix_z, _duffy_weight_scale
from solver.utils.quadrature.gauss_dunavant import dunavant_rule
from solver.utils.quadrature.duffy import duffy_source_nodes

EPSILON = 8.854e-12
MU = 1.257e-6
C = 1.0 / np.sqrt(EPSILON * MU)


def _strip_mesh(scale):
    """1x3 strip scaled by `scale`; triangle area = 0.5 * scale**2."""
    points = np.array([
        [0., 1., 2., 3., 0., 1., 2., 3.],
        [0., 0., 0., 0., 1., 1., 1., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
    ]) * scale
    triangles = np.array([
        [0, 1, 1, 2, 2, 3],
        [1, 5, 2, 6, 3, 7],
        [4, 4, 5, 5, 6, 6],
        [0, 0, 0, 0, 0, 0],
    ], dtype=int)
    p = Points(points)
    t = Triangles(triangles)
    t.calculate_triangles_area_and_center(p)
    e = t.get_edges()
    e.compute_edges_length(p)
    return p, t, e


def _real_part_spectrum(scale, freq):
    p, t, e = _strip_mesh(scale)
    omega = 2.0 * np.pi * freq
    Z = impedance_matrix_z(e, t, p, 1j * omega / C, omega, MU, EPSILON)
    R = (np.real(Z) + np.real(Z).T) * 0.5
    return np.linalg.eigvalsh(R)


def test_duffy_weights_rescale_to_the_dunavant_convention():
    """
    _duffy_weight_scale must map Duffy's physical-area weights onto the same
    normalisation the Dunavant weights use, since the SELF branch multiplies
    an outer Dunavant weight by an inner Duffy sum.
    """
    v = np.array([[0.0, 0.3, 0.05], [0.0, 0.0, 0.2], [0.0, 0.0, 0.0]]).T  # 3 vertices, columns
    p1, p2, p3 = v[:, 0], v[:, 1], v[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
    assert not np.isclose(area, 0.5), "test mesh must NOT have the degenerate area-1/2 that hides the bug"

    r_sing = (p1 + p2 + p3) / 3.0
    _pts, d_w = duffy_source_nodes(r_sing, p1, p2, p3, order=7)

    _xi, w_dunavant = dunavant_rule(6)
    scaled_sum = d_w.sum() * _duffy_weight_scale(area)

    np.testing.assert_allclose(scaled_sum, w_dunavant.sum(), rtol=1e-12)
    np.testing.assert_allclose(scaled_sum, 0.5, rtol=1e-12)


@pytest.mark.parametrize("scale", [0.1, 10.0])
def test_real_part_spectrum_is_scale_invariant(scale):
    """
    Shrinking the geometry by `scale` while raising the frequency by 1/scale
    is the SAME electrical problem, so the dimensionless shape of the R =
    Re(Z) spectrum (here min/max eigenvalue) must be unchanged.

    This is the invariant the missing 1/(2*A_s) broke: because the erroneous
    factor was 2*A_s, it depended on the absolute triangle size, so it
    perturbed the two geometrically-identical problems by different amounts.
    It is also exactly why the unit-square test meshes elsewhere in this
    directory could never expose it -- there 2*A = 1 and the bug is a no-op.
    """
    ref = _real_part_spectrum(1.0, 3e8)
    got = _real_part_spectrum(scale, 3e8 / scale)

    np.testing.assert_allclose(got[0] / got[-1], ref[0] / ref[-1], rtol=1e-6)


def test_real_part_is_positive_semidefinite_on_a_non_unit_mesh():
    """
    R = Re(Z) is the radiation operator: I^H R I = 2*P_rad >= 0 for a lossless
    PEC, so R must be positive semi-definite whatever the mesh scale.
    """
    ev = _real_part_spectrum(0.02, 1.5e9)
    assert ev[0] >= -1e-8 * ev[-1], f"min eig(Re Z) = {ev[0]:.4e} vs max {ev[-1]:.4e}: R is not PSD"
