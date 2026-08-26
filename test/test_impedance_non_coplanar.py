"""
End-to-end reciprocity guard on a NON-PLANAR mesh.

Every impedance-assembly test in this repo used to run on a flat strip. That is
a blind spot: the EDGE/VERTEX branch integrates the 1/R half of the kernel in
closed form over the source triangle, and the closed form's height term
vanishes identically when the observation point is coplanar with that triangle.
A version missing that term therefore passed every planar test while producing,
on a CubeSat model with panels at right angles to the chassis, an impedance
matrix with max|Z-Z^T|/max|Z| = 0.994 -- not reciprocal at all.

The mesh here is two plates meeting at 90 degrees along a shared edge: the
smallest geometry that exercises non-coplanar EDGE and VERTEX pairs. Reciprocity
is the right invariant to assert because the exact Galerkin MoM matrix is
complex-symmetric (Z = Z^T, NOT Hermitian) whatever the geometry, so any real
breakage shows up here regardless of mesh shape.
"""

import numpy as np
import pytest

from backend.rwg.rwg1 import Points, Triangles
from backend.utils.impmet import impedance_matrix_z

EPSILON = 8.854e-12
MU = 1.257e-6
C = 1.0 / np.sqrt(EPSILON * MU)
FREQ = 440e6


def _right_angle_junction_mesh():
    """
    Two strips meeting at 90 degrees along a shared edge.

    Row A lies on the seam (y=0, z=0); row B extends one strip out in +y at
    z=0, row C extends the other up in +z at y=0. Every edge along the seam is
    shared by exactly one triangle from each strip -- a clean bend, not a
    T-junction -- so those pairs are non-coplanar while the RWG pairing stays
    unambiguous.
    """
    a = 0.04
    points = np.array([
        [0.0, a, 2 * a, 0.0, a, 2 * a, 0.0, a, 2 * a],   # x
        [0.0, 0.0, 0.0, a,   a, a,     0.0, 0.0, 0.0],   # y: row B is out in +y
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, a,   a,   a],     # z: row C is up in +z
    ])
    triangles = np.array([
        [0, 1, 1, 2, 0, 1, 1, 2],
        [1, 4, 2, 5, 1, 7, 2, 8],
        [3, 3, 4, 4, 6, 6, 7, 7],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=int)

    p = Points(points)
    t = Triangles(triangles)
    t.calculate_triangles_area_and_center(p)
    e = t.get_edges()
    e.compute_edges_length(p)
    return p, t, e


@pytest.fixture(scope="module")
def junction_mesh():
    return _right_angle_junction_mesh()


@pytest.fixture(scope="module")
def junction_Z():
    p, t, e = _right_angle_junction_mesh()
    omega = 2.0 * np.pi * FREQ
    return impedance_matrix_z(e, t, p, 1j * omega / C, omega, MU, EPSILON)


def test_mesh_really_is_non_coplanar():
    """Guard the guard: if this mesh ever flattens, the test below stops testing anything."""
    p, _t, _e = _right_angle_junction_mesh()
    spans = p.points.max(axis=1) - p.points.min(axis=1)
    assert np.count_nonzero(spans > 1e-12) == 3, f"mesh is planar (spans {spans}), it must not be"


def test_impedance_matrix_is_exactly_symmetric(junction_Z):
    """
    Z = Z^T for a Galerkin MoM matrix, and the assembler now guarantees it by
    construction: it fills m <= n and mirrors. Exact equality is the only
    defensible target -- a nonzero value here means the mirroring is not
    reaching this code path at all.
    """
    assert np.array_equal(junction_Z, junction_Z.T), (
        f"max|Z-Z^T| = {np.abs(junction_Z - junction_Z.T).max():.3e}, expected exactly 0")


def test_the_two_halves_agree_when_computed_independently(junction_mesh):
    """
    The check above cannot fail even if the kernel is completely wrong, because
    mirroring imposes the answer. This one can, and it is the one that matters.

    Filling both halves independently makes the two quadrature schemes disagree
    by exactly their own error: the SELF and EDGE/VERTEX branches treat test and
    source triangles differently, so a small nonzero value is expected and
    healthy. This mesh currently reads 5.1e-5 (it was 1.3e-4 before impmet's
    NEAR_ORDER_DEFAULT rose from 11 to 16). A broken closed-form 1/R integral --
    one assuming the observation point is coplanar with the source triangle --
    reads ~1 here instead. That is the signal this test exists to preserve now
    that production runs are symmetric by construction and can no longer show it.

    The bound below is deliberately loose against that 5.1e-5: it is sized to
    separate "quadrature error" from "wrong kernel", two regimes four orders of
    magnitude apart, and a bound tightened onto the current measurement would
    fail on the next mesh someone points this at without anything being wrong.
    """
    p, t, e = junction_mesh
    omega = 2.0 * np.pi * FREQ
    Z_full = impedance_matrix_z(e, t, p, 1j * omega / C, omega, MU, EPSILON, full_fill=True)

    asymmetry = np.max(np.abs(Z_full - Z_full.T)) / np.max(np.abs(Z_full))
    assert asymmetry > 0, "full_fill=True still produced a symmetric matrix; it is not filling both halves"
    assert asymmetry < 5e-3, (
        f"max|Z-Z^T|/max|Z| = {asymmetry:.3e} on a 90-degree junction. A value near 1 "
        f"means the 1/R closed form is being evaluated as if the observation point were "
        f"coplanar with the source triangle."
    )


def test_real_part_stays_positive_semidefinite_across_a_right_angle(junction_Z):
    """R = Re(Z) is the radiation operator: I^H R I = 2*P_rad >= 0 for a lossless PEC."""
    R = (np.real(junction_Z) + np.real(junction_Z).T) * 0.5
    eig = np.linalg.eigvalsh(R)
    assert eig[0] >= -1e-8 * eig[-1], f"min eig(Re Z) = {eig[0]:.4e} vs max {eig[-1]:.4e}"
