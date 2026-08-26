"""
Tests for the settable outer quadrature orders of the two singular branches:
`near_order` (EDGE/VERTEX) and `self_outer_order` (SELF) of
backend/utils/impmet.py.

Two things need protecting here, and the existing regression suites cover
neither, because every one of them calls the assemblers at their DEFAULTS:

  1. The three assemblers -- impedance_matrix_z, impedance_matrix_z_sweep and
     impedance_matrix_z_sweep_gpu -- must agree at NON-default orders too. Each
     builds its own quadrature tables and its own SELF loop, so a parameter
     threaded through two of them and forgotten in the third would leave the
     existing cross-checks green (they only ever compare defaults) while a
     frequency sweep silently integrated at a different order than the
     single-frequency solve it is supposed to reproduce.

  2. Raising the order must actually change the matrix, and only where it can.
     The 1/R kernel extracted in closed form on EDGE/VERTEX pairs is real and
     its prefactor 4*pref_Phi is purely imaginary, so refining `near_order`
     moves Im(Z) and leaves Re(Z) alone up to the bounded remainder. A change
     that moved Re(Z) by as much as Im(Z) would mean the extraction is no
     longer doing what its derivation says.

See backend/utils/quadrature/outer_order_study.py for the convergence
measurements that motivated making these settable.
"""

import numpy as np
import pytest

from backend.rwg.rwg1 import Points, Triangles
from backend.utils.compute_backend import GPU_AVAILABLE
from backend.utils.impmet import (
    _default_near_order, impedance_matrix_z, impedance_matrix_z_sweep,
    impedance_matrix_z_sweep_gpu,
)

EPSILON = 8.854e-12
MU = 1.257e-6
C = 1.0 / np.sqrt(EPSILON * MU)
FREQ = 440e6


def _junction_mesh():
    """Two strips meeting at 90 degrees -- non-coplanar EDGE and VERTEX pairs.

    A flat mesh would not do: the closed-form 1/R integral's height term
    vanishes identically for a coplanar observation point, so the part of the
    EDGE/VERTEX branch that `near_order` most affects would go untested.
    """
    a = 0.04
    points = np.array([
        [0.0, a, 2 * a, 0.0, a, 2 * a, 0.0, a, 2 * a],
        [0.0, 0.0, 0.0, a,   a,   a,   0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, a,   a,   a],
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
def mesh():
    return _junction_mesh()


def _single(mesh, **kw):
    p, t, e = mesh
    omega = 2.0 * np.pi * FREQ
    return impedance_matrix_z(e, t, p, 1j * omega / C, omega, MU, EPSILON, **kw)


# ---------------------------------------------------------------------------
# The defaults must stay exactly what they were before the orders became settable
# ---------------------------------------------------------------------------

def test_explicit_defaults_reproduce_implicit_ones(mesh):
    """Passing the documented defaults explicitly must be a no-op, bit for bit."""
    implicit = _single(mesh, quad_order=6, duffy_order=7)
    explicit = _single(mesh, quad_order=6, duffy_order=7,
                       near_order=_default_near_order(6), self_outer_order=6)
    assert np.array_equal(implicit, explicit)


# ---------------------------------------------------------------------------
# All three assemblers must honour the parameters identically
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("near_order,self_outer_order", [(11, 6), (16, 6), (16, 7), (7, 4)])
def test_sweep_matches_single_frequency(mesh, near_order, self_outer_order):
    p, t, e = mesh
    omega = 2.0 * np.pi * FREQ
    Z_sweep = impedance_matrix_z_sweep(e, t, p, [FREQ], MU, EPSILON,
                                       near_order=near_order, self_outer_order=self_outer_order)
    Z_single = impedance_matrix_z(e, t, p, 1j * omega / C, omega, MU, EPSILON,
                                  near_order=near_order, self_outer_order=self_outer_order)
    np.testing.assert_allclose(
        Z_sweep[0], Z_single, rtol=1e-10, atol=1e-12,
        err_msg=f"sweep disagrees with the single-frequency assembler at "
                f"near_order={near_order}, self_outer_order={self_outer_order}")


@pytest.mark.parametrize("near_order,self_outer_order", [(16, 6), (7, 4)])
def test_batched_matches_sweep(mesh, near_order, self_outer_order):
    p, t, e = mesh
    kw = dict(near_order=near_order, self_outer_order=self_outer_order)
    Z_ref = impedance_matrix_z_sweep(e, t, p, [FREQ], MU, EPSILON, **kw)
    Z_bat = impedance_matrix_z_sweep_gpu(e, t, p, [FREQ], MU, EPSILON, use_gpu=False, **kw)
    np.testing.assert_allclose(
        Z_bat, Z_ref, rtol=1e-10, atol=1e-12,
        err_msg=f"batched assembly disagrees with the per-pair sweep at "
                f"near_order={near_order}, self_outer_order={self_outer_order}")


@pytest.mark.skipif(not GPU_AVAILABLE, reason="no usable CUDA GPU on this machine")
def test_batched_gpu_matches_sweep(mesh):
    p, t, e = mesh
    kw = dict(near_order=16, self_outer_order=6)
    Z_ref = impedance_matrix_z_sweep(e, t, p, [FREQ], MU, EPSILON, **kw)
    Z_gpu = impedance_matrix_z_sweep_gpu(e, t, p, [FREQ], MU, EPSILON, use_gpu=True, **kw)
    np.testing.assert_allclose(Z_gpu, Z_ref, rtol=1e-8, atol=1e-12)


# ---------------------------------------------------------------------------
# The parameters must do what their derivation says they do
# ---------------------------------------------------------------------------

def test_raising_near_order_moves_the_imaginary_part_far_more_than_the_real_one(mesh):
    """
    The EDGE/VERTEX 1/R extraction contributes a REAL integral through a purely
    IMAGINARY prefactor, so refining its outer rule is felt in Im(Z). Re(Z) sees
    only the bounded remainder, which a lower order already resolves. If this
    ratio ever collapses towards 1, the singular half is leaking into Re(Z) and
    the extraction is not the operation it is documented to be.
    """
    Z_low = _single(mesh, near_order=11, self_outer_order=6)
    Z_high = _single(mesh, near_order=24, self_outer_order=6)

    d_re = np.abs(Z_high.real - Z_low.real).max() / np.abs(Z_low).max()
    d_im = np.abs(Z_high.imag - Z_low.imag).max() / np.abs(Z_low).max()

    assert d_im > 1e-6, f"near_order had almost no effect on Im(Z) (moved {d_im:.2e})"
    assert d_im > 50 * d_re, (
        f"refining near_order moved Re(Z) by {d_re:.2e} and Im(Z) by {d_im:.2e}; the "
        f"extracted 1/R kernel is real with an imaginary prefactor, so Re(Z) should "
        f"barely move")


def test_unsupported_order_is_rejected(mesh):
    """A typo'd order must fail loudly, not silently fall back to a default."""
    with pytest.raises(ValueError, match="Unsupported Dunavant order"):
        _single(mesh, near_order=13)
    with pytest.raises(ValueError, match="Unsupported Dunavant order"):
        _single(mesh, self_outer_order=2)
