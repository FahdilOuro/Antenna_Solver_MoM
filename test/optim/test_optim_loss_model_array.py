"""
Step 5 building blocks: parallel_strip_array (box, ports), alpha_calibrated
(empirical loss factor), and G with alpha_cal from the reduced model against a
direct full solve.
"""

import numpy as np
import pytest

from solver.optim import ReducedMultiport
from solver.optim.farfield import RadiationOperator
from solver.optim.geometries import (
    BOX_WIDTH, BoxError, array_ports, parallel_strip_array,
)
from solver.optim.inner import optimize_loads
from solver.optim.loss_model import (
    ALPHA_UNCERTAINTY, CALIBRATION, MESH_RULE, alpha_band, alpha_calibrated,
)
from solver.utils.lossy_electric_conductor import calculate_normal_surface_impedance
from conftest import COPPER_SIGMA

DEG = np.pi / 180
PLUS_Y = (90 * DEG, 90 * DEG)


# --------------------------------------------------------------------------- geometry
def test_parallel_strip_array_ports_and_cuts():
    w = 4e-3
    m = parallel_strip_array([0.18, 0.18, 0.15], w, [0.035, 0.050], **MESH_RULE)
    assert set(m.cuts) == {"el1", "el2", "el3"}
    yc = m.params["y_centers"]
    assert np.allclose(np.diff(yc), [0.035, 0.050]) and abs(yc[0] + yc[-1]) < 1e-15   # centred
    for fed in range(3):
        ports = array_ports(m, fed)
        assert len(ports) == 3
        assert ports[0].name == f"el{fed + 1}"                     # fed element first
        for p in ports:
            assert p.n_edges == MESH_RULE["n_width"]
            assert abs(p.width(m.edges) - w) < 1e-12               # cut length = width
            assert np.all(p.signs == p.signs[0])
    # x = 0 is a grid line and every strip has the requested length
    pts = m.points.points
    for k, (c, L) in enumerate(zip(yc, m.params["lengths"])):
        on = np.abs(pts[1] - c) <= w / 2 + 1e-12
        assert abs(pts[0, on].max() - L / 2) < 1e-12 and abs(pts[0, on].min() + L / 2) < 1e-12


def test_parallel_strip_array_box_errors():
    w = 4e-3
    with pytest.raises(BoxError, match="length"):
        parallel_strip_array([0.181, 0.18], w, [0.05])
    d_max = BOX_WIDTH - w                                           # 116 mm for w = 4 mm
    parallel_strip_array([0.18, 0.18], w, [d_max], cell_length=0.03)  # exactly at the limit: OK
    with pytest.raises(BoxError, match="extent"):
        parallel_strip_array([0.18, 0.18], w, [d_max + 1e-4], cell_length=0.03)
    with pytest.raises(BoxError, match="extent"):
        parallel_strip_array([0.18] * 3, w, [0.06, 0.06], cell_length=0.03)
    with pytest.raises(ValueError, match="overlap"):
        parallel_strip_array([0.18, 0.18], w, [0.004])
    with pytest.raises(ValueError, match="spacings"):
        parallel_strip_array([0.18, 0.18], w, [0.03, 0.03])


# --------------------------------------------------------------------------- loss model
def test_alpha_calibrated_table_and_domain():
    for w_mm, r_cst, r_mom, a in CALIBRATION:
        assert alpha_calibrated(w_mm) == pytest.approx(a, abs=1e-12)
        assert r_cst / r_mom == pytest.approx(a, abs=0.006)        # the table is self-consistent
    assert alpha_calibrated(3.0) == pytest.approx(0.5 * (0.81 + 1.09))
    lo, mid, hi = alpha_band(4.0)
    assert (lo, mid, hi) == pytest.approx((1.09 * (1 - ALPHA_UNCERTAINTY), 1.09,
                                           1.09 * (1 + ALPHA_UNCERTAINTY)))
    for bad in (0.5, 0.99, 4.01, 6.0):
        with pytest.raises(ValueError, match="outside the calibrated domain"):
            alpha_calibrated(bad)


# --------------------------------------------------------------------------- G with alpha_cal
def test_gain_with_alpha_cal_matches_direct_solve():
    """Two strips (coarse mesh, the check is algebraic): G = 4 pi U / P_in from the
    reduced model with alpha_cal equals the same quantity from a direct solve of
    (Z + alpha Zs G + Z_gen a1 a1^T + jX a2 a2^T) I = a1 e."""
    f, w = 0.9e9, 4e-3
    m = parallel_strip_array([0.18, 0.18], w, [0.035], n_width=2, cell_length=0.0225)
    alpha = alpha_calibrated(w * 1e3)
    model = ReducedMultiport.from_mesh(m.points, m.triangles, m.edges, f, array_ports(m, 0),
                                       conductivity=COPPER_SIGMA, zs_scale=alpha, use_gpu=False)
    op = RadiationOperator(m.points, m.triangles, m.edges, f)
    res = optimize_loads(model, PLUS_Y, "phi", "G", operator=op, n_starts=6, seed=0)

    zs = calculate_normal_surface_impedance(f, COPPER_SIGMA)
    assert model.zs == pytest.approx(zs) and model.zs_scale == alpha
    a1, a2 = model.A[:, 0], model.A[:, 1]
    z_gen, e = 50.0, 1.0
    Zfull = (model.Z + alpha * zs * model.G + z_gen * np.outer(a1, a1)
             + 1j * res.x_opt[0] * np.outer(a2, a2))
    I = np.linalg.solve(Zfull, a1 * e)
    i1 = a1 @ I
    v1 = e - z_gen * i1
    p_in = 0.5 * np.real(v1 * np.conj(i1))
    u_phi = op.intensity(I, *PLUS_Y)[1][0]
    G_direct = 4 * np.pi * u_phi / p_in
    print(f"\n[alpha_cal = {alpha:.3f}] G reduced = {res.G:.10f}, G direct = {G_direct:.10f}, "
          f"X_opt = {res.x_opt[0]:+.3f} Ohm")
    assert abs(res.G - G_direct) / G_direct < 1e-10
