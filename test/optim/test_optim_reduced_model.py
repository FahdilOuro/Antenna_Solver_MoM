"""
Tests for solver.optim.reduced_model.ReducedMultiport.

  V3  test_reduction_exact       reduced model == full MoM solve with the loads in Z
  V1  test_power_balance         P_in = P_rad + P_cond + P_loads ; P_avail (1-|Gamma|^2) = P_in
  V4  test_admittance_symmetry   Y = Y^T (and the true reciprocity error with full_fill=True)

Run with  python -m pytest test/optim -v -s  to see the printed measurements.
"""

import time

import numpy as np
import pytest

from solver.optim import ReducedMultiport, port_from_edge
from solver.optim.threads import limit_blas_threads
from solver.rwg.rwg3 import calculate_z_matrice_lumped_elements, compute_lumped_impedance
from solver.utils.impmet import impedance_matrix_z
from solver.utils.lossy_electric_conductor import calculate_normal_surface_impedance
from conftest import FREQUENCY, X_HAT, rel_err

Z0 = 50.0
# Arbitrary complex terminations: generator Z0 at the feed, then an inductive
# lossy load and a capacitive lossy load.
Z_TERM = np.array([Z0, 12.5 + 40.0j, 3.0 - 75.0j])
EMF = np.array([0.7 * np.exp(0.3j), 0.0, 0.0])


def _rebuild(model, columns=None, Zs="same", zs_scale=None):
    """Same geometry/frequency, other port subset or loss model, reusing the stored Z and G."""
    A = model.A if columns is None else model.A[:, columns]
    return ReducedMultiport.from_matrices(
        model.Z, A, model.frequency, G=model.G,
        Zs=model.zs if Zs == "same" else Zs,
        zs_scale=model.zs_scale if zs_scale is None else zs_scale)


def _total_z(model):
    """Z_t = Z + alpha Zs G, assembled independently of the model's internals."""
    if model.zs is None:
        return model.Z.astype(complex)
    return model.Z + model.zs_scale * model.zs * model.G


# --------------------------------------------------------------------------- equivalence
def test_single_edge_equivalence(narrow_strip):
    """
    One-edge-wide strip: feed (generator Z0) + one RLC load, each on a single
    edge. Reference = rwg3.calculate_z_matrice_lumped_elements (Z_mm += l_m^2 Z_L
    for BOTH lumped elements, the generator's Z0 included) + direct solve with
    the Makarov RHS V_m = l_m e.
    """
    n = narrow_strip
    load_points = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    load_dirs = np.array([X_HAT, X_HAT])
    # (L, C, R) per element: generator = 50 Ohm, load = 20 nH + 3 pF + 5 Ohm in series.
    load_values = np.array([[0.0, None, Z0], [20e-9, 3e-12, 5.0]], dtype=object)

    feed = port_from_edge(n.points, n.triangles, n.edges, load_points[0], X_HAT, "feed")
    load = port_from_edge(n.points, n.triangles, n.edges, load_points[1], X_HAT, "load")
    assert feed.signs[0] == 1.0 and load.signs[0] == 1.0

    model = ReducedMultiport.from_mesh(n.points, n.triangles, n.edges, FREQUENCY, [feed, load],
                                       use_gpu=False)

    omega, *_, Z_lumped, imp_array = calculate_z_matrice_lumped_elements(
        n.points, n.triangles, n.edges, FREQUENCY, load_points, load_values, load_dirs,
        use_gpu=False)
    z_term = compute_lumped_impedance(load_values.T, omega)

    # Same edges selected by both rules.
    np.testing.assert_array_equal(imp_array, [feed.edge_indices[0], load.edge_indices[0]])

    e = 1.0 + 0.0j
    m = int(feed.edge_indices[0])
    l_m = n.edges.edges_length[m]
    V = np.zeros(n.edges.total_number_of_edges, dtype=complex)
    V[m] = l_m * e
    I_ref = np.linalg.solve(Z_lumped, V)
    i_ref = l_m * I_ref[m]
    zin_ref = (e - z_term[0] * i_ref) / i_ref

    sol = model.solve([e, 0.0], z_term)
    I = model.edge_currents(sol.v)

    assert rel_err(I, I_ref) < 1e-10
    assert rel_err(sol.i[0], i_ref) < 1e-10
    assert rel_err(sol.z_in[0], zin_ref) < 1e-10


def test_reduction_exact(wide_model):
    """
    V3. Wide strip (3 edges per cut), lossy copper, 1 driven + 2 loaded ports.
    Reference: (Z_t + sum_p Z_D,p a_p a_p^T) I = a_1 e_1, solved directly.
    """
    model = wide_model
    A = model.A
    assert all(np.count_nonzero(A[:, p]) == 3 for p in range(3))

    Z_full = _total_z(model) + (A * Z_TERM[None, :]) @ A.T     # sum_p Z_D,p a_p a_p^T
    I_ref = np.linalg.solve(Z_full, A[:, 0] * EMF[0])
    i_ref = A.T @ I_ref
    zin_ref = (EMF[0] - Z_TERM[0] * i_ref[0]) / i_ref[0]

    sol = model.solve(EMF, Z_TERM)
    I = model.edge_currents(sol.v)

    assert rel_err(I, I_ref) < 1e-10
    assert rel_err(sol.i, i_ref) < 1e-10
    assert rel_err(sol.z_in[0], zin_ref) < 1e-10
    # v = e - Z_D i, and the loaded ports really are loaded.
    assert rel_err(sol.v, EMF - Z_TERM * sol.i) < 1e-14
    assert np.all(np.abs(sol.i[1:]) > 0)


def test_short_circuit_load(wide_model):
    """Z_L = 0 on a declared port == that port not declared (continuous metal)."""
    with_short = _rebuild(wide_model, columns=[0, 1])
    without = _rebuild(wide_model, columns=[0])

    s2 = with_short.solve([1.0, 0.0], [Z0, 0.0])
    s1 = without.solve([1.0], [Z0])

    assert abs(s2.v[1]) == 0.0                                  # shorted port: no voltage
    assert rel_err(with_short.edge_currents(s2.v), without.edge_currents(s1.v)) < 1e-10
    assert rel_err(s2.i[0], s1.i[0]) < 1e-10
    assert rel_err(s2.z_in[0], s1.z_in[0]) < 1e-10


def test_open_circuit_rejected(wide_model):
    with pytest.raises(ValueError, match="open circuit"):
        wide_model.solve(EMF, [Z0, np.inf, 10.0])


# --------------------------------------------------------------------------- power
def test_power_balance(wide_model):
    """
    V1. Poor conductor (sigma = 1e5 S/m, zs_scale = 0.5) so that P_cond is a
    visible share, and resistive parts on both loads.
    """
    zs = calculate_normal_surface_impedance(FREQUENCY, 1e5)
    model = _rebuild(wide_model, Zs=zs, zs_scale=0.5)

    sol = model.solve(EMF, Z_TERM)
    assert sol.p_rad > 0 and sol.p_cond > 0 and sol.p_loads > 0
    p_in = sol.p_in_total
    assert abs(sol.power_balance_residual) / p_in < 1e-10
    # Real generator impedance: P_avail (1 - |Gamma|^2) = P_in.
    assert abs(sol.p_avail[0] * (1 - abs(sol.gamma[0]) ** 2) - p_in) / p_in < 1e-10
    assert np.isclose(sol.p_avail[0], abs(EMF[0]) ** 2 / (8 * Z0), rtol=1e-14)

    # The P x P forms agree with the N x N ones on the actual edge currents.
    I = model.edge_currents(sol.v)
    p_rad_direct = 0.5 * np.real(I.conj() @ model.Z.real @ I)
    p_cond_direct = 0.5 * np.real(0.5 * zs) * np.real(I.conj() @ model.G @ I)
    assert abs(sol.p_rad - p_rad_direct) / p_rad_direct < 1e-10
    assert abs(sol.p_cond - p_cond_direct) / p_cond_direct < 1e-10

    print(f"\n[power balance] P_in = {p_in:.6e} W, P_rad = {sol.p_rad:.6e}, "
          f"P_cond = {sol.p_cond:.6e}, P_loads = {sol.p_loads:.6e}, "
          f"residual/P_in = {sol.power_balance_residual / p_in:.2e}, "
          f"Zin = {sol.z_in[0]:.4f} Ohm")

    # Two driven ports: the balance and the per-port Thevenin identity still hold.
    emf2 = np.array([1.0, 0.4j, 0.0])
    z2 = np.array([Z0, 75.0, 20.0 - 10.0j])
    s2 = model.solve(emf2, z2)
    assert s2.driven.tolist() == [True, True, False]
    assert abs(s2.power_balance_residual) / s2.p_in_total < 1e-10
    np.testing.assert_allclose(s2.p_avail[:2] * (1 - np.abs(s2.gamma[:2]) ** 2), s2.p_in[:2],
                               rtol=1e-10)


# --------------------------------------------------------------------------- reciprocity
def test_admittance_symmetry(wide_strip, wide_model):
    """
    V4. With the default Z (fill_mode="full_sym": every pair, then (Z + Z^T)/2),
    Y is symmetric to rounding. The true reciprocity check needs both halves of
    Z computed independently and NOT symmetrised (full_fill=True): its asymmetry
    is measured and printed, with a loose bound.
    """
    Y = wide_model.Y
    y_asym = np.linalg.norm(Y - Y.T) / np.linalg.norm(Y)
    assert y_asym < 1e-12

    m = wide_strip
    c = wide_model.em_constants
    k = c["omega"] / c["light_speed_c"]
    Z_ff = impedance_matrix_z(m.edges, m.triangles, m.points, 1j * k, c["omega"], c["mu"],
                              c["epsilon"], full_fill=True)

    # Same kernel: the default Z (fill_mode="full_sym") IS the symmetric part of the full fill.
    assert np.array_equal(wide_model.Z, 0.5 * (Z_ff + Z_ff.T))

    D = Z_ff - Z_ff.T
    z_asym_max = np.max(np.abs(D)) / np.max(np.abs(Z_ff))
    z_asym_fro = np.linalg.norm(D) / np.linalg.norm(Z_ff)
    z_asym_re = np.max(np.abs(D.real)) / np.max(np.abs(Z_ff))
    z_asym_im = np.max(np.abs(D.imag)) / np.max(np.abs(Z_ff))

    model_ff = ReducedMultiport.from_matrices(Z_ff, wide_model.A, wide_model.frequency,
                                              G=wide_model.G, Zs=wide_model.zs)
    Y_ff = model_ff.Y
    y_ff_asym = np.linalg.norm(Y_ff - Y_ff.T) / np.linalg.norm(Y_ff)
    y_ff_asym_max = np.max(np.abs(Y_ff - Y_ff.T)) / np.max(np.abs(Y_ff))

    print(f"\n[reciprocity, full_fill=True, N = {Z_ff.shape[0]}]"
          f"\n  Z: max|Z-Z^T|/max|Z| = {z_asym_max:.3e}  (Re part {z_asym_re:.3e}, "
          f"Im part {z_asym_im:.3e}),  ||Z-Z^T||_F/||Z||_F = {z_asym_fro:.3e}"
          f"\n  Y: ||Y-Y^T||_F/||Y||_F = {y_ff_asym:.3e},  max|Y-Y^T|/max|Y| = {y_ff_asym_max:.3e}"
          f"\n  Y (default full_sym Z): ||Y-Y^T||_F/||Y||_F = {y_asym:.3e}")

    assert z_asym_max < 1e-3
    assert y_ff_asym < 1e-3


# --------------------------------------------------------------------------- functionals
def test_functional(wide_model):
    model = _rebuild(wide_model)
    rng = np.random.default_rng(12345)
    c = rng.standard_normal(model.n_edges) + 1j * rng.standard_normal(model.n_edges)
    model.add_linear_functional("random", c)

    sol = model.solve(EMF, Z_TERM)
    direct = c @ model.edge_currents(sol.v)          # c^T I, plain transpose
    assert abs(sol.functionals["random"] - direct) / abs(direct) < 1e-12

    with pytest.raises(ValueError):
        model.add_linear_functional("wrong_size", c[:-1])


# --------------------------------------------------------------------------- timings
def test_timings_report(wide_model):
    """Records the stage timings (for choosing DE vs. Bayesian optimisation)."""
    t = wide_model.timings
    for key in ("z_assembly", "gram_assembly", "lu_factorization", "precompute"):
        assert t[key] is not None and t[key] >= 0

    n_rep = 2000
    with limit_blas_threads(1):          # P x P work: the BLAS thread pool only costs here
        t0 = time.perf_counter()
        for _ in range(n_rep):
            wide_model.solve(EMF, Z_TERM)
        t_solve = (time.perf_counter() - t0) / n_rep

    print(f"\n[timings, N = {wide_model.n_edges}, P = {wide_model.n_ports}] "
          f"Z assembly {t['z_assembly']:.3f} s | Gram {t['gram_assembly'] * 1e3:.2f} ms | "
          f"LU {t['lu_factorization'] * 1e3:.2f} ms | precompute (B, Y, M) "
          f"{t['precompute'] * 1e3:.2f} ms | one solve {t_solve * 1e6:.1f} us")
