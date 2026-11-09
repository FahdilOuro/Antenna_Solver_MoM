"""
Tests for solver.optim.farfield (exact radiation integral) and objectives.

Three current distributions are used throughout, all at 300 MHz (lambda = 1 m):
  short      strip of length lambda/50, centre-fed            (Hertzian-dipole limit)
  half_wave  strip of length lambda/2,  centre-fed            (ordinary antenna)
  pair       two parallel strips, spacing lambda/20, port currents in quasi-opposition
             i_2 = -0.9 exp(+j k d) i_1, beam towards +y      (compensation case,
             representative of superdirectivity); also with ratio 0.99 (stronger
             compensation) in the power-identity and dipole-comparison tests

Run with  python -m pytest test/optim -v -s  to see the printed measurements.
"""

import numpy as np
import pytest

from solver.optim import ReducedMultiport
from solver.optim.farfield import RadiationOperator, spherical_unit_vectors
from solver.optim.geometries import parallel_strips, strip_dipole
from solver.optim.objectives import (
    directivity_from_R, gain, intensity_from_functional, realized_gain,
)
from solver.optim.threads import limit_blas_threads
from solver.utils.dipole_parameters import (
    compute_dipole_center_moment, compute_e_h_field, compute_spherical_e_field,
)
from solver.utils.lossy_electric_conductor import calculate_normal_surface_impedance
from conftest import COPPER_SIGMA

F0 = 3e8
DEG = np.pi / 180
PLUS_Y = (90 * DEG, 90 * DEG)       # (theta, phi) of the +y direction


class Case:
    """Mesh, PEC reduced model, edge currents and R = Re(Z) for one distribution."""

    def __init__(self, name, mesh, model, I):
        self.name, self.mesh, self.model, self.I = name, mesh, model, I
        self.R = model.Z.real

    @property
    def p_rad_R(self):
        return 0.5 * float(np.real(self.I.conj() @ self.R @ self.I))

    def operator(self, quad_order=7):
        m = self.mesh
        return RadiationOperator(m.points, m.triangles, m.edges, F0, quad_order=quad_order)


def _pec_model(mesh, ports):
    return ReducedMultiport.from_mesh(mesh.points, mesh.triangles, mesh.edges, F0, ports,
                                      use_gpu=False)


@pytest.fixture(scope="module")
def short_case():
    mesh = strip_dipole(length=1.0 / 50, width=1e-3, n_len=10, n_width=1)
    model = _pec_model(mesh, mesh.ports())
    return Case("short (L = lambda/50)", mesh, model, model.edge_currents([1.0]))


@pytest.fixture(scope="module")
def half_wave_case():
    mesh = strip_dipole(length=0.5, width=0.01, n_len=40, n_width=1)
    ports = mesh.ports()
    m = mesh
    model = ReducedMultiport.from_mesh(m.points, m.triangles, m.edges, F0, ports,
                                       conductivity=COPPER_SIGMA, use_gpu=False)
    # Edge currents of the PEC problem (for the lossless identities), via Z only.
    pec = ReducedMultiport.from_matrices(model.Z, model.A, F0)
    case = Case("half-wave (L = lambda/2)", mesh, pec, pec.edge_currents([1.0]))
    case.lossy_model = model
    return case


@pytest.fixture(scope="module")
def pair_case():
    d = 1.0 / 20
    mesh = parallel_strips(length=0.45, width=0.01, spacing=d, n_len=24, n_width=1)
    model = _pec_model(mesh, mesh.ports())
    k = 2 * np.pi * F0 / RadiationOperator(mesh.points, mesh.triangles, mesh.edges, F0).light_speed_c
    i_target = np.array([1.0, -0.9 * np.exp(1j * k * d)])
    v = np.linalg.solve(model.Y, i_target)              # port voltages giving those currents
    case = Case("pair, d = lambda/20, quasi-opposed", mesh, model, model.edge_currents(v))
    case.i_ports = model.A.T @ case.I
    return case


@pytest.fixture(scope="module")
def pair_strong_case(pair_case):
    """Same pair, stronger compensation: i_2 = -0.99 exp(+j k d) i_1 (no new assembly)."""
    m, model = pair_case.mesh, pair_case.model
    k = pair_case.operator().k
    v = np.linalg.solve(model.Y, np.array([1.0, -0.99 * np.exp(1j * k * m.params["spacing"])]))
    return Case("pair, d = lambda/20, ratio 0.99", m, model, model.edge_currents(v))


@pytest.fixture(scope="module")
def all_cases(short_case, half_wave_case, pair_case, pair_strong_case):
    return [short_case, half_wave_case, pair_case, pair_strong_case]


def _d_phi_plus_y(case, op, p_rad):
    _, u_phi = op.intensity(case.I, *PLUS_Y)
    return 4 * np.pi * float(u_phi[0]) / p_rad


# --------------------------------------------------------------------------- physics limits
def test_short_dipole_limit(short_case):
    case, op = short_case, short_case.operator()
    t = np.linspace(2, 178, 45) * DEG
    p = np.linspace(0, 358, 90) * DEG
    T, P = np.meshgrid(t, p, indexing="ij")
    u_t, u_p = op.intensity(case.I, T, P)
    U = (u_t + u_p).reshape(T.shape)
    pattern = 1.0 - (np.sin(T) * np.cos(P)) ** 2            # |x_hat x r_hat|^2
    assert np.max(np.abs(U / U.max() - pattern)) < 0.01

    u_t, u_p = op.intensity(case.I, *PLUS_Y)                  # broadside maximum
    d_max = 4 * np.pi * float(u_t[0] + u_p[0]) / case.p_rad_R
    print(f"\n[short dipole] D_max = {d_max:.5f} (Hertzian dipole: 1.5)")
    assert abs(d_max - 1.5) / 1.5 < 0.01


def test_polarization_end_fire(half_wave_case):
    case, op = half_wave_case, half_wave_case.operator()
    _, th_hat, ph_hat = spherical_unit_vectors(*PLUS_Y)
    np.testing.assert_allclose(ph_hat[0], [-1.0, 0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(th_hat[0], [0.0, 0.0, -1.0], atol=1e-15)
    n_t, n_p = op.components(case.I, *PLUS_Y)
    ratio = abs(n_t[0]) / abs(n_p[0])
    print(f"\n[polarization +y] |N_theta|/|N_phi| = {ratio:.2e}")
    assert ratio < 1e-6
    # N_phi = phi_hat . N = -N_x along +y
    N = op.n_vector(case.I, *PLUS_Y)[0]
    assert abs(n_p[0] + N[0]) <= 1e-14 * abs(N[0])


def test_translation_phase(half_wave_case):
    m = half_wave_case.mesh
    rng = np.random.default_rng(3)
    I = rng.standard_normal(m.n_edges) + 1j * rng.standard_normal(m.n_edges)
    d = np.array([0.31, -0.12, 0.07])
    op = RadiationOperator(m.points, m.triangles, m.edges, F0)
    shifted = type(m.points)(m.points.points + d[:, None])
    op_d = RadiationOperator(shifted, m.triangles, m.edges, F0)

    theta = rng.uniform(0, np.pi, 40)
    phi = rng.uniform(0, 2 * np.pi, 40)
    r_hat, _, _ = spherical_unit_vectors(theta, phi)
    N = op.n_vector(I, theta, phi)
    N_d = op_d.n_vector(I, theta, phi)
    expected = N * np.exp(1j * op.k * (r_hat @ d))[:, None]
    assert np.linalg.norm(N_d - expected) / np.linalg.norm(expected) < 1e-10

    # And the per-edge vectors agree with the summed form.
    f_t, f_p = op.vectors(theta, phi)
    n_t, n_p = op.components(I, theta, phi)
    assert np.linalg.norm(f_p @ I - n_p) / np.linalg.norm(n_p) < 1e-12
    assert np.linalg.norm(f_t @ I - n_t) / np.linalg.norm(n_t) < 1e-12


# --------------------------------------------------------------------------- V2
def test_power_identity(all_cases):
    """V2: P_rad by sphere integration of U == 1/2 I^H R I (R = Re Z from impmet)."""
    print("\n[V2 power identity]  P_sphere vs 1/2 I^H R I")
    for case in all_cases:
        op = case.operator()
        p_sphere = op.radiated_power_sphere(case.I)
        n_t, n_p = op.default_sphere_grid()
        p_sphere_fine = op.radiated_power_sphere(case.I, 2 * n_t, 2 * n_p)
        rel = (p_sphere - case.p_rad_R) / case.p_rad_R
        print(f"  {case.name:38s} P_R = {case.p_rad_R:.6e} W  P_sphere = {p_sphere:.6e} W "
              f"(grid {n_t}x{n_p})  rel. diff = {rel:+.2e}  "
              f"(sphere grid x2: {abs(p_sphere_fine - p_sphere) / p_sphere:.1e})")
        assert abs(rel) < 5e-3
        assert abs(p_sphere_fine - p_sphere) / p_sphere < 1e-10


def test_quadrature_convergence(pair_case):
    """D_phi(+y) of the compensated pair vs the triangle rule (degree 2 does not exist:
    gauss_dunavant offers 1, 3, 4, 5, 6, 7, 11, ...)."""
    case = pair_case
    orders = [1, 3, 4, 7, 11]
    d = {q: _d_phi_plus_y(case, case.operator(q), case.p_rad_R) for q in orders}
    ref = d[11]
    print(f"\n[quadrature convergence] {case.name}, port currents i = "
          f"{np.round(case.i_ports, 4)}")
    for q in orders:
        print(f"  quad_order {q:2d}: D_phi(+y) = {d[q]:.10f}  ({10 * np.log10(d[q]):.4f} dBi)  "
              f"rel. diff to order 11 = {(d[q] - ref) / ref:+.2e}")
    assert abs(d[7] - ref) / ref < 1e-3


# --------------------------------------------------------------------------- gains
def test_realized_gain_factorization(half_wave_case):
    case = half_wave_case
    model = ReducedMultiport.from_matrices(case.lossy_model.Z, case.lossy_model.A, F0,
                                           G=case.lossy_model.G, Zs=case.lossy_model.zs)
    op = case.operator()
    name = op.attach_functional(model, *PLUS_Y, component="phi")
    sol = model.solve([1.0], [50.0])

    U = intensity_from_functional(sol, name, op)
    _, u_direct = op.intensity(model.edge_currents(sol.v), *PLUS_Y)
    assert abs(U - u_direct[0]) / u_direct[0] < 1e-12          # functional == direct

    D = 4 * np.pi * U / sol.p_rad
    eta_rad = sol.p_rad / sol.p_in_total
    mismatch = 1 - abs(sol.gamma[0]) ** 2
    g_r = realized_gain(sol, U)
    assert abs(g_r - mismatch * eta_rad * D) / g_r < 1e-10
    assert abs(gain(sol, U) - eta_rad * D) / gain(sol, U) < 1e-12
    # D from the reduced model's M_rad == D from R on the edge currents
    D_R = directivity_from_R(U, model.edge_currents(sol.v), model.Z.real)
    assert abs(D - D_R) / D < 1e-10
    print(f"\n[realized gain, copper half-wave] Zin = {sol.z_in[0]:.3f} Ohm, "
          f"1-|G|^2 = {mismatch:.5f}, eta_rad = {eta_rad:.5f}, D_phi(+y) = {D:.5f}, "
          f"G_r = {g_r:.5f} ({10 * np.log10(g_r):.3f} dBi)")


# --------------------------------------------------------------------------- old chain
def _dipole_chain(case, op, r_obs=100.0):
    """
    The existing chain: compute_dipole_center_moment + compute_e_h_field at radius
    r_obs (efield2's sphere radius), but integrated on our Gauss-Legendre x trapezoid
    grid so that only the dipole approximation differs (efield2 additionally samples
    a flat-triangle sphere mesh at its centroids; not reproduced here).
    """
    centers, moments = compute_dipole_center_moment(case.mesh.triangles, case.mesh.edges, case.I)
    complex_k = 1j * op.k
    n_t, n_p = op.default_sphere_grid()
    u, w_u = np.polynomial.legendre.leggauss(n_t)
    phi = 2 * np.pi * np.arange(n_p) / n_p
    T, P = np.meshgrid(np.arccos(u), phi, indexing="ij")
    r_hat, _, _ = spherical_unit_vectors(T, P)
    U = np.empty(r_hat.shape[0])
    for i, rh in enumerate(r_hat):
        *_, U[i], _ = compute_e_h_field(r_obs * rh, op.eta, complex_k, moments, centers)
    p_rad = float(np.sum(w_u[:, None] * U.reshape(n_t, n_p)) * 2 * np.pi / n_p)

    point = r_obs * spherical_unit_vectors(*PLUS_Y)[0][0]
    e_tot, *_ = compute_e_h_field(point, op.eta, complex_k, moments, centers)
    _, e_phi = compute_spherical_e_field(point, r_obs, e_tot)
    u_phi = r_obs ** 2 * abs(e_phi) ** 2 / (2 * op.eta)
    return p_rad, 4 * np.pi * u_phi / p_rad


def test_compare_dipole_approximation(all_cases):
    print("\n[dipole approximation vs exact integral]  (exact: quad_order 7, P_rad from sphere)")
    print(f"  {'case':38s} {'P_rad exact':>12s} {'P_rad dip.':>12s} {'rel.':>9s}   "
          f"{'D_phi exact':>11s} {'D_phi dip.':>11s} {'rel.':>9s}")
    with limit_blas_threads(1):
        for case in all_cases:
            op = case.operator()
            p_exact = op.radiated_power_sphere(case.I)
            d_exact = _d_phi_plus_y(case, op, p_exact)
            p_dip, d_dip = _dipole_chain(case, op)
            print(f"  {case.name:38s} {p_exact:12.5e} {p_dip:12.5e} {(p_dip - p_exact) / p_exact:+9.2e}"
                  f"   {d_exact:11.5f} {d_dip:11.5f} {(d_dip - d_exact) / d_exact:+9.2e}")
            assert np.isfinite(p_dip) and np.isfinite(d_dip)
