"""
Tests for the impedance-matrix fill modes (impmet.FILL_MODES).

The scalar-potential part of the RWG EFIE matrix is Z_Phi = D^T K D (D: edge
currents -> triangle charges, K: one kernel per triangle pair), so for any
charge-free (loop) current u, u^H Z_Phi u = 0 exactly, whatever the quadrature.
Filling m <= n and mirroring (the former default) uses K_st instead of K_ts in
the lower half and breaks that structure; since Z_Phi ~ 1/(kh)^2 times Z_A,
the leak dominated the reactance of small loops. The default "full_sym" fills
every pair and returns (Z + Z^T)/2 = ... D^T [(K + K^T)/2] D: symmetric and
charge-neutral.

Meshes are built here with rwg1 only (no dependency on other packages).
Run with -s to see the printed measurements.
"""

import numpy as np
import pytest
from scipy.linalg import null_space

from solver.rwg.rwg1 import Points, Triangles
from solver.utils.compute_backend import GPU_AVAILABLE
from solver.utils.impmet import (
    FILL_MODES, impedance_matrix_z, impedance_matrix_z_sweep, impedance_matrix_z_sweep_gpu,
)
from solver.utils.quadrature.tests.test_impedance_matrix_regression import (
    Z_FOUR_TRI_GOLDEN, Z_SIX_TRI_GOLDEN, _compute_z, _four_triangle_mesh, _six_triangle_mesh,
)

EPS0 = 8.854e-12
MU0 = 1.257e-6
C0 = 1.0 / np.sqrt(EPS0 * MU0)
EPS_HUGE = 1e250          # pref_Phi = 1/(j w eps 4 pi) ~ 0: impedance_matrix_z returns Z_A alone

requires_gpu = pytest.mark.skipif(not GPU_AVAILABLE, reason="no usable CUDA GPU")


# --------------------------------------------------------------------------- meshes
class Mesh:
    def __init__(self, xs, ys, keep):
        nx, ny = len(xs) - 1, len(ys) - 1
        used = np.zeros((nx + 1, ny + 1), dtype=bool)
        ci, cj = np.nonzero(keep)
        for di in (0, 1):
            for dj in (0, 1):
                used[ci + di, cj + dj] = True
        vid = -np.ones(used.shape, dtype=int)
        vid[used] = np.arange(used.sum())
        I, J = np.nonzero(used)
        p = np.vstack([xs[I], ys[J], np.zeros(I.size)])
        tri = []
        for i, j in zip(ci, cj):
            v00, v10, v11, v01 = vid[i, j], vid[i + 1, j], vid[i + 1, j + 1], vid[i, j + 1]
            tri += [(v00, v10, v11), (v00, v11, v01)]
        t = np.vstack([np.array(tri).T, np.ones((1, len(tri)), dtype=int)])
        self.points = Points(p)
        self.triangles = Triangles(t)
        self.triangles.calculate_triangles_area_and_center(self.points)
        self.edges = self.triangles.get_edges()
        self.edges.compute_edges_length(self.points)
        self.N = self.edges.total_number_of_edges

    def cut_incidence(self, y_lo, y_hi):
        """a_n = s_n l_n for the edges on x = 0, y in [y_lo, y_hi]; s_n from +x."""
        p = self.points.points
        f, s = self.edges.first_points, self.edges.second_points
        on = ((np.abs(p[0, f]) < 1e-12) & (np.abs(p[0, s]) < 1e-12)
              & (p[1, f] >= y_lo - 1e-12) & (p[1, f] <= y_hi + 1e-12)
              & (p[1, s] >= y_lo - 1e-12) & (p[1, s] <= y_hi + 1e-12))
        c = self.triangles.triangles_center
        sign = np.sign(c[0, self.triangles.triangles_minus] - c[0, self.triangles.triangles_plus])
        a = np.where(on, sign * self.edges.edges_length, 0.0)
        assert np.count_nonzero(a) > 0 and np.all(sign[on] != 0)
        return a

    def charge_matrix(self):
        """D[t, n] = +-l_n/A_t: divergence of f_n on its two triangles."""
        tp, tm = self.triangles.triangles_plus, self.triangles.triangles_minus
        L, A = self.edges.edges_length, self.triangles.triangles_area
        n = np.arange(self.N)
        D = np.zeros((self.triangles.total_of_triangles, self.N))
        D[tp, n] += L / A[tp]
        D[tm, n] -= L / A[tm]
        return D


def ring(side_centerline, width, n_side, n_width):
    """Square loop in z = 0 (uniform cells), side measured on the centre line."""
    a_out, a_in = side_centerline / 2 + width / 2, side_centerline / 2 - width / 2
    lines = np.unique(np.concatenate([np.linspace(-a_out, -a_in, n_width + 1),
                                      np.linspace(-a_in, a_in, n_side + 1),
                                      np.linspace(a_in, a_out, n_width + 1)]))
    xc = 0.5 * (lines[:-1] + lines[1:])
    XC, YC = np.meshgrid(xc, xc, indexing="ij")
    m = Mesh(lines, lines, (np.abs(XC) > a_in) | (np.abs(YC) > a_in))
    m.feed = m.cut_incidence(-a_out, -a_in)
    return m


def strip(length, width, n_len, n_width):
    xs = np.linspace(-length / 2, length / 2, n_len + 1)
    ys = np.linspace(-width / 2, width / 2, n_width + 1)
    m = Mesh(xs, ys, np.ones((n_len, n_width), dtype=bool))
    m.feed = m.cut_incidence(-width / 2, width / 2)
    return m


def z_cpu(m, f, mu=MU0, eps=EPS0, **kw):
    w = 2 * np.pi * f
    return impedance_matrix_z(m.edges, m.triangles, m.points, 1j * w / C0, w, mu, eps, **kw)


def input_impedance(Z, a):
    return 1.0 / (a @ np.linalg.solve(Z, a))


@pytest.fixture(scope="module")
def small_loop():
    """20 mm loop (centre line), w = 2 mm, coarse: kh ~ 0.02 at 0.2 GHz, Z_Phi ~ 3e3 Z_A."""
    m = ring(0.020, 0.002, 4, 1)
    m.solenoidal = null_space(m.charge_matrix())
    m.Z_A = z_cpu(m, 0.2e9, eps=EPS_HUGE)
    assert np.all(np.isfinite(m.Z_A))
    return m


def _leak(m, Z_phi, n=5, seed=0):
    """max over random charge-free u of |u^T Z_phi u| / |u^T Z_A u|."""
    rng = np.random.default_rng(seed)
    out = 0.0
    for _ in range(n):
        u = m.solenoidal @ rng.standard_normal(m.solenoidal.shape[1])
        out = max(out, abs(u @ Z_phi @ u) / abs(u @ m.Z_A @ u))
    return out


# --------------------------------------------------------------------------- charge neutrality
@pytest.mark.parametrize("fill_mode", FILL_MODES)
def test_phi_leak_on_loop_currents_cpu(small_loop, fill_mode):
    m = small_loop
    Z_phi = z_cpu(m, 0.2e9, mu=0.0, fill_mode=fill_mode)          # pref_A = 0: Z_Phi alone
    leak = _leak(m, Z_phi)
    print(f"\n[CPU {fill_mode:8s}] N = {m.N}, {m.solenoidal.shape[1]} loop directions: "
          f"max |u^H Z_Phi u| / |u^H Z_A u| = {leak:.3e}")
    if fill_mode == "mirror":
        assert leak > 1e-3                          # documents the defect the default removes
    else:
        assert leak < 1e-10


@pytest.mark.parametrize("use_gpu", [False, pytest.param(True, marks=requires_gpu)])
@pytest.mark.parametrize("fill_mode", ["full_sym", "mirror"])
def test_phi_leak_on_loop_currents_batched(small_loop, use_gpu, fill_mode):
    """Batched assembler (GPU, or its NumPy fallback). It derives k from mu and eps, so Z_Phi
    cannot be isolated there: the leak is measured as u^H (Z - Z_A) u with the CPU Z_A."""
    m = small_loop
    Z = impedance_matrix_z_sweep_gpu(m.edges, m.triangles, m.points, [0.2e9], MU0, EPS0,
                                     use_gpu=use_gpu, fill_mode=fill_mode)[0]
    leak = _leak(m, Z - m.Z_A)
    print(f"\n[{'GPU' if use_gpu else 'batched CPU'} {fill_mode:8s}] leak = {leak:.3e}")
    if fill_mode == "mirror":
        assert leak > 1e-3
    else:
        assert leak < 1e-10


# --------------------------------------------------------------------------- symmetry, agreement
def test_full_sym_is_exactly_symmetric(small_loop):
    m = small_loop
    for Z in (z_cpu(m, 0.9e9),
              impedance_matrix_z_sweep(m.edges, m.triangles, m.points, [0.9e9], MU0, EPS0)[0],
              impedance_matrix_z_sweep_gpu(m.edges, m.triangles, m.points, [0.9e9], MU0, EPS0,
                                           use_gpu=False)[0]):
        assert np.array_equal(Z, Z.T)


def test_full_sym_is_the_symmetric_part_of_full(small_loop):
    m = small_loop
    Zf = z_cpu(m, 0.9e9, fill_mode="full")
    np.testing.assert_array_equal(z_cpu(m, 0.9e9), 0.5 * (Zf + Zf.T))
    np.testing.assert_array_equal(z_cpu(m, 0.9e9, full_fill=True), Zf)     # legacy spelling


@pytest.mark.parametrize("builder,golden", [(_four_triangle_mesh, Z_FOUR_TRI_GOLDEN),
                                            (_six_triangle_mesh, Z_SIX_TRI_GOLDEN)])
def test_default_matches_symmetrised_golden(builder, golden):
    """The frozen full-fill goldens of test_impedance_matrix_regression still pin the new default."""
    np.testing.assert_allclose(_compute_z(*builder()), 0.5 * (golden + golden.T),
                               rtol=1e-10, atol=1e-12)


def test_sweep_and_batched_cpu_match_single_frequency(small_loop):
    m = small_loop
    Z = z_cpu(m, 0.9e9)
    Zs = impedance_matrix_z_sweep(m.edges, m.triangles, m.points, [0.9e9], MU0, EPS0)[0]
    Zb = impedance_matrix_z_sweep_gpu(m.edges, m.triangles, m.points, [0.9e9], MU0, EPS0,
                                      use_gpu=False)[0]
    np.testing.assert_allclose(Zs, Z, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(Zb, Z, rtol=1e-10, atol=1e-12)


@requires_gpu
def test_cpu_and_gpu_agree_in_full_sym():
    m = ring(0.020, 0.002, 8, 2)
    Z_cpu = z_cpu(m, 0.9e9)
    Z_gpu = impedance_matrix_z_sweep_gpu(m.edges, m.triangles, m.points, [0.9e9], MU0, EPS0,
                                         use_gpu=True)[0]
    rel = np.linalg.norm(Z_gpu - Z_cpu) / np.linalg.norm(Z_cpu)
    print(f"\n[CPU vs GPU, full_sym, N = {m.N}] ||Z_gpu - Z_cpu|| / ||Z_cpu|| = {rel:.3e}")
    assert rel < 1e-8


def test_fill_mode_validation(small_loop):
    m = small_loop
    with pytest.raises(ValueError, match="fill_mode"):
        z_cpu(m, 0.9e9, fill_mode="upper")
    with pytest.raises(ValueError, match="conflicts"):
        z_cpu(m, 0.9e9, full_fill=True, fill_mode="mirror")


# --------------------------------------------------------------------------- physics
def test_loop_static_inductance_matches_grover():
    """Square loop, side 20 mm on the centre line, w = 2 mm, at 0.2 GHz, two meshes:
    L = Im(Zin)/w within 5 % of Grover's formula (flat strip, r_eq = w/4) and within 3 %
    of each other. With fill_mode='mirror' this read +84 / +35 nH (see impmet)."""
    s, w = 0.020, 0.002
    L_grover = 2 * MU0 * s / np.pi * (np.log(s / (w / 4)) - 0.774)
    f = 0.2e9
    L = {}
    for n_side, n_width in ((16, 2), (8, 4)):
        m = ring(s, w, n_side, n_width)
        Z = impedance_matrix_z_sweep_gpu(m.edges, m.triangles, m.points, [f], MU0, EPS0)[0]
        L[(n_side, n_width)] = input_impedance(Z, m.feed).imag / (2 * np.pi * f)
    vals = np.array(list(L.values()))
    print(f"\n[static L] Grover {L_grover * 1e9:.2f} nH; "
          + ", ".join(f"n_side={a}, n_width={b}: {v * 1e9:.2f} nH" for (a, b), v in L.items()))
    assert np.all(np.abs(vals / L_grover - 1) < 0.05)
    assert abs(vals[0] / vals[1] - 1) < 0.03


def test_dipole_input_impedance_old_vs_new():
    """V5a dipole (150 x 4 mm) at 0.9 GHz: open current, so the change should be small."""
    m = strip(0.150, 0.004, 40, 2)
    z = {}
    for mode in ("mirror", "full_sym"):
        Z = impedance_matrix_z_sweep_gpu(m.edges, m.triangles, m.points, [0.9e9], MU0, EPS0,
                                         fill_mode=mode)[0]
        z[mode] = input_impedance(Z, m.feed)
    rel = abs(z["full_sym"] - z["mirror"]) / abs(z["mirror"])
    print(f"\n[V5a dipole, N = {m.N}, 0.9 GHz] Zin mirror = {z['mirror']:.4f} Ohm, "
          f"full_sym = {z['full_sym']:.4f} Ohm, |dZ|/|Z| = {rel:.2e}")
    assert rel < 0.05
