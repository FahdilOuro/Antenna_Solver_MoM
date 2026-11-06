"""
Exact far-field radiation integral of RWG currents, by triangle quadrature.

Why not the existing dipole model
---------------------------------
solver.utils.dipole_parameters.compute_dipole_center_moment replaces each RWG
function by a point dipole m_n = l_n I_n (c_n^- - c_n^+) at (c_n^+ + c_n^-)/2.
Its moment is exact (it IS the integral of f_n), but its phase is sampled at a
single point. For a superdirective antenna the useful field is the small
residual of nearly opposed currents, so a small relative error on each
contribution can become a large one on the residual. Here the radiation
integral is evaluated with a Gauss-Dunavant rule whose order is a parameter,
so its error can be driven down and measured.

Formulation (time convention e^{jwt}, kernel e^{-jkR}/(4 pi R) as in impmet)
-------------------------------------------------------------------------
    E(r, r_hat) = -(j k eta0 / (4 pi r)) e^{-jkr} [N - (N . r_hat) r_hat]
    N(r_hat)    = sum_n I_n n_n(r_hat),
    n_n(r_hat)  = int_{T_n^+ U T_n^-} f_n(r') e^{j k r_hat . r'} dS'

with f_n = l_n/(2A^+) (r - v^+) on T_n^+ and -l_n/(2A^-) (r - v^-) on T_n^-
(v^+- the free vertices). With Dunavant weights w_q summing to 1/2 (the area
of the reference triangle, gauss_dunavant.py) the physical integral is
int_T g dS = 2A sum_q w_q g(r_q), so the 2A cancels:

    int_{T^+} f_n e^{jk r_hat . r'} dS' =  l_n sum_q w_q (r_q - v^+) e^{jk r_hat . r_q}
    int_{T^-} f_n e^{jk r_hat . r'} dS' = -l_n sum_q w_q (r_q - v^-) e^{jk r_hat . r_q}

Components and intensity
    theta_hat = (cos t cos p, cos t sin p, -sin t),  phi_hat = (-sin p, cos p, 0)
    N_theta = theta_hat . N,  N_phi = phi_hat . N
    U_theta,phi = r^2 |E_theta,phi|^2 / (2 eta0) = k^2 eta0 / (32 pi^2) |N_theta,phi|^2   (W/sr)

Constants: eps0 = 8.854e-12 F/m, mu0 = 1.257e-6 H/m, exactly those of
rwg3.calculate_z_matrice, so that k and eta0 are consistent with Z (and with
P_rad = 1/2 I^H Re(Z) I). These differ from CODATA by ~3e-4 relative in mu0;
they will be harmonised project-wide later.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss

from solver.utils.quadrature.gauss_dunavant import dunavant_rule

EPSILON_0 = 8.854e-12   # F/m, as in rwg3.calculate_z_matrice
MU_0 = 1.257e-6         # H/m, as in rwg3.calculate_z_matrice

# Complex entries kept in one direction chunk (memory bound for the phase tables).
_CHUNK_ENTRIES = 2_000_000


def spherical_unit_vectors(theta, phi):
    """
    r_hat, theta_hat, phi_hat as (n_dir, 3) arrays, for angles in radians
    (broadcast together, then flattened).
    """
    theta, phi = np.broadcast_arrays(np.asarray(theta, float), np.asarray(phi, float))
    t, p = theta.ravel(), phi.ravel()
    st, ct, sp, cp = np.sin(t), np.cos(t), np.sin(p), np.cos(p)
    r_hat = np.stack([st * cp, st * sp, ct], axis=1)
    theta_hat = np.stack([ct * cp, ct * sp, -st], axis=1)
    phi_hat = np.stack([-sp, cp, np.zeros_like(p)], axis=1)
    return r_hat, theta_hat, phi_hat


class RadiationOperator:
    """
    Far-field operator of an RWG mesh at one frequency.

    Parameters
    ----------
    points, triangles, edges : rwg1 objects (triangles_plus/minus set, edges_length set)
    frequency  : Hz
    quad_order : Gauss-Dunavant degree on each triangle (gauss_dunavant.dunavant_rule:
                 1, 3, 4, 5, 6, 7, 11, 16, 20, 24)
    origin     : (3,) phase reference of N (default: the coordinate origin).
                 Changing it multiplies N by exp(-jk r_hat . origin); |N| and U
                 are unaffected.

    Attributes
    ----------
    k, eta, light_speed_c, omega, frequency
    intensity_constant : k^2 eta0 / (32 pi^2), so that U = intensity_constant |N|^2
    """

    def __init__(self, points, triangles, edges, frequency, quad_order=7, origin=None):
        pts = np.asarray(points.points if hasattr(points, "points") else points, dtype=float)
        tri = np.asarray(triangles.triangles)[:3].astype(int)
        plus = np.asarray(triangles.triangles_plus).astype(int)
        minus = np.asarray(triangles.triangles_minus).astype(int)
        lengths = np.asarray(edges.edges_length, dtype=float)
        first = np.asarray(edges.first_points).astype(int)
        second = np.asarray(edges.second_points).astype(int)

        self.frequency = float(frequency)
        self.quad_order = int(quad_order)
        self.light_speed_c = 1.0 / np.sqrt(EPSILON_0 * MU_0)
        self.eta = np.sqrt(MU_0 / EPSILON_0)
        self.omega = 2.0 * np.pi * self.frequency
        self.k = self.omega / self.light_speed_c
        self.intensity_constant = self.k ** 2 * self.eta / (32.0 * np.pi ** 2)
        self.origin = np.zeros(3) if origin is None else np.asarray(origin, float).reshape(3)
        self.n_edges = lengths.size

        xi, w = dunavant_rule(self.quad_order)
        self._w = w
        tri_verts = pts[:, tri]                                    # (3 coord, 3 vertex, N_tri)
        # r_tri[t, :, q] = sum_v xi[q, v] * vertex_v(t)
        self._r_tri = np.einsum("qv,cvt->tcq", xi, tri_verts) - self.origin[None, :, None]
        self._plus, self._minus = plus, minus

        free_plus = self._free_vertex(pts, tri, plus, first, second, "T+")
        free_minus = self._free_vertex(pts, tri, minus, first, second, "T-")
        # W_+[n, :, q] = l_n w_q (r_q - v^+),  W_-[n, :, q] = -l_n w_q (r_q - v^-)
        origin_col = self.origin[None, :, None]
        self._W_plus = (lengths[:, None, None] * w[None, None, :]
                        * (self._r_tri[plus] + origin_col - free_plus.T[:, :, None]))
        self._W_minus = -(lengths[:, None, None] * w[None, None, :]
                          * (self._r_tri[minus] + origin_col - free_minus.T[:, :, None]))

        # Radius of the smallest origin-centred sphere around the quadrature points,
        # about the mesh centroid (used to size the sphere quadrature).
        r_abs = self._r_tri + origin_col
        self._centroid = r_abs.mean(axis=(0, 2))
        self._radius = float(np.max(np.linalg.norm(r_abs - self._centroid[None, :, None], axis=1)))

    @staticmethod
    def _free_vertex(pts, tri, tri_idx, first, second, label):
        verts = tri[:, tri_idx]                                     # (3, N)
        mask = (verts != first[None, :]) & (verts != second[None, :])
        if not np.all(mask.sum(axis=0) == 1):
            raise ValueError(f"Inconsistent mesh: some edge is not a side of its {label} triangle.")
        idx = np.argmax(mask, axis=0)
        return pts[:, verts[idx, np.arange(verts.shape[1])]]       # (3, N)

    # ------------------------------------------------------------ per-edge vectors
    def edge_vectors(self, theta, phi) -> np.ndarray:
        """n_n(r_hat) for every edge: complex (n_dir, N, 3)."""
        r_hat, _, _ = spherical_unit_vectors(theta, phi)
        nq = self._w.size
        chunk = max(1, _CHUNK_ENTRIES // max(1, self.n_edges * nq))
        out = np.empty((r_hat.shape[0], self.n_edges, 3), dtype=np.complex128)
        for s in range(0, r_hat.shape[0], chunk):
            rh = r_hat[s:s + chunk]
            phase = np.exp(1j * self.k * np.einsum("ci,tiq->ctq", rh, self._r_tri))
            out[s:s + chunk] = (np.einsum("cnq,niq->cni", phase[:, self._plus], self._W_plus)
                                + np.einsum("cnq,niq->cni", phase[:, self._minus], self._W_minus))
        return out

    def vectors(self, theta, phi):
        """
        f_theta, f_phi : complex (n_dir, N) with N_theta = f_theta @ I, N_phi = f_phi @ I
        (plain transpose, no conjugate). Angles in radians.
        """
        _, th_hat, ph_hat = spherical_unit_vectors(theta, phi)
        n = self.edge_vectors(theta, phi)
        return np.einsum("cni,ci->cn", n, th_hat), np.einsum("cni,ci->cn", n, ph_hat)

    # ------------------------------------------------------------ given currents
    def _sources(self, I, origin_shift=None):
        """Current-weighted quadrature sources per triangle: (N_tri*Nq, 3), positions (N_tri*Nq, 3)."""
        I = np.asarray(I, dtype=np.complex128).reshape(-1)
        if I.shape != (self.n_edges,):
            raise ValueError(f"I must have shape ({self.n_edges},), got {I.shape}.")
        n_tri, _, nq = self._r_tri.shape
        S = np.zeros((n_tri, 3, nq), dtype=np.complex128)
        np.add.at(S, self._plus, I[:, None, None] * self._W_plus)
        np.add.at(S, self._minus, I[:, None, None] * self._W_minus)
        pos = self._r_tri if origin_shift is None else self._r_tri - origin_shift[None, :, None]
        return S.transpose(0, 2, 1).reshape(-1, 3), pos.transpose(0, 2, 1).reshape(-1, 3)

    def _n_vector(self, I, r_hat, origin_shift=None):
        S, pos = self._sources(I, origin_shift)
        chunk = max(1, _CHUNK_ENTRIES // max(1, pos.shape[0]))
        out = np.empty((r_hat.shape[0], 3), dtype=np.complex128)
        for s in range(0, r_hat.shape[0], chunk):
            out[s:s + chunk] = np.exp(1j * self.k * (r_hat[s:s + chunk] @ pos.T)) @ S
        return out

    def n_vector(self, I, theta, phi) -> np.ndarray:
        """N(r_hat) = sum_n I_n n_n(r_hat), complex (n_dir, 3)."""
        r_hat, _, _ = spherical_unit_vectors(theta, phi)
        return self._n_vector(I, r_hat)

    def components(self, I, theta, phi):
        """N_theta, N_phi : complex (n_dir,)."""
        _, th_hat, ph_hat = spherical_unit_vectors(theta, phi)
        N = self.n_vector(I, theta, phi)
        return np.einsum("ci,ci->c", N, th_hat), np.einsum("ci,ci->c", N, ph_hat)

    def intensity(self, I, theta, phi):
        """U_theta, U_phi (W/sr) = k^2 eta0/(32 pi^2) |N_theta,phi|^2, for peak-phasor I."""
        n_t, n_p = self.components(I, theta, phi)
        return self.intensity_constant * np.abs(n_t) ** 2, self.intensity_constant * np.abs(n_p) ** 2

    def default_sphere_grid(self):
        """
        (n_theta, n_phi) sufficient for |N|^2: with a the radius of the mesh
        about its centroid, N is band-limited to angular degree ~ka, |N|^2 to ~2ka.
        Gauss-Legendre with n nodes is exact to degree 2n-1 in cos(theta), the
        periodic trapezoid with m nodes to Fourier order m-1 in phi.
        """
        L = int(np.ceil(self.k * self._radius)) + 12
        return L + 1, 2 * L + 2

    def radiated_power_sphere(self, I, n_theta=None, n_phi=None) -> float:
        """
        P_rad = int U dOmega (W), Gauss-Legendre in cos(theta) x periodic trapezoid in phi:

            P_rad = sum_i sum_j  w_i (2 pi / n_phi) [U_theta + U_phi](arccos u_i, phi_j)

        The phase reference is moved to the mesh centroid first (|N| is unchanged)
        so that the band limit, hence the default grid, is as small as possible.
        """
        d_t, d_p = self.default_sphere_grid()
        n_theta = d_t if n_theta is None else int(n_theta)
        n_phi = d_p if n_phi is None else int(n_phi)
        u, w_u = leggauss(n_theta)
        phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
        T, P = np.meshgrid(np.arccos(u), phi, indexing="ij")
        r_hat, _, _ = spherical_unit_vectors(T, P)
        shift = self._centroid - self.origin
        N = self._n_vector(I, r_hat, origin_shift=shift)
        N_perp2 = np.sum(np.abs(N) ** 2, axis=1) - np.abs(np.einsum("ci,ci->c", N, r_hat)) ** 2
        U = self.intensity_constant * N_perp2.reshape(n_theta, n_phi)
        return float(np.sum(w_u[:, None] * U) * 2.0 * np.pi / n_phi)

    # ------------------------------------------------------------ reduced model hook
    def attach_functional(self, model, theta, phi, component="phi", name=None, rtol=1e-12):
        """
        Register N_component(theta, phi) = f^T I on a ReducedMultiport as a linear
        functional h = B^T f, so that N = h^T v costs O(P) per load configuration.
        Returns the functional name. U then follows from
        objectives.intensity_from_functional.
        """
        if abs(model.frequency - self.frequency) > rtol * self.frequency:
            raise ValueError(f"Frequency mismatch: operator {self.frequency} Hz, "
                             f"model {model.frequency} Hz.")
        if model.n_edges != self.n_edges:
            raise ValueError(f"Edge-count mismatch: operator {self.n_edges}, model {model.n_edges}.")
        f_t, f_p = self.vectors(np.atleast_1d(theta)[:1], np.atleast_1d(phi)[:1])
        f = {"theta": f_t, "phi": f_p}[component][0]
        if name is None:
            name = (f"N_{component}(th={np.degrees(np.atleast_1d(theta)[0]):g},"
                    f"ph={np.degrees(np.atleast_1d(phi)[0]):g})")
        model.add_linear_functional(name, f)
        return name
