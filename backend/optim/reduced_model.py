"""
Exact reduction of an RWG MoM problem to a P-port network.

Once the MoM matrix has been factorised for a given geometry and frequency, the
response to ANY set of generators and loads connected at the P ports follows
from P x P algebra. This is what makes load optimisation (parasitic elements,
PIN diodes, varactors) cheap: the O(N^2) fill and O(N^3) factorisation are paid
once, every load configuration afterwards costs O(P^3).

Notation (SI units throughout)
------------------------------
N        number of RWG edges, P number of ports
Z        (N, N) PEC EFIE impedance matrix (rwg3.calculate_z_matrice); complex
         symmetric, Z_mn carries the factor l_m l_n
G        (N, N) RWG Gram matrix  G_mn = int f_m . f_n dS  (impmet.rwg_gram_matrix)
Zs       surface impedance of the conductor (lossy_electric_conductor), Ohm
alpha    zs_scale, see "Surface-impedance scaling" below
A        (N, P) port incidence matrix (backend.optim.ports): V = A v, i = A^T I

Reduction
---------
    Z_t = Z + alpha Zs G                       (MoM + conductor loss)
    B   = Z_t^{-1} A                           (one LU factorisation, P solves)
    Y   = A^T B                                (port admittance matrix, P x P)

so that for port voltages v:  I = B v  and  i = A^T I = Y v.
Y is symmetric because Z_t is (Galerkin testing + reciprocity).

Terminations
------------
Each port p is closed on a Thevenin source (emf e_p, internal impedance Z_D,p);
a passive load is the case e_p = 0, Z_D,p = Z_L,p:

    v = e - Z_D i,    Z_D = diag(Z_D,p).

Eliminating v with i = Y v, WITHOUT inverting Y:

    (1_P + Y Z_D) i = Y e,    v = e - Z_D i.

This handles a short circuit (Z_D,p = 0) exactly; a short-circuited port is
the same as no port at all, since an undeclared cut is continuous metal. An
open circuit (Z_D,p = infinity, i.e. i_p = 0) is NOT representable in this
form and is rejected: an open gap must be cut in the mesh itself (remove the
port's edges), or approximated by a large finite impedance.

It is algebraically identical to solving the full system

    (Z_t + A Z_D A^T) I = A e

(see test/optim, test_reduction_exact): no approximation is introduced.

Powers (time-harmonic peak phasors, hence the 1/2)
--------------------------------------------------
    R      = Re(Z)                       (PEC Z only: radiation, no loss)
    M_rad  = 1/2 B^H R B                 P_rad  = v^H M_rad  v
    M_cond = 1/2 Re(alpha Zs) B^H G B    P_cond = v^H M_cond v
    P_loads  = 1/2 sum_{p loaded} Re(Z_D,p) |i_p|^2
    P_in,p   = 1/2 Re(v_p i_p^*)          at driven ports
    P_avail,p = |e_p|^2 / (8 Re Z_D,p)    at driven ports

Power balance (exact, since Z_t is symmetric):
    sum_{p driven} P_in,p = P_rad + P_cond + P_loads.

Linear functionals
------------------
Any quantity linear in the edge currents, q = c^T I with c in C^N (e.g. one
far-field component, which the next step will provide), reduces to

    h = B^T c,    q = h^T v,

so that it too costs O(P) per load configuration (add_linear_functional).

Surface-impedance scaling (zs_scale = alpha) -- OPEN QUESTION
-------------------------------------------------------------
The mesh is a zero-thickness sheet carrying the total current J of a real strip
whose thickness t >> skin depth. Physically the current splits between the two
faces; if it splits equally, each face carries J/2 and the loss is
2 * 1/2 Rs |J/2|^2 = 1/2 (Rs/2) |J|^2, i.e. an effective Zs/2 (alpha = 0.5).
If instead the current crowds onto one face (e.g. above a ground plane), alpha
is closer to 1. The right value for a free-standing strip (edge crowding
included) is NOT settled; it will be fixed by comparison with CST. The default
alpha = 1.0 is a placeholder, not a recommendation.

Frequency sweeps
----------------
This class handles one frequency. The constructor takes already-assembled
matrices on purpose, so that a future ReducedMultiportSweep can build one
instance per frequency from calculate_z_matrice_sweep's (N_freq, N, N) output
(with Zs evaluated per frequency) sharing the same A, without re-assembling.

Free space only: Z is the free-space EFIE operator; no substrate, no ground
plane other than one that is meshed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from backend.optim.ports import Port, incidence_matrix


def terminated_input_impedance(Zp, z_loads, driven=0):
    """
    Input impedance at port `driven` of a P-port with open-circuit impedance
    matrix Zp (P, P), every other port closed on z_loads (P-1,), in port order:

        Z_in = Z_dd - Z_dL (Z_LL + diag(z_L))^{-1} Z_Ld

    This is the same network equation as ReducedMultiport.solve (v = Zp i,
    v_L = -z_L i_L) written in impedance form; it is used where Zp is known
    without a MoM model, e.g. interpolated in frequency. Accepts a stack of
    matrices (..., P, P) with z_loads (..., P-1) for vectorised sweeps.
    """
    Zp = np.asarray(Zp, dtype=np.complex128)
    P = Zp.shape[-1]
    others = [p for p in range(P) if p != driven]
    zl = np.asarray(z_loads, dtype=np.complex128)
    z_dd = Zp[..., driven, driven]
    z_dl = Zp[..., driven, others]
    z_ld = Zp[..., others, driven]
    z_ll = Zp[..., others, :][..., :, others] + zl[..., :, None] * np.eye(P - 1)
    x = np.linalg.solve(z_ll, z_ld[..., None])[..., 0]
    return z_dd - np.einsum("...i,...i->...", z_dl, x)


@dataclass(frozen=True)
class PortSolution:
    """
    Port quantities for one set of terminations. Arrays are indexed by port.

    Attributes
    ----------
    emf          : (P,) complex   generator emf e_p (V, peak); 0 at loaded ports
    z_term       : (P,) complex   termination Z_D,p: generator internal impedance at
                                  driven ports, load impedance elsewhere (Ohm)
    driven       : (P,) bool      True at driven ports
    v            : (P,) complex   port voltages v = e - Z_D i (V, peak)
    i            : (P,) complex   port currents i = Y v, positive along each port's
                                  direction (A, peak)
    z_in         : (P,) complex   input impedance v_p/i_p at driven ports (Ohm);
                                  the ACTIVE impedance if several ports are driven;
                                  NaN at loaded ports
    z_ref        : (P,) complex   reference impedance used for gamma (Ohm)
    gamma        : (P,) complex   power-wave reflection coefficient at driven ports,
                                  (z_in - conj(z_ref)) / (z_in + z_ref), which is
                                  (z_in - Z0)/(z_in + Z0) for a real Z0; NaN elsewhere
    p_rad        : float          radiated power (W)
    p_cond       : float          ohmic loss in the conductor (W)
    p_loads      : float          power dissipated in the loads of loaded ports (W)
    p_in         : (P,) float     power delivered to the structure at driven ports,
                                  1/2 Re(v_p i_p^*) (W); 0 at loaded ports
    p_avail      : (P,) float     available generator power |e_p|^2/(8 Re Z_D,p) (W);
                                  0 at loaded ports, inf if Re Z_D,p <= 0
    functionals  : dict           {name: h^T v} for each registered linear functional
    """
    emf: np.ndarray
    z_term: np.ndarray
    driven: np.ndarray
    v: np.ndarray
    i: np.ndarray
    z_in: np.ndarray
    z_ref: np.ndarray
    gamma: np.ndarray
    p_rad: float
    p_cond: float
    p_loads: float
    p_in: np.ndarray
    p_avail: np.ndarray
    functionals: Dict[str, complex] = field(default_factory=dict)

    @property
    def p_in_total(self) -> float:
        """sum_{p driven} P_in,p (W)."""
        return float(np.sum(self.p_in))

    @property
    def p_avail_total(self) -> float:
        """sum_{p driven} P_avail,p (W)."""
        return float(np.sum(self.p_avail))

    @property
    def radiation_efficiency(self) -> float:
        """P_rad / P_in_total (loads and conductor both count as loss)."""
        return self.p_rad / self.p_in_total

    @property
    def power_balance_residual(self) -> float:
        """P_in_total - (P_rad + P_cond + P_loads), W. Zero up to rounding."""
        return self.p_in_total - (self.p_rad + self.p_cond + self.p_loads)


class ReducedMultiport:
    """
    P-port reduced model of one geometry at one frequency (see module docstring).

    Attributes (after construction)
    -------------------------------
    frequency  : float            Hz
    A          : (N, P) float     port incidence matrix
    B          : (N, P) complex   Z_t^{-1} A  -- edge currents per volt at each port
    Y          : (P, P) complex   port admittance matrix A^T B (S)
    M_rad      : (P, P) complex   Hermitian, P_rad = v^H M_rad v
    M_cond     : (P, P) complex   Hermitian, P_cond = v^H M_cond v
    functionals: dict             {name: h (P,)}
    zs, zs_scale : surface impedance (Ohm, or None for PEC) and its scale alpha
    port_names : tuple of str
    timings    : dict of wall-clock seconds: 'z_assembly', 'gram_assembly'
                 (None when the matrices were given), 'lu_factorization',
                 'precompute' (the P solves for B, then Y, M_rad, M_cond)
    Z, G       : the matrices the model was built from (kept only if
                 keep_matrices=True; needed by nothing in the model itself)
    """

    def __init__(self, Z, A, frequency, G=None, Zs=None, zs_scale=1.0,
                 port_names: Optional[Sequence[str]] = None, timings=None,
                 keep_matrices: bool = True):
        Z = np.asarray(Z)
        A = np.asarray(A, dtype=np.float64)
        if Z.ndim != 2 or Z.shape[0] != Z.shape[1]:
            raise ValueError(f"Z must be square (N, N), got shape {Z.shape}.")
        n_edges = Z.shape[0]
        if A.ndim == 1:
            A = A[:, None]
        if A.ndim != 2 or A.shape[0] != n_edges:
            raise ValueError(f"A must have shape (N, P) with N = {n_edges}, got {A.shape}.")
        n_ports = A.shape[1]
        if np.any(np.all(A == 0.0, axis=0)):
            raise ValueError("A has an all-zero column: a port with no edge.")

        lossy = Zs is not None and Zs != 0
        if lossy:
            if G is None:
                raise ValueError("Zs was given without the Gram matrix G: conductor loss "
                                 "needs both (Z_t = Z + zs_scale * Zs * G).")
            G = np.asarray(G, dtype=np.float64)
            if G.shape != Z.shape:
                raise ValueError(f"G must have the same shape as Z {Z.shape}, got {G.shape}.")

        self.frequency = float(frequency)
        self.zs = None if Zs is None else complex(Zs)
        self.zs_scale = float(zs_scale)
        self.A = A
        if port_names is None:
            port_names = [f"port{p + 1}" for p in range(n_ports)]
        if len(port_names) != n_ports:
            raise ValueError(f"{len(port_names)} port names for {n_ports} ports.")
        self.port_names = tuple(port_names)
        self.ports = None  # set by from_mesh
        self.em_constants = None  # set by from_mesh
        self.mesh = None  # set by from_mesh: dict(points, triangles, edges)
        self.timings = {'z_assembly': None, 'gram_assembly': None,
                        'lu_factorization': None, 'precompute': None}
        if timings:
            self.timings.update(timings)

        # --- Z_t = Z + alpha Zs G, then LU --------------------------------
        t0 = time.perf_counter()
        z_eff = self.zs_scale * self.zs if lossy else 0.0
        if lossy:
            Zt = Z.astype(np.complex128) + z_eff * G      # fresh array: LU may overwrite it
        else:
            Zt = np.array(Z, dtype=np.complex128)         # copy: never overwrite the caller's Z
        lu = lu_factor(Zt, overwrite_a=True, check_finite=True)
        t1 = time.perf_counter()

        # --- B = Z_t^{-1} A, Y = A^T B, M_rad, M_cond --------------------
        self.B = lu_solve(lu, A.astype(np.complex128), check_finite=False)
        self.Y = A.T @ self.B
        # R = Re(Z), symmetrised so that v^H M_rad v is real even for a Z
        # assembled with full_fill=True (Re(Z) is symmetric to ~1e-15 there anyway).
        R = 0.5 * (Z.real + Z.real.T)
        self.M_rad = self._hermitian(0.5 * (self.B.conj().T @ (R @ self.B)))
        if lossy:
            self.M_cond = self._hermitian(
                0.5 * np.real(z_eff) * (self.B.conj().T @ (G @ self.B)))
        else:
            self.M_cond = np.zeros((n_ports, n_ports), dtype=np.complex128)
        t2 = time.perf_counter()

        self.timings['lu_factorization'] = t1 - t0
        self.timings['precompute'] = t2 - t1
        self.functionals: Dict[str, np.ndarray] = {}

        if keep_matrices:
            self.Z, self.G = Z, G
        else:
            self.Z = self.G = None

    # ------------------------------------------------------------------ build
    @classmethod
    def from_matrices(cls, Z, ports_A, frequency, G=None, Zs=None, zs_scale=1.0,
                      port_names=None, keep_matrices=True) -> "ReducedMultiport":
        """
        Build from an already-assembled Z (and optionally G, Zs).

        ports_A is the (N, P) incidence matrix (ports.incidence_matrix). Used by
        the tests and to reuse a Z assembled elsewhere (e.g. one slice of
        calculate_z_matrice_sweep).
        """
        return cls(Z, ports_A, frequency, G=G, Zs=Zs, zs_scale=zs_scale,
                   port_names=port_names, keep_matrices=keep_matrices)

    @classmethod
    def from_mesh(cls, points, triangles, edges, frequency, ports: Sequence[Port],
                  conductivity=None, zs_scale=1.0, use_gpu=None,
                  keep_matrices=True) -> "ReducedMultiport":
        """
        Assemble Z (rwg3.calculate_z_matrice), and, if `conductivity` (S/m) is
        given, G (impmet.rwg_gram_matrix) and Zs
        (lossy_electric_conductor.calculate_normal_surface_impedance), then reduce.
        conductivity=None means PEC (no conductor loss).
        """
        from backend.rwg.rwg3 import calculate_z_matrice
        from backend.utils.impmet import rwg_gram_matrix
        from backend.utils.lossy_electric_conductor import calculate_normal_surface_impedance

        ports = list(ports)
        A = incidence_matrix(ports, edges)

        t0 = time.perf_counter()
        omega, mu, epsilon, light_speed_c, eta, Z = calculate_z_matrice(
            points, triangles, edges, frequency, use_gpu=use_gpu)
        t_z = time.perf_counter() - t0

        G = Zs = None
        t_g = None
        if conductivity is not None:
            t0 = time.perf_counter()
            G = rwg_gram_matrix(edges, triangles, points)
            t_g = time.perf_counter() - t0
            Zs = calculate_normal_surface_impedance(frequency, conductivity)

        model = cls(Z, A, frequency, G=G, Zs=Zs, zs_scale=zs_scale,
                    port_names=[p.name for p in ports],
                    timings={'z_assembly': t_z, 'gram_assembly': t_g},
                    keep_matrices=keep_matrices)
        model.ports = tuple(ports)
        model.em_constants = dict(omega=omega, mu=mu, epsilon=epsilon,
                                  light_speed_c=light_speed_c, eta=eta)
        model.mesh = dict(points=points, triangles=triangles, edges=edges)
        return model

    # ------------------------------------------------------------ properties
    @property
    def n_ports(self) -> int:
        return self.A.shape[1]

    @property
    def n_edges(self) -> int:
        return self.A.shape[0]

    @property
    def Zp(self) -> np.ndarray:
        """Open-circuit port impedance matrix Y^{-1} (P x P). Diagnostic only:
        solve() never forms it."""
        return np.linalg.inv(self.Y)

    @staticmethod
    def _hermitian(M):
        return 0.5 * (M + M.conj().T)

    # -------------------------------------------------------------- functionals
    def add_linear_functional(self, name: str, c) -> None:
        """
        Register q = c^T I (c in C^N; plain transpose, not conjugate) as
        h = B^T c, so that q = h^T v for any port voltages v.
        """
        c = np.asarray(c, dtype=np.complex128).reshape(-1)
        if c.shape != (self.n_edges,):
            raise ValueError(f"Functional '{name}': c must have shape ({self.n_edges},), "
                             f"got {c.shape}.")
        self.functionals[name] = self.B.T @ c

    # ------------------------------------------------------------------ solve
    def solve(self, emf, z_term, driven=None, z_ref=None) -> PortSolution:
        """
        Port response for generators `emf` and terminations `z_term`.

        Solves (1_P + Y Z_D) i = Y e, then v = e - Z_D i (no inversion of Y,
        so Z_D,p = 0 -- a short circuit -- is exact).

        Parameters
        ----------
        emf    : (P,) complex  generator emf e_p (V, peak), 0 at loaded ports
        z_term : (P,) complex  Z_D,p: generator internal impedance at driven ports,
                 load impedance elsewhere (Ohm). Must be finite: an open circuit
                 is a port that is not declared.
        driven : (P,) bool or None  which ports are driven. Default: emf != 0.
        z_ref  : scalar or (P,) complex, or None  reference impedance for gamma.
                 Default: z_term at each driven port (the generator impedance),
                 for which P_avail (1 - |gamma|^2) = P_in exactly.
        """
        P = self.n_ports
        e = np.asarray(emf, dtype=np.complex128).reshape(-1)
        zd = np.asarray(z_term, dtype=np.complex128).reshape(-1)
        if e.shape != (P,) or zd.shape != (P,):
            raise ValueError(f"emf and z_term must have shape ({P},), got {e.shape} and {zd.shape}.")
        if not np.all(np.isfinite(zd)):
            bad = [self.port_names[p] for p in np.flatnonzero(~np.isfinite(zd))]
            raise ValueError(
                f"Non-finite termination at port(s) {bad}. An open circuit (Z = inf, i_p = 0) "
                f"is not supported by the reduced model. Note that leaving the port out is a "
                f"SHORT circuit (continuous metal), not an open one: to model an open gap, "
                f"cut it in the mesh (remove the port's edges), or use a large finite "
                f"impedance as an approximation.")
        if not np.all(np.isfinite(e)):
            raise ValueError("emf must be finite.")
        driven = (e != 0) if driven is None else np.asarray(driven, dtype=bool).reshape(-1)
        if driven.shape != (P,):
            raise ValueError(f"driven must have shape ({P},), got {driven.shape}.")

        K = np.eye(P, dtype=np.complex128) + self.Y * zd[None, :]     # 1 + Y diag(Z_D)
        i = np.linalg.solve(K, self.Y @ e)
        v = e - zd * i

        if z_ref is None:
            zr = zd.copy()
        else:
            zr = np.broadcast_to(np.asarray(z_ref, dtype=np.complex128), (P,)).copy()

        with np.errstate(divide='ignore', invalid='ignore'):
            z_in = np.where(driven, v / i, np.nan + 0j)
            gamma = np.where(driven, (z_in - np.conj(zr)) / (z_in + zr), np.nan + 0j)
            re_zd = zd.real
            p_avail = np.where(driven,
                               np.where(re_zd > 0, np.abs(e) ** 2 / (8.0 * re_zd), np.inf),
                               0.0)

        p_rad = float(np.real(v.conj() @ self.M_rad @ v))
        p_cond = float(np.real(v.conj() @ self.M_cond @ v))
        loaded = ~driven
        p_loads = float(0.5 * np.sum(zd.real[loaded] * np.abs(i[loaded]) ** 2))
        p_in = np.where(driven, 0.5 * np.real(v * np.conj(i)), 0.0)

        functionals = {name: complex(h @ v) for name, h in self.functionals.items()}

        return PortSolution(emf=e, z_term=zd, driven=driven, v=v, i=i, z_in=z_in,
                            z_ref=zr, gamma=gamma, p_rad=p_rad, p_cond=p_cond,
                            p_loads=p_loads, p_in=p_in, p_avail=p_avail,
                            functionals=functionals)

    def edge_currents(self, v) -> np.ndarray:
        """I = B v, shape (N,): RWG edge currents for port voltages v (plots/checks)."""
        v = np.asarray(v, dtype=np.complex128).reshape(-1)
        if v.shape != (self.n_ports,):
            raise ValueError(f"v must have shape ({self.n_ports},), got {v.shape}.")
        return self.B @ v
