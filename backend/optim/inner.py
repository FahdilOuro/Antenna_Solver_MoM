"""
Inner optimisation loop: best reactive loads for a fixed geometry.

Given a ReducedMultiport with one driven port and P-1 loaded ports, find the
load reactances X_p (Z_L,p = j X_p) maximising the gain in one direction and
polarisation. Each evaluation is a P x P solve on the reduced model (tens of
microseconds), so a multi-start local search is affordable:

  1. Latin-hypercube starting points in the box x_bounds^(P-1);
  2. L-BFGS-B on -log(objective) from each, in normalised variables
     u = (X - X_lo) / (X_hi - X_lo) in [0, 1] (finite-difference gradients
     then have a sensible step whatever the reactance scale);
  3. keep the best.

Objectives
----------
  "G"   : 4 pi U / P_in    -- gain with ideal matching (independent of the
                              generator impedance for lossless loads)
  "G_r" : 4 pi U / P_avail -- realized gain on a z_gen (default 50 Ohm) generator

Every evaluation runs under limit_blas_threads(1): the work is P x P.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

from backend.optim.farfield import RadiationOperator
from backend.optim.objectives import geometry_gain_bound
from backend.optim.threads import limit_blas_threads


@dataclass
class InnerResult:
    """
    Optimal loads and the figures of merit they give (frequency f0 = frequency).

    x_opt          : (P-1,) optimal reactances X_p (Ohm), loaded ports in port order
    z_loads        : (P-1,) complex, j X_p
    element_kind   : 'C' (X < 0), 'L' (X > 0) or 'short' (X = 0) per load
    element_value  : capacitance (F) or inductance (H) giving X_p at f0
    element_label  : e.g. '0.651 pF', '12.3 nH'
    G, G_r, D      : gain with ideal matching, realized gain on z_gen, directivity (linear)
    eta_rad        : P_rad / P_in
    z_in, gamma    : input impedance (Ohm) and reflection coefficient on z_gen
    g_max          : geometry bound (None if lossless or not computed)
    g_ratio        : G / g_max (None without bound)
    history        : one dict per start: x0, x, objective value, success, nfev
    """
    x_opt: np.ndarray
    z_loads: np.ndarray
    element_kind: List[str]
    element_value: np.ndarray
    element_label: List[str]
    G: float
    G_r: float
    D: float
    eta_rad: float
    z_in: complex
    gamma: complex
    g_max: Optional[float]
    g_ratio: Optional[float]
    objective: str
    direction: tuple
    polarization: str
    frequency: float
    z_gen: complex
    solution: object = None
    edge_currents: Optional[np.ndarray] = None
    history: list = field(default_factory=list)
    n_evaluations: int = 0

    def summary(self) -> str:
        loads = ", ".join(f"X={x:+.2f} Ohm ({lab})" for x, lab in zip(self.x_opt, self.element_label))
        ratio = "n/a" if self.g_ratio is None else f"{self.g_ratio:.3f}"
        return (f"G = {_db(self.G):.3f} dBi, G_r = {_db(self.G_r):.3f} dBi, D = {_db(self.D):.3f} dBi, "
                f"eta_rad = {self.eta_rad:.4f}, Zin = {self.z_in:.3f} Ohm, G/G_max = {ratio}; {loads}")


def _db(x):
    return 10 * np.log10(x)


def reactance_to_element(X, frequency):
    """('C', C) for X < 0 (X = -1/(w C)), ('L', L) for X > 0 (X = w L), ('short', 0) for 0."""
    w = 2 * np.pi * frequency
    if X < 0:
        c = -1.0 / (w * X)
        return "C", c, f"{c * 1e12:.4g} pF"
    if X > 0:
        ind = X / w
        return "L", ind, f"{ind * 1e9:.4g} nH"
    return "short", 0.0, "short"


def optimize_loads(model, direction, polarization="phi", objective="G",
                   x_bounds=(-2000.0, 2000.0), n_starts=32, seed=0,
                   operator: Optional[RadiationOperator] = None, z_gen=50.0,
                   driven_port=0, compute_bound=True) -> InnerResult:
    """
    Maximise `objective` ("G" or "G_r") in `direction` = (theta, phi) (rad) for the
    `polarization` component ("phi" or "theta") over the reactances of every port
    except `driven_port`. The far field comes from `operator`, or from a
    RadiationOperator built on model.mesh (models made by from_mesh).
    """
    if objective not in ("G", "G_r"):
        raise ValueError(f"objective must be 'G' or 'G_r', got {objective!r}.")
    P = model.n_ports
    if P < 2:
        raise ValueError("Need at least one loaded port.")
    loaded = [p for p in range(P) if p != driven_port]
    if operator is None:
        if model.mesh is None:
            raise ValueError("Pass `operator` (the model has no mesh reference).")
        m = model.mesh
        operator = RadiationOperator(m["points"], m["triangles"], m["edges"], model.frequency)
    theta, phi = direction
    name = operator.attach_functional(model, theta, phi, component=polarization)

    emf = np.zeros(P, dtype=complex)
    emf[driven_port] = 1.0
    lo, hi = map(float, x_bounds)
    z_term = np.zeros(P, dtype=complex)
    z_term[driven_port] = z_gen
    n_eval = [0]

    def evaluate(x):
        z_term[loaded] = 1j * np.asarray(x, dtype=float)
        sol = model.solve(emf, z_term)
        n_eval[0] += 1
        U = operator.intensity_constant * abs(sol.functionals[name]) ** 2
        return sol, 4 * np.pi * U / sol.p_in_total, 4 * np.pi * U / sol.p_avail_total

    def cost(u):
        _, g, g_r = evaluate(lo + u * (hi - lo))
        val = g if objective == "G" else g_r
        return -np.log(val) if val > 0 else 1e3

    d = len(loaded)
    starts = qmc.LatinHypercube(d=d, seed=seed).random(n_starts)
    history = []
    best = None
    with limit_blas_threads(1):
        for u0 in starts:
            res = minimize(cost, u0, method="L-BFGS-B", bounds=[(0.0, 1.0)] * d)
            x = lo + res.x * (hi - lo)
            history.append(dict(x0=(lo + u0 * (hi - lo)).tolist(), x=x.tolist(),
                                value=float(np.exp(-res.fun)), success=bool(res.success),
                                nfev=int(res.nfev)))
            if best is None or res.fun < best[1]:
                best = (x, res.fun)
        x_opt = best[0]
        sol, G, G_r = evaluate(x_opt)

    I = model.edge_currents(sol.v)
    U = operator.intensity_constant * abs(sol.functionals[name]) ** 2
    kinds, values, labels = zip(*(reactance_to_element(x, model.frequency) for x in x_opt))

    g_max = g_ratio = None
    if compute_bound and model.Z is not None:
        b = geometry_gain_bound(model, operator, theta, phi, polarization, verbose=False)
        if b is not None:
            g_max, g_ratio = b.g_max, G / b.g_max

    return InnerResult(
        x_opt=np.asarray(x_opt), z_loads=1j * np.asarray(x_opt), element_kind=list(kinds),
        element_value=np.asarray(values), element_label=list(labels),
        G=float(G), G_r=float(G_r), D=float(4 * np.pi * U / sol.p_rad),
        eta_rad=float(sol.p_rad / sol.p_in_total), z_in=complex(sol.z_in[driven_port]),
        gamma=complex(sol.gamma[driven_port]), g_max=g_max, g_ratio=g_ratio,
        objective=objective, direction=(float(theta), float(phi)), polarization=polarization,
        frequency=model.frequency, z_gen=complex(z_gen), solution=sol, edge_currents=I,
        history=history, n_evaluations=n_eval[0])
