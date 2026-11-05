"""
Per-geometry evaluation for the K-element campaigns (step 6, part C).

For one geometry (one Z at f0) and EVERY choice of fed element -- free, since Z
and the reduced model do not depend on it, only the role of the ports does:

  1. continuous optimal load reactances (inner.optimize_loads), objective G
     (ideal matching) towards +y, phi polarisation, alpha = alpha_cal(w);
  2. snapping to the capacitor kit (components.snap_to_kit), ESR = ESL = 0 in
     this loop, the best combination being chosen by the ROBUST OBJECTIVE:
         G_rob = mean of G (linear) over
                 {alpha_cal x 0.85, 1, 1.15} x {kit tolerance -, nominal, +}
         (the tolerance sign is applied to all loads together; 9 solves),
         times a soft matching penalty (min(1, Re Zin / r_in_min))^2 at the
         nominal point;
     the worst case of the 9 is reported too;
  3. on the retained parts: G, G_r(50 Ohm), D, eta, Zin, F/B, G_max and
     G/G_max (G_max does not depend on the fed element: computed once), the
     relative tolerance of every load and the gain spread between its two
     tolerance ends (others nominal), and an ESR x ESL sensitivity grid
     (components.PROVISIONAL_*_GRID).

Frequency drift and input Q need extra Z fills: post-checks on the best points
only (post_checks in the campaign script).
"""

from __future__ import annotations

import numpy as np

from backend.optim import ReducedMultiport, incidence_matrix
from backend.optim.components import (
    IDEAL, JOHANSON_S603DS, PROVISIONAL_ESL_GRID, PROVISIONAL_ESR_GRID, Capacitor,
    ComponentModel, IdealReactance, series_impedance, snap_to_kit,
)
from backend.optim.farfield import RadiationOperator
from backend.optim.inner import optimize_loads
from backend.optim.loss_model import ALPHA_UNCERTAINTY, alpha_calibrated
from backend.optim.objectives import geometry_gain_bound
from backend.optim.threads import limit_blas_threads, threads_for_size
from backend.utils.lossy_electric_conductor import calculate_normal_surface_impedance

DIRECTION = (np.pi / 2, np.pi / 2)        # +y
BACK = (np.pi / 2, 3 * np.pi / 2)         # -y
POLARIZATION = "phi"
Z_GEN = 50.0
SIGMA_CU = 5.8e7
ALPHA_FACTORS = (1 - ALPHA_UNCERTAINTY, 1.0, 1 + ALPHA_UNCERTAINTY)
TOL_SIGNS = (-1, 0, 1)


def _db(x):
    return float(10 * np.log10(x)) if x is not None and x > 0 else float("nan")


class DesignContext:
    """One geometry at one frequency: the three loss models (one LU each), the far-field
    operator with its +y functional, and the geometry bound G_max at alpha_cal."""

    def __init__(self, mesh, Z, G_gram, f0, width_mm, port_names=None,
                 alpha_factors=ALPHA_FACTORS, compute_bound=True):
        self.mesh, self.f0, self.width_mm = mesh, float(f0), float(width_mm)
        names = port_names or sorted(mesh.cuts, key=lambda n: int(n[2:]))      # el1, el2, ... (-y -> +y)
        self.port_names = list(names)
        self.K = len(names)
        self.A = incidence_matrix([mesh.port(n) for n in names], mesh.edges)
        self.alpha_cal = alpha_calibrated(width_mm)
        zs = calculate_normal_surface_impedance(f0, SIGMA_CU)
        self.op = RadiationOperator(mesh.points, mesh.triangles, mesh.edges, f0)
        self.models = {}
        with limit_blas_threads(threads_for_size(mesh.n_edges)):
            for a in alpha_factors:
                self.models[a] = ReducedMultiport.from_matrices(Z, self.A, f0, G=G_gram, Zs=zs,
                                                                zs_scale=a * self.alpha_cal,
                                                                keep_matrices=(a == 1.0 and compute_bound))
            bound = (geometry_gain_bound(self.models[1.0], self.op, *DIRECTION, POLARIZATION, verbose=False)
                     if compute_bound else None)
        self.g_max = bound.g_max if bound else None
        self.g_max_cond = bound.cond if bound else None
        self.fname = {}
        for a, m in self.models.items():
            self.fname[a] = self.op.attach_functional(m, *DIRECTION, component=POLARIZATION, name="Nphi_+y")
        self.f_back = self.op.vectors(np.atleast_1d(BACK[0]), np.atleast_1d(BACK[1]))[1][0]

    def solve(self, fed, load_z, alpha_factor=1.0):
        """Gain figures for the fed element `fed` (index) and load impedances (K-1, port order)."""
        m = self.models[alpha_factor]
        emf = np.zeros(self.K, complex)
        emf[fed] = 1.0
        zt = np.empty(self.K, complex)
        zt[fed] = Z_GEN
        zt[[q for q in range(self.K) if q != fed]] = load_z
        sol = m.solve(emf, zt)
        U = self.op.intensity_constant * abs(sol.functionals[self.fname[alpha_factor]]) ** 2
        p_in = sol.p_in_total
        return dict(G=4 * np.pi * U / p_in, G_r=4 * np.pi * U / sol.p_avail_total,
                    D=4 * np.pi * U / sol.p_rad, eta=sol.p_rad / p_in, zin=complex(sol.z_in[fed]),
                    sol=sol, U=U, model=m)

    def front_back_db(self, res):
        I = res["model"].edge_currents(res["sol"].v)
        n_back = self.f_back @ I
        u_back = self.op.intensity_constant * abs(n_back) ** 2
        return float(10 * np.log10(res["U"] / u_back)) if u_back > 0 else float("inf")


def load_impedances(parts, f, model: ComponentModel, tol_sign=0):
    return np.array([series_impedance(p, f, model, tol_sign) for p in parts])


def robust_score(ctx, fed, parts, r_in_min, model=IDEAL):
    """(objective, mean G, worst G, nominal result): see module docstring."""
    gs = []
    for a in ALPHA_FACTORS:
        for s in TOL_SIGNS:
            gs.append(ctx.solve(fed, load_impedances(parts, ctx.f0, model, s), a)["G"])
    nominal = ctx.solve(fed, load_impedances(parts, ctx.f0, model, 0), 1.0)
    penalty = min(1.0, nominal["zin"].real / r_in_min) ** 2 if r_in_min else 1.0
    mean_g = float(np.mean(gs))
    return mean_g * max(penalty, 0.0), mean_g, float(np.min(gs)), nominal, penalty


def part_record(p, f0):
    if isinstance(p, Capacitor):
        return dict(kind="C", label=p.label, value_F=p.value, tol_abs_F=p.tol_abs, tol_rel=p.tol_rel,
                    x_ohm=-1 / (2 * np.pi * f0 * p.value))
    if isinstance(p, IdealReactance):
        return dict(kind="ideal", label=p.label, value_F=None, tol_abs_F=0.0, tol_rel=0.0, x_ohm=p.x_ohm)
    return dict(kind=type(p).__name__, label=p.label, value=p.value, tol_abs=p.tol_abs, tol_rel=p.tol_rel)


def evaluate_fed(ctx, fed, kit=JOHANSON_S603DS, r_in_min=10.0, n_starts=32, seed=0):
    """Full record for one fed element (see module docstring)."""
    loaded = [q for q in range(ctx.K) if q != fed]
    res = optimize_loads(ctx.models[1.0], DIRECTION, POLARIZATION, "G", n_starts=n_starts, seed=seed,
                         operator=ctx.op, z_gen=Z_GEN, driven_port=fed, compute_bound=False)
    snap = snap_to_kit(res.x_opt, ctx.f0, IDEAL,
                       score=lambda parts: robust_score(ctx, fed, parts, r_in_min)[0], kit=kit)
    parts = snap.best
    obj, g_mean, g_worst, nom, penalty = robust_score(ctx, fed, parts, r_in_min)

    loads = []
    for k, (q, p) in enumerate(zip(loaded, parts)):
        rec = part_record(p, ctx.f0)
        rec.update(port=ctx.port_names[q], x_continuous_ohm=float(res.x_opt[k]))
        if isinstance(p, Capacitor):
            gk = []
            for s in (-1, 1):
                z = load_impedances(parts, ctx.f0, IDEAL, 0)
                z[k] = series_impedance(p, ctx.f0, IDEAL, s)
                gk.append(ctx.solve(fed, z, 1.0)["G"])
            rec["dG_tolerance_dB"] = _db(gk[1]) - _db(gk[0])            # G(+tol) - G(-tol)
        loads.append(rec)

    esr_esl = []
    for esr in PROVISIONAL_ESR_GRID:
        for esl in PROVISIONAL_ESL_GRID:
            r = ctx.solve(fed, load_impedances(parts, ctx.f0, ComponentModel(esr, esl), 0), 1.0)
            esr_esl.append(dict(esr_ohm=esr, esl_nH=esl * 1e9, G_dBi=_db(r["G"]), eta=r["eta"]))

    return dict(
        fed=ctx.port_names[fed], fed_index=fed, f0_GHz=ctx.f0 / 1e9, alpha_cal=ctx.alpha_cal,
        objective=obj, objective_dB=_db(obj), penalty=penalty,
        G_continuous_dBi=_db(res.G), G_kit_dBi=_db(nom["G"]), G_robust_mean_dBi=_db(g_mean),
        G_robust_worst_dBi=_db(g_worst), G_r_dBi=_db(nom["G_r"]), D_dBi=_db(nom["D"]), eta=nom["eta"],
        Zin_re_ohm=nom["zin"].real, Zin_im_ohm=nom["zin"].imag, FB_dB=ctx.front_back_db(nom),
        G_max_dBi=_db(ctx.g_max), G_over_G_max=(nom["G"] / ctx.g_max if ctx.g_max else None),
        loads=loads, notes=snap.notes, n_combinations=len(snap.combinations),
        esr_esl=esr_esl, esr_esl_G_min_dBi=min(r["G_dBi"] for r in esr_esl),
        n_evaluations_inner=res.n_evaluations)


def evaluate_design(ctx, kit=JOHANSON_S603DS, r_in_min=10.0, n_starts=32, seed=0):
    """All fed positions; returns (records, index of the best by robust objective)."""
    recs = [evaluate_fed(ctx, fed, kit, r_in_min, n_starts, seed) for fed in range(ctx.K)]
    best = int(np.argmax([r["objective"] for r in recs]))
    return recs, best
