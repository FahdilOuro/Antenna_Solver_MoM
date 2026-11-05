"""
Outer loop: Bayesian optimisation of the geometry, and the geometric
reparametrisation of K-element arrays.

Principle of Bayesian optimisation (BO)
---------------------------------------
Each evaluation of the objective is expensive (an impedance-matrix fill, ~1 min),
so instead of probing it blindly we build a cheap SURROGATE MODEL of it from the
points already evaluated: a Gaussian process (GP). For any untested x the GP
gives a prediction mu(x) AND an uncertainty sigma(x) -- small near evaluated
points, large far from them. The next point is the one that maximises the
EXPECTED IMPROVEMENT over the best value found so far, y*:

    EI(x) = E[max(f(x) - y* - xi, 0)] = (mu - y* - xi) Phi(z) + sigma phi(z),
    z = (mu - y* - xi) / sigma

EI is large where mu is high (exploitation: refine around good designs) OR where
sigma is high (exploration: look where we know little); the formula weighs the
two automatically. Each new evaluation refines the GP, and the loop stops after
n_iter points or when the best EI stays negligible (nothing left to gain).

Implementation: scikit-learn GaussianProcessRegressor, kernel
Constant x Matern(nu = 5/2, one length scale per variable: ARD) + WhiteKernel
(noise), normalised outputs; inputs mapped to the unit cube. EI is maximised by
dense random sampling followed by L-BFGS-B from the best candidates. Initial
design: Latin hypercube. Every evaluation is appended to a history (CSV + JSON);
a run restarted on the same history resumes where it stopped and, all random
draws being seeded by (seed, iteration), proposes the same points as an
uninterrupted run.

Geometric reparametrisation (constraints without penalties)
-----------------------------------------------------------
K parallel strips of width w, edge-to-edge extent s (<= 120 mm), K-1
centre-to-centre gaps d_i >= d_min. Variables: s and K-2 numbers u in [0, 1]:

    centre span        S = s - w = sum_i d_i
    free span          F = S - (K - 1) d_min   (>= 0 required: s >= w + (K-1) d_min)
    fractions          t = stick_breaking(u), t_i >= 0, sum t_i = 1
    gaps               d_i = d_min + t_i F

stick_breaking maps the cube [0,1]^(K-2) onto the simplex of K-1 fractions
smoothly and uniformly (Dirichlet(1,...,1)): t_1 = 1 - (1 - u_1)^(1/(K-2)),
then each next fraction takes a share of what is left. Every point of the box
is therefore a valid geometry: the box and non-overlap constraints hold by
construction and the GP never sees infeasible points.
"""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, qmc


# =============================================================================
# Geometry reparametrisation
# =============================================================================
def stick_breaking(u):
    """u in [0,1]^(n-1) -> fractions t (n,), t >= 0, sum = 1 (uniform on the simplex)."""
    u = np.clip(np.atleast_1d(np.asarray(u, dtype=float)), 0.0, 1.0)
    n = u.size + 1
    t = np.empty(n)
    rest = 1.0
    for i, ui in enumerate(u):
        b = 1.0 - (1.0 - ui) ** (1.0 / (n - 1 - i))     # Beta(1, n-1-i) quantile
        t[i] = rest * b
        rest -= t[i]
    t[-1] = rest
    return t


def inverse_stick_breaking(t):
    """Inverse of stick_breaking (for tests and for seeding with known geometries)."""
    t = np.asarray(t, dtype=float)
    n = t.size
    u = np.empty(n - 1)
    rest = 1.0
    for i in range(n - 1):
        b = t[i] / rest if rest > 0 else 0.0
        u[i] = 1.0 - (1.0 - b) ** (n - 1 - i)
        rest -= t[i]
    return u


def spacings_from_params(s, u, n_elements, width, d_min):
    """Centre-to-centre gaps (K-1,) from the extent s and the simplex coordinates u."""
    F = (s - width) - (n_elements - 1) * d_min
    if F < -1e-12:
        raise ValueError(f"extent s = {s * 1e3:.2f} mm < w + (K-1) d_min = "
                         f"{(width + (n_elements - 1) * d_min) * 1e3:.2f} mm")
    t = stick_breaking(u) if n_elements > 2 else np.ones(1)
    return d_min + t * max(F, 0.0)


def extent_bounds(n_elements, width, d_min, s_max):
    return (width + (n_elements - 1) * d_min, s_max)


# =============================================================================
# Bayesian optimiser
# =============================================================================
def expected_improvement(mu, sigma, y_best, xi=0.0):
    """EI for MAXIMISATION (see module docstring)."""
    sigma = np.maximum(sigma, 1e-12)
    z = (mu - y_best - xi) / sigma
    return (mu - y_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)


class BayesianOptimizer:
    """
    Maximise objective(x) over a box (bounds: (d, 2) in physical units).

    objective(x) -> float, or (float, dict) to store extra information with the point.
    history_path : base path; '<base>.csv' (one row per evaluation) and
                   '<base>.json' (full records) are rewritten after every evaluation
                   and read back at start-up to resume.
    ei_stop, ei_patience : stop when the best EI stays below ei_stop for ei_patience
                   consecutive iterations (in objective units).
    """

    def __init__(self, objective: Callable, bounds, n_init=12, n_iter=40, seed=0,
                 history_path: Optional[str] = None, xi=0.0, ei_stop=None, ei_patience=8,
                 n_candidates=4000, n_local=5, gp_restarts=3, names: Optional[Sequence[str]] = None,
                 log=print):
        self.objective = objective
        self.bounds = np.asarray(bounds, dtype=float)
        self.d = self.bounds.shape[0]
        self.n_init, self.n_iter, self.seed = int(n_init), int(n_iter), int(seed)
        self.xi, self.ei_stop, self.ei_patience = xi, ei_stop, int(ei_patience)
        self.n_candidates, self.n_local, self.gp_restarts = n_candidates, n_local, gp_restarts
        self.names = list(names) if names else [f"x{i}" for i in range(self.d)]
        self.history_path = Path(history_path) if history_path else None
        self.log = log
        self.X, self.y, self.records = [], [], []
        self.ei_trace = []
        if self.history_path:
            self._load()

    # ------------------------------------------------------------- unit cube <-> physical
    def _to_unit(self, x):
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return (np.asarray(x, float) - lo) / (hi - lo)

    def _from_unit(self, u):
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return lo + np.clip(u, 0, 1) * (hi - lo)

    # ------------------------------------------------------------- history
    def _load(self):
        p = self.history_path.with_suffix(".json")
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            self.records = data["records"]
            self.X = [np.asarray(r["x"], float) for r in self.records]
            self.y = [float(r["y"]) for r in self.records]
            self.ei_trace = [r["ei"] for r in self.records if r["phase"] == "bo"]
            self.log(f"[BO] resumed {len(self.y)} evaluations from {p}")

    def _save(self):
        if not self.history_path:
            return
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.with_suffix(".json").write_text(json.dumps(dict(
            bounds=self.bounds.tolist(), names=self.names, seed=self.seed, n_init=self.n_init,
            n_iter=self.n_iter, records=self.records), indent=1, default=float),
            encoding="utf-8")
        with open(self.history_path.with_suffix(".csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["eval", "phase", *self.names, "y", "best_so_far", "ei"])
            best = -np.inf
            for r in self.records:
                best = max(best, r["y"])
                w.writerow([r["eval"], r["phase"], *[f"{v:.10g}" for v in r["x"]],
                            f"{r['y']:.10g}", f"{best:.10g}", "" if r.get("ei") is None else f"{r['ei']:.4g}"])

    def _evaluate(self, x, phase, ei=None):
        out = self.objective(np.asarray(x, float))
        y, info = (out if isinstance(out, tuple) else (out, {}))
        self.X.append(np.asarray(x, float))
        self.y.append(float(y))
        self.records.append(dict(eval=len(self.y), phase=phase, x=list(map(float, x)), y=float(y),
                                 ei=None if ei is None else float(ei), info=info))
        self._save()
        self.log(f"[BO] eval {len(self.y):3d} ({phase}) x = {np.round(x, 6).tolist()} -> y = {y:.5f} "
                 f"(best {max(self.y):.5f})" + ("" if ei is None else f", EI = {ei:.3g}"))
        return y

    # ------------------------------------------------------------- GP and acquisition
    def _fit_gp(self, iteration):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                  * Matern(length_scale=np.full(self.d, 0.3), length_scale_bounds=(1e-2, 1e1), nu=2.5)
                  + WhiteKernel(1e-4, (1e-8, 1e-1)))
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                      n_restarts_optimizer=self.gp_restarts,
                                      random_state=self.seed * 100003 + iteration)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(np.array([self._to_unit(x) for x in self.X]), np.array(self.y))
        return gp

    def _propose(self, gp, iteration):
        rng = np.random.default_rng([self.seed, iteration])
        y_best = max(self.y)

        def neg_ei(u):
            mu, sd = gp.predict(np.atleast_2d(u), return_std=True)
            return -float(expected_improvement(mu, sd, y_best, self.xi)[0])

        cand = rng.random((self.n_candidates, self.d))
        mu, sd = gp.predict(cand, return_std=True)
        ei = expected_improvement(mu, sd, y_best, self.xi)
        starts = cand[np.argsort(-ei)[:self.n_local]]
        best_u, best_val = starts[0], -ei.max()
        for u0 in starts:
            res = minimize(neg_ei, u0, method="L-BFGS-B", bounds=[(0.0, 1.0)] * self.d)
            if res.fun < best_val:
                best_u, best_val = res.x, res.fun
        return self._from_unit(best_u), -best_val

    # ------------------------------------------------------------- main loop
    def _stalled(self):
        """True when the last ei_patience BO proposals all had EI < ei_stop."""
        if self.ei_stop is None:
            return False
        tail = self.ei_trace[-self.ei_patience:]
        return len(tail) == self.ei_patience and all(e < self.ei_stop for e in tail)

    def run(self):
        init = qmc.LatinHypercube(d=self.d, seed=self.seed).random(self.n_init)
        for k in range(len(self.y), self.n_init):
            self._evaluate(self._from_unit(init[k]), "init")
        # The records are the only source of truth: an interruption between a
        # proposal and its evaluation simply re-proposes the same point.
        self.ei_trace = [r["ei"] for r in self.records if r["phase"] == "bo"]
        for it in range(len(self.ei_trace), self.n_iter):
            if self._stalled():
                break
            gp = self._fit_gp(it)
            x, ei = self._propose(gp, it)
            self._evaluate(x, "bo", ei)
            self.ei_trace.append(float(ei))
        if self._stalled():
            self.log(f"[BO] stopped early: max EI < {self.ei_stop} for {self.ei_patience} iterations")
        return self.best()

    def best(self):
        k = int(np.argmax(self.y))
        return dict(x=self.X[k], y=self.y[k], record=self.records[k], n_evaluations=len(self.y))
