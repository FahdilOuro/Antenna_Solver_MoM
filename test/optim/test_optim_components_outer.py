"""
Step 6 building blocks: real components (kit, series model, snapping), power
balance with ESR, Bayesian optimisation on analytic functions (+ resume), the
geometric reparametrisation, and a mini 3-element design evaluation.
"""

import numpy as np
import pytest

from solver.optim import ReducedMultiport
from solver.optim.components import (
    IDEAL, JOHANSON_S603DS, Capacitor, ComponentModel, IdealReactance, bracketing_parts,
    effective_reactance, series_impedance, snap_to_kit,
)
from solver.optim.outer import (
    BayesianOptimizer, extent_bounds, inverse_stick_breaking, spacings_from_params, stick_breaking,
)

F0 = 0.9e9
W0 = 2 * np.pi * F0


# --------------------------------------------------------------------------- components
def test_kit_catalogue_and_series_model():
    kit = {c.label: c for c in JOHANSON_S603DS}
    assert len(kit) == 28
    assert kit["1 pF"].tol_abs == pytest.approx(0.10e-12) and kit["1 pF"].tol_rel == pytest.approx(0.10)
    assert kit["6.8 pF"].tol_abs == pytest.approx(0.25e-12)
    assert kit["10 pF"].tol_abs == pytest.approx(0.5e-12) and kit["82 pF"].tol_rel == pytest.approx(0.05)
    assert effective_reactance(kit["1 pF"], F0, IDEAL) == pytest.approx(-176.84, abs=0.01)
    m = ComponentModel(esr=0.3, esl=1.0e-9)
    z = series_impedance(kit["1 pF"], F0, m, tol_sign=+1)
    assert z == pytest.approx(0.3 + 1j * (W0 * 1e-9 - 1 / (W0 * 1.1e-12)))
    with pytest.raises(ValueError):
        ComponentModel(esr=-0.1, esl=0.0)
    # ideal reactances keep their L or C nature off f0
    assert series_impedance(IdealReactance(50.0, F0, "x"), 2 * F0, IDEAL) == pytest.approx(100j)
    assert series_impedance(IdealReactance(-50.0, F0, "x"), 2 * F0, IDEAL) == pytest.approx(-25j)


def test_bracketing_and_inductive_flag():
    parts, note = bracketing_parts(-150.0, F0, IDEAL)            # between 1.0 pF (-177) and 1.2 pF (-147)
    assert [p.label for p in parts] == ["1 pF", "1.2 pF"] and note is None
    x = [effective_reactance(p, F0, IDEAL) for p in parts]
    assert x[0] <= -150.0 <= x[1]
    parts, note = bracketing_parts(+40.0, F0, IDEAL)
    assert isinstance(parts[0], IdealReactance) and "inductor needed" in note
    parts, note = bracketing_parts(-900.0, F0, IDEAL)            # beyond 0.3 pF
    assert [p.label for p in parts] == ["0.3 pF"] and "smallest" in note
    # with ESL the bracketing uses the EFFECTIVE reactance
    m = ComponentModel(esr=0.0, esl=1e-9)
    parts, _ = bracketing_parts(-150.0, F0, m)
    x = [effective_reactance(p, F0, m) for p in parts]
    assert x[0] <= -150.0 <= x[-1]


def test_snap_to_kit_evaluates_all_combinations():
    targets = np.array([-150.0, -60.0, -20.0])
    alt = np.array([-170.0, -50.0, -25.0])                       # the score prefers these
    seen = []

    def score(parts):
        seen.append(parts)
        x = np.array([effective_reactance(p, F0, IDEAL) for p in parts])
        return -np.sum(np.abs(x - alt))

    res = snap_to_kit(targets, F0, IDEAL, score)
    assert len(res.combinations) == 8 == len(seen)
    assert res.best_score == max(s for _, s in res.combinations)
    res2 = snap_to_kit(np.array([-150.0, 30.0]), F0, IDEAL, score=lambda p: 0.0)
    assert any("inductor needed" in n for n in res2.notes)
    assert len(res2.combinations) == 2                           # 2 capacitors x 1 ideal inductor


def test_power_balance_with_esr(wide_model):
    """P_in = P_rad + P_cond + P_loads with real (ESR) capacitors on the loaded ports."""
    kit = {c.label: c for c in JOHANSON_S603DS}
    m = ComponentModel(esr=0.3, esl=0.5e-9)
    f = wide_model.frequency
    z_loads = [series_impedance(kit["2.2 pF"], f, m), series_impedance(kit["8.2 pF"], f, m)]
    sol = wide_model.solve([1.0, 0, 0], [50.0, *z_loads])
    assert sol.p_loads > 0
    assert sol.p_loads == pytest.approx(0.5 * 0.3 * np.sum(np.abs(sol.i[1:]) ** 2), rel=1e-12)
    assert abs(sol.power_balance_residual) / sol.p_in_total < 1e-10


# --------------------------------------------------------------------------- reparametrisation
def test_stick_breaking_and_spacings():
    rng = np.random.default_rng(0)
    for n in (2, 3, 4):
        for _ in range(20):
            u = rng.random(n - 1)
            t = stick_breaking(u)
            assert t.shape == (n,) and np.all(t >= 0) and t.sum() == pytest.approx(1.0)
            np.testing.assert_allclose(inverse_stick_breaking(t), u, atol=1e-12)
    w, d_min, s_max = 4e-3, 15e-3, 0.120
    lo, hi = extent_bounds(4, w, d_min, s_max)
    assert lo == pytest.approx(w + 3 * d_min) and hi == s_max
    for _ in range(50):
        s = rng.uniform(lo, hi)
        d = spacings_from_params(s, rng.random(2), 4, w, d_min)
        assert d.shape == (3,) and np.all(d >= d_min - 1e-15)
        assert d.sum() + w == pytest.approx(s)                   # edge-to-edge extent = s
    with pytest.raises(ValueError):
        spacings_from_params(lo - 1e-3, [0.5, 0.5], 4, w, d_min)


# --------------------------------------------------------------------------- Bayesian optimisation
def branin(x):
    a, b, c, r, s, t = 1, 5.1 / (4 * np.pi ** 2), 5 / np.pi, 6, 10, 1 / (8 * np.pi)
    return a * (x[1] - b * x[0] ** 2 + c * x[0] - r) ** 2 + s * (1 - t) * np.cos(x[0]) + s


def hartmann3(x):
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
    P = 1e-4 * np.array([[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]])
    return -np.sum(alpha * np.exp(-np.sum(A * (x - P) ** 2, axis=1)))


@pytest.mark.parametrize("seed", [0, 1])
def test_bo_branin(seed):
    bo = BayesianOptimizer(lambda x: -branin(x), [(-5, 10), (0, 15)], n_init=8, n_iter=32, seed=seed,
                           n_candidates=2000, n_local=3, gp_restarts=2, log=lambda *a: None)
    best = bo.run()
    print(f"\n[Branin, seed {seed}] min found {-best['y']:.4f} (true 0.3979) in {best['n_evaluations']} evaluations")
    assert best["n_evaluations"] <= 40
    assert -best["y"] - 0.397887 < 0.05


@pytest.mark.parametrize("seed", [0, 1])
def test_bo_hartmann3(seed):
    bo = BayesianOptimizer(lambda x: -hartmann3(x), [(0, 1)] * 3, n_init=10, n_iter=30, seed=seed,
                           n_candidates=2000, n_local=3, gp_restarts=2, log=lambda *a: None)
    best = bo.run()
    print(f"\n[Hartmann-3, seed {seed}] min found {-best['y']:.4f} (true -3.8628) at "
          f"{np.round(best['x'], 3)} in {best['n_evaluations']} evaluations")
    assert best["n_evaluations"] <= 40
    assert -best["y"] < -3.80


class Interrupted(Exception):
    pass


def test_bo_resume_reproduces_uninterrupted_run(tmp_path):
    kw = dict(bounds=[(-5, 10), (0, 15)], n_init=6, n_iter=8, seed=3, n_candidates=1000, n_local=2,
              gp_restarts=1, log=lambda *a: None)
    ref = BayesianOptimizer(lambda x: -branin(x), history_path=str(tmp_path / "ref"), **kw)
    ref.run()

    calls = {"n": 0}

    def flaky(x):
        calls["n"] += 1
        if calls["n"] == 10:                                     # crash during the 10th evaluation
            raise Interrupted()
        return -branin(x)

    with pytest.raises(Interrupted):
        BayesianOptimizer(flaky, history_path=str(tmp_path / "run"), **kw).run()
    resumed = BayesianOptimizer(lambda x: -branin(x), history_path=str(tmp_path / "run"), **kw)
    assert len(resumed.y) == 9
    resumed.run()
    np.testing.assert_allclose(np.array(resumed.X), np.array(ref.X), rtol=0, atol=1e-12)
    np.testing.assert_allclose(resumed.y, ref.y, rtol=0, atol=1e-12)
    assert (tmp_path / "run.csv").is_file()


# --------------------------------------------------------------------------- mini design evaluation
def test_mini_three_element_design():
    from solver.optim.design_eval import DesignContext, evaluate_design
    from solver.optim.geometries import parallel_strip_array
    from solver.rwg.rwg3 import calculate_z_matrice
    from solver.utils.impmet import rwg_gram_matrix

    w = 4e-3
    m = parallel_strip_array([0.18] * 3, w, [0.03, 0.03], n_width=2, cell_length=0.0225)
    *_, Z = calculate_z_matrice(m.points, m.triangles, m.edges, F0, use_gpu=False)
    ctx = DesignContext(m, Z, rwg_gram_matrix(m.edges, m.triangles, m.points), F0, w * 1e3)
    recs, best = evaluate_design(ctx, n_starts=6, seed=0)
    assert len(recs) == 3 and best == int(np.argmax([r["objective"] for r in recs]))
    for r in recs:
        assert len(r["loads"]) == 2 and r["fed"] not in {ld["port"] for ld in r["loads"]}
        assert r["G_kit_dBi"] <= r["G_continuous_dBi"] + 1e-6    # snapped point is feasible in the continuum
        assert r["G_robust_worst_dBi"] <= r["G_robust_mean_dBi"] + 1e-12
        assert len(r["esr_esl"]) == 9
        assert r["G_kit_dBi"] <= r["G_max_dBi"] + 1e-9
    b = recs[best]
    print(f"\n[mini 3 elements] best fed {b['fed']}: G cont {b['G_continuous_dBi']:.2f}, kit {b['G_kit_dBi']:.2f}, "
          f"robust {b['G_robust_mean_dBi']:.2f} (worst {b['G_robust_worst_dBi']:.2f}) dBi, loads "
          + ", ".join(f"{ld['port']}: {ld['label']}" for ld in b["loads"]) + f"; notes {b['notes']}")
