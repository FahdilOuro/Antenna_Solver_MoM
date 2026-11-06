"""
Tests for the rank-truncated CMA eigensolver and the per-mode spurious-mode
diagnostic.

The situation being modelled throughout: R = Re(Z) is the radiation operator and
is badly rank-deficient on any real mesh, because most RWG basis functions
describe currents that radiate essentially nothing. Regularizing (R + eps*I)
makes the solve succeed but populates those directions with modes whose
eigenvalue is a function of eps rather than of the structure. The synthetic
matrices here reproduce that exactly -- R with a known rank, X full rank -- so
the expected answer is known in advance rather than inferred from the solver.
"""

import numpy as np
import pytest

from solver.cma.cma_parameters import CMAParameters
from solver.cma.eigen_solver import GeneralizedEigenSolver
from solver.cma.mode_diagnostics import mode_report, format_mode_report

N = 24
RANK = 7


def _rank_deficient_system(seed=0):
    """R with exactly RANK nonzero eigenvalues; X symmetric and full rank."""
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(rng.standard_normal((N, N)))[0]
    mu = np.concatenate([rng.uniform(0.5, 5.0, RANK), np.zeros(N - RANK)])
    R = Q @ np.diag(mu) @ Q.T
    R = (R + R.T) * 0.5
    B = rng.standard_normal((N, N))
    X = (B + B.T) * 0.5
    return R, X, Q


class TestRankTruncatedSolver:

    def test_returns_exactly_rank_modes(self):
        R, X, _ = _rank_deficient_system()
        params = CMAParameters(n_modes=N, solver='rank_truncated')
        with pytest.warns(UserWarning, match="numerical rank"):
            eigenvalues, eigenvectors, _ = GeneralizedEigenSolver.solve(R, X, params)
        assert eigenvalues.shape == (RANK,), (
            f"asked for {N} modes on a rank-{RANK} R and got {eigenvalues.shape[0]}; "
            "the null space is being populated again")
        assert eigenvectors.shape == (N, RANK)

    def test_modes_are_r_orthonormal(self):
        """W^T R W = I is what the whitening buys; check it survives back-transform."""
        R, X, _ = _rank_deficient_system()
        params = CMAParameters(n_modes=RANK, solver='rank_truncated', normalize=False)
        _, J, _ = GeneralizedEigenSolver.solve(R, X, params)
        np.testing.assert_allclose(J.T @ R @ J, np.eye(RANK), atol=1e-9)

    def test_eigenvalue_equation_holds_on_the_retained_subspace(self):
        """
        X J = lambda R J holds where R is invertible, and CANNOT hold anywhere
        else: on the null space the equation reads X J = lambda * 0, which no
        finite lambda satisfies. So the residual is projected onto R's range --
        and the same modes measured on the full space give O(1), which the
        companion test below pins down so this projection is not mistaken for a
        loosened tolerance.
        """
        R, X, _ = _rank_deficient_system()
        params = CMAParameters(n_modes=RANK, solver='rank_truncated')
        lam, J, _ = GeneralizedEigenSolver.solve(R, X, params)

        mu, U = np.linalg.eigh(R)
        range_R = U[:, mu > 1e-6 * mu[-1]]
        for n in range(RANK):
            residual = range_R.T @ (X @ J[:, n] - lam[n] * R @ J[:, n])
            assert np.linalg.norm(residual) < 1e-9 * np.linalg.norm(range_R.T @ (X @ J[:, n])), (
                f"mode {n}: X J != lambda R J on the radiating subspace")

    def test_full_space_residual_is_large_and_that_is_correct(self):
        """
        Guard the guard for the projection above. If this ever starts passing at
        a tight tolerance, R is no longer rank-deficient in the fixture and the
        projected test stops proving anything.
        """
        R, X, _ = _rank_deficient_system()
        params = CMAParameters(n_modes=RANK, solver='rank_truncated')
        lam, J, _ = GeneralizedEigenSolver.solve(R, X, params)
        full = [np.linalg.norm(X @ J[:, n] - lam[n] * R @ J[:, n]) / np.linalg.norm(X @ J[:, n])
                for n in range(RANK)]
        assert min(full) > 0.1, (
            f"full-space residuals {full} are small, so the null space is not being "
            "exercised and the projected residual test is vacuous")

    def test_no_mode_lies_in_the_null_space(self):
        """The whole point: every returned mode must actually radiate."""
        R, X, _ = _rank_deficient_system()
        params = CMAParameters(n_modes=RANK, solver='rank_truncated')
        lam, J, _ = GeneralizedEigenSolver.solve(R, X, params)
        rep = mode_report(J, R, X, lam)
        assert not rep['spurious'].any(), (
            f"rank truncation returned spurious modes: {format_mode_report(rep, lam)}")
        assert rep['rank_R'] == RANK

    def test_result_does_not_depend_on_the_regularization_parameter(self):
        """
        The defining difference from 'regularized'. eps is swept over four orders
        of magnitude; a truncated solve cannot see it at all.
        """
        R, X, _ = _rank_deficient_system()
        ref = None
        for eps in (1e-14, 1e-12, 1e-10, 1e-8):
            params = CMAParameters(n_modes=RANK, solver='rank_truncated',
                                   regularization_epsilon=eps)
            lam, _, _ = GeneralizedEigenSolver.solve(R, X, params)
            if ref is None:
                ref = lam
            else:
                np.testing.assert_allclose(lam, ref, rtol=1e-12)

    def test_regularized_solver_by_contrast_does_depend_on_it(self):
        """
        Guard the guard: if this ever stops failing, the synthetic system is no
        longer rank-deficient and the test above proves nothing.
        """
        R, X, _ = _rank_deficient_system()
        out = []
        for eps in (1e-12, 1e-8):
            params = CMAParameters(n_modes=RANK, solver='regularized',
                                   regularization_epsilon=eps)
            lam, _, _ = GeneralizedEigenSolver.solve(R, X, params)
            out.append(lam)
        assert not np.allclose(out[0], out[1], rtol=1e-6), (
            "the regularized solver gave eps-independent eigenvalues on a "
            "rank-deficient R, so this fixture is not exercising the null space")

    def test_unknown_solver_is_rejected(self):
        R, X, _ = _rank_deficient_system()
        with pytest.raises(ValueError, match="Unknown solver"):
            GeneralizedEigenSolver.solve(R, X, CMAParameters(solver='magic'))


class TestModeReport:

    def test_flags_a_pure_null_space_current_as_spurious(self):
        R, X, Q = _rank_deficient_system()
        J = np.column_stack([Q[:, -1], Q[:, 0]])   # null-space direction, then a radiating one
        rep = mode_report(J, R)
        assert rep['verdict'][0] == 'SPURIOUS'
        assert rep['verdict'][1] == 'PHYSICAL'
        assert rep['leak'][0] > 0.99, "a pure null-space vector must leak ~1"
        assert rep['leak'][1] < 1e-8

    def test_leakage_catches_a_mode_power_alone_would_miss(self):
        """
        A mode that is 99% null space but carries a sliver of radiating content
        keeps P_rel above 1e-6. Leakage is what still flags it -- which is why
        both criteria are applied, not just the power one.
        """
        R, X, Q = _rank_deficient_system()
        J = np.column_stack([0.03 * Q[:, 0] + Q[:, -1], Q[:, 0]])
        rep = mode_report(J, R)
        assert rep['p_rel'][0] > 1e-6, "fixture no longer isolates the leakage criterion"
        assert rep['leak'][0] > 0.5
        assert rep['verdict'][0] == 'SPURIOUS'
        assert "leak" in rep['reason'][0]

    def test_residual_is_small_for_a_genuine_eigenpair(self):
        R, X, _ = _rank_deficient_system()
        params = CMAParameters(n_modes=RANK, solver='rank_truncated')
        lam, J, _ = GeneralizedEigenSolver.solve(R, X, params)
        rep = mode_report(J, R, X, lam)
        assert np.all(rep['residual'] < 1e-8), f"residuals {rep['residual']}"

    def test_accepts_a_single_mode_as_1d(self):
        R, X, Q = _rank_deficient_system()
        rep = mode_report(Q[:, 0], R)
        assert rep['p_rel'].shape == (1,)

    def test_power_criterion_survives_the_eigensolver_normalisation(self):
        """
        The regression this exists for. scipy's eigh(X, R) returns eigenvectors
        normalised so that J^T R J = 1, and on a real mesh that holds for the
        near-null modes too: they are not pure null-space vectors but mixtures,
        97% null-space by norm with the remaining 3% carrying the whole R-norm.
        Measuring the raw J^T R J then reports 1.0 across the board and the power
        criterion silently stops discriminating -- measured doing exactly that on
        the CubeSat mesh at 440/300/233/500 MHz, where every mode read
        P_rel = 1.000e+00. Dividing by J^T J is what restores it.

        The property, stated directly: two modes carrying the SAME J^T R J but
        different amounts of null-space content must not receive the same score.
        Built explicitly here rather than fished out of a spectrum, so the test
        pins the metric rather than a particular fixture's eigenvalues.
        """
        R, _X, Q = _rank_deficient_system()
        v = Q[:, 0]                                  # radiating direction
        u = Q[:, -1]                                 # null direction
        a = 1.0 / np.sqrt(v @ R @ v)                 # scale to J^T R J = 1

        clean = a * v
        padded = a * v + 300.0 * u                   # same R-norm, far bigger ||J||
        J = np.column_stack([clean, padded])

        # Precondition: the naive metric cannot tell these apart at all.
        raw = np.einsum('ik,ik->k', J, R @ J)
        np.testing.assert_allclose(raw, [1.0, 1.0], rtol=1e-8,
                                   err_msg="the two modes no longer share an R-norm")

        rep = mode_report(J, R)
        assert rep['p_rel'][1] < 1e-4 * rep['p_rel'][0], (
            f"P_rel = {rep['p_rel']} barely separates a clean mode from one padded "
            "with 300x its own norm of non-radiating current; the power criterion "
            "is measuring the normalisation convention, not the physics")
        assert rep['verdict'][0] == 'PHYSICAL'
        assert rep['verdict'][1] == 'SPURIOUS'
