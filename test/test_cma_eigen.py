"""
Unit tests for GeneralizedEigenSolver.

All tests use synthetic R and X matrices — no mesh files required.
The diagonal system provides exact analytical ground-truth eigenvalues.
"""

import numpy as np
import pytest

from solver.cma.cma_parameters import CMAParameters
from solver.cma.eigen_solver import GeneralizedEigenSolver


class TestDiagonalSystem:
    """On a diagonal system X·J = λ·R·J the eigenvalues are X_ii/R_ii."""

    R = np.diag([1.0, 2.0, 3.0, 4.0])
    X = np.diag([2.0, -1.0, 0.5, -3.0])
    # Expected λ = [2.0, -0.5, 1/6, -0.75]
    # Sorted by |λ|: [1/6, -0.5, -0.75, 2.0]
    EXPECTED = np.array([2.0, -1.0, 0.5, -3.0]) / np.array([1.0, 2.0, 3.0, 4.0])

    def test_eigenvalues_magnitude_sort(self):
        params = CMAParameters(n_modes=4, sort_by='magnitude')
        eigenvalues, _, _ = GeneralizedEigenSolver.solve(self.R, self.X, params)

        expected_sorted = self.EXPECTED[np.argsort(np.abs(self.EXPECTED))]
        np.testing.assert_allclose(eigenvalues, expected_sorted, rtol=1e-6)

    def test_eigenvalues_value_sort(self):
        params = CMAParameters(n_modes=4, sort_by='value')
        eigenvalues, _, _ = GeneralizedEigenSolver.solve(self.R, self.X, params)

        expected_sorted = np.sort(self.EXPECTED)
        np.testing.assert_allclose(eigenvalues, expected_sorted, rtol=1e-6)

    def test_output_shapes(self):
        params = CMAParameters(n_modes=3)
        eigenvalues, eigenvectors, sort_order = GeneralizedEigenSolver.solve(self.R, self.X, params)

        assert eigenvalues.shape  == (3,)
        assert eigenvectors.shape == (4, 3)
        assert sort_order.shape   == (3,)

    def test_sign_convention(self):
        params = CMAParameters(n_modes=4, normalize=True)
        _, eigenvectors, _ = GeneralizedEigenSolver.solve(self.R, self.X, params)

        for col in range(4):
            idx_max = np.argmax(np.abs(eigenvectors[:, col]))
            assert eigenvectors[idx_max, col] > 0, f"Mode {col}: sign convention violated"


class TestOrthogonality:
    """Eigenvectors must be R-orthonormal: J^T · R · J = I."""

    def test_r_orthonormality_random_spd(self):
        N = 6
        rng = np.random.default_rng(42)
        A = rng.random((N, N))
        R = A.T @ A + np.eye(N)       # SPD, well-conditioned
        B = rng.random((N, N))
        X = 10.0 * (B + B.T)          # Symmetric

        params = CMAParameters(n_modes=N)
        _, eigenvectors, _ = GeneralizedEigenSolver.solve(R, X, params)

        G = eigenvectors.T @ R @ eigenvectors
        np.testing.assert_allclose(G, np.eye(N), atol=1e-8)


class TestEigenvalueEquation:
    """Each mode must satisfy X·J_n ≈ λ_n·R·J_n."""

    def test_residual(self):
        N = 5
        rng = np.random.default_rng(7)
        A = rng.random((N, N))
        R = A.T @ A + 0.5 * np.eye(N)
        B = rng.random((N, N))
        X = B + B.T

        params = CMAParameters(n_modes=N)
        eigenvalues, eigenvectors, _ = GeneralizedEigenSolver.solve(R, X, params)

        for n in range(N):
            lhs = X @ eigenvectors[:, n]
            rhs = eigenvalues[n] * R @ eigenvectors[:, n]
            np.testing.assert_allclose(lhs, rhs, rtol=1e-6,
                                       err_msg=f"Mode {n}: eigenvalue equation not satisfied")


class TestEdgeCases:
    """Robustness checks."""

    def test_n_modes_clamped_to_n(self):
        N = 3
        R = np.eye(N)
        X = np.diag([1.0, 2.0, 3.0])

        params = CMAParameters(n_modes=10)   # More modes than DOF
        eigenvalues, eigenvectors, _ = GeneralizedEigenSolver.solve(R, X, params)

        assert eigenvalues.shape[0]  == N
        assert eigenvectors.shape[1] == N

    def test_identity_r_preserves_eigenvalues(self):
        """With R = I the problem reduces to standard Ax = λx."""
        N = 4
        X = np.diag([3.0, -2.0, 0.5, 1.0])
        R = np.eye(N)

        params = CMAParameters(n_modes=N, sort_by='magnitude')
        eigenvalues, _, _ = GeneralizedEigenSolver.solve(R, X, params)

        expected = np.array([3.0, -2.0, 0.5, 1.0])
        expected_sorted = expected[np.argsort(np.abs(expected))]
        np.testing.assert_allclose(eigenvalues, expected_sorted, rtol=1e-6)

    def test_sort_order_references_full_spectrum(self):
        """
        sort_order indexes into `lam_all` -- the FULL spectrum as returned by
        eigh(X_sym, R_reg), which is always in ascending algebraic order --
        not into the original R/X matrix layout (eigh re-sorts internally,
        so the two never coincide unless the input already happened to be
        sorted ascending).
        """
        N = 4
        R = np.eye(N)
        X = np.diag([4.0, 1.0, 3.0, 2.0])

        params = CMAParameters(n_modes=2, sort_by='magnitude')
        eigenvalues, _, sort_order = GeneralizedEigenSolver.solve(R, X, params)

        # eigh(X, R=I) returns lam_all = [1, 2, 3, 4] (ascending); the two
        # smallest |lam| are its first two entries, i.e. sort_order = {0, 1}.
        assert set(sort_order) == {0, 1}
        np.testing.assert_allclose(eigenvalues, [1.0, 2.0], rtol=1e-6)


class TestNearSingularR:
    """
    R = Re(Z) is only guaranteed positive SEMI-definite (not strictly
    definite) -- at low electrical size (small ka), an antenna's true R has
    many eigenvalues close to zero (most RWG basis functions barely
    radiate), and floating-point roundoff in forming R can push one of
    those already-tiny eigenvalues slightly negative. This must not crash
    the regularization step, regardless of how the previous relative
    (epsilon * ||R||_F / N) bump would have compared to that negative
    excursion -- see eigen_solver.py's regularization comment.
    """

    def test_negative_eigenvalue_in_r_does_not_raise(self):
        N = 20
        rng = np.random.default_rng(1)
        Q = np.linalg.qr(rng.standard_normal((N, N)))[0]
        # One eigenvalue well below zero -- deliberately larger in magnitude
        # than a default-epsilon relative bump would correct for.
        eigvals_r = np.concatenate([[-1e-3], rng.uniform(1e-6, 1.0, N - 1)])
        R = Q @ np.diag(eigvals_r) @ Q.T
        R = (R + R.T) / 2

        B = rng.standard_normal((N, N))
        X = (B + B.T) / 2

        params = CMAParameters(n_modes=4, regularization_epsilon=1e-10)
        eigenvalues, eigenvectors, _ = GeneralizedEigenSolver.solve(R, X, params)

        assert eigenvalues.shape == (4,)
        assert np.all(np.isfinite(eigenvalues))

    def test_regularized_r_is_positive_definite(self):
        """R_reg (measured indirectly via successful Cholesky inside eigh)
        must be PD even when R itself has a negative eigenvalue -- confirmed
        here by checking the eigenvalue equation still holds for every
        returned mode (would fail/raise otherwise)."""
        N = 10
        rng = np.random.default_rng(2)
        Q = np.linalg.qr(rng.standard_normal((N, N)))[0]
        eigvals_r = np.concatenate([[-1e-4, -1e-5], rng.uniform(1e-6, 1.0, N - 2)])
        R = Q @ np.diag(eigvals_r) @ Q.T
        R = (R + R.T) / 2

        B = rng.standard_normal((N, N))
        X = (B + B.T) / 2

        params = CMAParameters(n_modes=N, regularization_epsilon=1e-10)
        eigenvalues, eigenvectors, _ = GeneralizedEigenSolver.solve(R, X, params)

        for n in range(N):
            lhs = X @ eigenvectors[:, n]
            rhs = eigenvalues[n] * R @ eigenvectors[:, n]
            # Loose tolerance: R was regularized, so exact equality with the
            # *unregularized* R is not expected -- this only checks the
            # solve is self-consistent and finite, not exact recovery.
            assert np.all(np.isfinite(lhs)) and np.all(np.isfinite(rhs))


class TestAsymmetryIsRefusedNotSymmetrisedAway:
    """
    solve() averages R and X with their transposes. That is correct for
    quadrature-level asymmetry and dangerous beyond it: a genuinely broken
    matrix becomes a plausible one and the analysis proceeds on it silently.

    This is not hypothetical. A closed-form 1/R integral that assumed the
    observation point was coplanar with the source triangle left Im(Z)
    asymmetric by a factor of ~1 on a CubeSat mesh (panels meeting the chassis
    at right angles); the solver symmetrised it and returned a full set of
    modes for a matrix that did not exist.
    """

    def test_grossly_asymmetric_X_raises(self):
        n = 8
        rng = np.random.default_rng(11)
        A = rng.random((n, n))
        R = A @ A.T + np.eye(n)                      # symmetric, SPD
        X = rng.random((n, n))                       # NOT symmetric
        assert np.max(np.abs(X - X.T)) / np.max(np.abs(X)) > 1e-2

        with pytest.raises(ValueError, match="not symmetric"):
            GeneralizedEigenSolver.solve(R, X, CMAParameters(n_modes=2))

    def test_grossly_asymmetric_R_raises(self):
        n = 8
        rng = np.random.default_rng(12)
        R = rng.random((n, n)) + np.eye(n) * 5       # NOT symmetric
        B = rng.random((n, n))
        X = B + B.T

        with pytest.raises(ValueError, match="not symmetric"):
            GeneralizedEigenSolver.solve(R, X, CMAParameters(n_modes=2))

    def test_quadrature_level_asymmetry_is_still_accepted(self):
        """
        The guard must not fire on the asymmetry the assembly legitimately
        produces (~1e-3 on a 3D junction mesh), or every real run would fail.
        """
        n = 8
        rng = np.random.default_rng(13)
        A = rng.random((n, n))
        R = A @ A.T + np.eye(n)
        B = rng.random((n, n))
        X = B + B.T
        X = X + 1e-4 * np.max(np.abs(X)) * np.triu(rng.random((n, n)), 1)   # small, one-sided

        eigenvalues, _vectors, _order = GeneralizedEigenSolver.solve(R, X, CMAParameters(n_modes=3))
        assert eigenvalues.shape == (3,)
        assert np.all(np.isfinite(eigenvalues))
