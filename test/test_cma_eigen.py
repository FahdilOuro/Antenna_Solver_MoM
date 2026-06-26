"""
Unit tests for GeneralizedEigenSolver.

All tests use synthetic R and X matrices — no mesh files required.
The diagonal system provides exact analytical ground-truth eigenvalues.
"""

import numpy as np

from backend.cma.cma_parameters import CMAParameters
from backend.cma.eigen_solver import GeneralizedEigenSolver


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
        N = 4
        R = np.eye(N)
        X = np.diag([4.0, 1.0, 3.0, 2.0])

        params = CMAParameters(n_modes=2, sort_by='magnitude')
        _, _, sort_order = GeneralizedEigenSolver.solve(R, X, params)

        # The two smallest |λ| are 1.0 (index 1) and 2.0 (index 3)
        assert set(sort_order) == {1, 3}
