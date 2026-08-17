import warnings

import numpy as np
from scipy.linalg import eigh

from backend.cma.cma_parameters import CMAParameters


class GeneralizedEigenSolver:
    """
    Solve the CMA generalized eigenvalue problem  X·J = λ·R·J

    R = Re(Z)  radiation-resistance matrix  (symmetric, PSD for lossless PEC)
    X = Im(Z)  reactance matrix             (symmetric for reciprocal structures)
    """

    @staticmethod
    def solve(R: np.ndarray, X: np.ndarray, params: CMAParameters):
        """
        Parameters
        ----------
        R : float64 (N, N)  Re(Z), must be positive semi-definite
        X : float64 (N, N)  Im(Z), symmetric
        params : CMAParameters

        Returns
        -------
        eigenvalues  : float64 (K,)    λ_n sorted according to params.sort_by
        eigenvectors : float64 (N, K)  J_n as columns, R-orthonormal
        sort_order   : int    (K,)     indices into the full spectrum before selection
        """
        N = R.shape[0]
        n_modes = min(params.n_modes, N)

        # --- 1. Symmetrize (small asymmetries arise from numerical integration) ---
        R_sym = (R + R.T) * 0.5
        X_sym = (X + X.T) * 0.5

        # --- 2. Adaptive regularization of R ---
        # R = Re(Z) is only guaranteed positive SEMI-definite (radiated power
        # P_rad = 0.5*I^H*R*I >= 0), and at low electrical size (small ka) most
        # RWG basis functions correspond to current distributions that barely
        # radiate at all -- so most of R's true eigenvalues are already close
        # to zero there. That is a physical fact about the structure, not a
        # quadrature error: even a perfectly-integrated Z (Duffy for the
        # singular SELF pairs, Gauss-Dunavant for FAR/EDGE/VERTEX -- see
        # impmet.py) still yields a near-singular R at low ka. Floating-point
        # roundoff in forming R then just as easily pushes one of those
        # already-tiny eigenvalues slightly NEGATIVE. A blind relative bump
        # (eps ~ epsilon * ||R||_F / N) can be smaller than that negative
        # excursion and still leave R_reg non-positive-definite -- exactly the
        # `eigh` Cholesky failure this guards against. Instead, measure R's
        # actual smallest eigenvalue and shift by exactly enough to clear it.
        frob_R = np.linalg.norm(R_sym, 'fro')
        floor_eps = params.regularization_epsilon * frob_R / N

        min_eig_R = eigh(R_sym, eigvals_only=True, subset_by_index=[0, 0])[0]
        eps_adapt = max(floor_eps, floor_eps - min_eig_R)
        R_reg = R_sym + eps_adapt * np.eye(N)

        if eps_adapt > 1e-4 * frob_R / N:
            warnings.warn(
                f"CMA: large regularization eps_adapt={eps_adapt:.2e} "
                f"(R's smallest eigenvalue was {min_eig_R:.2e}). R may be poorly "
                "conditioned -- check mesh quality, or the structure may be too "
                "electrically small at this frequency for a meaningful CMA "
                "(most basis functions barely radiate at low ka).",
                UserWarning,
                stacklevel=2,
            )

        # --- 3. Generalized symmetric eigenvalue problem: X·J = λ·R·J ---
        # scipy.linalg.eigh(a, b) solves a·x = λ·b·x with b SPD.
        # Returns eigenvalues sorted algebraically (ascending).
        try:
            lam_all, J_all = eigh(X_sym, R_reg)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                "CMA: eigenvalue decomposition failed. R is not positive definite "
                "even after regularization. Try increasing regularization_epsilon."
            ) from exc

        # --- 4. Select n_modes according to sort criterion ---
        if params.sort_by == 'magnitude':
            order = np.argsort(np.abs(lam_all))
        elif params.sort_by == 'value':
            order = np.argsort(lam_all)
        else:
            raise ValueError(f"Unknown sort_by='{params.sort_by}'. Use 'magnitude' or 'value'.")

        selected = order[:n_modes]
        eigenvalues = lam_all[selected]
        eigenvectors = J_all[:, selected].copy()

        # --- 5. Sign convention: component with largest magnitude is positive ---
        if params.normalize:
            for col in range(n_modes):
                idx_max = np.argmax(np.abs(eigenvectors[:, col]))
                if eigenvectors[idx_max, col] < 0:
                    eigenvectors[:, col] *= -1

        return eigenvalues, eigenvectors, selected
