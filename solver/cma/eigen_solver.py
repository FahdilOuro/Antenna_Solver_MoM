import warnings

import numpy as np
from scipy.linalg import eigh

from solver.cma.cma_parameters import CMAParameters
from solver.cma.z_matrix_diagnostics import RECIPROCITY_TOL


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

        # --- 0. Refuse to symmetrise away a real defect ---
        # The step below averages R and X with their transposes, which is the
        # right thing to do for quadrature-level asymmetry -- and a dangerous
        # thing to do for anything larger, because it turns a broken matrix
        # into a plausible-looking one and the whole analysis proceeds on it
        # without complaint. That is exactly what happened once: a closed-form
        # integral evaluated with a coplanarity assumption left Im(Z)
        # asymmetric by a factor of 1 on a CubeSat mesh, and this function
        # quietly symmetrised it and returned modes for it.
        #
        # Since impmet now fills only m <= n and mirrors, a matrix coming
        # straight from the assembler satisfies this exactly. The guard is kept
        # for matrices arriving from anywhere else -- a stale .mat file, a
        # MATLAB import, a hand-built test case.
        for name, M in (('R = Re(Z)', R), ('X = Im(Z)', X)):
            scale = np.max(np.abs(M))
            if scale == 0:
                continue
            asymmetry = np.max(np.abs(M - M.T)) / scale
            if asymmetry > RECIPROCITY_TOL:
                raise ValueError(
                    f"CMA: {name} is not symmetric (max|M-M^T|/max|M| = {asymmetry:.3e}, "
                    f"tolerance {RECIPROCITY_TOL:.0e}). A Galerkin MoM matrix is "
                    f"complex-symmetric whatever the geometry, so this is an assembly "
                    f"defect, not a conditioning problem -- symmetrising it here would "
                    f"hide the fault and yield modes for a matrix that does not exist. "
                    f"Check the impedance assembly (solver.utils.impmet) before reading "
                    f"any result."
                )

        # --- 1. Symmetrize (small asymmetries arise from numerical integration) ---
        R_sym = (R + R.T) * 0.5
        X_sym = (X + X.T) * 0.5

        if params.solver == 'rank_truncated':
            lam_all, J_all = GeneralizedEigenSolver._solve_rank_truncated(
                R_sym, X_sym, params)
            return GeneralizedEigenSolver._select_and_sign(lam_all, J_all, params, n_modes)
        if params.solver != 'regularized':
            raise ValueError(
                f"Unknown solver={params.solver!r}. Use 'regularized' or 'rank_truncated'.")

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
        #
        # Note what this does NOT fix: shifting makes the solve succeed, it does
        # not make the null-space directions meaningful. There X J = lambda R J
        # degenerates into X J = lambda * eps * J, so the eigenvalue reported for
        # such a mode is a function of eps rather than of the structure. Use
        # solver='rank_truncated' to remove those directions instead.
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

        return GeneralizedEigenSolver._select_and_sign(lam_all, J_all, params, n_modes)

    # ------------------------------------------------------------------

    @staticmethod
    def _solve_rank_truncated(R_sym, X_sym, params):
        """
        Solve X J = lambda R J on the subspace that R actually spans.

        R = Re(Z) is the radiation operator, and on any real mesh it is badly
        rank-deficient: most RWG basis functions describe currents that radiate
        essentially nothing, so most of R's eigenvalues sit near float64 noise
        (measured on the CubeSat mesh at 440 MHz: numerical rank 93 out of 760).
        The generalized problem is ill-posed on those directions -- no finite
        lambda satisfies X J = lambda R J when R J = 0 -- and the usual remedy,
        replacing R by R + eps*I, does not remove the difficulty so much as
        answer it arbitrarily: in the null space the equation becomes
        X J = lambda * eps * J, so the lambda that comes back is set by eps, not
        by the structure. Those modes radiate nothing yet are reported with
        ordinary-looking eigenvalues, and because modal significance depends only
        on lambda, one of them can carry MS ~ 0.99. They also move erratically
        from run to run, since eps is being divided into noise.

        Truncation deletes the null space instead of populating it. With
        R = U diag(mu) U^T and the columns kept where mu > rank_tol * mu_max,

            W     = U_r diag(mu_r)^(-1/2)      so that   W^T R W = I_r
            X_r   = W^T X W
            X_r y = lambda y                   (ordinary symmetric problem)
            J     = W y

        The recovered J are automatically R-orthonormal (J^T R J = I), no
        regularization parameter enters anywhere, and only rank(R) modes come
        back -- which is the honest count of what the structure supports at this
        frequency.

        Returns
        -------
        (lam_all ascending (r,), J_all (N, r))
        """
        N = R_sym.shape[0]
        mu, U = eigh(R_sym)                       # ascending
        mu_max = mu[-1]
        if mu_max <= 0:
            raise RuntimeError(
                "CMA: R = Re(Z) has no positive eigenvalue, so the structure radiates "
                "nothing at this frequency and no characteristic mode is defined. "
                "Check the frequency and the impedance assembly."
            )

        keep = mu > params.rank_tol * mu_max
        rank = int(np.count_nonzero(keep))
        if rank == 0:
            raise RuntimeError(
                f"CMA: no eigenvalue of R exceeds rank_tol={params.rank_tol:.1e} x "
                f"lambda_max(R). Nothing radiates enough to define a mode."
            )
        if rank < min(params.n_modes, N):
            warnings.warn(
                f"CMA: R has numerical rank {rank} of {N}, so only {rank} characteristic "
                f"modes exist at this frequency ({params.n_modes} were requested). The "
                f"remaining directions radiate nothing; the 'regularized' solver would "
                f"have returned modes for them anyway, with eigenvalues set by the "
                f"regularization rather than by the structure.",
                UserWarning,
                stacklevel=3,
            )

        W = U[:, keep] / np.sqrt(mu[keep])        # (N, r), W^T R W = I_r
        X_r = W.T @ X_sym @ W
        X_r = (X_r + X_r.T) * 0.5                 # kill projection roundoff
        lam_all, Y = eigh(X_r)
        return lam_all, W @ Y

    # ------------------------------------------------------------------

    @staticmethod
    def _select_and_sign(lam_all, J_all, params, n_modes):
        """Order the spectrum, keep n_modes of it, apply the sign convention."""
        n_modes = min(n_modes, lam_all.shape[0])

        if params.sort_by == 'magnitude':
            order = np.argsort(np.abs(lam_all))
        elif params.sort_by == 'value':
            order = np.argsort(lam_all)
        else:
            raise ValueError(f"Unknown sort_by='{params.sort_by}'. Use 'magnitude' or 'value'.")

        selected = order[:n_modes]
        eigenvalues = lam_all[selected]
        eigenvectors = J_all[:, selected].copy()

        # Sign convention: component with largest magnitude is positive.
        if params.normalize:
            for col in range(eigenvectors.shape[1]):
                idx_max = np.argmax(np.abs(eigenvectors[:, col]))
                if eigenvectors[idx_max, col] < 0:
                    eigenvectors[:, col] *= -1

        return eigenvalues, eigenvectors, selected
