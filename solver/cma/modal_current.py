import numpy as np

from solver.rwg.rwg5 import calculate_current_density


class ModalCurrent:
    """
    Reconstruct the surface current density on each triangle for one characteristic mode.

    The formula is identical to the one used in rwg5 for physical currents:

        J⃗(T) = Σ_{m∈T}  J_n[m] · (l_m / (2·A_T)) · ρ⃗_m(T)

    where J_n[m] is the m-th component of eigenvector n (real, in A).
    """

    @staticmethod
    def reconstruct(
        modal_coefficient: np.ndarray,
        triangles,
        edges,
        vecteurs_rho,
    ):
        """
        Parameters
        ----------
        modal_coefficient : float64 (N_edges,)  — eigenvectors[:, n]
        triangles, edges, vecteurs_rho          — mesh objects from DataManager_rwg2

        Returns
        -------
        surface_current_density : float64 (N_triangles,)     — |J_surface| per triangle in A/m
        surface_current_vector  : complex (3, N_triangles)   — J_surface before taking its
            norm, so callers can draw current-direction arrows. Characteristic
            eigenvectors are real (eigh on real symmetric R, X), so this vector
            is real up to dtype: its direction is the mode's standing-wave
            current pattern, with no phase to disambiguate.
        """
        return calculate_current_density(modal_coefficient, triangles, edges, vecteurs_rho)
