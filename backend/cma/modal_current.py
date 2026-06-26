import numpy as np

from backend.rwg.rwg5 import calculate_current_density


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
    ) -> np.ndarray:
        """
        Parameters
        ----------
        modal_coefficient : float64 (N_edges,)  — eigenvectors[:, n]
        triangles, edges, vecteurs_rho          — mesh objects from DataManager_rwg2

        Returns
        -------
        float64 (N_triangles,)  — |J_surface| per triangle in A/m
        """
        return calculate_current_density(modal_coefficient, triangles, edges, vecteurs_rho)
