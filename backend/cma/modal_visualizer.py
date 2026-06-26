from backend.rwg.rwg5 import visualize_surface_current
from backend.cma.modal_current import ModalCurrent


class ModalVisualizer:
    """Plotly visualisation of characteristic mode currents on the mesh."""

    @staticmethod
    def plot_mode(points, triangles, surface_current, mode_index, eigenvalue, title_prefix="CMA"):
        """
        Plot the surface current of a single characteristic mode.

        Parameters
        ----------
        points, triangles       — mesh objects from DataManager_rwg2
        surface_current         — float64 (N_triangles,) from ModalCurrent.reconstruct
        mode_index              — 0-based mode index (displayed as mode_index+1)
        eigenvalue              — λ_n for this mode
        title_prefix            — string prefix for the figure title

        Returns
        -------
        plotly.graph_objects.Figure
        """
        title = f"{title_prefix} — Mode {mode_index + 1},  λ = {eigenvalue:.4f}"
        return visualize_surface_current(points, triangles, surface_current, feed_point=None, title=title)

    @staticmethod
    def plot_all_modes(points, triangles, edges, vecteurs_rho, modal_data):
        """
        Reconstruct and plot every mode stored in modal_data.

        Returns
        -------
        list[plotly.graph_objects.Figure]  — one figure per mode
        """
        figures = []
        for n in range(modal_data.n_modes):
            J_n = modal_data.eigenvectors[:, n]
            J_surface = ModalCurrent.reconstruct(J_n, triangles, edges, vecteurs_rho)
            fig = ModalVisualizer.plot_mode(points, triangles, J_surface, n, modal_data.eigenvalues[n])
            figures.append(fig)
        return figures
