import os
import warnings

import numpy as np

from backend.rwg.rwg3 import DataManager_rwg3
from backend.rwg.rwg2 import DataManager_rwg2
from backend.utils.mesh_io import verify_mesh_fingerprint
from backend.cma.cma_parameters import CMAParameters
from backend.cma.eigen_solver import GeneralizedEigenSolver
from backend.cma.modal_data import ModalData, DataManager_cma
from backend.cma.modal_current import ModalCurrent
from backend.cma.modal_visualizer import ModalVisualizer


class CharacteristicModeSolver:
    """
    Orchestrates the full CMA pipeline:

        path.mat_impedance  →  Z  →  R, X  →  eigenvalue problem
            →  ModalData  →  surface currents  →  Plotly figures

    Pre-conditions
    --------------
    - path.mat_impedance exists  (run radiation_algorithm first)
    - path.mat_mesh2    exists  (run rwg1 + rwg2 first)
    """

    def __init__(self, path, params: CMAParameters):
        # Extend the path namespace with the CMA output path (no modification of file_path.py)
        cma_dir = 'data/antennas_cma'
        os.makedirs(cma_dir, exist_ok=True)
        path.mat_cma = os.path.join(cma_dir, f"{path.name}_cma.mat")

        self.path   = path
        self.params = params

        # Cached mesh objects (loaded once on first use)
        self._points      = None
        self._triangles   = None
        self._edges       = None
        self._vecteurs_rho = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self) -> ModalData:
        """
        Run the CMA analysis and save results to path.mat_cma.

        Returns
        -------
        ModalData
        """
        # 1. Load impedance matrix
        # (guard against mixing a stale impedance file with a regenerated mesh)
        verify_mesh_fingerprint(self.path.mat_mesh2, self.path.mat_impedance)
        impedance_data = DataManager_rwg3.load_data(self.path.mat_impedance)
        frequency, _omega, _mu, _epsilon, _c, _eta, Z = impedance_data

        # 2. Load mesh (needed for current reconstruction later)
        self._load_mesh()

        # 3. Decompose Z = R + jX
        R = np.real(Z)
        X = np.imag(Z)

        # 4. Clamp n_modes to available DOF
        n_dof = R.shape[0]
        if self.params.n_modes > n_dof:
            warnings.warn(
                f"n_modes={self.params.n_modes} exceeds n_edges={n_dof}. "
                f"Clamping to {n_dof}.",
                UserWarning,
                stacklevel=2,
            )
            self.params.n_modes = n_dof

        # 5. Solve generalized eigenvalue problem X·J = λ·R·J
        eigenvalues, eigenvectors, sort_order = GeneralizedEigenSolver.solve(R, X, self.params)

        # 6. Store and save
        modal_data = ModalData(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            frequency=float(frequency),
            n_modes=self.params.n_modes,
            sort_order=sort_order,
        )
        DataManager_cma.save_data(self.path, modal_data)
        print(f"CMA: {self.params.n_modes} modes computed at f = {float(frequency)/1e9:.4f} GHz")
        print(f"  λ range: [{eigenvalues.min():.4f}, {eigenvalues.max():.4f}]")

        return modal_data

    def reconstruct_modal_currents(self, modal_data: ModalData) -> list:
        """
        Reconstruct surface current density |J| on each triangle for every mode.

        Returns
        -------
        list of float64 arrays, each (N_triangles,)
        """
        self._load_mesh()
        surface_currents = []
        for n in range(modal_data.n_modes):
            J_n = modal_data.eigenvectors[:, n]
            J_surface = ModalCurrent.reconstruct(J_n, self._triangles, self._edges, self._vecteurs_rho)
            surface_currents.append(J_surface)
        return surface_currents

    def visualize(self, modal_data: ModalData, show: bool = True) -> list:
        """
        Plot each characteristic mode current on the mesh.

        Returns
        -------
        list of plotly.graph_objects.Figure
        """
        self._load_mesh()
        surface_currents = self.reconstruct_modal_currents(modal_data)
        figures = []
        for n, J_surface in enumerate(surface_currents):
            fig = ModalVisualizer.plot_mode(
                self._points, self._triangles, J_surface,
                mode_index=n, eigenvalue=modal_data.eigenvalues[n],
            )
            if show:
                fig.show()
            figures.append(fig)
        return figures

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_mesh(self):
        if self._triangles is not None:
            return
        mesh_data = DataManager_rwg2.load_data(self.path.mat_mesh2)
        points, triangles, edges, vecteurs_rho = mesh_data
        self._points       = points
        self._triangles    = triangles
        self._edges        = edges
        self._vecteurs_rho = vecteurs_rho
