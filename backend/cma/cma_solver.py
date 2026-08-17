import os
import warnings

import numpy as np

from backend.rwg.rwg3 import calculate_z_matrice, DataManager_rwg3
from backend.utils.mesh_io import MeshDataError
from backend.cma.cma_mesh import prepare_cma_mesh
from backend.cma.cma_parameters import CMAParameters
from backend.cma.eigen_solver import GeneralizedEigenSolver
from backend.cma.modal_data import ModalData, DataManager_cma
from backend.cma.modal_current import ModalCurrent
from backend.cma.modal_visualizer import ModalVisualizer


class CharacteristicModeSolver:
    """
    Orchestrates the full CMA pipeline:

        path.msh  →  mesh  →  Z(frequency)  →  R, X  →  eigenvalue problem
            →  ModalData  →  surface currents  →  Plotly figures

    Self-sufficient, like cma_frequency_sweep(): both the mesh AND the
    impedance matrix are always (re)computed here at the requested
    frequency, never loaded from a possibly-stale save left by a previous
    radiation_algorithm() run. See "Freshness" below for why.

    Pre-conditions
    --------------
    - path.msh OR path.mat exists. When path.msh is present the mesh is
      re-derived from it; when it is absent (mesh not produced by gmsh,
      e.g. a structured pixel grid exported from MATLAB) path.mat is used
      as the source of truth. See cma_mesh.prepare_cma_mesh.
    - Either `frequency` is passed explicitly, or path.mat_impedance
      already exists from a previous run (its stored frequency is then
      reused as the analysis point -- everything else in that file is
      still recomputed fresh, see below).

    Freshness
    ---------
    _load_mesh() re-derives points/triangles/edges/vecteurs_rho from
    path.msh every time it can, and solve() then always recomputes Z at
    `frequency` from that fresh mesh via calculate_z_matrice() -- neither
    is loaded from disk and trusted as-is. This closes the staleness gap a
    fingerprint check can only detect after the fact: previously, if
    path.mat_mesh2 and path.mat_impedance were BOTH left stale together
    (neither regenerated after a .msh edit), their fingerprints matched
    each other and a consistency check would pass silently even though
    neither matched the CURRENT .msh. Always recomputing both from path.msh
    makes that mismatch structurally impossible instead of merely detected.
    """

    def __init__(self, path, params: CMAParameters, frequency: float = None, use_gpu: bool = None):
        # Extend the path namespace with the CMA output path (no modification of file_path.py)
        cma_dir = 'data/antennas_cma'
        os.makedirs(cma_dir, exist_ok=True)
        path.mat_cma = os.path.join(cma_dir, f"{path.name}_cma.mat")

        self.path      = path
        self.params    = params
        self.frequency = frequency
        self.use_gpu   = use_gpu

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
        # 1. Refresh the mesh from path.msh (see class docstring, "Freshness").
        self._load_mesh()

        # 2. Resolve the analysis frequency: explicit if given, otherwise
        #    fall back to whatever path.mat_impedance was last computed at
        #    (that file must then already exist) -- only the frequency
        #    VALUE is reused here, everything else (Z included) is about to
        #    be recomputed from the fresh mesh regardless.
        frequency = self.frequency
        if frequency is None:
            try:
                frequency, *_rest = DataManager_rwg3.load_data(self.path.mat_impedance)
            except MeshDataError as exc:
                raise MeshDataError(
                    "CharacteristicModeSolver: no frequency given and "
                    f"'{self.path.mat_impedance}' does not exist either. Pass "
                    "frequency=... explicitly, or run radiation_algorithm() "
                    "first at the frequency you want the analysis at."
                ) from exc

        # 3. (Re)assemble Z at that frequency from the FRESH mesh -- never
        #    trust a saved path.mat_impedance as-is, even if frequency was
        #    inferred from it (see class docstring, "Freshness").
        omega, mu, epsilon, light_speed_c, eta, Z = calculate_z_matrice(
            self._points, self._triangles, self._edges, frequency, use_gpu=self.use_gpu)
        DataManager_rwg3.save_data(self.path, frequency, omega, mu, epsilon, light_speed_c, eta, Z)

        # 4. Decompose Z = R + jX
        R = np.real(Z)
        X = np.imag(Z)

        # 5. Clamp n_modes to available DOF
        n_dof = R.shape[0]
        if self.params.n_modes > n_dof:
            warnings.warn(
                f"n_modes={self.params.n_modes} exceeds n_edges={n_dof}. "
                f"Clamping to {n_dof}.",
                UserWarning,
                stacklevel=2,
            )
            self.params.n_modes = n_dof

        # 6. Solve generalized eigenvalue problem X·J = λ·R·J
        eigenvalues, eigenvectors, sort_order = GeneralizedEigenSolver.solve(R, X, self.params)

        # 7. Store and save
        modal_data = ModalData(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            frequency=float(frequency),
            n_modes=self.params.n_modes,
            sort_order=sort_order,
        )
        DataManager_cma.save_data(self.path, modal_data)
        print(f"CMA: {self.params.n_modes} modes computed at f = {float(frequency)/1e9:.4f} GHz")
        # ASCII only: a bare Windows console is cp1252 and raises
        # UnicodeEncodeError on 'lambda' as a Greek letter (same trap as the
        # 'Ohm' fix in rwg4.compute_port_results).
        print(f"  lambda range: [{eigenvalues.min():.4f}, {eigenvalues.max():.4f}]")

        return modal_data

    def reconstruct_modal_currents(self, modal_data: ModalData):
        """
        Reconstruct the surface current on each triangle for every mode.

        Returns
        -------
        surface_currents : list of float64 (N_triangles,)   — |J| per mode
        current_vectors  : list of complex (3, N_triangles) — J per mode, before
            taking the norm, for the current-direction arrows
        """
        self._load_mesh()
        surface_currents = []
        current_vectors = []
        for n in range(modal_data.n_modes):
            J_n = modal_data.eigenvectors[:, n]
            J_surface, J_vector = ModalCurrent.reconstruct(
                J_n, self._triangles, self._edges, self._vecteurs_rho)
            surface_currents.append(J_surface)
            current_vectors.append(J_vector)
        return surface_currents, current_vectors

    def visualize(self, modal_data: ModalData, show: bool = True, save_image: bool = False,
                  show_current_arrows: bool = True) -> list:
        """
        Plot each characteristic mode current on the mesh.

        Parameters
        ----------
        modal_data          : ModalData from solve()
        show                : display each figure
        save_image          : also write each mode to data/fig_image/ as PDF
        show_current_arrows : overlay black arrows along the modal current
                              direction (see rwg5.visualize_surface_current)

        Returns
        -------
        list of plotly.graph_objects.Figure
        """
        self._load_mesh()
        surface_currents, current_vectors = self.reconstruct_modal_currents(modal_data)

        if save_image:
            output_dir_fig_image = "data/fig_image/"
            os.makedirs(output_dir_fig_image, exist_ok=True)

        figures = []
        for n, J_surface in enumerate(surface_currents):
            fig = ModalVisualizer.plot_mode(
                self._points, self._triangles, J_surface,
                mode_index=n, eigenvalue=modal_data.eigenvalues[n],
                current_vector=current_vectors[n] if show_current_arrows else None,
            )
            if show:
                fig.show()

            if save_image:
                pdf_name = f"cma_mode{n + 1}_current_{self.path.name}.pdf".replace(" ", "_")
                pdf_path = os.path.join(output_dir_fig_image, pdf_name)
                # Transparent background and trimmed margins, as in
                # radiation_algorithm(), so the figure drops cleanly into a
                # report -- but keeping a top margin here, unlike there: the
                # title is what says which mode and which lambda this is, and
                # t=0 clips it off the canvas.
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=0, t=40, b=0),
                )
                fig.write_image(pdf_path, format="pdf")
                print(f"Image saved in PDF format: {pdf_path}")

            figures.append(fig)
        return figures

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_mesh(self):
        if self._triangles is not None:
            return

        # Re-derive from path.msh (when there is one) rather than trusting
        # whatever path.mat_mesh2 last happened to contain -- see class
        # docstring, "Freshness". Cached on self for the lifetime of this
        # solver instance (solve()/reconstruct_modal_currents()/visualize()
        # all call this, but only the first call does real work).
        (self._points, self._triangles,
         self._edges, self._vecteurs_rho) = prepare_cma_mesh(self.path, verbose=False)
