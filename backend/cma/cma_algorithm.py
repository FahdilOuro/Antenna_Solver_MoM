from backend.cma.cma_parameters import CMAParameters
from backend.cma.cma_solver import CharacteristicModeSolver
from backend.cma.modal_data import ModalData


def cma_algorithm(
    path,
    frequency: float = None,
    n_modes: int = 10,
    regularization_epsilon: float = 1e-10,
    sort_by: str = 'magnitude',
    show: bool = True,
    save_image: bool = False,
    show_current_arrows: bool = True,
    use_gpu: bool = None,
) -> ModalData:
    """
    High-level CMA pipeline — mirror of radiation_algorithm(), but self-
    sufficient: the mesh and the impedance matrix are always (re)computed
    from path.msh at `frequency`, never loaded from a possibly-stale save
    left by a previous radiation_algorithm() run (see
    CharacteristicModeSolver's docstring, "Freshness").

    Pre-conditions
    --------------
    path.msh must exist. `frequency` is required UNLESS path.mat_impedance
    already exists from a previous run, in which case its stored frequency
    is reused as the analysis point (everything else is still recomputed
    fresh from the current mesh).

    Parameters
    ----------
    path                    : SimpleNamespace from setup_save_file_paths()
    frequency               : float, Hz -- analysis frequency. Optional if
                              path.mat_impedance already exists (see above).
    n_modes                 : number of characteristic modes to extract
    regularization_epsilon  : relative perturbation applied to R  (default 1e-10)
    sort_by                 : 'magnitude'  →  modes nearest resonance first (|λ| ascending)
                              'value'      →  algebraic sort (most negative first)
    show                    : display Plotly figures for each mode
    save_image              : also write each mode's current plot to
                              data/fig_image/ as cma_mode<n>_current_<name>.pdf
    show_current_arrows     : overlay black arrows along the modal current
                              direction on each mode plot (default True)
    use_gpu                 : bool or None, passed to calculate_z_matrice
                              (None = auto-detect, see backend.utils.compute_backend)

    Returns
    -------
    ModalData  with eigenvalues, eigenvectors, frequency, n_modes

    Example
    -------
    >>> from backend.utils.file_path import setup_save_file_paths
    >>> from backend.cma.cma_algorithm import cma_algorithm
    >>> path = setup_save_file_paths("dipole")
    >>> modal_data = cma_algorithm(path, frequency=1e9, n_modes=5, show=True)
    """
    params = CMAParameters(
        n_modes=n_modes,
        regularization_epsilon=regularization_epsilon,
        sort_by=sort_by,
    )
    solver = CharacteristicModeSolver(path, params, frequency=frequency, use_gpu=use_gpu)
    modal_data = solver.solve()

    # `show or save_image`, not just `show`: exporting the mode plots without
    # popping them up is a legitimate (and headless-friendly) combination.
    if show or save_image:
        solver.visualize(modal_data, show=show, save_image=save_image,
                         show_current_arrows=show_current_arrows)

    return modal_data
