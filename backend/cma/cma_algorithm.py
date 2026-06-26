from backend.cma.cma_parameters import CMAParameters
from backend.cma.cma_solver import CharacteristicModeSolver
from backend.cma.modal_data import ModalData


def cma_algorithm(
    path,
    n_modes: int = 10,
    regularization_epsilon: float = 1e-10,
    sort_by: str = 'magnitude',
    show: bool = True,
    save_image: bool = False,
) -> ModalData:
    """
    High-level CMA pipeline — mirror of radiation_algorithm().

    Pre-conditions
    --------------
    path.mat_impedance and path.mat_mesh2 must already exist.
    Run radiation_algorithm() (or at least rwg1–rwg3) before calling this.

    Parameters
    ----------
    path                    : SimpleNamespace from setup_save_file_paths()
    n_modes                 : number of characteristic modes to extract
    regularization_epsilon  : relative perturbation applied to R  (default 1e-10)
    sort_by                 : 'magnitude'  →  modes nearest resonance first (|λ| ascending)
                              'value'      →  algebraic sort (most negative first)
    show                    : display Plotly figures for each mode
    save_image              : reserved for V2 (PDF export)

    Returns
    -------
    ModalData  with eigenvalues, eigenvectors, frequency, n_modes

    Example
    -------
    >>> from backend.utils.file_path import setup_save_file_paths
    >>> from backend.cma.cma_algorithm import cma_algorithm
    >>> path = setup_save_file_paths("dipole")
    >>> modal_data = cma_algorithm(path, n_modes=5, show=True)
    """
    params = CMAParameters(
        n_modes=n_modes,
        regularization_epsilon=regularization_epsilon,
        sort_by=sort_by,
    )
    solver = CharacteristicModeSolver(path, params)
    modal_data = solver.solve()

    if show:
        solver.visualize(modal_data, show=True)

    return modal_data
