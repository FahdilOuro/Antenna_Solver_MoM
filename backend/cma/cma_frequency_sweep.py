import os
import time
from dataclasses import dataclass

import numpy as np
from scipy.io import savemat, loadmat

from backend.rwg.rwg3 import calculate_z_matrice_sweep
from backend.utils.compute_backend import describe_backend
from backend.utils.frequency_sweep import generate_freq_points
from backend.cma.cma_mesh import prepare_cma_mesh
from backend.cma.cma_parameters import CMAParameters
from backend.cma.eigen_solver import GeneralizedEigenSolver


@dataclass
class CMASweepData:
    """Container for CMA results across a frequency sweep."""

    frequencies:  np.ndarray   # float64, (n_freq,)             — Hz
    eigenvalues:  np.ndarray   # float64, (n_freq, n_modes)      — lambda_n(f)
    eigenvectors: np.ndarray   # float64, (n_freq, n_edges, n_modes) — J_n(f) as columns
    n_modes:      int


class DataManager_cma_sweep:
    """Save and load CMA frequency-sweep results, following the DataManager_* pattern."""

    @staticmethod
    def save_data(path, sweep_data: CMASweepData) -> None:
        savemat(path.mat_cma_sweep, {
            'frequencies':  sweep_data.frequencies,
            'eigenvalues':  sweep_data.eigenvalues,
            'eigenvectors': sweep_data.eigenvectors,
            'n_modes':      int(sweep_data.n_modes),
        })

    @staticmethod
    def load_data(filename: str) -> CMASweepData:
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"CMA sweep file '{filename}' does not exist.")
        data = loadmat(filename)
        return CMASweepData(
            frequencies=data['frequencies'].squeeze(),
            eigenvalues=data['eigenvalues'],
            eigenvectors=data['eigenvectors'],
            n_modes=int(data['n_modes'].squeeze()),
        )


def cma_frequency_sweep(path, freq_start, freq_stop, n_points=41, n_modes=6,
                        regularization_epsilon=1e-10, sort_by='magnitude', use_gpu=None):
    """
    Compute characteristic mode eigenvalues AND eigenvectors across a
    frequency sweep, reusing calculate_z_matrice_sweep's batched Z(f)
    assembly -- exactly like frequency_sweep() does for physical antennas --
    instead of re-assembling Z from scratch at every frequency.

    Storing eigenvectors for the whole sweep (not just eigenvalues) is cheap
    for the mode counts CMA actually uses (a few to a few dozen), and means
    the "surface current for a given frequency and eigenvalue" step never
    needs a second solve: just index eigenvectors[freq_idx, :, mode_idx].

    Mode-ordering caveat
    --------------------
    At each frequency, modes are selected/sorted independently by |lambda_n|
    (see GeneralizedEigenSolver.solve) -- there is no cross-frequency mode
    tracking/correlation here. So "mode index n" is not guaranteed to follow
    the same physical mode across the sweep when two eigenvalue curves
    cross; reading the lambda_n(f) plot by proximity to resonance
    (lambda_n = 0) is fine, but curve identity can swap at a crossing.

    Mesh freshness
    --------------
    Unlike cma_algorithm()/CharacteristicModeSolver (which load the already
    -saved path.mat_mesh2 and rely on radiation_algorithm() having been run
    on the current geometry), this function does NOT trust a possibly-stale
    path.mat_mesh2: it re-derives points/triangles/edges/vecteurs_rho from
    path.msh itself, exactly like frequency_sweep() does for physical
    antennas. Reason: this function assembles its own Z(f) from scratch and
    never reads path.mat_impedance, so there is no other check (like
    CharacteristicModeSolver.solve()'s verify_mesh_fingerprint) that would
    catch an updated .msh whose mesh1/mesh2 hadn't been regenerated to
    match -- silently sweeping the WRONG geometry otherwise.

    Pre-conditions
    --------------
    path.msh OR path.mat must exist. When path.msh is present the mesh is
    always re-derived from it, so path.mat/path.mat_mesh2 need not be up to
    date beforehand; when it is absent (mesh not produced by gmsh, e.g. a
    structured pixel grid exported from MATLAB) path.mat is used as the
    source of truth and left untouched. See cma_mesh.prepare_cma_mesh.

    Parameters
    ----------
    path                    : SimpleNamespace from setup_save_file_paths()
    freq_start, freq_stop   : float, sweep bounds in Hz
    n_points                : int, number of frequency points (>=2, both ends included)
    n_modes                 : int, number of characteristic modes to track per frequency
    regularization_epsilon  : float, relative perturbation applied to R before eigh
    sort_by                 : 'magnitude' | 'value', see GeneralizedEigenSolver
    use_gpu                 : bool or None, passed to calculate_z_matrice_sweep

    Returns
    -------
    CMASweepData
    """
    # Re-derive the mesh from path.msh when there is one -- see "Mesh
    # freshness" above -- instead of trusting whatever path.mat_mesh2 last
    # happened to contain. Meshes that never came from gmsh (no .msh, e.g. a
    # pixel grid exported straight to .mat) are loaded from path.mat as-is.
    points, triangles, edges, _vecteurs_rho = prepare_cma_mesh(path)

    frequencies = generate_freq_points(freq_start, freq_stop, n_points)
    n_freq = len(frequencies)
    n_dof = edges.total_number_of_edges
    n_modes_eff = min(n_modes, n_dof)

    print(f"Assembling Z(f) for {n_freq} frequencies ({n_dof}x{n_dof} edges) on "
          f"{describe_backend()} -- geometry reused across the whole sweep...")
    t0 = time.time()
    _omega_arr, _mu, _epsilon, _light_speed_c, _eta, Z_all = calculate_z_matrice_sweep(
        points, triangles, edges, frequencies, use_gpu=use_gpu)
    dt = time.time() - t0
    print(f"Z(f) assembly done in {dt:.1f} s ({dt/n_freq:.3f} s/frequency amortised)\n")

    params = CMAParameters(
        n_modes=n_modes_eff,
        regularization_epsilon=regularization_epsilon,
        sort_by=sort_by,
    )

    eigenvalues_arr  = np.zeros((n_freq, n_modes_eff))
    eigenvectors_arr = np.zeros((n_freq, n_dof, n_modes_eff))

    for idx in range(n_freq):
        R = np.real(Z_all[idx])
        X = np.imag(Z_all[idx])
        eigenvalues, eigenvectors, _sort_order = GeneralizedEigenSolver.solve(R, X, params)
        eigenvalues_arr[idx, :]  = eigenvalues
        eigenvectors_arr[idx, :, :] = eigenvectors

        modes_str = ", ".join(f"{lam:.3f}" for lam in eigenvalues)
        print(f"Step {idx+1}/{n_freq} | {frequencies[idx]/1e6:8.2f} MHz | lambda: {modes_str}")

    sweep_data = CMASweepData(
        frequencies=frequencies,
        eigenvalues=eigenvalues_arr,
        eigenvectors=eigenvectors_arr,
        n_modes=n_modes_eff,
    )

    cma_dir = 'data/antennas_cma'
    os.makedirs(cma_dir, exist_ok=True)
    path.mat_cma_sweep = os.path.join(cma_dir, f"{path.name}_cma_sweep.mat")
    DataManager_cma_sweep.save_data(path, sweep_data)
    print(f"\nSweep complete. CMA results for {n_modes_eff} modes saved to: {path.mat_cma_sweep}")

    return sweep_data
