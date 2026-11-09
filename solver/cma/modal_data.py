import os
from dataclasses import dataclass

import numpy as np
from scipy.io import savemat, loadmat


@dataclass
class ModalData:
    """Container for CMA results at a single frequency."""

    eigenvalues:  np.ndarray   # float64, (K,)      — characteristic values λ_n
    eigenvectors: np.ndarray   # float64, (N_edges, K) — modal currents J_n as columns
    frequency:    float        # Hz
    n_modes:      int          # K
    sort_order:   np.ndarray   # int, (K,)  — indices in the full spectrum before selection


class DataManager_cma:
    """Save and load CMA results following the DataManager_rwg* pattern."""

    @staticmethod
    def save_data(path, modal_data: ModalData) -> None:
        data = {
            'eigenvalues':  modal_data.eigenvalues,
            'eigenvectors': modal_data.eigenvectors,
            'frequency':    float(modal_data.frequency),
            'n_modes':      int(modal_data.n_modes),
            'sort_order':   modal_data.sort_order.astype(np.int32),
        }
        savemat(path.mat_cma, data)

    @staticmethod
    def load_data(filename: str) -> ModalData:
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"CMA file '{filename}' does not exist.")

        data = loadmat(filename)

        eigenvalues = data['eigenvalues'].squeeze()

        # Avoid squeezing a (N, 1) eigenvector array into (N,)
        evecs = data['eigenvectors']
        if evecs.ndim == 1:
            evecs = evecs[:, np.newaxis]

        frequency  = float(data['frequency'].squeeze())
        n_modes    = int(data['n_modes'].squeeze())
        sort_order = data['sort_order'].squeeze().astype(int)

        return ModalData(
            eigenvalues=eigenvalues,
            eigenvectors=evecs,
            frequency=frequency,
            n_modes=n_modes,
            sort_order=sort_order,
        )
