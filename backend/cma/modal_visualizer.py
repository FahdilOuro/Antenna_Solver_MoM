import os

import numpy as np
import matplotlib.pyplot as plt

from backend.rwg.rwg5 import visualize_surface_current
from backend.efield.efield2 import deform_sphere_for_radiation_pattern
from backend.efield.efield2 import visualize_surface_current as visualize_sphere_gain
from backend.cma.modal_current import ModalCurrent
from backend.cma.modal_significance import modal_significance
from backend.cma.characteristic_angle import characteristic_angle


class ModalVisualizer:
    """Plotly/matplotlib visualisation of characteristic mode currents, far fields and spectra."""

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

    @staticmethod
    def plot_mode_farfield(sphere_points, sphere_triangles, gain_db, mode_index, eigenvalue,
                           title_prefix="CMA", dynamic_range=20):
        """
        Plot the 3D far-field directivity pattern radiated by a single characteristic mode,
        using the same deformed-sphere rendering as the physical-antenna gain patterns
        (backend.efield.efield2.render_and_save_pattern).

        Parameters
        ----------
        sphere_points, sphere_triangles, gain_db — from modal_farfield.compute_modal_farfield()
        mode_index, eigenvalue, title_prefix     — see plot_mode()
        dynamic_range                            — dB range shown before clipping (see
                                                    deform_sphere_for_radiation_pattern)

        Returns
        -------
        plotly.graph_objects.Figure
        """
        sphere_points_deformed = deform_sphere_for_radiation_pattern(sphere_points, gain_db, dynamic_range)
        gain_faces_db = np.mean(gain_db[sphere_triangles], axis=0)
        title = f"{title_prefix} — Mode {mode_index + 1} far field,  λ = {eigenvalue:.4f}"
        return visualize_sphere_gain(sphere_points_deformed, sphere_triangles, gain_faces_db, title)

    @staticmethod
    def plot_eigenvalues_vs_frequency(sweep_data, n_modes_to_plot=None, freq_unit='MHz',
                                      save_pdf=False, save_folder=None, antenna_name="",
                                      yscale='symlog', linthresh=1.0):
        """
        Plot characteristic eigenvalues lambda_n(f) for the first `n_modes_to_plot` modes
        across a CMA frequency sweep (backend.cma.cma_frequency_sweep.cma_frequency_sweep).

        A mode is resonant where its curve crosses lambda_n = 0 (dashed reference line).
        See cma_frequency_sweep()'s docstring for the mode-tracking/crossing caveat.

        Why the default y axis is symlog
        --------------------------------
        lambda_n is unbounded: a barely-radiating mode reaches ~1e5 while the
        near-resonant ones sit around 1, so on a linear axis one curve sets the
        scale and flattens every other mode onto the zero line -- the exact
        complaint AToM's own documentation makes about this plot. symlog keeps
        both visible, staying linear within +/-`linthresh` so the crossings
        through zero (the resonances) are not distorted. Pass yscale='linear'
        for the raw view. For a bounded alternative see
        plot_characteristic_angle_vs_frequency().

        Parameters
        ----------
        sweep_data       — CMASweepData
        n_modes_to_plot  — int or None (defaults to all modes stored in sweep_data)
        freq_unit        — 'Hz', 'kHz', 'MHz', or 'GHz'
        save_pdf         — bool, save as PDF if True
        save_folder      — str, destination folder (defaults to 'data/fig_image/')
        antenna_name     — str, used in the title and the saved filename
        yscale           — 'symlog' (default) or 'linear'
        linthresh        — half-width of the linear region when yscale='symlog'

        Returns
        -------
        matplotlib.figure.Figure
        """
        n_modes_to_plot = n_modes_to_plot or sweep_data.n_modes
        n_modes_to_plot = min(n_modes_to_plot, sweep_data.n_modes)

        unit_factors = {'Hz': 1, 'kHz': 1e3, 'MHz': 1e6, 'GHz': 1e9}
        freq_divisor = unit_factors.get(freq_unit, 1e6)
        freq_converted = sweep_data.frequencies / freq_divisor

        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.family'] = 'serif'

        fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
        colors = plt.cm.tab10(np.linspace(0, 1, 10))

        for n in range(n_modes_to_plot):
            ax.plot(freq_converted, sweep_data.eigenvalues[:, n], color=colors[n % 10],
                    linewidth=2, label=rf'Mode {n + 1}')

        ax.axhline(0, color="#2c3e50", linestyle='--', linewidth=1.2, label='Resonance (λ=0)', alpha=0.6)

        if yscale == 'symlog':
            ax.set_yscale('symlog', linthresh=linthresh)
        else:
            ax.set_yscale(yscale)

        ax.set_xlabel(f'Frequency ({freq_unit})', fontsize=13, labelpad=10)
        ax.set_ylabel(r'Characteristic value $\lambda_n$', fontsize=13, labelpad=10)
        ax.set_title(f'Characteristic Eigenvalues vs Frequency — {antenna_name}'.strip(' —'), fontsize=14)
        ax.grid(True, which='both', linestyle=':', alpha=0.5, color='#bdc3c7')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, fontsize=10)
        plt.tight_layout()

        if save_pdf:
            if save_folder is None:
                save_folder = "data/fig_image/"
            os.makedirs(save_folder, exist_ok=True)
            pdf_path = os.path.join(save_folder, f"cma_eigenvalues_{antenna_name}.pdf".replace(" ", "_"))
            plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
            print(f"Figure exported to: {pdf_path}")

        plt.show()

    @staticmethod
    def plot_modal_significance(sweep_data, n_modes_to_plot=None, freq_unit='MHz',
                                save_pdf=False, save_folder=None, antenna_name=""):
        """
        Plot Modal Significance MS_n(f) = 1/sqrt(1 + lambda_n(f)^2) for the first
        `n_modes_to_plot` modes across a CMA frequency sweep.

        The MS >= 1/sqrt(2) reference line marks the conventional "significantly
        contributing" threshold (the CMA analogue of a -3 dB bandwidth).

        Parameters mirror plot_eigenvalues_vs_frequency().

        Returns
        -------
        matplotlib.figure.Figure
        """
        n_modes_to_plot = n_modes_to_plot or sweep_data.n_modes
        n_modes_to_plot = min(n_modes_to_plot, sweep_data.n_modes)

        unit_factors = {'Hz': 1, 'kHz': 1e3, 'MHz': 1e6, 'GHz': 1e9}
        freq_divisor = unit_factors.get(freq_unit, 1e6)
        freq_converted = sweep_data.frequencies / freq_divisor

        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.family'] = 'serif'

        fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
        colors = plt.cm.tab10(np.linspace(0, 1, 10))

        for n in range(n_modes_to_plot):
            ms_n = modal_significance(sweep_data.eigenvalues[:, n])
            ax.plot(freq_converted, ms_n, color=colors[n % 10], linewidth=2, label=rf'Mode {n + 1}')

        ax.axhline(1 / np.sqrt(2), color="#2c3e50", linestyle='--', linewidth=1.2,
                   label=r'MS = $1/\sqrt{2}$', alpha=0.6)

        ax.set_xlabel(f'Frequency ({freq_unit})', fontsize=13, labelpad=10)
        ax.set_ylabel('Modal Significance', fontsize=13, labelpad=10)
        ax.set_ylim(0, 1.05)
        ax.set_title(f'Modal Significance vs Frequency — {antenna_name}'.strip(' —'), fontsize=14)
        ax.grid(True, which='both', linestyle=':', alpha=0.5, color='#bdc3c7')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, fontsize=10)
        plt.tight_layout()

        if save_pdf:
            if save_folder is None:
                save_folder = "data/fig_image/"
            os.makedirs(save_folder, exist_ok=True)
            pdf_path = os.path.join(save_folder, f"cma_modal_significance_{antenna_name}.pdf".replace(" ", "_"))
            plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
            print(f"Figure exported to: {pdf_path}")

        plt.show()

    @staticmethod
    def plot_characteristic_angle_vs_frequency(sweep_data, n_modes_to_plot=None, freq_unit='MHz',
                                               save_pdf=False, save_folder=None, antenna_name=""):
        """
        Plot the characteristic angle delta_n(f) = 180 deg - arctan(lambda_n(f))
        for the first `n_modes_to_plot` modes across a CMA frequency sweep.

        This is the representation to read first. delta_n is confined to
        (90 deg, 270 deg), so every mode stays visible at the same scale --
        unlike lambda_n, where one barely-radiating mode reaching ~1e5 flattens
        all the others onto zero (see plot_eigenvalues_vs_frequency).

            delta = 90 deg   : lambda -> +inf, inductive
            delta = 180 deg  : RESONANCE (dashed reference line)
            delta = 270 deg  : lambda -> -inf, capacitive

        A mode resonates where its curve crosses 180 deg; the steeper the
        crossing, the higher (qualitatively) its Q, i.e. the narrower its band.

        Parameters mirror plot_eigenvalues_vs_frequency().

        Returns
        -------
        matplotlib.figure.Figure
        """
        n_modes_to_plot = n_modes_to_plot or sweep_data.n_modes
        n_modes_to_plot = min(n_modes_to_plot, sweep_data.n_modes)

        unit_factors = {'Hz': 1, 'kHz': 1e3, 'MHz': 1e6, 'GHz': 1e9}
        freq_divisor = unit_factors.get(freq_unit, 1e6)
        freq_converted = sweep_data.frequencies / freq_divisor

        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.family'] = 'serif'

        fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
        colors = plt.cm.tab10(np.linspace(0, 1, 10))

        for n in range(n_modes_to_plot):
            delta_n = characteristic_angle(sweep_data.eigenvalues[:, n])
            ax.plot(freq_converted, delta_n, color=colors[n % 10], linewidth=2,
                    label=rf'Mode {n + 1}')

        ax.axhline(180, color="#2c3e50", linestyle='--', linewidth=1.2,
                   label='Resonance (δ=180°)', alpha=0.6)

        ax.set_xlabel(f'Frequency ({freq_unit})', fontsize=13, labelpad=10)
        ax.set_ylabel(r'Characteristic angle $\delta_n$ (deg)', fontsize=13, labelpad=10)
        ax.set_ylim(90, 270)
        ax.set_yticks([90, 120, 150, 180, 210, 240, 270])
        ax.set_title(f'Characteristic Angle vs Frequency — {antenna_name}'.strip(' —'), fontsize=14)
        ax.grid(True, which='both', linestyle=':', alpha=0.5, color='#bdc3c7')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, fontsize=10)
        plt.tight_layout()

        if save_pdf:
            if save_folder is None:
                save_folder = "data/fig_image/"
            os.makedirs(save_folder, exist_ok=True)
            pdf_path = os.path.join(save_folder, f"cma_characteristic_angle_{antenna_name}.pdf".replace(" ", "_"))
            plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
            print(f"Figure exported to: {pdf_path}")

        plt.show()
