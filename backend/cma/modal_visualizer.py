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
    def plot_mode(points, triangles, surface_current, mode_index, eigenvalue, title_prefix="CMA",
                  current_vector=None):
        """
        Plot the surface current of a single characteristic mode.

        Parameters
        ----------
        points, triangles       — mesh objects from DataManager_rwg2
        surface_current         — float64 (N_triangles,) from ModalCurrent.reconstruct
        mode_index              — 0-based mode index (displayed as mode_index+1)
        eigenvalue              — λ_n for this mode
        title_prefix            — string prefix for the figure title
        current_vector          — complex (3, N_triangles) from ModalCurrent.reconstruct,
                                  or None. When given, black arrows show the direction the
                                  modal current flows in (see rwg5.visualize_surface_current).

        Returns
        -------
        plotly.graph_objects.Figure
        """
        title = f"{title_prefix} — Mode {mode_index + 1},  λ = {eigenvalue:.4f}"
        return visualize_surface_current(points, triangles, surface_current, feed_point=None,
                                         title=title, current_vector=current_vector)

    @staticmethod
    def plot_all_modes(points, triangles, edges, vecteurs_rho, modal_data, show_current_arrows=True):
        """
        Reconstruct and plot every mode stored in modal_data.

        Returns
        -------
        list[plotly.graph_objects.Figure]  — one figure per mode
        """
        figures = []
        for n in range(modal_data.n_modes):
            J_n = modal_data.eigenvectors[:, n]
            J_surface, J_vector = ModalCurrent.reconstruct(J_n, triangles, edges, vecteurs_rho)
            fig = ModalVisualizer.plot_mode(points, triangles, J_surface, n, modal_data.eigenvalues[n],
                                            current_vector=J_vector if show_current_arrows else None)
            figures.append(fig)
        return figures

    @staticmethod
    def plot_mode_farfield(sphere_points, sphere_triangles, gain_db, mode_index, eigenvalue,
                           title_prefix="CMA", dynamic_range=20,
                           save_image=False, save_folder=None, antenna_name="",
                           directivity_max_db=None, show_annotation=True):
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
        save_image                               — also write the figure to
                                                    cma_mode<n>_farfield_<antenna_name>.pdf
        save_folder                              — destination (defaults to 'data/fig_image/')
        antenna_name                             — used in the saved filename
        directivity_max_db                       — peak directivity in dBi shown in the in-figure
                                                    annotation; defaults to max(gain_db)
        show_annotation                          — draw the mode number, λ and peak directivity
                                                    inside the figure body (survives the PDF export,
                                                    unlike the title)

        Returns
        -------
        plotly.graph_objects.Figure
        """
        sphere_points_deformed = deform_sphere_for_radiation_pattern(sphere_points, gain_db, dynamic_range)
        gain_faces_db = np.mean(gain_db[sphere_triangles], axis=0)
        if directivity_max_db is None:
            directivity_max_db = float(np.max(gain_db))
        title = f"{title_prefix} — Mode {mode_index + 1} far field,  λ = {eigenvalue:.4f}"
        fig = visualize_sphere_gain(sphere_points_deformed, sphere_triangles, gain_faces_db, title)

        if show_annotation:
            # efield2's renderer forces title='', so the mode identity has to be written
            # into the figure body to be readable interactively and in the PDF export
            # below, which crops the top margin away.
            fig.add_annotation(
                xref="paper", yref="paper", x=0.02, y=0.98,
                xanchor="left", yanchor="top", align="left", showarrow=False,
                text=(f"Mode {mode_index + 1}<br>"
                      f"λ = {eigenvalue:.4f}<br>"
                      f"Peak directivity = {directivity_max_db:.2f} dBi"),
                font=dict(size=14, color="black"),
                bgcolor="rgba(255,255,255,0.7)",
                bordercolor="black", borderwidth=1, borderpad=4,
            )

        if save_image:
            if save_folder is None:
                save_folder = "data/fig_image/"
            os.makedirs(save_folder, exist_ok=True)
            pdf_name = f"cma_mode{mode_index + 1}_farfield_{antenna_name}.pdf".replace(" ", "_")
            pdf_path = os.path.join(save_folder, pdf_name)
            # t=0 here, unlike the modal-current export: efield2's renderer
            # forces title='', so there is no title to clip and the tighter
            # crop is free. The mode, λ and peak directivity are carried by the
            # in-figure annotation above (and the mode also by the filename).
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=0, b=0),
            )
            fig.write_image(pdf_path, format="pdf")
            print(f"Image saved in PDF format: {pdf_path}")

        return fig

    @staticmethod
    def plot_eigenvalues_vs_frequency(sweep_data, n_modes_to_plot=None, freq_unit='MHz',
                                      save_pdf=False, save_folder=None, antenna_name="",
                                      yscale='symlog', linthresh=1.0):
        """
        Plot characteristic eigenvalues lambda_n(f) for the first `n_modes_to_plot` modes
        across a CMA frequency sweep (backend.cma.cma_frequency_sweep.cma_frequency_sweep).

        A mode is resonant where its curve crosses lambda_n = 0 (dashed reference line).
        Mode identity across the sweep
        ------------------------------
        `sweep_data` straight out of cma_frequency_sweep() is in the solver's
        per-frequency order, so a curve here can jump from one physical mode to
        another where two eigenvalues cross. Pass `sweep_data.tracked()`
        instead -- same object type, modes followed across the sweep by the
        R-weighted correlation -- and read `sweep_data.tracking.events` for the
        bands where that identity is not decidable (see
        backend.cma.mode_tracking). The same applies to every plotter below.

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
