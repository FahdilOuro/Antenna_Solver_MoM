import os
import time
from dataclasses import dataclass

import numpy as np
from scipy.io import savemat, loadmat

from solver.rwg.rwg3 import calculate_z_matrice_sweep
from solver.utils.compute_backend import describe_backend
from solver.utils.frequency_sweep import generate_freq_points
from solver.cma.cma_mesh import prepare_cma_mesh
from solver.cma.cma_parameters import CMAParameters
from solver.cma.cma_report import write_sweep_report
from solver.cma.eigen_solver import GeneralizedEigenSolver
from solver.cma.mode_tracking import ModeTracker, ModeTrackingResult, TrackingParameters

# How many tracking events the console prints before summarising the rest.
_MAX_EVENTS_PRINTED = 12


@dataclass
class CMASweepData:
    """
    Container for CMA results across a frequency sweep.

    `eigenvalues` / `eigenvectors` are always stored in the eigensolver's own
    per-frequency ordering (sorted by `|lambda|` at each point independently),
    whether or not tracking ran. `tracking`, when present, is the permutation
    that turns those labels into mode identities that hold across the sweep;
    `tracked()` applies it. Keeping the raw arrays as the stored form means a
    saved sweep reads the same as it always did, and the untracked view stays
    available for comparison.
    """

    frequencies:  np.ndarray   # float64, (n_freq,)             — Hz
    eigenvalues:  np.ndarray   # float64, (n_freq, n_modes)      — lambda_n(f)
    eigenvectors: np.ndarray   # float64, (n_freq, n_edges, n_modes) — J_n(f) as columns
    n_modes:      int
    # ModeTrackingResult or None — see solver.cma.mode_tracking
    tracking:     object = None

    def tracked(self) -> 'CMASweepData':
        """
        The same sweep with the modes re-indexed so that column `k` is one
        physical mode from end to end.

        Returns a NEW container; this one is left in raw solver order. Every
        plotter and report takes a CMASweepData, so the tracked view is used by
        passing `sweep_data.tracked()` wherever `sweep_data` went before.
        Idempotent: the returned object carries a consumed permutation, so
        calling `tracked()` on it again changes nothing.

        Raises
        ------
        RuntimeError if the sweep was computed with track=False, since there is
        then no permutation to apply.
        """
        if self.tracking is None:
            raise RuntimeError(
                "tracked(): this sweep carries no tracking data. Re-run "
                "cma_frequency_sweep(..., track=True) (the default), or track a saved "
                "sweep with solver.cma.mode_tracking.track_sweep() -- which needs "
                "Re(Z) per frequency, not stored in the .mat.")
        if self.tracking.is_applied:
            return self
        return CMASweepData(
            frequencies=self.frequencies,
            eigenvalues=self.tracking.reorder(self.eigenvalues),
            eigenvectors=self.tracking.reorder_eigenvectors(self.eigenvectors),
            n_modes=self.n_modes,
            tracking=self.tracking.applied(),
        )


class DataManager_cma_sweep:
    """Save and load CMA frequency-sweep results, following the DataManager_* pattern."""

    @staticmethod
    def save_data(path, sweep_data: CMASweepData) -> None:
        data = {
            'frequencies':  sweep_data.frequencies,
            'eigenvalues':  sweep_data.eigenvalues,
            'eigenvectors': sweep_data.eigenvectors,
            'n_modes':      int(sweep_data.n_modes),
        }
        track = sweep_data.tracking
        if track is not None:
            # Flat arrays plus one JSON blob for the events: savemat has no
            # natural representation for a list of records with a string field,
            # and a struct array of them would come back needing more unpacking
            # than it saves.
            data.update({
                'track_order':       np.asarray(track.order, dtype=np.int32),
                'track_sign':        np.asarray(track.sign, dtype=np.float64),
                'track_correlation': np.asarray(track.correlation, dtype=np.float64),
                'track_margin':      np.asarray(track.margin, dtype=np.float64),
                'track_metric':      str(track.metric),
                'track_correction':  str(track.correction),
                'track_events':      track.events_json(),
            })
            if track.corrected_eigenvalues is not None:
                data['track_corrected_eigenvalues'] = np.asarray(
                    track.corrected_eigenvalues, dtype=np.float64)
        savemat(path.mat_cma_sweep, data)

    @staticmethod
    def load_data(filename: str) -> CMASweepData:
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"CMA sweep file '{filename}' does not exist.")
        data = loadmat(filename)
        frequencies = data['frequencies'].squeeze()

        tracking = None
        if 'track_order' in data:
            def _text(key):
                return str(np.atleast_1d(data[key]).ravel()[0])

            corrected = data.get('track_corrected_eigenvalues')
            tracking = ModeTrackingResult(
                frequencies=np.asarray(frequencies, dtype=float),
                order=np.asarray(data['track_order'], dtype=int),
                sign=np.asarray(data['track_sign'], dtype=float),
                correlation=np.asarray(data['track_correlation'], dtype=float),
                margin=np.asarray(data['track_margin'], dtype=float),
                events=ModeTrackingResult.events_from_json(_text('track_events')),
                metric=_text('track_metric'),
                correction=_text('track_correction'),
                corrected_eigenvalues=None if corrected is None else np.asarray(
                    corrected, dtype=float),
            )

        return CMASweepData(
            frequencies=frequencies,
            eigenvalues=data['eigenvalues'],
            eigenvectors=data['eigenvectors'],
            n_modes=int(data['n_modes'].squeeze()),
            tracking=tracking,
        )


def cma_frequency_sweep(path, freq_start, freq_stop, n_points=41, n_modes=6,
                        regularization_epsilon=1e-10, sort_by='magnitude',
                        cma_solver='regularized', rank_tol=1e-6, use_gpu=None,
                        write_report=True, track=True, tracking_params=None):
    """
    Compute characteristic mode eigenvalues AND eigenvectors across a
    frequency sweep, reusing calculate_z_matrice_sweep's batched Z(f)
    assembly -- exactly like frequency_sweep() does for physical antennas --
    instead of re-assembling Z from scratch at every frequency.

    Storing eigenvectors for the whole sweep (not just eigenvalues) is cheap
    for the mode counts CMA actually uses (a few to a few dozen), and means
    the "surface current for a given frequency and eigenvalue" step never
    needs a second solve: just index eigenvectors[freq_idx, :, mode_idx].

    Mode ordering and tracking
    --------------------------
    At each frequency, modes are selected/sorted independently by |lambda_n|
    (see GeneralizedEigenSolver.solve), so "mode index n" in the returned
    arrays is a per-frequency label, not a mode identity: where two
    eigenvalue curves cross, the label follows whichever mode happens to be
    n-th closest to resonance there, and swaps.

    With track=True (the default) the sweep is additionally correlated
    across adjacent frequencies (solver.cma.mode_tracking) and the
    permutation that restores mode identity is attached to the result as
    `sweep_data.tracking`. The stored eigenvalues/eigenvectors stay in raw
    solver order -- applying the permutation is `sweep_data.tracked()`,
    which returns a second CMASweepData that every plotter and the report
    accept unchanged:

        sweep = cma_frequency_sweep(path, f1, f2)
        ModalVisualizer.plot_eigenvalues_vs_frequency(sweep.tracked())
        print(sweep.tracking.summary())

    Tracking reports what it cannot decide rather than hiding it: quasi-
    degenerate pairs, avoided crossings and steps where the correlation is
    weak come back as `sweep.tracking.events`. See the mode_tracking module
    docstring.

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
    cma_solver              : 'regularized' (default) or 'rank_truncated' -- see
                              CMAParameters. Truncation returns only rank(R) modes,
                              and rank(R) varies across the sweep, so n_modes must
                              stay below the rank at the LOWEST frequency; the loop
                              raises with that number if it does not.
    rank_tol                : null-space threshold for 'rank_truncated' (default 1e-6)
    use_gpu                 : bool or None, passed to calculate_z_matrice_sweep
    write_report            : also write a human-readable Markdown report next
                              to the .mat (resonances, modal Q, bandwidths,
                              health checks) -- see solver.cma.cma_report
    track                   : correlate the modes across adjacent frequencies and
                              attach the resulting permutation as
                              sweep_data.tracking (default True). Costs one
                              (n_modes x n_modes) correlation per step -- negligible
                              next to the Z(f) assembly -- and does not alter the
                              stored arrays; see "Mode ordering and tracking" above.
    tracking_params         : TrackingParameters, if the defaults need changing
                              (correlation metric, degeneracy tolerance, the TECA
                              -style correction hook) -- see
                              solver.cma.mode_tracking

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
        solver=cma_solver,
        rank_tol=rank_tol,
    )

    eigenvalues_arr  = np.zeros((n_freq, n_modes_eff))
    eigenvectors_arr = np.zeros((n_freq, n_dof, n_modes_eff))

    # Tracking runs inside this loop rather than as a pass over the finished
    # arrays because it needs R(f), and R(f) is only conveniently available here
    # -- the saved .mat holds modes, not the operator they came from. The
    # tracker keeps just two frequencies at a time, so this adds nothing to peak
    # memory beyond what Z_all already costs.
    tracker = ModeTracker(tracking_params or TrackingParameters()) if track else None

    for idx in range(n_freq):
        R = np.real(Z_all[idx])
        X = np.imag(Z_all[idx])
        eigenvalues, eigenvectors, _sort_order = GeneralizedEigenSolver.solve(R, X, params)

        # 'rank_truncated' returns at most rank(R) modes, and rank(R) grows with
        # frequency (measured on the CubeSat mesh: 41 at 233 MHz, 64 at 440,
        # 71 at 500). The sweep arrays are sized once, so a frequency whose rank
        # falls below n_modes cannot be stored -- and silently padding it with
        # zeros would put fictitious modes into every downstream plot. Say so
        # instead, and name the fix.
        if eigenvalues.shape[0] < n_modes_eff:
            raise RuntimeError(
                f"CMA sweep: at {frequencies[idx]/1e6:.3f} MHz, R has numerical rank "
                f"{eigenvalues.shape[0]}, so only that many characteristic modes exist "
                f"there -- fewer than the {n_modes_eff} this sweep is collecting. Lower "
                f"n_modes to {eigenvalues.shape[0]} or below, raise the lower frequency "
                f"bound, or refine the mesh (rank(R) grows with electrical size)."
            )

        eigenvalues_arr[idx, :]  = eigenvalues
        eigenvectors_arr[idx, :, :] = eigenvectors

        if tracker is not None:
            tracker.push(frequencies[idx], eigenvalues, eigenvectors, R=R, X=X)

        modes_str = ", ".join(f"{lam:.3f}" for lam in eigenvalues)
        print(f"Step {idx+1}/{n_freq} | {frequencies[idx]/1e6:8.2f} MHz | lambda: {modes_str}")

    tracking = tracker.finalize() if tracker is not None else None
    if tracking is not None:
        print()
        print(tracking.summary())
        # A degenerate band flags the same pair at every step inside it, so the
        # list can run long; the console gets the head of it and the report and
        # sweep_data.tracking.events keep all of it.
        for event in tracking.events[:_MAX_EVENTS_PRINTED]:
            print(f"  {event.describe()}")
        if len(tracking.events) > _MAX_EVENTS_PRINTED:
            print(f"  ... {len(tracking.events) - _MAX_EVENTS_PRINTED} more "
                  "(see sweep_data.tracking.events and the report)")

    sweep_data = CMASweepData(
        frequencies=frequencies,
        eigenvalues=eigenvalues_arr,
        eigenvectors=eigenvectors_arr,
        n_modes=n_modes_eff,
        tracking=tracking,
    )

    cma_dir = 'data/antennas_cma'
    os.makedirs(cma_dir, exist_ok=True)
    path.mat_cma_sweep = os.path.join(cma_dir, f"{path.name}_cma_sweep.mat")
    DataManager_cma_sweep.save_data(path, sweep_data)
    print(f"\nSweep complete. CMA results for {n_modes_eff} modes saved to: {path.mat_cma_sweep}")

    if write_report:
        # The .mat holds eigenvalues and eigenvectors, which say little on their
        # own; the report turns them into resonances, Q, bandwidths and the
        # sanity checks, so a run can be interpreted later without re-deriving
        # any of it. Z_all lets it check the matrix the modes came from.
        #
        # The report gets the TRACKED view when there is one: its tables are
        # read by a human as mode curves, and a column that changes mode
        # halfway down is exactly what tracking exists to remove. It says which
        # order it is in, and the .mat beside it keeps the raw one.
        write_sweep_report(path, sweep_data.tracked() if tracking is not None else sweep_data,
                           points, triangles, edges, Z_all=Z_all, params=params)

    return sweep_data
