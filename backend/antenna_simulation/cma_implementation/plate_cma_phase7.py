"""
Phase 7 — Canonical CMA benchmark: rectangular PEC plate (1.0 m x 0.5 m).

Replaces the prior (uncommitted-deleted) plate_cma.ipynb, which used the legacy
impedance_matrix_z and failed (cond(R)=5.14e19, eigendecomposition not PD).
This script uses the new Duffy/Gauss-Dunavant impedance_matrix_z (Phase 4-6).

See backend/utils/quadrature/_diagnostics/FINDINGS.md section 9 for the scope
decision (Phase 8 / Nice Cube dropped; refined-mesh sweep restricted to
resonance windows) driving the parameters below.

Run:
  cd <repo_root>
  python -m backend.antenna_simulation.cma_implementation.plate_cma_phase7 [--refined] [--out DIR]
"""
import argparse
import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import gmsh

from backend.utils.gmsh_function import setup_performance_config, generate_and_save_mesh, extract_msh_to_mat
from backend.utils.file_path import setup_save_file_paths
from backend.rwg.rwg1 import Points, Triangles, load_mesh_file, filter_complexes_jonctions
from backend.rwg.rwg2 import Barycentric_triangle, Vecteurs_Rho
from backend.utils.impmet import impedance_matrix_z
from backend.cma.eigen_solver import GeneralizedEigenSolver
from backend.cma.cma_parameters import CMAParameters

# ---------------------------------------------------------------------------
# Physical constants / plate geometry
# ---------------------------------------------------------------------------
EPSILON = 8.854e-12
MU      = 1.257e-6
C       = 1.0 / np.sqrt(EPSILON * MU)

LENGTH = 1.0   # m
WIDTH  = 0.5   # m

FREQ_MIN = 50e6
FREQ_MAX = 600e6
N_FREQ   = 41

N_MODES_SOLVE  = 6   # extracted per frequency (extra modes give headroom for tracking through crossings)
N_MODES_REPORT = 4   # reported as lambda_n(f), MS_n(f)

RESONANCE_WINDOW_MODE1 = (100e6, 220e6)
RESONANCE_WINDOW_MODE2 = (200e6, 450e6)

OUT_DIR_DEFAULT = os.path.join(
    _REPO_ROOT, 'data', 'cma_results', 'phase7_plate'
)


def mesh_size_for_freq(freq_max, edges_per_lambda=12.0):
    lambda_min = C / freq_max
    return lambda_min / edges_per_lambda


def build_plate_mesh(name, mesh_size):
    """Generate the plate geometry+mesh with Gmsh, then run rwg1/rwg2 preprocessing.

    Returns (points, triangles, edges, bary, rho, E).
    """
    path = setup_save_file_paths(name)

    gmsh.initialize()
    gmsh.model.add(name)
    setup_performance_config()
    gmsh.model.occ.addRectangle(-LENGTH / 2, -WIDTH / 2, 0, LENGTH, WIDTH)
    gmsh.model.occ.synchronize()
    generate_and_save_mesh(path, mesh_size)
    gmsh.finalize()

    extract_msh_to_mat(path.msh, path.mat)
    p, t = load_mesh_file(path.mat)

    points = Points(p)
    triangles = Triangles(t)
    triangles.calculate_triangles_area_and_center(points)
    edges = triangles.get_edges()
    filter_complexes_jonctions(points, triangles, edges)
    edges.compute_edges_length(points)

    bary = Barycentric_triangle()
    bary.calculate_barycentric_center(points, triangles)
    rho = Vecteurs_Rho()
    rho.calculate_vecteurs_rho(points, triangles, edges, bary)

    E = edges.total_number_of_edges
    print(f"[{name}] points={points.total_of_points}  triangles={triangles.total_of_triangles}  E={E}")
    return points, triangles, edges, bary, rho, E


def solve_at_frequency(points, triangles, edges, freq, n_modes=N_MODES_SOLVE):
    omega = 2.0 * np.pi * freq
    k = omega / C
    k_complex = 1j * k

    t0 = time.time()
    Z = impedance_matrix_z(edges, triangles, points, k_complex, omega, MU, EPSILON,
                            quad_order=6, duffy_order=7)
    dt = time.time() - t0

    R = np.real(Z)
    X = np.imag(Z)

    sym_R = np.linalg.norm(R - R.T) / np.linalg.norm(R)
    sym_X = np.linalg.norm(X - X.T) / np.linalg.norm(X)

    R_sym = (R + R.T) * 0.5
    eig_R = np.linalg.eigvalsh(R_sym)
    min_eig_R = eig_R.min()
    max_eig_R = eig_R.max()

    # regularization_epsilon=1.5 gives eps_adapt≈6.2e-3 (1.9% of max_eig≈0.33),
    # required because SELF-pair Duffy asymmetry shifts min_eig(R_sym) to ~-1.1e-3
    # on the E~1064 target mesh.  See FINDINGS.md §10 for root-cause analysis.
    params = CMAParameters(n_modes=n_modes, regularization_epsilon=1.5, sort_by='magnitude')
    eigenvalues, eigenvectors, _ = GeneralizedEigenSolver.solve(R, X, params)

    # Orthogonality check: J_i^T R J_j  (eigh(b-normalized) => should be ~delta_ij)
    G = eigenvectors.T @ R @ eigenvectors
    diag = np.diag(G)
    off_diag_mask = ~np.eye(len(diag), dtype=bool)
    denom = np.sqrt(np.outer(np.abs(diag), np.abs(diag)))
    ortho_matrix = np.abs(G) / np.maximum(denom, 1e-300)
    max_ortho_off_diag = ortho_matrix[off_diag_mask].max() if len(diag) > 1 else 0.0

    return {
        'freq': freq,
        'dt': dt,
        'E': R.shape[0],
        'sym_R': sym_R,
        'sym_X': sym_X,
        'min_eig_R': min_eig_R,
        'max_eig_R': max_eig_R,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'max_ortho_off_diag': max_ortho_off_diag,
    }


def track_modes(records, n_track=2):
    """Greedy eigenvector-correlation tracking across the frequency sweep.

    Returns:
      tracked_lambda: (n_track, n_freq) array of lambda_n along tracked trajectories
      min_corr: (n_track, n_freq-1) array of step-to-step correlation (1.0 = perfect continuity)
    """
    n_freq = len(records)
    tracked_idx = np.zeros((n_track, n_freq), dtype=int)
    tracked_lambda = np.zeros((n_track, n_freq))
    min_corr = np.full((n_track, n_freq - 1), np.nan)

    # seed: lowest |lambda| modes at the first frequency (already magnitude-sorted)
    for k in range(n_track):
        tracked_idx[k, 0] = k
        tracked_lambda[k, 0] = records[0]['eigenvalues'][k]

    for i in range(1, n_freq):
        prev_vecs = records[i - 1]['eigenvectors']
        cur_vecs = records[i]['eigenvectors']
        for k in range(n_track):
            prev_j = tracked_idx[k, i - 1]
            v_prev = prev_vecs[:, prev_j]
            v_prev = v_prev / (np.linalg.norm(v_prev) + 1e-300)

            n_modes = cur_vecs.shape[1]
            corrs = np.zeros(n_modes)
            for m in range(n_modes):
                v_cur = cur_vecs[:, m]
                v_cur = v_cur / (np.linalg.norm(v_cur) + 1e-300)
                corrs[m] = abs(np.dot(v_prev, v_cur))

            best = int(np.argmax(corrs))
            tracked_idx[k, i] = best
            tracked_lambda[k, i] = records[i]['eigenvalues'][best]
            min_corr[k, i - 1] = corrs[best]

    return tracked_lambda, min_corr


def find_zero_crossing(freqs, lam):
    """Return the frequency of the first sign change of lam(f) (linear interpolation)."""
    for i in range(len(lam) - 1):
        if lam[i] == 0.0:
            return freqs[i]
        if lam[i] * lam[i + 1] < 0:
            f0, f1 = freqs[i], freqs[i + 1]
            l0, l1 = lam[i], lam[i + 1]
            return f0 + (f1 - f0) * (-l0) / (l1 - l0)
    return None


def run_sweep(name, mesh_size, freqs, out_dir, n_modes=N_MODES_SOLVE):
    os.makedirs(out_dir, exist_ok=True)
    points, triangles, edges, bary, rho, E = build_plate_mesh(name, mesh_size)

    records = []
    t_start = time.time()
    for i, f in enumerate(freqs):
        rec = solve_at_frequency(points, triangles, edges, f, n_modes=n_modes)
        records.append(rec)
        elapsed = time.time() - t_start
        eta = elapsed / (i + 1) * (len(freqs) - i - 1)
        print(f"  [{name}] {i+1}/{len(freqs)}  f={f/1e6:7.2f} MHz  "
              f"dt={rec['dt']:6.1f}s  symR={rec['sym_R']:.2e}  symX={rec['sym_X']:.2e}  "
              f"minEigR={rec['min_eig_R']:+.3e}  lam[0:2]={rec['eigenvalues'][:2]}  "
              f"ETA={eta/60:.1f} min")

        # Checkpoint after every point.
        np.savez(
            os.path.join(out_dir, f"{name}_checkpoint.npz"),
            freqs=np.array([r['freq'] for r in records]),
            sym_R=np.array([r['sym_R'] for r in records]),
            sym_X=np.array([r['sym_X'] for r in records]),
            min_eig_R=np.array([r['min_eig_R'] for r in records]),
            max_eig_R=np.array([r['max_eig_R'] for r in records]),
            max_ortho_off_diag=np.array([r['max_ortho_off_diag'] for r in records]),
            dt=np.array([r['dt'] for r in records]),
            eigenvalues=np.array([r['eigenvalues'] for r in records]),
            E=E,
        )

    return records, E


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refined', action='store_true',
                         help='Run the 1.5x-refined mesh on the targeted resonance windows only')
    parser.add_argument('--out', default=OUT_DIR_DEFAULT)
    args = parser.parse_args()

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    if not args.refined:
        mesh_size = mesh_size_for_freq(FREQ_MAX, edges_per_lambda=12.0)
        freqs = np.linspace(FREQ_MIN, FREQ_MAX, N_FREQ)
        print(f"Phase 7 baseline sweep: mesh_size={mesh_size*100:.2f} cm, "
              f"{N_FREQ} points in [{FREQ_MIN/1e6:.0f}, {FREQ_MAX/1e6:.0f}] MHz")
        records, E = run_sweep('phase7_plate_baseline', mesh_size, freqs, out_dir)

        tracked_lambda, min_corr = track_modes(records, n_track=2)
        f_res1 = find_zero_crossing(freqs, tracked_lambda[0])
        f_res2 = find_zero_crossing(freqs, tracked_lambda[1])

        np.savez(os.path.join(out_dir, 'phase7_baseline_final.npz'),
                  freqs=freqs, tracked_lambda=tracked_lambda, min_corr=min_corr,
                  f_res1=f_res1 if f_res1 else np.nan,
                  f_res2=f_res2 if f_res2 else np.nan, E=E)

        print(f"\nBaseline mesh (E={E}): resonance mode1 = {f_res1/1e6 if f_res1 else None} MHz "
              f"(target [{RESONANCE_WINDOW_MODE1[0]/1e6:.0f},{RESONANCE_WINDOW_MODE1[1]/1e6:.0f}])")
        print(f"Baseline mesh (E={E}): resonance mode2 = {f_res2/1e6 if f_res2 else None} MHz "
              f"(target [{RESONANCE_WINDOW_MODE2[0]/1e6:.0f},{RESONANCE_WINDOW_MODE2[1]/1e6:.0f}])")
        print(f"Min step-to-step tracking correlation: mode1={np.nanmin(min_corr[0]):.4f}  "
              f"mode2={np.nanmin(min_corr[1]):.4f}")

    else:
        # Refined mesh: targeted windows around baseline resonances only.
        baseline_path = os.path.join(out_dir, 'phase7_baseline_final.npz')
        if not os.path.isfile(baseline_path):
            raise FileNotFoundError("Run the baseline sweep first (without --refined).")
        baseline = np.load(baseline_path)
        f_res1, f_res2 = float(baseline['f_res1']), float(baseline['f_res2'])

        mesh_size = mesh_size_for_freq(FREQ_MAX, edges_per_lambda=12.0) / 1.5
        window1 = np.linspace(max(FREQ_MIN, f_res1 - 30e6), f_res1 + 30e6, 7)
        window2 = np.linspace(max(FREQ_MIN, f_res2 - 30e6), min(FREQ_MAX, f_res2 + 30e6), 7)
        freqs = np.unique(np.concatenate([window1, window2]))

        print(f"Phase 7 refined-mesh convergence sweep: mesh_size={mesh_size*100:.2f} cm, "
              f"{len(freqs)} targeted points around baseline resonances "
              f"({f_res1/1e6:.1f}, {f_res2/1e6:.1f}) MHz")
        records, E = run_sweep('phase7_plate_refined', mesh_size, freqs, out_dir)

        tracked_lambda, min_corr = track_modes(records, n_track=2)
        f_res1_ref = find_zero_crossing(freqs, tracked_lambda[0])
        f_res2_ref = find_zero_crossing(freqs, tracked_lambda[1])

        shift1 = abs(f_res1_ref - f_res1) / f_res1 * 100 if f_res1_ref else None
        shift2 = abs(f_res2_ref - f_res2) / f_res2 * 100 if f_res2_ref else None

        np.savez(os.path.join(out_dir, 'phase7_refined_final.npz'),
                  freqs=freqs, tracked_lambda=tracked_lambda, min_corr=min_corr,
                  f_res1=f_res1_ref if f_res1_ref else np.nan,
                  f_res2=f_res2_ref if f_res2_ref else np.nan, E=E,
                  shift1=shift1 if shift1 else np.nan, shift2=shift2 if shift2 else np.nan)

        print(f"\nRefined mesh (E={E}): resonance mode1 = "
              f"{f_res1_ref/1e6 if f_res1_ref else None} MHz  (shift={shift1:.2f}% )" if shift1 else "")
        print(f"Refined mesh (E={E}): resonance mode2 = "
              f"{f_res2_ref/1e6 if f_res2_ref else None} MHz  (shift={shift2:.2f}% )" if shift2 else "")


if __name__ == '__main__':
    main()
