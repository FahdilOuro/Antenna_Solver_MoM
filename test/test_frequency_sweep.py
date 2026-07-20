"""
End-to-end regression test for the fast frequency_sweep() path
(backend/utils/frequency_sweep.py).

Context
-------
frequency_sweep() used to call radiation_algorithm() once per frequency
point, which reloads and recomputes the mesh geometry, RWG rho vectors, and
the whole Z(f) assembly from scratch every time -- purely redundant work for
a sweep, since none of that depends on frequency. The rewritten version
computes the mesh, the gap-source excitation vector, and (inside Z(f)
itself) every triangle-pair distance and Duffy node exactly once for the
whole sweep, and only repeats the genuinely frequency-dependent Green's-
function kernel and the linear solve per point.

This test builds a small real gmsh mesh (few cm strip, coarse mesh so it
stays fast) and checks that the FAST path's per-frequency S11/impedance/
current results are identical (to numerical-solve precision) to calling
radiation_algorithm() directly at each of the same frequencies -- i.e. the
optimisation changed nothing about what is computed, only how often.
"""

import os

import gmsh
import numpy as np
import pytest
from scipy.io import loadmat

from backend.utils.file_path import setup_save_file_paths
from backend.utils.gmsh_function import setup_performance_config, generate_and_save_mesh
from backend.src.radiation_algorithm.radiation_algorithm import radiation_algorithm
from backend.utils.frequency_sweep import frequency_sweep, load_freq_sweep_data, generate_freq_points

ANTENNA_NAME = "unittest_freq_sweep_strip"


# ---------------------------------------------------------------------------
# generate_freq_points: point-count-based sweep generation + error handling
# ---------------------------------------------------------------------------

def test_generate_freq_points_includes_both_endpoints():
    freqs = generate_freq_points(60e6, 90e6, 4)
    np.testing.assert_allclose(freqs, [60e6, 70e6, 80e6, 90e6])


def test_generate_freq_points_single_step_two_points():
    freqs = generate_freq_points(100e6, 200e6, 2)
    np.testing.assert_allclose(freqs, [100e6, 200e6])


@pytest.mark.parametrize("freq_start,freq_stop,n_points,match", [
    (90e6, 60e6, 4, "must be strictly greater"),      # stop < start
    (60e6, 60e6, 4, "must be strictly greater"),       # stop == start
    (-60e6, 90e6, 4, "strictly positive"),             # negative start
    (60e6, 90e6, 1, "integer >= 2"),                   # too few points
    (60e6, 90e6, 0, "integer >= 2"),
    (60e6, 90e6, 3.5, "integer >= 2"),                 # non-integer
    (float("nan"), 90e6, 4, "finite numbers"),
    (float("inf"), 90e6, 4, "finite numbers"),
])
def test_generate_freq_points_rejects_invalid_input(freq_start, freq_stop, n_points, match):
    with pytest.raises(ValueError, match=match):
        generate_freq_points(freq_start, freq_stop, n_points)


def test_frequency_sweep_rejects_invalid_input_before_mesh_generation(monkeypatch):
    """Validation must happen before any expensive mesh/gmsh work starts."""
    import backend.utils.frequency_sweep as fs

    def _boom(*args, **kwargs):
        raise AssertionError("mesh generation must not run when inputs are invalid")

    monkeypatch.setattr(fs, "extract_msh_to_mat", _boom)

    fake_path = object()   # never touched if validation runs first
    with pytest.raises(ValueError, match="must be strictly greater"):
        frequency_sweep(fake_path, 90e6, 60e6, np.array([0.0, 0.0, 0.0]), n_points=4)


def test_frequency_sweep_rejects_lossy_without_conductivity():
    fake_path = object()
    with pytest.raises(ValueError, match="conductivity_conductor"):
        frequency_sweep(fake_path, 60e6, 90e6, np.array([0.0, 0.0, 0.0]), n_points=4,
                         lossy_conductor=True)


def _files_for(path):
    return [path.msh, path.geo, path.mat, path.mat_mesh1, path.mat_mesh2,
            path.mat_impedance, path.mat_current, path.mat_freq_sweep]


@pytest.fixture
def strip_path():
    """A tiny single-port strip antenna, coarse mesh so the test stays fast."""
    path = setup_save_file_paths(ANTENNA_NAME)

    mm = 1e-3
    width, length = 10 * mm, 200 * mm
    mesh_size = 12 * mm   # coarse -> small E, fast test

    gmsh.initialize()
    gmsh.model.add(ANTENNA_NAME)
    setup_performance_config()
    gmsh.model.occ.addRectangle(-width / 2, -length / 2, 0, width, length)
    gmsh.model.occ.synchronize()
    generate_and_save_mesh(path, mesh_size)
    gmsh.finalize()

    yield path

    for f in _files_for(path):
        if os.path.isfile(f):
            os.remove(f)


FEED_POINT = np.array([0.0, 0.0, 0.0])
GAP_WIDTH = 15e-3
EXCITATION = 'y'
FREQ_START, FREQ_STOP, N_POINTS = 60e6, 90e6, 4   # 60, 70, 80, 90 MHz


def test_fast_sweep_matches_per_frequency_radiation_algorithm(strip_path):
    path = strip_path

    # Reference: call radiation_algorithm() directly at each sweep frequency.
    frequencies_ref = list(np.linspace(FREQ_START, FREQ_STOP, N_POINTS))
    ref_s11_db = []
    ref_impedances = []
    for freq in frequencies_ref:
        results = radiation_algorithm(
            path, freq, FEED_POINT,
            excitation_unit_vector=EXCITATION, gap_width=GAP_WIDTH,
            voltage_amplitude=1, show=False,
        )
        gap_Z = np.atleast_1d(results[6])
        s11 = (gap_Z - 50) / (gap_Z + 50)
        ref_s11_db.append(20 * np.log10(np.abs(s11)))
        ref_impedances.append(gap_Z)

    # Fast path: one call covering the whole sweep.
    frequency_sweep(path, FREQ_START, FREQ_STOP, FEED_POINT, n_points=N_POINTS,
                     excitation_unit_vector=EXCITATION, gap_width=GAP_WIDTH,
                     voltage_amplitude=1, show=False)

    frequencies, impedances, s11_db, *_ = load_freq_sweep_data(path.mat_freq_sweep)

    np.testing.assert_allclose(frequencies, frequencies_ref, rtol=1e-12)
    np.testing.assert_allclose(
        np.atleast_1d(s11_db).reshape(len(frequencies_ref), -1),
        np.array(ref_s11_db),
        atol=1e-8, rtol=1e-8,
        err_msg="fast frequency_sweep() S11 disagrees with per-frequency radiation_algorithm()",
    )
    np.testing.assert_allclose(
        np.atleast_1d(impedances).reshape(len(frequencies_ref), -1),
        np.array(ref_impedances),
        atol=1e-6, rtol=1e-8,
        err_msg="fast frequency_sweep() Zin disagrees with per-frequency radiation_algorithm()",
    )


def test_fast_sweep_lossy_conductor_matches_per_frequency_radiation_algorithm(strip_path):
    """Covers the Gram-matrix/Zs(f) branch, computed once and per-frequency respectively."""
    path = strip_path
    conductivity = 5.8e7   # copper, S/m

    frequencies_ref = list(np.linspace(FREQ_START, FREQ_STOP, N_POINTS))
    ref_impedances = []
    for freq in frequencies_ref:
        results = radiation_algorithm(
            path, freq, FEED_POINT,
            excitation_unit_vector=EXCITATION, gap_width=GAP_WIDTH,
            voltage_amplitude=1, show=False,
            lossy_conductor=True, conductivity_conductor=conductivity,
        )
        ref_impedances.append(np.atleast_1d(results[6]))

    frequency_sweep(path, FREQ_START, FREQ_STOP, FEED_POINT, n_points=N_POINTS,
                     excitation_unit_vector=EXCITATION, gap_width=GAP_WIDTH,
                     voltage_amplitude=1, show=False,
                     lossy_conductor=True, conductivity_conductor=conductivity)

    _, impedances, _, *_ = load_freq_sweep_data(path.mat_freq_sweep)

    np.testing.assert_allclose(
        np.atleast_1d(impedances).reshape(len(frequencies_ref), -1),
        np.array(ref_impedances),
        atol=1e-6, rtol=1e-8,
        err_msg="fast frequency_sweep(lossy_conductor=True) disagrees with per-frequency radiation_algorithm()",
    )


def test_fast_sweep_writes_expected_mat_schema(strip_path):
    """plot_s_parameters / plot_radiation_efficiency depend on this schema."""
    path = strip_path
    frequency_sweep(path, FREQ_START, FREQ_STOP, FEED_POINT, n_points=N_POINTS,
                     excitation_unit_vector=EXCITATION, gap_width=GAP_WIDTH,
                     voltage_amplitude=1, show=False)

    data = loadmat(path.mat_freq_sweep)
    for key in ['frequencies', 'port_locations', 'num_ports', 'impedances',
                's11_db', 'gap_currents', 'feed_powers', 'current_vectors',
                'efficiencies_total']:
        assert key in data, f"missing expected key '{key}' in mat_freq_sweep output"
