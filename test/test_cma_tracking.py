"""
Tests for cross-frequency mode tracking (solver/cma/mode_tracking.py).

The thing being tested is an identity claim -- "column k is the same physical
mode at every frequency" -- so the fixtures are built the only way that claim
can be checked: from currents whose identity is known by construction, run
through the same relabelling the eigensolver performs (sort by |lambda| at each
frequency, sign convention applied independently), and then asked to be
recovered.

The failure these guard against is silent. Nothing crashes when tracking gets an
assignment wrong; a lambda(f) curve simply jumps to a different mode, and the
plot still looks like a plausible antenna.
"""

import os
import types

import numpy as np
import pytest

from solver.cma.cma_frequency_sweep import CMASweepData, DataManager_cma_sweep

from solver.cma.mode_tracking import (
    EVENT_AVOIDED_CROSSING, EVENT_DEGENERACY, EVENT_LOW_CORRELATION, EVENT_SWAP,
    ModeTracker, ModeTrackingResult, TECACorrection, TrackingParameters,
    correlation_matrix, get_correction, track_sweep,
)

N_DOF = 24
N_MODES = 4


def _spd(seed, n=N_DOF, spread=(1.0, 5.0)):
    """A symmetric positive-definite stand-in for R = Re(Z)."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return Q @ np.diag(np.linspace(*spread, n)) @ Q.T


def _basis(seed, n=N_DOF, k=N_MODES):
    """k fixed, linearly independent current shapes."""
    rng = np.random.default_rng(seed)
    return np.linalg.qr(rng.standard_normal((n, n)))[0][:, :k]


def _relabel(lam, J):
    """
    Apply exactly what GeneralizedEigenSolver._select_and_sign does at one
    frequency: order by |lambda|, then force the largest component positive.
    This is the relabelling tracking has to undo.
    """
    order = np.argsort(np.abs(lam))
    lam_r, J_r = lam[order], J[:, order].copy()
    for c in range(J_r.shape[1]):
        if J_r[np.argmax(np.abs(J_r[:, c])), c] < 0:
            J_r[:, c] *= -1
    return lam_r, J_r, order


# ----------------------------------------------------------------------
# The correlation metric
# ----------------------------------------------------------------------

def test_correlation_is_one_on_the_diagonal_and_ignores_scale_and_sign():
    R = _spd(0)
    J = _basis(1)
    scaled = J * np.array([1.0, -2.0, 3.0, -0.5])

    C = correlation_matrix(J, scaled, R)

    assert np.allclose(np.diag(C), 1.0)
    assert C.max() <= 1.0 + 1e-12
    assert np.allclose(C, correlation_matrix(J, J, R))


def test_correlation_recovers_a_permutation():
    R = _spd(2)
    J = _basis(3)
    perm = np.array([2, 0, 3, 1])

    # Column n of the second set is mode perm[n] of the first, so it is the
    # argmax DOWN each column that reads back the permutation.
    C = correlation_matrix(J, J[:, perm] * np.array([1.0, -2.0, 3.0, -1.0]), R)

    assert np.array_equal(np.argmax(C, axis=0), perm)
    assert np.allclose(C[perm, np.arange(len(perm))], 1.0)


def test_R_weighting_and_euclidean_overlap_are_different_metrics():
    """
    The point of weighting by R. Two currents that agree in the radiating
    directions and differ in the ones R cannot see are the same mode physically,
    and the R-weighted correlation says so while the Euclidean one does not.
    """
    n = 12
    rng = np.random.default_rng(4)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    # A deliberately rank-deficient R: the last 8 directions radiate nothing,
    # the way most RWG basis functions do not at low ka.
    R = Q @ np.diag([1.0, 1.0, 1.0, 1.0] + [0.0] * 8) @ Q.T

    radiating = Q[:, 0]
    invisible = Q[:, 6]
    a = radiating[:, None]
    b = (radiating + 3.0 * invisible)[:, None]     # same radiating content

    assert correlation_matrix(a, b, R)[0, 0] == pytest.approx(1.0)
    assert correlation_matrix(a, b, None)[0, 0] < 0.4


def test_correlation_rejects_mismatched_bases():
    with pytest.raises(ValueError, match="same RWG basis"):
        correlation_matrix(np.ones((6, 2)), np.ones((7, 2)), None)


def test_a_non_radiating_column_correlates_with_nothing():
    """R's null space has no directions, so a mode lying in it gets C = 0."""
    R = np.diag([1.0, 1.0, 0.0])
    J = np.eye(3)[:, :3]

    C = correlation_matrix(J, J, R)

    assert np.allclose(C[2, :], 0.0)
    assert np.allclose(C[:, 2], 0.0)


# ----------------------------------------------------------------------
# Tracking through a crossing
# ----------------------------------------------------------------------

def _crossing_sweep(n_freq=11):
    """
    Two modes whose eigenvalues cross, plus two that do not.

    The currents are FIXED in frequency, so the correct tracking answer is known
    exactly: every track holds the same column of `J` throughout. Only the
    labels move, because lambda_0 falls through lambda_1 mid-band and the
    solver sorts by |lambda|.
    """
    R = _spd(5)
    J = _basis(6)
    freqs = np.linspace(1e9, 2e9, n_freq)
    t = np.linspace(-1.0, 1.0, n_freq)

    lam_true = np.column_stack([
        -3.0 * t,                       # falls through zero
        1.0 + 0.0 * t,                  # constant, crossed by the first
        8.0 + 2.0 * t,
        -20.0 + 3.0 * t,
    ])

    lam_raw = np.empty_like(lam_true)
    vec_raw = np.empty((n_freq, J.shape[0], J.shape[1]))
    order_true = np.empty((n_freq, J.shape[1]), dtype=int)
    for i in range(n_freq):
        lam_raw[i], vec_raw[i], order_true[i] = _relabel(lam_true[i], J)
    return freqs, lam_true, lam_raw, vec_raw, order_true, R


def test_hungarian_tracking_undoes_the_solver_relabelling():
    freqs, lam_true, lam_raw, vec_raw, order_true, R = _crossing_sweep()

    result = track_sweep(freqs, lam_raw, vec_raw, R_list=[R] * len(freqs))

    # A track is named by the raw slot it started in, so track k holds true mode
    # order_true[0][k]; at frequency i that mode sits in raw slot
    # argsort(order_true[i])[...]. That composition is what result.order must be.
    expected = np.stack([np.argsort(order_true[i])[order_true[0]]
                         for i in range(len(freqs))])

    assert np.array_equal(result.order, expected)
    # And the payoff: each tracked column is one mode's own lambda(f), smooth
    # through the crossing.
    assert np.allclose(result.reorder(lam_raw), lam_true[:, order_true[0]])


def test_tracking_reports_the_swap_it_followed_through():
    freqs, _lam_true, lam_raw, vec_raw, _order, R = _crossing_sweep()

    result = track_sweep(freqs, lam_raw, vec_raw, R_list=[R] * len(freqs))
    swaps = result.events_of(EVENT_SWAP)

    assert swaps, "the |lambda| labels of modes 1 and 2 invert mid-band"
    assert all(set(e.modes) == {0, 1} for e in swaps)
    # Followed through with confidence: the currents never actually changed.
    assert result.correlation.min() == pytest.approx(1.0)


def test_untracked_labels_really_do_break_without_tracking():
    """
    Guards the premise. If the raw |lambda|-sorted labels happened to be
    continuous already, every other test here would pass vacuously.
    """
    _freqs, lam_true, lam_raw, _vec, _order, _R = _crossing_sweep()

    assert not np.allclose(lam_raw, lam_true)


def test_sign_continuity_along_a_track():
    """
    The per-frequency sign convention flips when the largest component changes
    edge. Tracking removes that, so a modal current plotted across the sweep
    does not reverse under one label.
    """
    R = _spd(7)
    J = _basis(8)
    freqs = np.array([1e9, 1.1e9, 1.2e9])
    lam = np.tile(np.array([0.1, 2.0, 5.0, 9.0]), (3, 1))

    vec = np.stack([J, -J, J], axis=0)          # a pure sign flip at step 2
    result = track_sweep(freqs, lam, vec, R_list=[R] * 3)

    tracked = result.reorder_eigenvectors(vec)
    for i in (1, 2):
        assert np.allclose(tracked[i], tracked[0])


# ----------------------------------------------------------------------
# What tracking flags rather than decides
# ----------------------------------------------------------------------

def test_quasi_degenerate_pair_is_flagged():
    R = _spd(9)
    J = _basis(10)
    freqs = np.array([1e9, 1.1e9])
    # Modes 1 and 2 sit 0.001 apart in lambda: far inside the angle tolerance.
    lam = np.array([[0.500, 0.501, 6.0, 12.0],
                    [0.500, 0.501, 6.0, 12.0]])

    result = track_sweep(freqs, lam, np.stack([J, J]), R_list=[R, R])
    degenerate = result.events_of(EVENT_DEGENERACY)

    assert degenerate
    assert set(degenerate[0].modes) == {0, 1}
    assert "not decidable from correlation alone" in degenerate[0].detail


def test_two_far_from_resonance_modes_are_not_called_degenerate():
    """
    Pins the degeneracy measure against the tempting wrong one. lambda = 9 and
    lambda = 30 are a factor of three apart in stored energy and separate
    cleanly, but their characteristic angles differ by only 4.4 deg, so a
    tolerance on delta would flag them as a degenerate pair.
    """
    R = _spd(24)
    J = _basis(25)
    freqs = np.array([1e9, 1.1e9])
    lam = np.array([[0.2, 3.0, 9.0, 30.0], [0.2, 3.0, 9.0, 30.0]])

    delta = 180.0 - np.degrees(np.arctan([9.0, 30.0]))
    assert abs(delta[0] - delta[1]) < 5.0, "the wrong measure would flag this pair"

    result = track_sweep(freqs, lam, np.stack([J, J]), R_list=[R, R])

    assert result.events_of(EVENT_DEGENERACY) == []


def test_avoided_crossing_is_flagged_when_the_currents_exchange_character():
    """
    Quasi-degenerate, no label inversion, and the currents rotate into each
    other over the step -- the correlation drops and the margin collapses,
    which is what crossing avoidance looks like from here.
    """
    R = _spd(11)
    J = _basis(12)
    freqs = np.array([1e9, 1.1e9])
    lam = np.array([[0.500, 0.501, 6.0, 12.0],
                    [0.500, 0.501, 6.0, 12.0]])

    # Rotate the degenerate pair by 45 deg: neither mode matches its partner
    # better than the other one does.
    mixed = J.copy()
    c = np.sqrt(0.5)
    mixed[:, 0] = c * (J[:, 0] + J[:, 1])
    mixed[:, 1] = c * (J[:, 0] - J[:, 1])

    result = track_sweep(freqs, lam, np.stack([J, mixed]), R_list=[R, R])

    avoided = result.events_of(EVENT_AVOIDED_CROSSING)
    assert avoided
    assert set(avoided[0].modes) == {0, 1}
    # The margin can go negative: the Hungarian assignment maximises the TOTAL
    # correlation, so one track may be handed less than its own best match. That
    # is the signature being detected -- the pair is no longer separable.
    assert result.margin.min() < TrackingParameters().margin_threshold


def test_low_correlation_is_flagged_when_the_step_is_too_coarse():
    R = _spd(13)
    freqs = np.array([1e9, 2e9])
    lam = np.array([[0.1, 2.0, 6.0, 12.0], [0.1, 2.0, 6.0, 12.0]])
    # Unrelated current shapes at the two frequencies.
    vec = np.stack([_basis(14), _basis(15)])

    result = track_sweep(freqs, lam, vec, R_list=[R, R])

    assert result.events_of(EVENT_LOW_CORRELATION)
    assert result.correlation.min() < 0.7


def test_a_clean_sweep_flags_nothing():
    R = _spd(16)
    J = _basis(17)
    freqs = np.linspace(1e9, 1.2e9, 5)
    lam = np.tile(np.array([0.2, 3.0, 9.0, 30.0]), (5, 1))

    result = track_sweep(freqs, lam, np.stack([J] * 5), R_list=[R] * 5)

    assert result.events == []
    assert "no swaps" in result.summary()


# ----------------------------------------------------------------------
# Contracts: ordering, misuse, the correction hook
# ----------------------------------------------------------------------

def test_frequencies_must_increase():
    tracker = ModeTracker(TrackingParameters(metric='identity'))
    J = _basis(18)
    tracker.push(2e9, np.arange(N_MODES, dtype=float), J)
    with pytest.raises(ValueError, match="must increase"):
        tracker.push(1e9, np.arange(N_MODES, dtype=float), J)


def test_R_metric_requires_R():
    with pytest.raises(ValueError, match="needs Re\\(Z\\)"):
        track_sweep(np.array([1e9, 2e9]), np.zeros((2, 2)), np.zeros((2, 5, 2)))


def test_mode_count_must_not_change_across_the_sweep():
    tracker = ModeTracker(TrackingParameters(metric='identity'))
    tracker.push(1e9, np.arange(4, dtype=float), _basis(19, k=4))
    with pytest.raises(ValueError, match="same number of modes"):
        tracker.push(2e9, np.arange(3, dtype=float), _basis(19, k=3))


def test_teca_hook_is_registered_and_declares_itself_unimplemented():
    correction = get_correction('teca')

    assert isinstance(correction, TECACorrection)
    with pytest.raises(NotImplementedError, match="Chen, Pan & Su"):
        correction.refine(None)


def test_unknown_correction_names_the_registered_ones():
    with pytest.raises(ValueError, match="Registered:"):
        get_correction('nope')


def test_a_correction_can_override_the_assignment():
    """The hook is load-bearing, not decorative: what it returns is what is used."""

    class Reverse:
        def refine(self, step):
            step.assignment = step.assignment[::-1].copy()
            return step

    R = _spd(20)
    J = _basis(21)
    tracker = ModeTracker(TrackingParameters())
    tracker._correction = Reverse()
    lam = np.array([0.1, 2.0, 6.0, 12.0])
    tracker.push(1e9, lam, J, R=R)
    tracker.push(1.1e9, lam, J, R=R)

    assert np.array_equal(tracker.finalize().order[1], np.array([3, 2, 1, 0]))


def test_a_correction_returning_a_non_permutation_is_refused():
    class Broken:
        def refine(self, step):
            step.assignment = np.zeros_like(step.assignment)
            return step

    R = _spd(22)
    J = _basis(23)
    tracker = ModeTracker(TrackingParameters())
    tracker._correction = Broken()
    lam = np.array([0.1, 2.0, 6.0, 12.0])
    tracker.push(1e9, lam, J, R=R)
    with pytest.raises(ValueError, match="not a permutation"):
        tracker.push(1.1e9, lam, J, R=R)


# ----------------------------------------------------------------------
# Applying the permutation
# ----------------------------------------------------------------------

def test_applying_the_permutation_twice_changes_nothing():
    freqs, lam_true, lam_raw, vec_raw, _order, R = _crossing_sweep()
    result = track_sweep(freqs, lam_raw, vec_raw, R_list=[R] * len(freqs))

    once = result.reorder(lam_raw)
    applied = result.applied()

    assert applied.is_applied
    assert np.allclose(applied.reorder(once), once)
    assert applied.events == result.events


def test_reorder_refuses_an_array_of_the_wrong_shape():
    freqs, _lam_true, lam_raw, vec_raw, _order, R = _crossing_sweep()
    result = track_sweep(freqs, lam_raw, vec_raw, R_list=[R] * len(freqs))

    with pytest.raises(ValueError, match="n_freq, n_modes"):
        result.reorder(lam_raw[:, :2])


def test_events_survive_a_json_round_trip():
    freqs, _lam_true, lam_raw, vec_raw, _order, R = _crossing_sweep()
    result = track_sweep(freqs, lam_raw, vec_raw, R_list=[R] * len(freqs))

    restored = ModeTrackingResult.events_from_json(result.events_json())

    assert [e.as_dict() for e in restored] == [e.as_dict() for e in result.events]


# ----------------------------------------------------------------------
# Integration with CMASweepData, its .mat and the report
# ----------------------------------------------------------------------

def _tracked_sweep():
    freqs, lam_true, lam_raw, vec_raw, order_true, R = _crossing_sweep()
    tracking = track_sweep(freqs, lam_raw, vec_raw, R_list=[R] * len(freqs))
    sweep = CMASweepData(frequencies=freqs, eigenvalues=lam_raw,
                         eigenvectors=vec_raw, n_modes=N_MODES, tracking=tracking)
    return sweep, lam_true[:, order_true[0]]


def test_tracked_view_leaves_the_raw_sweep_untouched():
    sweep, lam_expected = _tracked_sweep()
    raw_before = sweep.eigenvalues.copy()

    tracked = sweep.tracked()

    assert np.allclose(tracked.eigenvalues, lam_expected)
    assert np.allclose(sweep.eigenvalues, raw_before), "tracked() must not mutate"
    assert tracked is not sweep


def test_tracked_is_idempotent():
    sweep, lam_expected = _tracked_sweep()

    twice = sweep.tracked().tracked()

    assert np.allclose(twice.eigenvalues, lam_expected)


def test_tracked_refuses_a_sweep_that_was_not_tracked():
    sweep, _ = _tracked_sweep()
    sweep.tracking = None

    with pytest.raises(RuntimeError, match="no tracking data"):
        sweep.tracked()


def test_tracking_survives_the_mat_round_trip(tmp_path):
    sweep, lam_expected = _tracked_sweep()
    path = types.SimpleNamespace(
        name="unit_test", mat_cma_sweep=str(tmp_path / "sweep.mat"))

    DataManager_cma_sweep.save_data(path, sweep)
    loaded = DataManager_cma_sweep.load_data(path.mat_cma_sweep)

    assert loaded.tracking is not None
    assert np.array_equal(loaded.tracking.order, sweep.tracking.order)
    assert np.allclose(loaded.tracking.sign, sweep.tracking.sign)
    assert np.allclose(loaded.tracking.correlation, sweep.tracking.correlation)
    assert loaded.tracking.metric == 'R'
    assert loaded.tracking.correction == 'none'
    assert ([e.as_dict() for e in loaded.tracking.events]
            == [e.as_dict() for e in sweep.tracking.events])
    # And the permutation still means the same thing after the trip.
    assert np.allclose(loaded.tracked().eigenvalues, lam_expected)


def test_a_sweep_mat_without_tracking_still_loads(tmp_path):
    """Backward compatibility: files written before tracking existed."""
    from scipy.io import savemat

    filename = str(tmp_path / "old.mat")
    savemat(filename, {'frequencies': np.linspace(1e9, 2e9, 3),
                       'eigenvalues': np.zeros((3, 2)),
                       'eigenvectors': np.zeros((3, 5, 2)),
                       'n_modes': 2})

    loaded = DataManager_cma_sweep.load_data(filename)

    assert loaded.tracking is None
    assert loaded.n_modes == 2


# ----------------------------------------------------------------------
# End to end, on a real mesh
# ----------------------------------------------------------------------

ANTENNA_NAME = "unittest_cma_tracking_strip"


@pytest.fixture
def strip_path():
    """A tiny strip, coarse mesh so a real Z(f) sweep stays fast."""
    import gmsh
    from solver.utils.file_path import setup_save_file_paths
    from solver.utils.gmsh_function import setup_performance_config, generate_and_save_mesh

    path = setup_save_file_paths(ANTENNA_NAME)
    mm = 1e-3
    gmsh.initialize()
    gmsh.model.add(ANTENNA_NAME)
    setup_performance_config()
    gmsh.model.occ.addRectangle(-5 * mm, -100 * mm, 0, 10 * mm, 200 * mm)
    gmsh.model.occ.synchronize()
    generate_and_save_mesh(path, 12 * mm)
    gmsh.finalize()

    yield path

    for f in (path.msh, path.geo, path.mat, path.mat_mesh1, path.mat_mesh2,
              path.mat_impedance, getattr(path, 'mat_cma_sweep', None)):
        if f and os.path.isfile(f):
            os.remove(f)


def test_cma_frequency_sweep_tracks_a_real_structure(strip_path):
    """
    The whole path: mesh -> Z(f) -> modes -> tracking -> .mat -> report.

    The assertions are about the tracking CONTRACT, not about this strip's
    physics: raw arrays preserved, a permutation attached that survives the
    round trip, and tracked eigenvalue curves that vary less step to step than
    the untracked ones wherever the labels move at all.
    """
    from solver.cma.cma_frequency_sweep import cma_frequency_sweep

    sweep = cma_frequency_sweep(strip_path, 300e6, 900e6, n_points=7, n_modes=4,
                                use_gpu=False, write_report=True)

    assert sweep.tracking is not None
    assert sweep.tracking.n_freq == 7 and sweep.tracking.n_modes == 4
    assert sweep.tracking.correlation.shape == (6, 4)
    # A real structure over a 3:1 band: adjacent modes must still be
    # recognisable, or the metric is not doing its job.
    assert sweep.tracking.correlation.min() > 0.5

    tracked = sweep.tracked()
    assert tracked.eigenvalues.shape == sweep.eigenvalues.shape
    # Same set of modes at each frequency, only relabelled.
    for i in range(7):
        assert np.allclose(np.sort(tracked.eigenvalues[i]),
                           np.sort(sweep.eigenvalues[i]))

    # Where tracking actually moved a label, it did so to make the curves
    # smoother; where it did not, the two views coincide.
    if not np.array_equal(sweep.tracking.order,
                          np.tile(np.arange(4), (7, 1))):
        jump_raw = np.abs(np.diff(sweep.eigenvalues, axis=0)).sum()
        jump_tracked = np.abs(np.diff(tracked.eigenvalues, axis=0)).sum()
        assert jump_tracked < jump_raw

    # The permutation reached the .mat, and the report has the section.
    reloaded = DataManager_cma_sweep.load_data(strip_path.mat_cma_sweep)
    assert reloaded.tracking is not None
    assert np.array_equal(reloaded.tracking.order, sweep.tracking.order)

    report = os.path.join('data/antennas_cma', f"{strip_path.name}_cma_sweep_report.md")
    with open(report, encoding='utf-8') as fh:
        assert "## Mode tracking" in fh.read()
    os.remove(report)
