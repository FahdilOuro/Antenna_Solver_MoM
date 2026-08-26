"""
Tests for the CMA Markdown data sheets (backend/cma/cma_report.py).

These check that the report records the run faithfully -- every mode, every
swept frequency, the mesh it came from and the solver settings -- because the
whole point of the file is to still be readable once the session that produced
it is gone.

They also pin the boundary the reports deliberately do not cross: the file
states measured quantities and defines them, and does not decide for the reader
where the antenna resonates. Automating that reading is a trap (a sign change
in lambda is a resonance through zero but a pole through infinity, and the two
mean opposite things about excitability), so a regression that starts emitting
verdicts should fail here.
"""

import os
import types

import numpy as np
import pytest

from backend.cma.cma_parameters import CMAParameters
from backend.cma.cma_report import write_single_frequency_report, write_sweep_report
from backend.cma.cma_frequency_sweep import CMASweepData
from backend.cma.mode_tracking import track_sweep
from backend.cma.modal_data import ModalData
from backend.rwg.rwg1 import Points, Triangles


@pytest.fixture
def mesh():
    """A 1x3 strip: small, planar, and enough edges to exercise the tables."""
    points = np.array([
        [0., 1., 2., 3., 0., 1., 2., 3.],
        [0., 0., 0., 0., 1., 1., 1., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
    ]) * 0.01
    triangles = np.array([
        [0, 1, 1, 2, 2, 3],
        [1, 5, 2, 6, 3, 7],
        [4, 4, 5, 5, 6, 6],
        [0, 0, 0, 0, 0, 0],
    ], dtype=int)
    p = Points(points)
    t = Triangles(triangles)
    t.calculate_triangles_area_and_center(p)
    e = t.get_edges()
    e.compute_edges_length(p)
    return p, t, e


@pytest.fixture
def path_in(tmp_path, monkeypatch):
    """
    The writers resolve 'data/antennas_cma' relative to the working directory,
    matching every other DataManager in the project, so the test moves there
    rather than reaching into the repo's own data tree.
    """
    monkeypatch.chdir(tmp_path)
    return types.SimpleNamespace(name="unit_test_antenna")


def _read(p):
    with open(p, encoding='utf-8') as fh:
        return fh.read()


def test_single_frequency_report_records_every_mode(path_in, mesh):
    points, triangles, edges = mesh
    lam = np.array([-0.5, 2.0, -30.0])
    modal = ModalData(eigenvalues=lam,
                      eigenvectors=np.zeros((edges.total_number_of_edges, 3)),
                      frequency=1.5e9, n_modes=3, sort_order=np.arange(3))

    out = write_single_frequency_report(path_in, modal, points, triangles, edges)
    assert os.path.isfile(out)
    text = _read(out)

    # Every eigenvalue must appear, with its derived MS and characteristic angle.
    for value in lam:
        assert f"{value:+.6g}" in text
    assert "0.8944" in text                      # MS for lambda = -0.5
    assert "| 3 |" in text                       # the third mode has a row

    assert "unit_test_antenna" in text
    assert "1.5000 GHz" in text
    assert str(edges.total_number_of_edges) in text


def test_sweep_report_records_every_frequency_and_all_three_quantities(path_in, mesh):
    points, triangles, edges = mesh
    freqs = np.linspace(1e9, 2e9, 5)
    lam = np.outer(np.linspace(-2.0, 2.0, 5), np.array([1.0, -3.0]))
    sweep = CMASweepData(frequencies=freqs, eigenvalues=lam,
                         eigenvectors=np.zeros((5, edges.total_number_of_edges, 2)),
                         n_modes=2)

    text = _read(write_sweep_report(path_in, sweep, points, triangles, edges))

    # One row per swept frequency, in each of the three tables.
    for f in freqs:
        label = f"{f / 1e9:.4f} GHz"
        assert text.count(label) >= 3, f"{label} missing from one of the tables"

    assert "Characteristic values lambda" in text
    assert "Modal significance MS" in text
    assert "Characteristic angle delta" in text

    # Electrical size is stated at both ends, since it differs across the band.
    assert "band start" in text and "band stop" in text


def test_solver_settings_are_recorded_when_supplied(path_in, mesh):
    points, triangles, edges = mesh
    modal = ModalData(eigenvalues=np.array([1.0]),
                      eigenvectors=np.zeros((edges.total_number_of_edges, 1)),
                      frequency=1e9, n_modes=1, sort_order=np.array([0]))
    params = CMAParameters(n_modes=1, regularization_epsilon=1e-9, sort_by='value')

    text = _read(write_single_frequency_report(path_in, modal, points, triangles, edges,
                                               params=params))
    assert "1.0e-09" in text, "the regularization actually used must be on record"
    assert "`value`" in text, "the mode ordering must be on record"


def test_health_section_appears_only_with_an_impedance_matrix(path_in, mesh):
    points, triangles, edges = mesh
    n = edges.total_number_of_edges
    modal = ModalData(eigenvalues=np.array([1.0]), eigenvectors=np.zeros((n, 1)),
                      frequency=1e9, n_modes=1, sort_order=np.array([0]))

    without = _read(write_single_frequency_report(path_in, modal, points, triangles, edges))
    assert "Impedance matrix" not in without

    rng = np.random.default_rng(0)
    A = rng.random((n, n))
    Z = (A @ A.T + np.eye(n)) + 1j * np.eye(n)          # symmetric, PSD real part
    with_z = _read(write_single_frequency_report(path_in, modal, points, triangles, edges, Z=Z))
    assert "Impedance matrix" in with_z
    assert "Reciprocity" in with_z
    assert "Numerical rank of R" in with_z


def test_report_states_quantities_without_ruling_on_them(path_in, mesh):
    """
    The reports define lambda / MS / delta and record their values; they must
    not announce resonances, bandwidths or a modal Q. Deriving those reliably
    needs judgement the writer does not have -- most sharply, telling a zero
    crossing from a pole -- so emitting them would put confident but
    potentially wrong conclusions in a file meant to be trusted later.
    """
    points, triangles, edges = mesh
    freqs = np.linspace(1e9, 2e9, 5)
    lam = np.linspace(-2.0, 2.0, 5).reshape(-1, 1)      # crosses zero mid-band
    sweep = CMASweepData(frequencies=freqs, eigenvalues=lam,
                         eigenvectors=np.zeros((5, edges.total_number_of_edges, 1)),
                         n_modes=1)

    text = _read(write_sweep_report(path_in, sweep, points, triangles, edges)).lower()

    assert "modal q" not in text
    assert "fractional bandwidth" not in text
    assert "resonances found" not in text
    # The definitions still have to be there -- that is what makes the data usable.
    assert "modal significance" in text
    assert "characteristic angle" in text


def test_sweep_report_records_what_tracking_decided_and_what_it_could_not(path_in, mesh):
    """
    The tracking section exists so a reader can tell a followed crossing from an
    undecidable one. Both must reach the file.
    """
    points, triangles, edges = mesh
    n_edges = edges.total_number_of_edges
    freqs = np.linspace(1e9, 2e9, 4)

    rng = np.random.default_rng(0)
    R = np.eye(n_edges)
    J = np.linalg.qr(rng.standard_normal((n_edges, n_edges)))[0][:, :2]
    # Two modes that stay 0.001 apart: quasi-degenerate at every step.
    lam = np.tile(np.array([0.500, 0.501]), (4, 1))

    tracking = track_sweep(freqs, lam, np.stack([J] * 4), R_list=[R] * 4)
    sweep = CMASweepData(frequencies=freqs, eigenvalues=lam,
                         eigenvectors=np.stack([J] * 4), n_modes=2,
                         tracking=tracking)

    text = _read(write_sweep_report(path_in, sweep.tracked(), points, triangles, edges))

    assert "## Mode tracking" in text
    assert "Hungarian" in text
    assert "degeneracy" in text
    assert "Accepted correlation per step" in text
    # The tracked report must not still carry the untracked caveat.
    assert "have been tracked across the sweep" in text
    assert "no attempt" not in text


def test_sweep_report_keeps_the_untracked_caveat_when_nothing_was_tracked(path_in, mesh):
    points, triangles, edges = mesh
    freqs = np.linspace(1e9, 2e9, 3)
    sweep = CMASweepData(frequencies=freqs, eigenvalues=np.zeros((3, 2)),
                         eigenvectors=np.zeros((3, edges.total_number_of_edges, 2)),
                         n_modes=2)

    text = _read(write_sweep_report(path_in, sweep, points, triangles, edges))

    assert "## Mode tracking" not in text
    assert "no attempt" in text
