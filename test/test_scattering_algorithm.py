"""
Regression tests for the high-level scattering pipeline
(solver/src/scattering_algorithm/scattering_algorithm.py).

Context
-------
scattering_algorithm() had fallen behind the DataManager_rwg1..3 API, which
moved to the `path` namespace of setup_save_file_paths(): it still called
DataManager_rwg1.save_data(mesh, folder, points, triangles, edges) and raised a
TypeError before computing anything. These tests pin the two supported calling
conventions (a `path` namespace, as radiation_algorithm() takes, and the legacy
.mat filename used by the scattering notebooks) and check that the returned
current actually solves Z I = V.

The mesh is built in NumPy (no gmsh) and every file is written under tmp_path.
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import savemat

from solver.rwg.rwg4 import DataManager_rwg4
from solver.src.scattering_algorithm.scattering_algorithm import scattering_algorithm

FREQUENCY = 300e6                     # lambda = 1 m
DIRECTION = [0.0, 0.0, -1.0]          # normal incidence from +z
POLARIZATION = [1.0, 0.0, 0.0]        # E along x


def _plate_mesh(side=0.5, n=4):
    """Square plate in z = 0, n x n cells split in two triangles each.

    Returns p (3, N) and t (4, M) with 1-based vertex indices and domain
    label 1 in the 4th row, i.e. the AEMM/MATLAB .mat convention.
    """
    xs = np.linspace(-side / 2, side / 2, n + 1)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    p = np.vstack([X.ravel(), Y.ravel(), np.zeros(X.size)])

    def idx(i, j):
        return i * (n + 1) + j

    tris = []
    for i in range(n):
        for j in range(n):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            tris += [(a, b, c), (a, c, d)]
    t = np.array(tris).T + 1                     # 1-based
    t = np.vstack([t, np.ones((1, t.shape[1]), dtype=int)])
    return p, t


def _path_namespace(folder, name="plate"):
    """The subset of setup_save_file_paths() that the scattering pipeline uses."""
    folder = str(folder)
    return SimpleNamespace(
        name=name,
        msh=os.path.join(folder, f"{name}.msh"),             # absent -> .mat used as is
        mat=os.path.join(folder, f"{name}.mat"),
        mat_mesh1=os.path.join(folder, f"{name}_mesh1.mat"),
        mat_mesh2=os.path.join(folder, f"{name}_mesh2.mat"),
        mat_impedance=os.path.join(folder, f"{name}_impedance.mat"),
        mat_current=os.path.join(folder, f"{name}_current.mat"),
    )


@pytest.fixture
def plate_path(tmp_path):
    path = _path_namespace(tmp_path)
    p, t = _plate_mesh()
    savemat(path.mat, {"p": p, "t": t})
    return path


def test_path_namespace_runs_and_solves_the_mom_system(plate_path):
    Z, V, I, J = scattering_algorithm(plate_path, FREQUENCY, DIRECTION, POLARIZATION,
                                      show=False, use_gpu=False)

    assert Z.shape == (I.size, I.size)
    assert np.all(np.isfinite(I)) and np.any(I != 0)
    # The returned current is the solution of Z I = V, not just some vector.
    np.testing.assert_allclose(Z @ I, V, rtol=1e-9, atol=1e-12 * np.max(np.abs(V)))
    assert J.ndim == 1 and np.all(np.isfinite(J))

    for f in (plate_path.mat_mesh1, plate_path.mat_mesh2,
              plate_path.mat_impedance, plate_path.mat_current):
        assert os.path.isfile(f), f

    *_, wave_dir, pol, V_saved, I_saved = DataManager_rwg4.load_data(
        plate_path.mat_current, scattering=True)
    np.testing.assert_allclose(I_saved, I)
    np.testing.assert_allclose(V_saved, V)
    np.testing.assert_allclose(wave_dir, DIRECTION)
    np.testing.assert_allclose(pol, POLARIZATION)


def test_legacy_mat_filename_matches_path_namespace(tmp_path, monkeypatch, plate_path):
    """The legacy string call writes to data/... under the CWD and gives the same current."""
    _, _, I_ref, _ = scattering_algorithm(plate_path, FREQUENCY, DIRECTION, POLARIZATION,
                                          show=False, use_gpu=False)

    work = tmp_path / "legacy"
    work.mkdir()
    monkeypatch.chdir(work)
    p, t = _plate_mesh()
    os.makedirs("meshes")
    savemat(os.path.join("meshes", "plate.mat"), {"p": p, "t": t})

    _, _, I_legacy, _ = scattering_algorithm(os.path.join("meshes", "plate.mat"), FREQUENCY,
                                             DIRECTION, POLARIZATION, show=False,
                                             load_from_matlab=True, use_gpu=False)

    np.testing.assert_allclose(I_legacy, I_ref, rtol=1e-12)
    assert os.path.isfile(os.path.join("data", "antennas_current", "plate_current.mat"))


def test_accepts_the_h_refinement_call_signature(plate_path):
    """run_refinement_cycle passes radiation-only keywords to every solver function."""
    out = scattering_algorithm(plate_path, show=False, excitation_unit_vector=None, gap_width=0.001,
                               frequency=FREQUENCY, wave_incident_direction=DIRECTION,
                               polarization=POLARIZATION, use_gpu=False)
    assert len(out) == 4


def test_rejects_an_unsupported_mesh_argument():
    with pytest.raises(TypeError, match="setup_save_file_paths"):
        scattering_algorithm(42, FREQUENCY, DIRECTION, POLARIZATION, show=False)
