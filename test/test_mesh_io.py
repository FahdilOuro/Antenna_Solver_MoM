"""
Tests for solver/utils/mesh_io.py and the error-handling / consistency-check
behaviour it brought to DataManager_rwg1..4 (solver/rwg/rwg1..4.py).

Background
----------
Every DataManager_rwgN.load_data used to catch
FileNotFoundError/KeyError/ValueError/Exception, print it, and implicitly
`return None`. A caller doing `a, b, c = load_data(...)` then failed with a
confusing `TypeError: cannot unpack non-iterable NoneType object` instead of
the original, more informative error. These tests confirm load_data now
raises `MeshDataError` (chained from the real cause) instead.

They also cover the mesh-fingerprint consistency check: nothing previously
prevented combining a `_mesh2.mat` regenerated from a new mesh with a stale
`_impedance.mat` computed on the old one; `verify_mesh_fingerprint` now
catches that at the point rwg4 combines the two files.
"""

import os
import tempfile

import numpy as np
import pytest
from scipy.io import savemat

from solver.rwg.rwg1 import Points, Triangles, DataManager_rwg1
from solver.rwg.rwg2 import Vecteurs_Rho, DataManager_rwg2
from solver.rwg.rwg3 import DataManager_rwg3
from solver.rwg.rwg4 import DataManager_rwg4
from solver.utils.mesh_io import (
    MeshDataError, MeshConsistencyError,
    compute_mesh_fingerprint, read_mesh_fingerprint, verify_mesh_fingerprint,
)


class _FakePath:
    """Minimal stand-in for utils.file_path.FilePath."""
    def __init__(self, directory, name="test"):
        self.name = name
        self.mat_mesh1 = os.path.join(directory, f"{name}_mesh1.mat")
        self.mat_mesh2 = os.path.join(directory, f"{name}_mesh2.mat")
        self.mat_impedance = os.path.join(directory, f"{name}_impedance.mat")


def _four_triangle_mesh():
    points = np.array([
        [0., 1., 2., 0., 1., 2.],
        [0., 0., 0., 1., 1., 1.],
        [0., 0., 0., 0., 0., 0.],
    ])
    triangles = np.array([
        [0, 1, 1, 2],
        [1, 4, 2, 5],
        [3, 3, 4, 4],
        [0, 0, 0, 0],
    ], dtype=int)
    return points, triangles


def _build_rwg_objects(raw_points, raw_triangles):
    points_obj = Points(raw_points)
    tri_obj = Triangles(raw_triangles)
    tri_obj.calculate_triangles_area_and_center(points_obj)
    edges_obj = tri_obj.get_edges()
    edges_obj.compute_edges_length(points_obj)
    return points_obj, tri_obj, edges_obj


def _run_pipeline_stage1_2_3(tmpdir, raw_points, raw_triangles, frequency=3e8):
    """Save mesh1 -> mesh2 -> impedance for one mesh, mirroring
    radiation_algorithm's save sequence. Returns the FakePath used."""
    path = _FakePath(tmpdir)
    points_obj, tri_obj, edges_obj = _build_rwg_objects(raw_points, raw_triangles)

    DataManager_rwg1.save_data(path, points_obj, tri_obj, edges_obj)

    vecteurs_rho = Vecteurs_Rho()
    vecteurs_rho.calculate_vecteurs_rho(points_obj, tri_obj, edges_obj)
    DataManager_rwg2.save_data(path, vecteurs_rho)

    matrice_z = np.eye(edges_obj.total_number_of_edges, dtype=complex)
    DataManager_rwg3.save_data(path, frequency, 2 * np.pi * frequency,
                                1.257e-6, 8.854e-12, 3e8, 377.0, matrice_z)
    return path


# ---------------------------------------------------------------------------
# 1. Missing file -> MeshDataError, not a silent None
# ---------------------------------------------------------------------------

def test_rwg1_load_missing_file_raises():
    with pytest.raises(MeshDataError, match="does not exist"):
        DataManager_rwg1.load_data("no/such/file_mesh1.mat")


def test_rwg2_load_missing_file_raises():
    with pytest.raises(MeshDataError, match="does not exist"):
        DataManager_rwg2.load_data("no/such/file_mesh2.mat")


def test_rwg3_load_missing_file_raises():
    with pytest.raises(MeshDataError, match="does not exist"):
        DataManager_rwg3.load_data("no/such/file_impedance.mat")


def test_rwg4_load_missing_file_raises():
    with pytest.raises(MeshDataError, match="does not exist"):
        DataManager_rwg4.load_data("no/such/file_current.mat", radiation=True)


def test_rwg4_load_without_mode_flag_raises_valueerror():
    """Misuse (neither radiation nor scattering set) must surface clearly,
    not be swallowed by the generic except-and-print handler."""
    with pytest.raises(ValueError, match="radiation=True.*scattering=True"):
        DataManager_rwg4.load_data("irrelevant.mat")


# ---------------------------------------------------------------------------
# 2. Malformed file (missing expected key) -> MeshDataError, chained
# ---------------------------------------------------------------------------

def test_rwg2_load_malformed_file_raises_chained():
    with tempfile.TemporaryDirectory() as tmpdir:
        bogus = os.path.join(tmpdir, "bogus_mesh2.mat")
        savemat(bogus, {"unrelated_key": np.array([1.0])})
        with pytest.raises(MeshDataError, match="missing expected key"):
            DataManager_rwg2.load_data(bogus)


# ---------------------------------------------------------------------------
# 3. Round trip: save then load must succeed and drop no data
# ---------------------------------------------------------------------------

def test_full_pipeline_round_trip_loads_cleanly():
    raw_points, raw_triangles = _four_triangle_mesh()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _run_pipeline_stage1_2_3(tmpdir, raw_points, raw_triangles)

        points, triangles, edges = DataManager_rwg1.load_data(path.mat_mesh1)
        assert edges.total_number_of_edges == 3

        points2, triangles2, edges2, vecteurs_rho = DataManager_rwg2.load_data(path.mat_mesh2)
        assert vecteurs_rho.vecteur_rho_plus.shape == (3, 3)

        frequency, omega, mu, epsilon, c, eta, Z = DataManager_rwg3.load_data(path.mat_impedance)
        assert Z.shape == (3, 3)


# ---------------------------------------------------------------------------
# 4. Mesh-fingerprint consistency check
# ---------------------------------------------------------------------------

def test_fingerprint_deterministic_and_sensitive_to_content():
    raw_points, raw_triangles = _four_triangle_mesh()
    fp1 = compute_mesh_fingerprint(raw_points, raw_triangles)
    fp2 = compute_mesh_fingerprint(raw_points, raw_triangles)
    assert fp1 == fp2

    raw_points_shifted = raw_points.copy()
    raw_points_shifted[0, 0] += 1e-9
    fp3 = compute_mesh_fingerprint(raw_points_shifted, raw_triangles)
    assert fp3 != fp1


def test_verify_mesh_fingerprint_passes_for_consistent_pipeline():
    raw_points, raw_triangles = _four_triangle_mesh()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _run_pipeline_stage1_2_3(tmpdir, raw_points, raw_triangles)

        fp_mesh2 = read_mesh_fingerprint(path.mat_mesh2)
        fp_impedance = read_mesh_fingerprint(path.mat_impedance)
        assert fp_mesh2 is not None
        assert fp_mesh2 == fp_impedance

        # Must not raise: same mesh underlies both files.
        verify_mesh_fingerprint(path.mat_mesh2, path.mat_impedance)


def test_verify_mesh_fingerprint_raises_on_stale_impedance():
    """Regenerate mesh2 from a DIFFERENT mesh without recomputing impedance
    -- verify_mesh_fingerprint must catch the mismatch."""
    raw_points, raw_triangles = _four_triangle_mesh()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _run_pipeline_stage1_2_3(tmpdir, raw_points, raw_triangles)

        # Simulate re-meshing: perturb geometry and re-save mesh1 -> mesh2 only.
        perturbed_points = raw_points.copy()
        perturbed_points[1, :] *= 2.0  # stretch the mesh
        points_obj, tri_obj, edges_obj = _build_rwg_objects(perturbed_points, raw_triangles)
        DataManager_rwg1.save_data(path, points_obj, tri_obj, edges_obj)
        vecteurs_rho = Vecteurs_Rho()
        vecteurs_rho.calculate_vecteurs_rho(points_obj, tri_obj, edges_obj)
        DataManager_rwg2.save_data(path, vecteurs_rho)
        # path.mat_impedance is now stale (still describes the original mesh).

        with pytest.raises(MeshConsistencyError, match="different meshes"):
            verify_mesh_fingerprint(path.mat_mesh2, path.mat_impedance)


def test_verify_mesh_fingerprint_skips_when_fingerprint_missing():
    """Older files predating this field must not break the check."""
    with tempfile.TemporaryDirectory() as tmpdir:
        f_a = os.path.join(tmpdir, "a.mat")
        f_b = os.path.join(tmpdir, "b.mat")
        savemat(f_a, {"some_field": np.array([1.0])})
        savemat(f_b, {"some_field": np.array([2.0])})
        # Neither file has a mesh_fingerprint key -> must not raise.
        verify_mesh_fingerprint(f_a, f_b)
