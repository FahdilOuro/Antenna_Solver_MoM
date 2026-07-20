"""
Shared error handling and mesh-identity utilities for the staged rwg1->rwg4
file-based pipeline (DataManager_rwg1..4 in backend/rwg/).

Each pipeline stage caches its output to a .mat file and the next stage
loads it back. Two failure modes used to be silent:

  1. A missing or malformed file: every DataManager_rwgN.load_data caught
     FileNotFoundError/KeyError/ValueError/Exception, printed it, and fell
     through to an implicit `return None`. A caller doing
     `a, b, c = load_data(...)` then failed with a confusing
     `TypeError: cannot unpack non-iterable NoneType object` instead of the
     original, far more informative error.
  2. Mixing artifacts from different mesh generations: nothing prevented
     loading a `_mesh2.mat` regenerated from a new mesh together with a
     stale `_impedance.mat` computed on the old one, silently combining
     mismatched matrices.

`load_matfile_or_raise` fixes (1): it replaces the catch-print-None pattern
with a raised, chained `MeshDataError`. `compute_mesh_fingerprint` /
`read_mesh_fingerprint` / `verify_mesh_fingerprint` address (2): each stage
embeds a lightweight identity hash of the underlying mesh, and the functions
that combine two stages (e.g. mesh2 + impedance) cross-check it.
"""

import os
import hashlib

import numpy as np
from scipy.io import loadmat


class MeshDataError(Exception):
    """Raised when a pipeline-stage .mat file is missing or malformed."""


class MeshConsistencyError(MeshDataError):
    """Raised when two pipeline-stage files were generated from different meshes."""


def load_matfile_or_raise(filename: str) -> dict:
    """
    Load a .mat file, raising MeshDataError with full context on any failure.
    """
    if not os.path.isfile(filename):
        raise MeshDataError(f"File '{filename}' does not exist.")
    try:
        return loadmat(filename)
    except Exception as e:
        raise MeshDataError(f"Failed to load '{filename}': {e}") from e


def compute_mesh_fingerprint(points: np.ndarray, triangles: np.ndarray) -> str:
    """Short, deterministic identity hash of a mesh's raw geometry arrays."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(points).tobytes())
    h.update(np.ascontiguousarray(triangles).tobytes())
    return h.hexdigest()[:16]


def read_mesh_fingerprint(filename: str):
    """
    Best-effort read of the 'mesh_fingerprint' key from a .mat file.

    Returns None (rather than raising) if the file does not exist, fails to
    load, or predates this field, so that older data files degrade
    gracefully instead of hard-failing a consistency check that is an
    additional safety net, not a hard schema requirement.
    """
    if not os.path.isfile(filename):
        return None
    try:
        data = loadmat(filename)
    except Exception:
        return None
    fp = data.get('mesh_fingerprint')
    if fp is None:
        return None
    return str(np.asarray(fp).squeeze())


def verify_mesh_fingerprint(filename_a: str, filename_b: str) -> None:
    """
    Raise MeshConsistencyError if the mesh fingerprints stored in two
    pipeline-stage files are both present and differ.

    Silently passes if either fingerprint is missing (older files predating
    this field) -- this is an additional safety check on top of the
    existing pipeline, not a hard requirement that would break older data.
    """
    fp_a = read_mesh_fingerprint(filename_a)
    fp_b = read_mesh_fingerprint(filename_b)
    if fp_a is None or fp_b is None:
        return
    if fp_a != fp_b:
        raise MeshConsistencyError(
            f"Mesh mismatch: '{filename_a}' and '{filename_b}' were generated "
            f"from different meshes (fingerprints {fp_a} != {fp_b}). "
            f"Re-run the pipeline stages in order so they share the same mesh."
        )
