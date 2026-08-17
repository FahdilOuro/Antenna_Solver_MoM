import os
import shutil
import tempfile
import warnings

from scipy.io import loadmat

from backend.rwg.rwg1 import (
    Points, Triangles, load_mesh_file, filter_complexes_jonctions, DataManager_rwg1,
)
from backend.rwg.rwg2 import Vecteurs_Rho, DataManager_rwg2
from backend.utils.gmsh_function import extract_msh_to_mat


def _try_regenerate_mat_from_msh(path) -> bool:
    """
    Convert path.msh -> path.mat, but only overwrite path.mat once the result
    is known to contain an actual surface mesh.

    A .msh file can legitimately exist while holding no elements at all: gmsh
    writes `$MeshFormat` / `$PhysicalNames` / `$Entities` for a geometry that
    was never meshed (or whose meshing step was not re-run). extract_msh_to_mat
    does not fail on those -- it happily writes an EMPTY p/t. Doing that
    straight onto path.mat destroys a perfectly good mesh, and since .mat files
    are gitignored there is no undo. So: convert into a temporary file first,
    inspect it, and only then replace path.mat.

    Returns
    -------
    True  if path.mat was regenerated from path.msh
    False if path.msh held no usable surface mesh (path.mat left untouched)
    """
    tmp_dir = tempfile.mkdtemp(prefix='cma_mesh_')
    tmp_mat = os.path.join(tmp_dir, 'candidate.mat')
    try:
        extract_msh_to_mat(path.msh, tmp_mat)
        data = loadmat(tmp_mat)
        p, t = data.get('p'), data.get('t')
        if p is None or t is None or p.size == 0 or t.size == 0:
            warnings.warn(
                f"'{path.msh}' holds no surface elements (geometry only, or an unmeshed "
                f"model), so it cannot define the mesh. Keeping '{path.mat}' and leaving "
                f"it untouched. Re-run the gmsh meshing step if the .msh was meant to be "
                f"the source of truth.",
                UserWarning,
                stacklevel=3,
            )
            return False
        shutil.move(tmp_mat, path.mat)
        return True
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def prepare_cma_mesh(path, verbose=True):
    """
    Build the RWG mesh objects a CMA run needs, preferring a freshly
    re-derived mesh but tolerating meshes that never came from gmsh.

    Two situations have to coexist:

    1. The geometry was built with gmsh, so `path.msh` exists and is the
       source of truth. Re-deriving `path.mat` (and mesh1/mesh2) from it on
       every run is what keeps an edited .msh from being silently ignored --
       CMA assembles its own Z, so nothing else downstream would notice a
       stale cached mesh.

    2. The mesh came from somewhere gmsh cannot produce, e.g. a structured
       pixel-grid exported to `.mat` from MATLAB. There is no `.msh` at all,
       and `path.mat` IS the source of truth. Calling extract_msh_to_mat()
       here would raise (missing file) -- or worse, overwrite the only copy
       of the mesh if a stale .msh happened to be lying around under the
       same name.

    So the .msh is used when present and skipped when not, instead of
    requiring the caller to comment the conversion in and out by hand.

    Parameters
    ----------
    path    : SimpleNamespace from setup_save_file_paths()
    verbose : print a one-line mesh summary and which source was used

    Returns
    -------
    points, triangles, edges, vecteurs_rho
    """
    regenerated = os.path.isfile(path.msh) and _try_regenerate_mat_from_msh(path)

    if regenerated:
        source = f"regenerated from {path.msh}"
    else:
        if not os.path.isfile(path.mat):
            raise FileNotFoundError(
                f"Neither '{path.msh}' nor '{path.mat}' provides a usable mesh -- CMA has "
                f"nothing to work from. Generate and mesh the geometry with gmsh, or "
                f"supply the .mat directly."
            )
        source = f"loaded from {path.mat}"

    p, t = load_mesh_file(path.mat)
    points = Points(p)
    triangles = Triangles(t)
    triangles.calculate_triangles_area_and_center(points)
    edges = triangles.get_edges()
    filter_complexes_jonctions(points, triangles, edges)
    edges.compute_edges_length(points)
    DataManager_rwg1.save_data(path, points, triangles, edges)

    vecteurs_rho = Vecteurs_Rho()
    vecteurs_rho.calculate_vecteurs_rho(points, triangles, edges)
    DataManager_rwg2.save_data(path, vecteurs_rho)

    if verbose:
        print(f"Mesh ready: {points.total_of_points} points, {triangles.total_of_triangles} "
              f"triangles, {edges.total_number_of_edges} edges -- {source}\n")

    return points, triangles, edges, vecteurs_rho
