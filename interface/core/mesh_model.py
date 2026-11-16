"""In-memory surface mesh, read from a Gmsh .msh file.

Milestone 1 only needs to *display* a mesh, so this module reads the triangles
straight from the .msh file with the Gmsh API. Unlike
solver.utils.gmsh_function.extract_msh_to_mat, it writes nothing to disk:
importing a file in the interface must never create or overwrite files.

Units: the solver works in SI, so mesh coordinates are taken to be in metres.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class MeshImportError(ValueError):
    """The file could not be read as a triangular surface mesh."""


@dataclass(frozen=True)
class SurfaceMesh:
    """A triangular surface mesh.

    points        (N, 3) float  vertex coordinates in metres
    triangles     (M, 3) int    0-based vertex indices of each triangle
    surface_tags  (M,)   int    Gmsh surface (entity) each triangle belongs to
    source        Path          file the mesh was read from
    """

    points: np.ndarray
    triangles: np.ndarray
    surface_tags: np.ndarray
    source: Path

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def n_triangles(self) -> int:
        return int(self.triangles.shape[0])

    @property
    def n_surfaces(self) -> int:
        return int(np.unique(self.surface_tags).size)

    def bounding_box(self):
        """(min corner, max corner), each a (3,) array in metres."""
        return self.points.min(axis=0), self.points.max(axis=0)

    def edge_length_range(self):
        """(shortest, longest) triangle edge length in metres."""
        p = self.points[self.triangles]                    # (M, 3 vertices, 3 coords)
        edges = p[:, [1, 2, 0], :] - p                     # the 3 edges of every triangle
        lengths = np.linalg.norm(edges, axis=2)
        return float(lengths.min()), float(lengths.max())

    def n_shared_edges(self) -> int:
        """Number of edges shared by at least two triangles.

        This is only an indication of the size of the MoM problem: the solver
        builds one RWG function per pair of triangles sharing an edge, so the
        exact number of unknowns can be larger at junctions (see
        solver/rwg/rwg1.py). The exact count will come from the solver itself
        in a later milestone.
        """
        t = self.triangles
        edges = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        return int(np.count_nonzero(counts >= 2))


def load_msh(path) -> SurfaceMesh:
    """Read the triangles of a Gmsh .msh file.

    Only 3-node triangles (Gmsh element type 2) on surfaces are kept; lines,
    points and volume elements are ignored, as in the solver.
    """
    import gmsh  # imported here so that `import interface.core` stays light

    path = Path(path)
    if not path.is_file():
        raise MeshImportError(f"File not found: {path}")
    if path.suffix.lower() != ".msh":
        raise MeshImportError(f"Expected a .msh file, got '{path.suffix}'.")

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)  # no console output from Gmsh
        gmsh.open(str(path))

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        if len(node_tags) == 0:
            raise MeshImportError(f"'{path.name}' contains no mesh nodes.")
        points = np.asarray(node_coords, dtype=float).reshape(-1, 3)

        # Gmsh node tags are arbitrary integers: map them to 0..N-1.
        tag_to_index = np.full(int(np.max(node_tags)) + 1, -1, dtype=np.int64)
        tag_to_index[np.asarray(node_tags, dtype=np.int64)] = np.arange(len(node_tags))

        triangle_blocks, tag_blocks = [], []
        for dim, entity in gmsh.model.getEntities(2):
            elem_types, _, elem_nodes = gmsh.model.mesh.getElements(dim, entity)
            for etype, nodes in zip(elem_types, elem_nodes):
                if etype != 2:  # 2 = 3-node triangle
                    continue
                tri = tag_to_index[np.asarray(nodes, dtype=np.int64).reshape(-1, 3)]
                triangle_blocks.append(tri)
                tag_blocks.append(np.full(tri.shape[0], entity, dtype=np.int64))
    except MeshImportError:
        raise
    except Exception as error:  # Gmsh raises plain Exceptions with its own message
        raise MeshImportError(f"Gmsh could not read '{path.name}': {error}") from error
    finally:
        gmsh.finalize()

    if not triangle_blocks:
        raise MeshImportError(f"'{path.name}' contains no surface triangles.")

    triangles = np.concatenate(triangle_blocks)
    if np.any(triangles < 0):
        raise MeshImportError(f"'{path.name}' has triangles referring to unknown nodes.")

    return SurfaceMesh(
        points=points,
        triangles=triangles,
        surface_tags=np.concatenate(tag_blocks),
        source=path.resolve(),
    )
