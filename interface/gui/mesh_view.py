"""Conversion of interface.core objects into PyVista objects for the 3-D view."""

import numpy as np
import pyvista as pv

from interface.core import SurfaceMesh


# Muted RGB colours, one per Gmsh surface (cycled when there are more surfaces).
SURFACE_COLOURS = np.array([
    (0xB8, 0x7A, 0x3D), (0x5B, 0x8D, 0xB8), (0x8A, 0xA6, 0x5C), (0xB0, 0x6E, 0x8C),
    (0xC9, 0xA2, 0x4A), (0x6E, 0x9E, 0x9A), (0x9A, 0x7E, 0xB8), (0xA8, 0x6A, 0x5A),
], dtype=np.uint8)


def to_polydata(mesh: SurfaceMesh) -> pv.PolyData:
    """Build a PyVista surface from a SurfaceMesh.

    VTK stores the faces as one flat array: for every triangle, the number of
    vertices (3) followed by the three vertex indices -> [3, i, j, k, 3, ...].
    The Gmsh surface number is attached to each triangle as cell data so that
    the view can colour the mesh by surface.
    """
    n = mesh.n_triangles
    faces = np.hstack([np.full((n, 1), 3, dtype=np.int64), mesh.triangles.astype(np.int64)]).ravel()
    polydata = pv.PolyData(mesh.points, faces)
    polydata.cell_data["surface"] = mesh.surface_tags

    # One colour per surface, computed here rather than through a colour map:
    # PyVista's "categorical" option was removed in recent versions (0.49).
    _, index = np.unique(mesh.surface_tags, return_inverse=True)
    polydata.cell_data["colour"] = SURFACE_COLOURS[index % len(SURFACE_COLOURS)]
    return polydata
