"""
Flat-strip mesh for the backend.optim tests: a thin wrapper around
backend.optim.geometries.planar.tensor_grid_mesh with uniform grid lines
(np.linspace), keeping the small helper API the step-1 tests use.

Rectangle [-Lx/2, Lx/2] x [-Ly/2, Ly/2] in z = 0, (nx x ny) cells, 2 triangles
per cell. Vertices and triangles are numbered x-major, so on every vertical
grid line the left triangle is T+ (Triangles.get_edges): the RWG reference
direction across a transverse cut is +x.
"""

from dataclasses import dataclass

import numpy as np

from backend.optim.geometries.planar import tensor_grid_mesh


@dataclass
class StripMesh:
    points: object
    triangles: object
    edges: object
    vecteurs_rho: object
    xs: np.ndarray      # (nx+1,) x coordinates of the vertical grid lines
    ys: np.ndarray      # (ny+1,) y coordinates of the horizontal grid lines
    length: float
    width: float

    def cut_at(self, x):
        """(p_start, p_end) of the transverse cut at abscissa x (full strip width)."""
        return (np.array([x, -self.width / 2, 0.0]),
                np.array([x, self.width / 2, 0.0]))

    def nearest_grid_x(self, x):
        """Grid-line abscissa closest to x (so that a cut there lies on edges)."""
        return float(self.xs[np.argmin(np.abs(self.xs - x))])


def make_strip_mesh(length, width, nx, ny):
    """
    Flat strip of size length (x) by width (y), (nx x ny) cells.

    RWG edge count: N = nx*(ny-1) + (nx-1)*ny + nx*ny.
    """
    xs = np.linspace(-length / 2, length / 2, nx + 1)
    ys = np.linspace(-width / 2, width / 2, ny + 1)
    m = tensor_grid_mesh(xs, ys)
    return StripMesh(m.points, m.triangles, m.edges, m.vecteurs_rho, xs, ys, length, width)
