"""
Structured planar meshes (z = 0) for port-based MoM studies, built without Gmsh.

Every geometry here is a subset of the cells of one tensor-product grid
(x-lines xs, y-lines ys), each kept cell split into two triangles along its
(i, j) -> (i+1, j+1) diagonal. Consequences, by construction:

  * the mesh is conformal: two kept cells that touch share a full grid edge,
    so there is no T-junction and no edge has more than two triangles;
  * every grid line is a line of mesh edges, so a port cut placed on a grid
    line (solver.optim.ports.port_from_cut) falls exactly on RWG edges.

The RWG data go through the production classes (Points, Triangles.get_edges,
Edges.compute_edges_length, filter_complexes_jonctions, Vecteurs_Rho).

Width grading
-------------
The surface current on a thin strip is singular at the strip edges,
J ~ 1/sqrt(distance to the edge). Conductor loss scales as |J|^2 and is
therefore concentrated at the edges: a uniform, coarse discretisation of the
width under-resolves exactly where the loss happens and underestimates it.
`edge_grading` clusters the cells towards both edges of each strip:

    None / 'uniform' : t_k = k / n
    'cosine'         : t_k = (1 - cos(pi k / n)) / 2      (Chebyshev-Lobatto)
    callable         : n -> array of n+1 increasing values from 0 to 1

with the strip's cross-coordinate = edge_low + t_k * width.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np

from solver.rwg.rwg1 import Points, Triangles, filter_complexes_jonctions
from solver.rwg.rwg2 import Vecteurs_Rho

Grading = Union[None, str, Callable[[int], np.ndarray]]
Cut = Tuple[np.ndarray, np.ndarray, np.ndarray]     # (p_start, p_end, direction)


@dataclass
class PlanarMesh:
    """
    Mesh plus the port cuts defined on it.

    Attributes
    ----------
    points, triangles, edges, vecteurs_rho : production rwg1/rwg2 objects
    cuts      : {name: (p_start, p_end, direction)}, all lying on grid lines
    xs, ys    : grid lines (m)
    params    : construction parameters (for metadata files)
    """
    points: Points
    triangles: Triangles
    edges: object
    vecteurs_rho: Vecteurs_Rho
    cuts: Dict[str, Cut]
    xs: np.ndarray
    ys: np.ndarray
    params: dict = field(default_factory=dict)

    @property
    def feed_cut(self) -> Cut:
        """(p_start, p_end, direction) of the feed port."""
        return self.cuts["feed"]

    @property
    def n_edges(self) -> int:
        return int(self.edges.total_number_of_edges)

    def port(self, name="feed", tol=1e-9):
        """solver.optim.ports.Port for the named cut."""
        from solver.optim.ports import port_from_cut
        p0, p1, d = self.cuts[name]
        return port_from_cut(self.points, self.triangles, self.edges, p0, p1, d, name, tol=tol)

    def ports(self, tol=1e-9):
        """All cuts as Ports, 'feed' first."""
        names = ["feed"] + [n for n in self.cuts if n != "feed"]
        return [self.port(n, tol=tol) for n in names if n in self.cuts]


# --------------------------------------------------------------------------- grading
def grading_nodes(n: int, edge_grading: Grading = None) -> np.ndarray:
    """n+1 increasing nodes from 0 to 1 (see module docstring)."""
    if n < 1:
        raise ValueError(f"Need at least one cell, got n = {n}.")
    k = np.arange(n + 1)
    if edge_grading is None or edge_grading == "uniform":
        t = k / n
    elif edge_grading == "cosine":
        t = 0.5 * (1.0 - np.cos(np.pi * k / n))
    elif callable(edge_grading):
        t = np.asarray(edge_grading(n), dtype=float)
    else:
        raise ValueError(f"Unknown edge_grading {edge_grading!r}: use None, 'uniform', "
                         f"'cosine' or a callable n -> nodes.")
    if t.shape != (n + 1,) or not np.all(np.diff(t) > 0):
        raise ValueError("Grading nodes must be n+1 strictly increasing values.")
    t = (t - t[0]) / (t[-1] - t[0])
    t[0], t[-1] = 0.0, 1.0
    return t


def _segment(a, b, n, grading=None):
    return a + (b - a) * grading_nodes(n, grading)


def _merge_lines(*parts, tol=1e-12):
    """Concatenate grid-line pieces, dropping duplicated junction values."""
    x = np.sort(np.concatenate(parts))
    keep = np.concatenate([[True], np.diff(x) > tol])
    return x[keep]


# --------------------------------------------------------------------------- builder
def tensor_grid_mesh(xs, ys, keep: Optional[np.ndarray] = None, cuts=None, params=None) -> PlanarMesh:
    """
    Mesh of the kept cells of the (xs x ys) grid in the plane z = 0.

    keep : (len(xs)-1, len(ys)-1) bool, True for cells that are metal
           (default: all). Unused vertices are dropped.
    Vertices and triangles are numbered x-major, so on a vertical grid line the
    -x triangle is the lower-indexed one (T+ in Triangles.get_edges).
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    nx, ny = len(xs) - 1, len(ys) - 1
    if keep is None:
        keep = np.ones((nx, ny), dtype=bool)
    keep = np.asarray(keep, dtype=bool)
    if keep.shape != (nx, ny):
        raise ValueError(f"keep must have shape {(nx, ny)}, got {keep.shape}.")
    if not keep.any():
        raise ValueError("No cell kept.")

    used = np.zeros((nx + 1, ny + 1), dtype=bool)
    ci, cj = np.nonzero(keep)
    for di in (0, 1):
        for dj in (0, 1):
            used[ci + di, cj + dj] = True
    vid = -np.ones((nx + 1, ny + 1), dtype=int)
    vid[used] = np.arange(used.sum())            # x-major numbering (C order)
    I, J = np.nonzero(used)
    p = np.vstack([xs[I], ys[J], np.zeros(I.size)])

    tri = []
    for i, j in zip(ci, cj):                     # np.nonzero is x-major too
        v00, v10, v11, v01 = vid[i, j], vid[i + 1, j], vid[i + 1, j + 1], vid[i, j + 1]
        tri.append((v00, v10, v11))
        tri.append((v00, v11, v01))
    tri = np.array(tri, dtype=int).T
    t = np.vstack([tri, np.ones((1, tri.shape[1]), dtype=int)])   # Gmsh-like (4, M)

    points = Points(p)
    triangles = Triangles(t)
    triangles.calculate_triangles_area_and_center(points)
    edges = triangles.get_edges()
    edges.compute_edges_length(points)
    filter_complexes_jonctions(points, triangles, edges)
    vecteurs_rho = Vecteurs_Rho()
    vecteurs_rho.calculate_vecteurs_rho(points, triangles, edges)

    return PlanarMesh(points, triangles, edges, vecteurs_rho, dict(cuts or {}), xs, ys,
                      dict(params or {}))


def _cut(x, y0, y1, direction=(1.0, 0.0, 0.0)) -> Cut:
    return (np.array([x, y0, 0.0]), np.array([x, y1, 0.0]), np.array(direction, dtype=float))


# --------------------------------------------------------------------------- geometries
def strip_dipole(length, width, n_len, n_width, edge_grading: Grading = None) -> PlanarMesh:
    """
    Strip along x, [-length/2, length/2] x [-width/2, width/2], centred at the origin.

    n_len   : cells along x (must be even: x = 0 is the feed line)
    n_width : cells across y (graded by `edge_grading`)
    Feed cut: x = 0, full width, positive direction +x.
    """
    if n_len % 2:
        raise ValueError(f"n_len must be even so that x = 0 is a grid line, got {n_len}.")
    half = n_len // 2
    xs = _merge_lines(_segment(-length / 2, 0.0, half), _segment(0.0, length / 2, half))
    ys = _segment(-width / 2, width / 2, n_width, edge_grading)
    params = dict(geometry="strip_dipole", length=length, width=width, n_len=n_len,
                  n_width=n_width, edge_grading=_grading_name(edge_grading))
    return tensor_grid_mesh(xs, ys, cuts={"feed": _cut(0.0, -width / 2, width / 2)},
                            params=params)


def square_loop(side_centerline, width, n_side, n_width, edge_grading: Grading = None) -> PlanarMesh:
    """
    Square loop in the xy plane, centred at the origin; `side_centerline` is the
    side length measured on the strip's centre line. Outer edges at
    +-(s/2 + w/2), inner edges at +-(s/2 - w/2).

    n_side  : cells along each side between the two corner squares (must be even:
              x = 0 is the feed line on the side y = -s/2)
    n_width : cells across the strip (graded by `edge_grading`); each corner is an
              n_width x n_width block, so the corners are conformal.
    Feed cut: x = 0 across the side y = -s/2, positive direction +x
    (counter-clockwise seen from +z).
    Load cut ('load'): x = 0 across the opposite side y = +s/2, positive direction
    -x (same circulation). Unused unless a two-port model is built (V5b').
    """
    s, w = float(side_centerline), float(width)
    if not 0 < w < s:
        raise ValueError(f"Need 0 < width < side_centerline, got w = {w}, s = {s}.")
    if n_side % 2:
        raise ValueError(f"n_side must be even so that x = 0 is a grid line, got {n_side}.")
    a_out, a_in = s / 2 + w / 2, s / 2 - w / 2
    half = n_side // 2
    lines = _merge_lines(
        _segment(-a_out, -a_in, n_width, edge_grading),
        _segment(-a_in, 0.0, half), _segment(0.0, a_in, half),
        _segment(a_in, a_out, n_width, edge_grading),
    )
    xs = ys = lines
    xc = 0.5 * (xs[:-1] + xs[1:])
    yc = 0.5 * (ys[:-1] + ys[1:])
    XC, YC = np.meshgrid(xc, yc, indexing="ij")
    keep = (np.abs(XC) > a_in) | (np.abs(YC) > a_in)      # ring = outer square minus inner
    params = dict(geometry="square_loop", side_centerline=s, width=w, n_side=n_side,
                  n_width=n_width, edge_grading=_grading_name(edge_grading))
    cuts = {"feed": _cut(0.0, -a_out, -a_in),
            "load": _cut(0.0, a_in, a_out, direction=(-1.0, 0.0, 0.0))}
    return tensor_grid_mesh(xs, ys.copy(), keep, cuts=cuts, params=params)


def parallel_strips(length, width, spacing, n_len, n_width, edge_grading: Grading = None) -> PlanarMesh:
    """
    Two strips along x, centred at y = -spacing/2 ('feed') and y = +spacing/2
    ('parasite'), centre-to-centre distance `spacing`. Each has a cut at x = 0,
    positive direction +x. Representative of a two-element end-fire array along y.
    """
    if n_len % 2:
        raise ValueError(f"n_len must be even, got {n_len}.")
    if spacing <= width:
        raise ValueError(f"The strips overlap: spacing {spacing} <= width {width}.")
    half = n_len // 2
    xs = _merge_lines(_segment(-length / 2, 0.0, half), _segment(0.0, length / 2, half))
    yc = (-spacing / 2, spacing / 2)
    ys = _merge_lines(*[_segment(c - width / 2, c + width / 2, n_width, edge_grading) for c in yc])
    ymid = 0.5 * (ys[:-1] + ys[1:])
    in_strip = np.zeros(ymid.shape, dtype=bool)
    for c in yc:
        in_strip |= np.abs(ymid - c) < width / 2
    keep = np.broadcast_to(in_strip[None, :], (len(xs) - 1, len(ys) - 1)).copy()
    cuts = {"feed": _cut(0.0, yc[0] - width / 2, yc[0] + width / 2),
            "parasite": _cut(0.0, yc[1] - width / 2, yc[1] + width / 2)}
    params = dict(geometry="parallel_strips", length=length, width=width, spacing=spacing,
                  n_len=n_len, n_width=n_width, edge_grading=_grading_name(edge_grading))
    return tensor_grid_mesh(xs, ys, keep, cuts=cuts, params=params)


#: Challenge box (z = 0 plane), with the margins of step 5: 187 x 125 mm minus
#: 3.5 mm at each end along x and 2.5 mm at each side along y.
BOX_LENGTH = 0.180          # max strip length along x (m)
BOX_WIDTH = 0.120           # max edge-to-edge extent along y, widths included (m)


class BoxError(ValueError):
    """Raised when a geometry does not fit in the challenge box."""


def parallel_strip_array(lengths, width, spacings, n_width=4, edge_grading: Grading = "cosine",
                         cell_length=4.5e-3, box_length=BOX_LENGTH, box_width=BOX_WIDTH) -> PlanarMesh:
    """
    K strips parallel to x, centred at x = 0, in z = 0, stacked along y.

    lengths     : K strip lengths (m), each <= box_length
    width       : common strip width (m)
    spacings    : K-1 centre-to-centre distances along y (m), each > width; the array
                  is centred on y = 0 and its edge-to-edge extent
                  (y_last - y_first) + width must be <= box_width
    n_width, edge_grading : cells across each strip (default: the calibrated rule of
                  solver.optim.loss_model.MESH_RULE)
    cell_length : target cell length along x (m); each strip gets an even number of
                  cells so that x = 0 is a grid line

    Cuts 'el1' ... 'elK' (from -y to +y) at x = 0, full width, positive direction +x.
    Use array_ports(mesh, fed) for the Port list with the fed element first.
    Raises BoxError if the array does not fit the box, ValueError if strips overlap.
    """
    lengths = [float(L) for L in np.atleast_1d(lengths)]
    spacings = [float(d) for d in np.atleast_1d(spacings)] if len(lengths) > 1 else []
    K = len(lengths)
    if len(spacings) != K - 1:
        raise ValueError(f"{K} strips need {K - 1} spacings, got {len(spacings)}.")
    for i, L in enumerate(lengths):
        if L > box_length + 1e-12:
            raise BoxError(f"strip el{i + 1}: length {L * 1e3:.2f} mm exceeds the box "
                           f"({box_length * 1e3:.1f} mm along x).")
    for i, d in enumerate(spacings):
        if d <= width:
            raise ValueError(f"spacing {i + 1} ({d * 1e3:.2f} mm) <= width ({width * 1e3:.2f} mm): "
                             f"strips overlap.")
    yc = np.concatenate([[0.0], np.cumsum(spacings)])
    yc -= 0.5 * (yc[0] + yc[-1])
    extent = (yc[-1] - yc[0]) + width
    if extent > box_width + 1e-12:
        raise BoxError(f"array extent along y is {extent * 1e3:.2f} mm (edge to edge) > "
                       f"{box_width * 1e3:.1f} mm.")

    xs_parts, n_cells = [], []
    for L in lengths:
        n = max(2, 2 * int(np.ceil(L / (2 * cell_length))))
        n_cells.append(n)
        xs_parts += [_segment(-L / 2, 0.0, n // 2), _segment(0.0, L / 2, n // 2)]
    xs = _merge_lines(*xs_parts)
    ys = _merge_lines(*[_segment(c - width / 2, c + width / 2, n_width, edge_grading) for c in yc])

    xm = 0.5 * (xs[:-1] + xs[1:])
    ym = 0.5 * (ys[:-1] + ys[1:])
    keep = np.zeros((len(xm), len(ym)), dtype=bool)
    for c, L in zip(yc, lengths):
        keep |= (np.abs(xm)[:, None] < L / 2) & (np.abs(ym - c)[None, :] < width / 2)

    cuts = {f"el{k + 1}": _cut(0.0, c - width / 2, c + width / 2) for k, c in enumerate(yc)}
    params = dict(geometry="parallel_strip_array", lengths=lengths, width=width,
                  spacings=spacings, y_centers=yc.tolist(), extent_y=extent,
                  n_len=n_cells, n_width=n_width, edge_grading=_grading_name(edge_grading),
                  cell_length=cell_length, box=(box_length, box_width))
    return tensor_grid_mesh(xs, ys, keep, cuts=cuts, params=params)


def array_ports(mesh: PlanarMesh, fed: int, tol=1e-9):
    """Ports of a parallel_strip_array mesh, fed element (0-based index, -y to +y) first,
    the others in y order."""
    K = len(mesh.params["y_centers"])
    if not 0 <= fed < K:
        raise ValueError(f"fed index {fed} out of range for {K} elements.")
    order = [fed] + [k for k in range(K) if k != fed]
    return [mesh.port(f"el{k + 1}", tol=tol) for k in order]


def _grading_name(g):
    if g is None:
        return "uniform"
    return g if isinstance(g, str) else getattr(g, "__name__", "custom")
