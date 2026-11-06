"""
Ports of an RWG mesh, defined as cuts across a conductor.

A port p is a set of RWG edges S_p that together form a cut across a strip,
plus a unit vector u_p giving the positive current direction across that cut.
Its incidence vector a_p in R^N is

    (a_p)_n = s_n * l_n   if n in S_p,   0 otherwise,
    s_n     = sign( (c_n^- - c_n^+) . u_p ),

where l_n is the edge length and c_n^+/- the centroids of the triangles T_n^+/-.

Why this is the right vector (conventions checked in solver.utils.impmet):

  * The RWG function f_n carries current from T_n^+ to T_n^- across edge n,
    and its normal component on the edge is 1, so the total current crossing
    edge n in the T^+ -> T^- direction is l_n * I_n. Hence the current across
    the port, counted positive along u_p, is

        i_p = a_p^T I.

  * A delta-gap voltage v_p across the cut, i.e. E_inc = v_p delta(n) u_p,
    tested with f_m gives  <f_m, E_inc> = s_m l_m v_p  (Makarov's V_m = l_m V
    generalised to several edges and to either edge orientation), so

        V = a_p v_p.

  * A lumped load Z_L on a single-edge port is therefore  Z += Z_L a_p a_p^T,
    i.e. Z_mm += l_m^2 Z_L -- exactly rwg3.calculate_z_matrice_lumped_elements.

With P ports, A = [a_1 ... a_P] (N x P):  V = A v,  i = A^T I.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


class PortDefinitionError(ValueError):
    """Raised when a port cannot be defined unambiguously on the RWG mesh."""


# Relative threshold under which (c^- - c^+) . u is considered to be zero, i.e.
# the requested current direction does not cross the edge at all.
_SIGN_ZERO_RTOL = 1e-9

# Below this norm, the component of `direction` perpendicular to the cut is
# considered to vanish: the direction runs ALONG the cut, not across it.
_PARALLEL_TOL = 1e-6


def _points_array(points):
    """Accept a Points object or a raw (3, N_points) array."""
    return np.asarray(points.points if hasattr(points, "points") else points, dtype=float)


def _unit(vector, what):
    v = np.asarray(vector, dtype=float).reshape(-1)
    if v.shape != (3,):
        raise PortDefinitionError(f"{what} must be a 3-vector, got shape {np.shape(vector)}.")
    norm = np.linalg.norm(v)
    if not np.isfinite(norm) or norm == 0.0:
        raise PortDefinitionError(f"{what} must be a non-zero finite 3-vector, got {v}.")
    return v / norm


def _distance_to_segment(p, a, b):
    """
    Distance from each column of p (3, K) to the segment [a, b]:

        t* = clamp( (p - a).(b - a) / |b - a|^2, 0, 1 ),   d = | p - (a + t* (b - a)) |.
    """
    ab = b - a
    t = np.clip(((p - a[:, None]).T @ ab) / (ab @ ab), 0.0, 1.0)
    closest = a[:, None] + ab[:, None] * t[None, :]
    return np.linalg.norm(p - closest, axis=0)


def _edge_signs(triangles, edge_indices, direction, port_name):
    """
    s_n = sign( (c_n^- - c_n^+) . u ) for each selected edge.

    Raises PortDefinitionError if the projection vanishes (relative to
    |c_n^- - c_n^+|): `direction` then has no component across edge n and the
    port orientation is undefined there.
    """
    centers = np.asarray(triangles.triangles_center, dtype=float)
    plus = np.asarray(triangles.triangles_plus)[edge_indices]
    minus = np.asarray(triangles.triangles_minus)[edge_indices]
    delta = centers[:, minus] - centers[:, plus]                 # (3, n_p)
    proj = direction @ delta
    scale = np.linalg.norm(delta, axis=0)
    zero = np.abs(proj) <= _SIGN_ZERO_RTOL * scale
    if np.any(zero):
        bad = np.asarray(edge_indices)[zero]
        raise PortDefinitionError(
            f"Port '{port_name}': direction {direction} does not cross edge(s) {bad.tolist()} "
            f"(sign of (c^- - c^+).direction is 0). The direction must point ACROSS the cut, "
            f"tangent to the conductor -- e.g. along the strip axis for a transverse cut -- "
            f"not along the cut nor normal to the surface."
        )
    return np.sign(proj)


@dataclass(frozen=True, eq=False)
class Port:
    """
    One port: a cut made of RWG edges with a positive current direction.

    Attributes
    ----------
    name         : str               label used in results
    edge_indices : (n_p,) int        RWG edges forming the cut
    signs        : (n_p,) float      s_n = +1/-1, orientation of each edge w.r.t. `direction`
    direction    : (3,) float        unit vector u_p, positive current direction across the cut

    The arrays are stored as read-only copies so a Port can be shared safely.
    """
    name: str
    edge_indices: np.ndarray
    signs: np.ndarray
    direction: np.ndarray

    def __post_init__(self):
        idx = np.array(self.edge_indices, dtype=np.int64).reshape(-1)
        sg = np.array(self.signs, dtype=np.float64).reshape(-1)
        u = _unit(self.direction, f"Port '{self.name}' direction")
        if idx.size == 0:
            raise PortDefinitionError(f"Port '{self.name}' has no edge.")
        if sg.shape != idx.shape:
            raise PortDefinitionError(
                f"Port '{self.name}': {idx.size} edge indices but {sg.size} signs.")
        if not np.all(np.isin(sg, (-1.0, 1.0))):
            raise PortDefinitionError(f"Port '{self.name}': signs must all be +1 or -1, got {sg}.")
        if np.any(idx < 0):
            raise PortDefinitionError(f"Port '{self.name}': negative edge index in {idx}.")
        if np.unique(idx).size != idx.size:
            raise PortDefinitionError(f"Port '{self.name}': the same edge is listed twice in {idx}.")
        for arr in (idx, sg, u):
            arr.setflags(write=False)
        object.__setattr__(self, "edge_indices", idx)
        object.__setattr__(self, "signs", sg)
        object.__setattr__(self, "direction", u)

    @property
    def n_edges(self) -> int:
        return int(self.edge_indices.size)

    def width(self, edges) -> float:
        """Total length of the cut, sum_{n in S_p} l_n (m)."""
        return float(np.sum(np.asarray(edges.edges_length)[self.edge_indices]))

    def incidence_vector(self, edges) -> np.ndarray:
        """
        a_p, shape (N,):  (a_p)_n = s_n l_n for n in S_p, 0 elsewhere.

        Delta-gap excitation V = a_p v_p ; port current i_p = a_p^T I.
        """
        n_total = int(edges.total_number_of_edges)
        if self.edge_indices.max() >= n_total:
            raise PortDefinitionError(
                f"Port '{self.name}' refers to edge {int(self.edge_indices.max())} but the mesh "
                f"has only {n_total} RWG edges. Was the port built on a different mesh (or "
                f"before filter_complexes_jonctions removed edges)?")
        a = np.zeros(n_total, dtype=np.float64)
        a[self.edge_indices] = self.signs * np.asarray(edges.edges_length)[self.edge_indices]
        return a


def port_from_cut(points, triangles, edges, p_start, p_end, direction, name, tol=1e-9) -> Port:
    """
    Port made of all RWG edges lying ON the segment [p_start, p_end].

    An edge is selected when both its endpoints are within `tol` (metres) of
    the segment. Its sign is s_n = sign((c_n^- - c_n^+) . u), u = direction/|direction|.

    Raises PortDefinitionError, with a message saying what to change, if
      * no edge lies on the segment (cut off the mesh, or not on mesh lines);
      * `direction` runs along the cut instead of across it, or gives a zero
        sign on some edge (no component across that edge);
      * the sign from `direction` disagrees with the sign from its component
        perpendicular to the cut on some edge (direction too oblique to define
        an orientation unambiguously);
      * the selected edges do not cover the segment:
        | sum_n l_n - |p_end - p_start| | > tol  (cut partly off the conductor,
        or crossing a T-junction whose duplicated edges would be counted twice).
    """
    pts = _points_array(points)
    a = np.asarray(p_start, dtype=float).reshape(3)
    b = np.asarray(p_end, dtype=float).reshape(3)
    seg = b - a
    seg_len = float(np.linalg.norm(seg))
    if seg_len <= tol:
        raise PortDefinitionError(
            f"Port '{name}': p_start and p_end coincide (|p_end - p_start| = {seg_len:.3g} m); "
            f"give the two ends of the cut across the conductor.")
    u = _unit(direction, f"Port '{name}' direction")
    seg_hat = seg / seg_len
    u_perp = u - (u @ seg_hat) * seg_hat
    if np.linalg.norm(u_perp) < _PARALLEL_TOL:
        raise PortDefinitionError(
            f"Port '{name}': direction {u} is parallel to the cut {a} -> {b}. The direction "
            f"is the current flow ACROSS the port: for a transverse cut of a strip it must "
            f"point along the strip.")
    u_perp /= np.linalg.norm(u_perp)

    first = np.asarray(edges.first_points)
    second = np.asarray(edges.second_points)
    on_cut = ((_distance_to_segment(pts[:, first], a, b) <= tol)
              & (_distance_to_segment(pts[:, second], a, b) <= tol))
    idx = np.flatnonzero(on_cut)
    if idx.size == 0:
        raise PortDefinitionError(
            f"Port '{name}': no RWG edge lies on the segment {a} -> {b} (tol = {tol:g} m). "
            f"The cut must follow mesh edges: place it on a line of mesh vertices across "
            f"the conductor (and inside the mesh, not on its boundary -- boundary edges "
            f"carry no RWG function), or increase tol if the coordinates are only "
            f"approximately aligned.")

    signs = _edge_signs(triangles, idx, u, name)
    signs_perp = _edge_signs(triangles, idx, u_perp, name)
    if np.any(signs != signs_perp):
        bad = idx[signs != signs_perp]
        raise PortDefinitionError(
            f"Port '{name}': direction {u} is too oblique to the cut to orient edge(s) "
            f"{bad.tolist()} unambiguously. Use a direction perpendicular to the cut.")

    covered = float(np.sum(np.asarray(edges.edges_length)[idx]))
    if abs(covered - seg_len) > tol:
        raise PortDefinitionError(
            f"Port '{name}': the {idx.size} edge(s) found on the cut have total length "
            f"{covered:.12g} m but the cut is {seg_len:.12g} m long. Either the cut extends "
            f"beyond the conductor (shorten it to the conductor's width), part of it does "
            f"not lie on mesh edges, or it crosses a junction whose duplicated edges are "
            f"counted twice.")

    return Port(name=name, edge_indices=idx, signs=signs, direction=u)


def port_from_edge(points, triangles, edges, point, direction, name, orientation_tol=1e-3) -> Port:
    """
    Single-edge port, selected with the same rule as
    rwg3.calculate_z_matrice_lumped_elements: among the edges whose unit vector
    e_n = (P1 - P2)/l_n satisfies |e_n . u| < orientation_tol (edge perpendicular
    to the current direction), take the one whose midpoint is closest to `point`.

    The direction is normalised first; for a unit `direction` (the normal use of
    LoadDir) the selected edge is the one rwg3 picks. The sign is
    s = sign((c^- - c^+) . u), as for port_from_cut.

    Raises PortDefinitionError if no edge is perpendicular to `direction`, or if
    the selected edge's sign is 0.
    """
    pts = _points_array(points)
    p0 = np.asarray(point, dtype=float).reshape(3)
    u = _unit(direction, f"Port '{name}' direction")
    first = np.asarray(edges.first_points)
    second = np.asarray(edges.second_points)
    lengths = np.asarray(edges.edges_length)

    # Same expressions as rwg3.calculate_z_matrice_lumped_elements, so ties in
    # distance are broken identically by np.argsort.
    edge_centers = 0.5 * (pts[:, first] + pts[:, second])
    edge_vectors = (pts[:, first] - pts[:, second]) / lengths[np.newaxis, :]
    dist = np.linalg.norm(edge_centers - p0[:, np.newaxis], axis=0)
    orientation = np.abs(np.einsum('ij,i->j', edge_vectors, u))

    selected = None
    for idx in np.argsort(dist):
        if orientation[idx] < orientation_tol:
            selected = int(idx)
            break
    if selected is None:
        raise PortDefinitionError(
            f"Port '{name}': no RWG edge is perpendicular to direction {u} "
            f"(|e_n . u| < {orientation_tol:g}). Give the direction of current flow at the "
            f"port, i.e. along the strip, so that the feed edge lies across it.")

    signs = _edge_signs(triangles, np.array([selected]), u, name)
    return Port(name=name, edge_indices=np.array([selected]), signs=signs, direction=u)


def incidence_matrix(ports: Sequence[Port], edges) -> np.ndarray:
    """
    A = [a_1 ... a_P], shape (N, P).

    Raises PortDefinitionError if the list is empty, if two ports share a name,
    or if two ports share an edge (the ports would not be independent
    terminals: V = A v would add both voltages on the shared edge).
    """
    ports = list(ports)
    if not ports:
        raise PortDefinitionError("At least one port is required.")
    names = [p.name for p in ports]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise PortDefinitionError(f"Port names must be unique; duplicated: {dupes}.")

    owner = {}
    for p in ports:
        for e in p.edge_indices.tolist():
            if e in owner:
                raise PortDefinitionError(
                    f"Ports '{owner[e]}' and '{p.name}' share edge {e}. Two ports cannot "
                    f"overlap: move one cut to another line of edges.")
            owner[e] = p.name

    return np.column_stack([p.incidence_vector(edges) for p in ports])
