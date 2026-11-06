"""
Lumped matching network between the 50-Ohm port and the fed gap (step 6b).

Antenna as a one-port
---------------------
With its parasitic loads frozen (kit parts), the antenna seen from the fed gap
is a one-port: input impedance Z_A(f) and gain G_A(f) = 4 pi U(+y) / P_A, where
P_A is the power accepted by the gap (antenna conductor losses included). The
far field is proportional to the gap current, so with a network between the
generator (internal impedance Z0, available power P_avail) and the gap

    G_r(f) = G_A(f) * P_A / P_avail = G_A(f) * (1 - |S11|^2) * eta_net,
    eta_net = P_A / P_in   (power delivered to the gap / power entering the network).

Z_A(f) and G_A(f) come from the MoM frequency checks of the design (Z refilled
at each frequency, reduced model with the frozen parts) and are interpolated by
cubic splines between those frequencies (AntennaOnePort; no extrapolation).

Network
-------
Cascade of series and shunt elements listed from the PORT to the GAP; two-port
ABCD matrices: series Z -> [[1, Z], [0, 1]], shunt Z -> [[1, 0], [1/Z, 1]].
With V_s = 1:  Z_port = (A Z_A + B) / (C Z_A + D),  I1 = 1 / (Z0 + Z_port),
V1 = Z_port I1, and (reciprocal network, AD - BC = 1) V2 = D V1 - B I1,
I2 = -C V1 + A I1;  P_in = Re(V1 I1*)/2, P_A = Re(V2 I2*)/2, P_avail = 1/(8 Z0).

Topologies (elements from port to gap)
    a     : series C                          (cancels the input reactance only)
    b_sp  : shunt X1, series X2               (L section, shunt on the 50-Ohm side)
    b_ps  : series X1, shunt X2               (L section, shunt on the gap side)
    c_sp  : shunt X1, series X2, series C     (series C at the gap, then L section)
    c_ps  : series X1, shunt X2, series C
X1, X2: any kit part (capacitor 0603 / 0402, inductor chip 0402 / L402W);
C: kit capacitor. Parts that are electrically identical in the model (same
value and tolerance, e.g. a 1 pF 0603 and a 1 pF 0402) are merged into one
"electrical part" listing the physical alternatives.

Robust criterion
----------------
Monte-Carlo over the part tolerances (each value uniform in [nominal - tol,
nominal + tol], independent, n_mc draws with a fixed seed; the SAME draws for
every candidate -- common random numbers) times a frequency grid f0 (1 + o),
o in f_offsets (default -0.5 ... +0.5 %, 5 points): mean of G_r (linear) and
10th percentile. The capacitor ESR/ESL and inductor Q / C_par are passed
explicitly (ComponentModel / InductorModel), never defaulted.

Why the network loss grows with the Q of the load (order of magnitude)
---------------------------------------------------------------------
To match Z_A = R + jX the network must cancel X with a component of opposite
reactance in series with R. A component of quality factor Q_c = |X|/R_loss adds
R_loss = |X|/Q_c = R Q_X/Q_c (Q_X = |X|/R), so the fraction of the power lost is
R_loss/(R + R_loss) = 1/(1 + Q_c/Q_X) ~ Q_X/Q_c for Q_X << Q_c. For an
electrically small antenna far from resonance Q_X ~ Q_in = (w/2R)|dX/dw| and
the usual rule "loss ~ Q_in/Q_c" follows. When the antenna reactance varies
faster than that of a lumped element (Q_in > Q_X, e.g. a superdirective array
near its own resonances) the rule overestimates the loss at f0, while Q_in
still sets the bandwidth (|S11| <= -10 dB: fractional band ~ (S-1)/(Q_in sqrt(S)),
S = VSWR = 1.925). Approximations only: they ignore the impedance-transforming
section and the frequency dependence of Q_c.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import CubicSpline

from solver.optim.components import (
    CAPACITOR_KITS, INDUCTOR_KIT, Capacitor, ComponentModel, Inductor, InductorModel,
    capacitor_impedance, inductor_impedance,
)

Z0 = 50.0
TOPOLOGIES: Dict[str, Tuple[str, ...]] = {
    "a": ("series",),
    "b_sp": ("shunt", "series"),
    "b_ps": ("series", "shunt"),
    "c_sp": ("shunt", "series", "series"),
    "c_ps": ("series", "shunt", "series"),
}
TOPOLOGY_TEXT = {
    "a": "series C at the gap",
    "b_sp": "L section: shunt (port side) + series (gap side)",
    "b_ps": "L section: series (port side) + shunt (gap side)",
    "c_sp": "shunt (port side) + series + series C (gap)",
    "c_ps": "series (port side) + shunt + series C (gap)",
}
F_OFFSETS_ROBUST = tuple(np.linspace(-0.005, 0.005, 5))


# =============================================================================
# Antenna one-port
# =============================================================================
class AntennaOnePort:
    """Z_A(f) and G_A(f) (linear) interpolated by cubic splines in (f/f0 - 1); G in dB."""

    def __init__(self, f_hz, zin, gain_lin, f0):
        f_hz = np.asarray(f_hz, float)
        order = np.argsort(f_hz)
        self.f_hz = f_hz[order]
        self.zin_data = np.asarray(zin, complex)[order]
        self.gain_data = np.asarray(gain_lin, float)[order]
        self.f0 = float(f0)
        x = self.f_hz / self.f0 - 1
        self._re = CubicSpline(x, self.zin_data.real)
        self._im = CubicSpline(x, self.zin_data.imag)
        self._g = CubicSpline(x, 10 * np.log10(self.gain_data))
        self.f_min, self.f_max = float(self.f_hz[0]), float(self.f_hz[-1])
        self.dz = 0j

    @classmethod
    def from_freq_rows(cls, rows, f0):
        """rows: dicts with f_GHz, Zin_re_ohm, Zin_im_ohm, G_dBi (campaign post-checks / V6 CSV)."""
        return cls([float(r["f_GHz"]) * 1e9 for r in rows],
                   [float(r["Zin_re_ohm"]) + 1j * float(r["Zin_im_ohm"]) for r in rows],
                   [10 ** (float(r["G_dBi"]) / 10) for r in rows], f0)

    def _x(self, f):
        f = np.asarray(f, float)
        tol = 1e-9 * self.f0
        if np.any(f < self.f_min - tol) or np.any(f > self.f_max + tol):
            raise ValueError(f"frequency outside the MoM data range [{self.f_min / 1e9:.4f}, "
                             f"{self.f_max / 1e9:.4f}] GHz (no extrapolation)")
        return f / self.f0 - 1

    def zin(self, f):
        x = self._x(f)
        return self._re(x) + 1j * self._im(x) + self.dz

    def gain(self, f):
        return 10 ** (self._g(self._x(f)) / 10)

    def shifted(self, dz):
        """Copy with Z_A(f) + dz at every frequency (e.g. the fine-mesh shift at f0)."""
        other = AntennaOnePort(self.f_hz, self.zin_data, self.gain_data, self.f0)
        other.dz = complex(dz)
        return other

    def leave_one_out(self):
        """Interpolation error estimate: drop each interior point, re-spline, compare."""
        out = []
        for k in range(1, len(self.f_hz) - 1):
            keep = np.arange(len(self.f_hz)) != k
            sub = AntennaOnePort(self.f_hz[keep], self.zin_data[keep], self.gain_data[keep], self.f0)
            out.append(dict(f_GHz=self.f_hz[k] / 1e9, dZ_ohm=abs(sub.zin(self.f_hz[k]) - self.zin_data[k]),
                            dG_dB=float(10 * np.log10(sub.gain(self.f_hz[k]) / self.gain_data[k]))))
        return out


# =============================================================================
# Parts and networks
# =============================================================================
@dataclass(frozen=True)
class ElectricalPart:
    """One value/tolerance as seen by the model, with the physical kit parts that realise it."""
    is_inductor: bool
    value: float
    tol_abs: float
    alternatives: Tuple = ()

    @property
    def label(self) -> str:
        return self.alternatives[0].label

    @property
    def tol_rel(self) -> float:
        return self.tol_abs / self.value

    def describe(self) -> str:
        sizes = "/".join(sorted({p.size for p in self.alternatives}))
        return f"{self.label} ({'L' if self.is_inductor else 'C'} {sizes}, +-{100 * self.tol_rel:.0f}%)"


def electrical_parts(parts: Sequence) -> List[ElectricalPart]:
    groups: Dict[tuple, list] = {}
    for p in parts:
        key = (isinstance(p, Inductor), round(p.value * 1e15), round(p.tol_abs * 1e18))
        groups.setdefault(key, []).append(p)
    out = [ElectricalPart(isinstance(g[0], Inductor), g[0].value, g[0].tol_abs, tuple(g)) for g in groups.values()]
    return sorted(out, key=lambda e: (e.is_inductor, e.value, e.tol_abs))


@dataclass(frozen=True)
class Network:
    topology: str
    parts: Tuple[ElectricalPart, ...]          # port -> gap, one per position of TOPOLOGIES[topology]

    @property
    def positions(self):
        return TOPOLOGIES[self.topology]

    @property
    def has_inductor(self) -> bool:
        return any(p.is_inductor for p in self.parts)

    def describe(self) -> str:
        return " | ".join(f"{pos} {p.describe()}" for pos, p in zip(self.positions, self.parts))

    def key(self):
        """Electrical equivalence key: adjacent series elements commute."""
        groups, cur = [], None
        for pos, p in zip(self.positions, self.parts):
            item = (p.is_inductor, p.value, p.tol_abs)
            if pos == "series" and cur is not None and cur[0] == "series":
                cur[1].append(item)
            else:
                cur = [pos, [item]]
                groups.append(cur)
        return tuple((g[0], tuple(sorted(g[1]))) for g in groups)


# =============================================================================
# Circuit core (vectorised)
# =============================================================================
def _element_z(is_ind, value, f, cap_model, ind_model):
    is_ind = np.asarray(is_ind, bool)
    zc = capacitor_impedance(value, f, cap_model) if not np.all(is_ind) else None
    zl = inductor_impedance(value, f, ind_model) if np.any(is_ind) else None
    if zc is None:
        return zl
    if zl is None:
        return zc
    return np.where(is_ind, zl, zc)


def abcd_cascade(positions, zs):
    """ABCD of series/shunt elements listed port -> load (broadcast over the arrays in zs)."""
    A, B, C, D = 1.0 + 0j, 0j, 0j, 1.0 + 0j
    for pos, z in zip(positions, zs):
        if pos == "series":
            A, B, C, D = A, A * z + B, C, C * z + D
        elif pos == "shunt":
            y = 1.0 / z
            A, B, C, D = A + B * y, B, C + D * y, D
        else:
            raise ValueError(pos)
    return A, B, C, D


def two_port_response(A, B, C, D, z_load, z0=Z0):
    """S11, power ratios and port impedance for a generator (V_s = 1, Z0) -> network -> z_load."""
    zp = (A * z_load + B) / (C * z_load + D)
    i1 = 1.0 / (z0 + zp)
    v1 = zp * i1
    v2 = D * v1 - B * i1
    i2 = -C * v1 + A * i1
    p_avail = 1.0 / (8 * z0)
    p_in = 0.5 * np.real(v1 * np.conj(i1))
    p_load = 0.5 * np.real(v2 * np.conj(i2))
    return dict(z_port=zp, s11=(zp - z0) / (zp + z0), p_in=p_in / p_avail, p_load=p_load / p_avail,
                eta_net=p_load / p_in, v2=v2, i2=i2)


def respond(positions, is_ind, values, f, antenna: AntennaOnePort, cap_model, ind_model, z0=Z0):
    """
    Network response for value arrays `values` (..., n_el) (is_ind broadcastable to it) and
    frequencies f broadcastable to values[..., 0]. Returns two_port_response + G_r (linear).
    """
    values = np.asarray(values, float)
    is_ind = np.broadcast_to(np.asarray(is_ind, bool), values.shape)
    f = np.asarray(f, float)
    zs = [_element_z(is_ind[..., k], values[..., k], f, cap_model, ind_model) for k in range(len(positions))]
    r = two_port_response(*abcd_cascade(positions, zs), antenna.zin(f), z0)
    r["G_r"] = antenna.gain(f) * r["p_load"]
    return r


def network_response(net: Network, f, antenna, cap_model, ind_model, values=None, z0=Z0):
    vals = np.array([p.value for p in net.parts]) if values is None else values
    return respond(net.positions, [p.is_inductor for p in net.parts], vals, f, antenna, cap_model, ind_model, z0)


# =============================================================================
# Robust metrics
# =============================================================================
def tolerance_draws(n_mc, n_slots=3, seed=0):
    """Common random numbers u in [-1, 1] (n_mc, n_slots): value = nominal + u * tol."""
    return np.random.default_rng(seed).uniform(-1.0, 1.0, size=(n_mc, n_slots))


def _db(x):
    x = np.asarray(x, float)
    return np.where(x > 0, 10 * np.log10(np.maximum(x, 1e-300)), -np.inf)


def robust_stats(positions, is_ind, nominal, tol, f0, antenna, cap_model, ind_model, u,
                 f_offsets=F_OFFSETS_ROBUST, z0=Z0, chunk=50):
    """
    nominal, tol, is_ind: (n_cand, n_el). Returns (mean G_r, 10th percentile G_r), linear,
    over the draws u (n_mc, >= n_el) x frequencies f0 (1 + f_offsets).
    """
    n_el = len(positions)
    f = f0 * (1 + np.asarray(f_offsets))[None, None, :]
    means, p10s = [], []
    for s in range(0, len(nominal), chunk):
        sl = slice(s, s + chunk)
        vals = nominal[sl, None, None, :] + u[None, :, None, :n_el] * tol[sl, None, None, :]
        g = respond(positions, is_ind[sl, None, None, :], vals, f, antenna, cap_model, ind_model, z0)["G_r"]
        g = g.reshape(g.shape[0], -1)
        means.append(g.mean(axis=1))
        p10s.append(np.percentile(g, 10, axis=1))
    return np.concatenate(means), np.concatenate(p10s)


def band_10db(net, antenna, cap_model, ind_model, n=801, level_db=-10.0, z0=Z0):
    """Contiguous band around f0 where |S11| <= level (nominal values), within the data range."""
    f = np.linspace(antenna.f_min, antenna.f_max, n)
    s = 20 * np.log10(np.maximum(np.abs(network_response(net, f, antenna, cap_model, ind_model, z0=z0)["s11"]),
                                 1e-15))
    k0 = int(np.argmin(np.abs(f - antenna.f0)))
    if s[k0] > level_db:
        return dict(f_lo_GHz=None, f_hi_GHz=None, bw_MHz=0.0, bw_pct=0.0, clipped=False)
    lo = k0
    while lo > 0 and s[lo - 1] <= level_db:
        lo -= 1
    hi = k0
    while hi < n - 1 and s[hi + 1] <= level_db:
        hi += 1

    def edge(i, j):          # linear interpolation of the crossing between samples i (in) and j (out)
        return f[i] + (f[j] - f[i]) * (level_db - s[i]) / (s[j] - s[i])

    f_lo = f[0] if lo == 0 else edge(lo, lo - 1)
    f_hi = f[-1] if hi == n - 1 else edge(hi, hi + 1)
    return dict(f_lo_GHz=f_lo / 1e9, f_hi_GHz=f_hi / 1e9, bw_MHz=(f_hi - f_lo) / 1e6,
                bw_pct=100 * (f_hi - f_lo) / antenna.f0, clipped=bool(lo == 0 or hi == n - 1))


def evaluate_solution(net, antenna, cap_model, ind_model, u, f_offsets=F_OFFSETS_ROBUST, z0=Z0):
    """Full metric record of one network (nominal at f0, robust mean / P10, band, losses)."""
    f0 = antenna.f0
    r = network_response(net, f0, antenna, cap_model, ind_model, z0=z0)
    pos, is_ind = net.positions, np.array([[p.is_inductor for p in net.parts]])
    nom = np.array([[p.value for p in net.parts]])
    tol = np.array([[p.tol_abs for p in net.parts]])
    mean, p10 = robust_stats(pos, is_ind, nom, tol, f0, antenna, cap_model, ind_model, u, f_offsets, z0)
    s11 = abs(complex(r["s11"]))
    return dict(
        topology=net.topology, description=net.describe(), has_inductor=net.has_inductor,
        G_r_nominal_dBi=float(_db(r["G_r"])), G_r_mean_dBi=float(_db(mean[0])), G_r_p10_dBi=float(_db(p10[0])),
        S11_dB=float(20 * np.log10(s11)), mismatch_dB=float(_db(1 - s11 ** 2)),
        network_loss_dB=float(-_db(r["eta_net"])), eta_net=float(r["eta_net"]),
        z_port_ohm=[float(np.real(r["z_port"])), float(np.imag(r["z_port"]))],
        band=band_10db(net, antenna, cap_model, ind_model, z0=z0))


# =============================================================================
# Search
# =============================================================================
@dataclass
class SearchResult:
    counts: Dict[str, int]                     # value combinations with |S11(f0)| <= level, per topology
    counts_total: Dict[str, int]
    ranked: List[Tuple[Network, float, float]] = field(default_factory=list)   # (net, mean, p10) linear


def _candidate_indices(topology, n_parts, cap_idx):
    pos = TOPOLOGIES[topology]
    if topology == "a":
        return np.array(cap_idx)[:, None]
    grids = [np.arange(n_parts)] * 2 + ([np.array(cap_idx)] if len(pos) == 3 else [])
    mesh = np.meshgrid(*grids, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=1)


def search(antenna: AntennaOnePort, cap_model: ComponentModel, ind_model: InductorModel,
           capacitors=CAPACITOR_KITS, inductors=INDUCTOR_KIT, topologies=tuple(TOPOLOGIES),
           s11_max_db=-10.0, n_pre=300, n_mc=500, seed=0, f_offsets=F_OFFSETS_ROBUST,
           n_keep=10, z0=Z0) -> SearchResult:
    """
    1. every value combination of every topology at f0 (nominal values): keep |S11| <= s11_max_db;
    2. per topology, the n_pre best by nominal G_r averaged over f_offsets;
    3. Monte-Carlo (common draws, seed) -> robust mean and P10 of G_r; rank by mean,
       merge electrically equivalent networks, keep n_keep.
    """
    parts = electrical_parts(list(capacitors) + list(inductors))
    is_ind_all = np.array([p.is_inductor for p in parts])
    val_all = np.array([p.value for p in parts])
    tol_all = np.array([p.tol_abs for p in parts])
    cap_idx = [k for k, p in enumerate(parts) if not p.is_inductor]
    u = tolerance_draws(n_mc, 3, seed)
    f0 = antenna.f0
    counts, totals, pool = {}, {}, []
    for topo in topologies:
        pos = TOPOLOGIES[topo]
        idx = _candidate_indices(topo, len(parts), cap_idx)
        r = respond(pos, is_ind_all[idx], val_all[idx], f0, antenna, cap_model, ind_model, z0)
        ok = 20 * np.log10(np.abs(r["s11"])) <= s11_max_db
        totals[topo], counts[topo] = int(len(idx)), int(ok.sum())
        idx = idx[ok]
        if not len(idx):
            continue
        fo = f0 * (1 + np.asarray(f_offsets))[None, :]
        g = respond(pos, is_ind_all[idx][:, None, :], val_all[idx][:, None, :], fo, antenna,
                    cap_model, ind_model, z0)["G_r"].mean(axis=1)
        idx = idx[np.argsort(-g)[:n_pre]]
        mean, p10 = robust_stats(pos, is_ind_all[idx], val_all[idx], tol_all[idx], f0, antenna,
                                 cap_model, ind_model, u, f_offsets, z0)
        for row, m, p in zip(idx, mean, p10):
            pool.append((Network(topo, tuple(parts[i] for i in row)), float(m), float(p)))
    pool.sort(key=lambda t: -t[1])
    seen, ranked = set(), []
    for net, m, p in pool:
        k = net.key()
        if k in seen:
            continue
        seen.add(k)
        ranked.append((net, m, p))
    return SearchResult(counts=counts, counts_total=totals, ranked=ranked[:n_keep] if n_keep else ranked)


def part_records(net: Network):
    """JSON-ready description of the elements (port -> gap) with every physical alternative."""
    out = []
    for pos, p in zip(net.positions, net.parts):
        out.append(dict(
            position=pos, type="inductor" if p.is_inductor else "capacitor", label=p.label,
            value=p.value, unit="H" if p.is_inductor else "F", tolerance_abs=p.tol_abs, tolerance_rel=p.tol_rel,
            kits=[dict(kit=a.kit, size=a.size) for a in p.alternatives]))
    return out
