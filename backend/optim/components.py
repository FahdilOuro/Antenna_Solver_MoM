"""
Real lumped components for the loaded ports: kit catalogue, series model, snapping.

Series model of a chip capacitor (no hidden defaults: ESR and ESL are always
passed explicitly through a ComponentModel)

    Z(f) = R_ESR + j w L_ESL + 1 / (j w C)

Kits
----
JOHANSON_S603DS : Johanson S-Series 0603 NP0/C0G kit available at LEAT
(notes_comprehension/06_composants_disponibles.md, photo of 5/11):
    0.3 ... 4.7 pF  +- 0.10 pF
    5.6, 6.8, 8.2 pF  +- 0.25 pF
    10 ... 82 pF  +- 5 %
INDUCTOR_KIT : empty until the inductor kit values are known. A port whose
continuous optimum is inductive (X > 0) is NOT rounded to a capacitor: it is
kept as an ideal reactance and flagged "inductor needed: kit to be confirmed".

ESR / ESL: PROVISIONAL sensitivity grids (ESR in {0, 0.1, 0.3} Ohm, ESL in
{0, 0.5, 1.0} nH), to be replaced by values read on the Johanson curves.

Snapping (snap_to_kit): for each loaded port, the (up to) two kit values whose
effective reactance at f0 brackets the continuous optimum; all combinations
(at most 2^(P-1)) are scored by a caller-supplied objective (the robust gain of
backend.optim.design_eval) and the best one is kept.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- parts
@dataclass(frozen=True)
class Capacitor:
    """Kit capacitor: nominal value (F) and absolute tolerance half-range (F)."""
    value: float
    tol_abs: float
    label: str

    @property
    def tol_rel(self) -> float:
        return self.tol_abs / self.value

    def value_at(self, tol_sign: int = 0) -> float:
        """C at the low (-1), nominal (0) or high (+1) end of the tolerance."""
        return self.value + tol_sign * self.tol_abs


@dataclass(frozen=True)
class Inductor:
    """Kit inductor (structure ready for the inductor kit; no values yet)."""
    value: float
    tol_abs: float
    label: str

    @property
    def tol_rel(self) -> float:
        return self.tol_abs / self.value

    def value_at(self, tol_sign: int = 0) -> float:
        return self.value + tol_sign * self.tol_abs


@dataclass(frozen=True)
class IdealReactance:
    """Continuous reactance kept as is (inductive optimum, or out of the kit range).
    Frequency behaviour: an inductor L = X/w0 if X > 0, a capacitor C = -1/(w0 X) if X < 0."""
    x_ohm: float
    f0: float
    note: str

    @property
    def label(self) -> str:
        w0 = 2 * np.pi * self.f0
        if self.x_ohm > 0:
            return f"ideal L = {self.x_ohm / w0 * 1e9:.4g} nH ({self.note})"
        if self.x_ohm < 0:
            return f"ideal C = {-1 / (w0 * self.x_ohm) * 1e12:.4g} pF ({self.note})"
        return f"short ({self.note})"


def _cap(pf, tol_pf=None, tol_rel=None):
    tol = tol_pf * 1e-12 if tol_pf is not None else tol_rel * pf * 1e-12
    return Capacitor(value=pf * 1e-12, tol_abs=tol, label=f"{pf:g} pF")


JOHANSON_S603DS: Tuple[Capacitor, ...] = tuple(
    [_cap(v, tol_pf=0.10) for v in (0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.6, 3.9, 4.7)]
    + [_cap(v, tol_pf=0.25) for v in (5.6, 6.8, 8.2)]
    + [_cap(v, tol_rel=0.05) for v in (10, 12, 15, 18, 20, 24, 27, 33, 39, 47, 68, 82)]
)

#: Inductor kit: to be filled when the values are known (Inductor(value_H, tol_abs_H, label)).
INDUCTOR_KIT: Tuple[Inductor, ...] = ()


# --------------------------------------------------------------------------- series model
@dataclass(frozen=True)
class ComponentModel:
    """Parasitics of a chip component, always explicit: ESR (Ohm), ESL (H)."""
    esr: float
    esl: float

    def __post_init__(self):
        if self.esr < 0 or self.esl < 0:
            raise ValueError("ESR and ESL must be >= 0.")


IDEAL = ComponentModel(esr=0.0, esl=0.0)
#: PROVISIONAL sensitivity grids -- to be replaced by values read on the Johanson S-Series curves.
PROVISIONAL_ESR_GRID = (0.0, 0.1, 0.3)
PROVISIONAL_ESL_GRID = (0.0, 0.5e-9, 1.0e-9)


def series_impedance(part, f, model: ComponentModel, tol_sign: int = 0) -> complex:
    """
    Z(f) = R_ESR + j w L_ESL + Z_part(f):
        Capacitor  -> 1/(j w C)        (C at the requested tolerance end)
        Inductor   -> j w L
        IdealReactance -> X scaled as an ideal L or C fixed at its f0 (no parasitics added)
    """
    w = 2 * np.pi * f
    if isinstance(part, IdealReactance):
        if part.x_ohm == 0:
            return 0j
        w0 = 2 * np.pi * part.f0
        return 1j * part.x_ohm * (w / w0 if part.x_ohm > 0 else w0 / w)
    z = model.esr + 1j * w * model.esl
    if isinstance(part, Capacitor):
        return z + 1.0 / (1j * w * part.value_at(tol_sign))
    if isinstance(part, Inductor):
        return z + 1j * w * part.value_at(tol_sign)
    raise TypeError(f"Unknown part {part!r}")


def effective_reactance(part, f, model: ComponentModel) -> float:
    return float(np.imag(series_impedance(part, f, model)))


# --------------------------------------------------------------------------- snapping
def bracketing_parts(x_target, f0, model: ComponentModel, kit: Sequence[Capacitor] = JOHANSON_S603DS):
    """
    (parts, note): parts whose effective reactance at f0 brackets x_target.
        x_target >= 0 -> [IdealReactance], note "inductor needed: kit to be confirmed"
                         (never rounded to a capacitor)
        x_target outside the kit's capacitive range -> [the end value of the kit], note
        otherwise -> the two neighbouring kit capacitors (one if exactly equal), note None
    """
    if x_target >= 0:
        p = IdealReactance(x_target, f0, "inductor needed: kit to be confirmed")
        return [p], p.label
    parts = sorted(kit, key=lambda c: effective_reactance(c, f0, model))   # most negative first
    xs = np.array([effective_reactance(c, f0, model) for c in parts])
    below = np.flatnonzero(xs <= x_target)
    above = np.flatnonzero(xs >= x_target)
    if not below.size:
        return [parts[0]], (f"X = {x_target:.1f} Ohm is beyond the smallest kit value "
                            f"({parts[0].label}, {xs[0]:.1f} Ohm): end value used")
    if not above.size:
        return [parts[-1]], (f"X = {x_target:.1f} Ohm is beyond the largest kit value "
                             f"({parts[-1].label}, {xs[-1]:.1f} Ohm): end value used")
    out = [parts[below[-1]]]
    if above[0] != below[-1]:
        out.append(parts[above[0]])
    return out, None


@dataclass
class SnapResult:
    """best: chosen parts (one per loaded port); score of each evaluated combination."""
    best: Tuple
    best_score: float
    combinations: List[Tuple[Tuple, float]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def snap_to_kit(x_opt, f0, model: ComponentModel, score: Callable[[Tuple], float],
                kit: Sequence[Capacitor] = JOHANSON_S603DS) -> SnapResult:
    """
    Evaluate every combination of bracketing parts (<= 2^(P-1) when all optima are
    capacitive and inside the kit range) with `score(parts) -> float` (higher is
    better) and keep the best. Ports needing an inductor keep their continuous
    reactance (IdealReactance) and are listed in `notes`.
    """
    options, notes = [], []
    for k, x in enumerate(np.atleast_1d(x_opt)):
        opts, note = bracketing_parts(float(x), f0, model, kit)
        options.append(opts)
        if note:
            notes.append(f"loaded port {k}: {note}")
    combos = []
    for parts in itertools.product(*options):
        combos.append((parts, float(score(parts))))
    best, best_score = max(combos, key=lambda c: c[1])
    return SnapResult(best=best, best_score=best_score, combinations=combos, notes=notes)
