"""
Real lumped components for the loaded ports and the matching network: kit
catalogue, circuit models, snapping.

Circuit models (no hidden defaults: the parasitics are always passed explicitly)

    capacitor (ComponentModel(esr, esl)):
        Z(f) = R_ESR + j w L_ESL + 1 / (j w C)
    inductor (InductorModel(q, c_par)), constant-Q series branch shunted by the
    parasitic capacitance (self-resonance f_SRF = 1 / (2 pi sqrt(L C_par))):
        Z_s(f) = w L / Q + j w L,      Z(f) = Z_s / (1 + j w C_par Z_s)

Kits (notes_comprehension/06_composants_disponibles.md)
----
JOHANSON_S603DS : Johanson S-Series 0603 NP0/C0G (photo of 5/11):
    0.3 ... 4.7 pF  +- 0.10 pF ;  5.6, 6.8, 8.2 pF  +- 0.25 pF ;  10 ... 82 pF  +- 5 %
JOHANSON_S402 : capacitors of the Johanson IOT-L/C402D-1A kit, S-Series 0402 (6/11):
    0.3 ... 4.7 pF  +- 0.10 pF (no 3.6 pF) ;  5.6 pF  +- 0.25 pF ;  6.8 pF  +- 5 %
JOHANSON_CHIP_L0402 : chip inductors of the same kit, 0402 (6/11):
    1.0 ... 4.7 nH  +- 0.3 nH ;  5.6 ... 15 nH  +- 5 %
JOHANSON_L402W : Johanson L402W wire-wound high-frequency inductors, 0402 (6/11):
    1.0 ... 5.6 nH  +- 0.2 nH ;  6.8 ... 120 nH  +- 5 %
INDUCTOR_KIT = JOHANSON_CHIP_L0402 + JOHANSON_L402W.

The inductor Q and parasitic capacitance at 1 GHz are NOT known yet (to be read
on the Johanson datasheets): InductorModel has no default, and the grids below
are declared RANGES for sensitivity studies, not data. For the loaded ports of
the antenna, an inductive continuous optimum (X > 0) is still NOT rounded to a
kit inductor: it is kept as an ideal reactance and flagged "inductor needed:
kit to be confirmed" (until the inductor Q is known).

Capacitor ESR / ESL: PROVISIONAL sensitivity grids (ESR in {0, 0.1, 0.3} Ohm,
ESL in {0, 0.5, 1.0} nH). MANUFACTURER DATA (step 9): JOHANSON_R14S_RF_1GHZ, the
jtisoft values at 1 GHz of the S-Series 0603 (R14S) 0.5 / 1.0 / 1.2 / 2.7 pF (source
PDF per value in Fiche_johanson_technologie/); the other kit values use ESR =
0.145 Ohm, ESL = 0.3 nH, EXTRAPOLATED from those four. johanson_r14s_model() gives
the per-value model (TabulatedCapacitorModel) accepted by capacitor_impedance.

Snapping (snap_to_kit): for each loaded port, the (up to) two kit values whose
effective reactance at f0 brackets the continuous optimum; all combinations
(at most 2^(P-1)) are scored by a caller-supplied objective (the robust gain of
solver.optim.design_eval) and the best one is kept.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- parts
@dataclass(frozen=True)
class Capacitor:
    """Kit capacitor: nominal value (F), absolute tolerance half-range (F), kit name and EIA size."""
    value: float
    tol_abs: float
    label: str
    kit: str = ""
    size: str = ""

    @property
    def tol_rel(self) -> float:
        return self.tol_abs / self.value

    def value_at(self, tol_sign: int = 0) -> float:
        """C at the low (-1), nominal (0) or high (+1) end of the tolerance."""
        return self.value + tol_sign * self.tol_abs


@dataclass(frozen=True)
class Inductor:
    """Kit inductor: nominal value (H), absolute tolerance half-range (H), kit name and EIA size."""
    value: float
    tol_abs: float
    label: str
    kit: str = ""
    size: str = ""

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


KIT_S603DS = "Johanson S603DS (S-Series 0603 C0G)"
KIT_S402 = "Johanson IOT-L/C402D-1A, S-Series 0402 capacitors"
KIT_CHIP_L0402 = "Johanson IOT-L/C402D-1A, chip inductors 0402"
KIT_L402W = "Johanson L402W wire-wound inductors 0402"


def _cap(pf, tol_pf=None, tol_rel=None, kit=KIT_S603DS, size="0603"):
    tol = tol_pf * 1e-12 if tol_pf is not None else tol_rel * pf * 1e-12
    return Capacitor(value=pf * 1e-12, tol_abs=tol, label=f"{pf:g} pF", kit=kit, size=size)


def _ind(nh, tol_nh=None, tol_rel=None, kit=KIT_L402W, size="0402"):
    tol = tol_nh * 1e-9 if tol_nh is not None else tol_rel * nh * 1e-9
    return Inductor(value=nh * 1e-9, tol_abs=tol, label=f"{nh:g} nH", kit=kit, size=size)


JOHANSON_S603DS: Tuple[Capacitor, ...] = tuple(
    [_cap(v, tol_pf=0.10) for v in (0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.6, 3.9, 4.7)]
    + [_cap(v, tol_pf=0.25) for v in (5.6, 6.8, 8.2)]
    + [_cap(v, tol_rel=0.05) for v in (10, 12, 15, 18, 20, 24, 27, 33, 39, 47, 68, 82)]
)

JOHANSON_S402: Tuple[Capacitor, ...] = tuple(
    [_cap(v, tol_pf=0.10, kit=KIT_S402, size="0402")
     for v in (0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7)]
    + [_cap(5.6, tol_pf=0.25, kit=KIT_S402, size="0402"), _cap(6.8, tol_rel=0.05, kit=KIT_S402, size="0402")]
)

JOHANSON_CHIP_L0402: Tuple[Inductor, ...] = tuple(
    [_ind(v, tol_nh=0.3, kit=KIT_CHIP_L0402) for v in (1.0, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7)]
    + [_ind(v, tol_rel=0.05, kit=KIT_CHIP_L0402) for v in (5.6, 6.8, 8.2, 10, 12, 15)]
)

JOHANSON_L402W: Tuple[Inductor, ...] = tuple(
    [_ind(v, tol_nh=0.2) for v in (1.0, 1.8, 2.0, 2.7, 3.3, 3.9, 4.7, 5.6)]
    + [_ind(v, tol_rel=0.05) for v in (6.8, 7.5, 8.2, 9.0, 10, 12, 15, 18, 20, 22, 24, 27, 30, 33, 39, 47,
                                       56, 82, 100, 120)]
)

CAPACITOR_KITS: Tuple[Capacitor, ...] = JOHANSON_S603DS + JOHANSON_S402
INDUCTOR_KIT: Tuple[Inductor, ...] = JOHANSON_CHIP_L0402 + JOHANSON_L402W


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


@dataclass(frozen=True)
class InductorModel:
    """
    Parasitics of a chip inductor, both MANDATORY (no default):
        q     : quality factor w L / R of the series branch (constant-Q model; np.inf = lossless)
        c_par : parallel (self-resonance) capacitance, F (0 = not modelled)
    """
    q: float
    c_par: float

    def __post_init__(self):
        if not self.q > 0:
            raise ValueError("Inductor Q must be > 0 (np.inf for a lossless inductor).")
        if self.c_par < 0:
            raise ValueError("Parasitic capacitance must be >= 0.")


#: Inductor Q at 1 GHz: a RANGE for the sensitivity study, NOT data (to be read on the
#: Johanson chip-0402 / L402W datasheets).
Q_L_RANGE = (20.0, 40.0, 80.0)
#: Parasitic capacitance of an 0402 inductor: RANGE, NOT data (from the datasheet SRF).
C_PAR_RANGE = (0.0, 0.1e-12)


# --------------------------------------------------------------------------- manufacturer data
@dataclass(frozen=True)
class CapacitorRFData:
    """RF parameters of one capacitor as given by the manufacturer's tool, at frequency f_hz."""
    value_pf: float
    part_number: str
    f_hz: float
    esr_ohm: float
    esl_nh: float
    srf_ghz: float
    q: float
    ceff_pf: float
    z_mag_ohm: float
    source: str
    conditions: str


_JTI_DIR = "PUBLICATIONS/7-EurAAP Small Antenna Super-Gain Challenge/Fiche_johanson_technologie"
_JTI_CONDITIONS = ("Johanson MLC Soft (jtisoft.johansontechnology.com), High-Q Capacitor Selection, "
                   "EIA 0603 R14S-series (S-Series), +-0.1 pF, 250 V, Ni/Sn; frequency 1000 MHz, "
                   "ambient 65 degC, max 125 degC; pages printed 07/11/2026 19:07-19:13")

#: MANUFACTURER DATA (not a range): Johanson S-Series 0603 (R14S) at 1 GHz, read on the jtisoft
#: pages saved as PDF in Fiche_johanson_technologie/ (one PDF per value).
JOHANSON_R14S_RF_1GHZ: Tuple[CapacitorRFData, ...] = (
    CapacitorRFData(0.5, "251R14S0R5BV4T", 1e9, 0.14547, 0.38, 11.56, 2172.92, 0.50, 316.1,
                    f"{_JTI_DIR}/0.5pF - MLC Soft _ Johanson Technology.pdf", _JTI_CONDITIONS),
    CapacitorRFData(1.0, "251R14S1R0BV4T", 1e9, 0.14591, 0.25, 10.00, 1080.44, 1.01, 157.6,
                    f"{_JTI_DIR}/1.0pF - MLC Soft _ Johanson Technology.pdf", _JTI_CONDITIONS),
    CapacitorRFData(1.2, "251R14S1R2BV4T", 1e9, 0.14608, 0.28, 8.70, 896.37, 1.22, 130.9,
                    f"{_JTI_DIR}/1.2pF - MLC Soft _ Johanson Technology.pdf", _JTI_CONDITIONS),
    CapacitorRFData(2.7, "251R14S2R7BV4T", 1e9, 0.14336, 0.19, 7.03, 403.05, 2.76, 57.8,
                    f"{_JTI_DIR}/2.7pF - MLC Soft _ Johanson Technology.pdf", _JTI_CONDITIONS),
)
#: Kit values without a jtisoft page: EXTRAPOLATED from the 4 measured values (not data).
JOHANSON_R14S_EXTRAPOLATED = dict(esr_ohm=0.145, esl_nh=0.3,
                                  status="extrapolated from the 4 measured values (0.5, 1.0, 1.2, 2.7 pF)")


@dataclass(frozen=True)
class TabulatedCapacitorModel:
    """
    Capacitor parasitics that depend on the nominal value: ESR / ESL of the kit value nearest
    to C (nearest-nominal lookup, so a value drawn within its tolerance keeps the parasitics
    of its nominal part: the kit tolerances never reach half the gap to the next value),
    with (default_esr, default_esl) for values absent from the table.
        Z(f) = R_ESR(C) + j w L_ESL(C) + 1 / (j w C)
    """
    values: Tuple[float, ...]            # F, tabulated nominal values
    esr: Tuple[float, ...]               # Ohm
    esl: Tuple[float, ...]               # H
    default_esr: float
    default_esl: float
    kit_values: Tuple[float, ...]        # F, every nominal of the kit (defines the lookup cells)
    label: str = ""

    def parasitics(self, c):
        c = np.asarray(c, float)
        kv = np.asarray(sorted(set(self.kit_values) | set(self.values)))
        nominal = kv[np.argmin(np.abs(c[..., None] - kv), axis=-1)]
        esr = np.full(nominal.shape, self.default_esr)
        esl = np.full(nominal.shape, self.default_esl)
        for v, r, l in zip(self.values, self.esr, self.esl):
            hit = np.isclose(nominal, v, rtol=1e-9, atol=0)
            esr = np.where(hit, r, esr)
            esl = np.where(hit, l, esl)
        return esr, esl


def johanson_r14s_model(kit: Sequence[Capacitor] = None) -> TabulatedCapacitorModel:
    """Johanson R14S manufacturer data at 1 GHz + extrapolated values for the rest of the kit."""
    kit = CAPACITOR_KITS if kit is None else kit
    d = JOHANSON_R14S_RF_1GHZ
    return TabulatedCapacitorModel(
        values=tuple(r.value_pf * 1e-12 for r in d), esr=tuple(r.esr_ohm for r in d),
        esl=tuple(r.esl_nh * 1e-9 for r in d), default_esr=JOHANSON_R14S_EXTRAPOLATED["esr_ohm"],
        default_esl=JOHANSON_R14S_EXTRAPOLATED["esl_nh"] * 1e-9,
        kit_values=tuple(sorted({c.value for c in kit})),
        label="Johanson R14S, jtisoft 1 GHz (0.5/1.0/1.2/2.7 pF) + extrapolated ESR 0.145 Ohm, ESL 0.3 nH")


def _cap_parasitics(model, c):
    if isinstance(model, TabulatedCapacitorModel):
        return model.parasitics(c)
    if isinstance(model, ComponentModel):
        return model.esr, model.esl
    raise TypeError("A capacitor needs a ComponentModel(esr, esl) or a TabulatedCapacitorModel.")


def capacitor_impedance(c, f, model):
    """R_ESR + j w L_ESL + 1/(j w C); c and f broadcast (numpy arrays allowed).
    model: ComponentModel (one ESR / ESL) or TabulatedCapacitorModel (per nominal value)."""
    w = 2 * np.pi * np.asarray(f)
    esr, esl = _cap_parasitics(model, c)
    return esr + 1j * w * esl + 1.0 / (1j * w * np.asarray(c))


def inductor_impedance(l, f, model: InductorModel):
    """(w L/Q + j w L) shunted by C_par; l and f broadcast (numpy arrays allowed)."""
    if not isinstance(model, InductorModel):
        raise TypeError("An inductor needs an explicit InductorModel(q, c_par).")
    w = 2 * np.pi * np.asarray(f)
    xl = w * np.asarray(l)
    zs = xl / model.q + 1j * xl
    if model.c_par == 0:
        return zs
    return zs / (1 + 1j * w * model.c_par * zs)


def series_impedance(part, f, model, tol_sign: int = 0) -> complex:
    """
    Impedance of one part:
        Capacitor  -> capacitor_impedance with a ComponentModel (C at the requested tolerance end)
        Inductor   -> inductor_impedance with an InductorModel (L at the requested tolerance end)
        IdealReactance -> X scaled as an ideal L or C fixed at its f0 (no parasitics added)
    """
    w = 2 * np.pi * f
    if isinstance(part, IdealReactance):
        if part.x_ohm == 0:
            return 0j
        w0 = 2 * np.pi * part.f0
        return 1j * part.x_ohm * (w / w0 if part.x_ohm > 0 else w0 / w)
    if isinstance(part, Capacitor):
        if not isinstance(model, (ComponentModel, TabulatedCapacitorModel)):
            raise TypeError("A capacitor needs a ComponentModel(esr, esl) or a TabulatedCapacitorModel.")
        return complex(capacitor_impedance(part.value_at(tol_sign), f, model))
    if isinstance(part, Inductor):
        return complex(inductor_impedance(part.value_at(tol_sign), f, model))
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
