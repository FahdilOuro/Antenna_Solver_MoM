"""
Johanson R14S manufacturer data (jtisoft, 1 GHz) in solver.optim.components (step 9, part A).
"""

from pathlib import Path

import numpy as np
import pytest

from solver.optim.components import (
    CAPACITOR_KITS, JOHANSON_R14S_EXTRAPOLATED, JOHANSON_R14S_RF_1GHZ, ComponentModel, TabulatedCapacitorModel,
    capacitor_impedance, johanson_r14s_model,
)

EXPECTED = {   # C pF: (part number, ESR Ohm, ESL nH, SRF GHz)  -- step-9 table
    0.5: ("251R14S0R5BV4T", 0.1455, 0.38, 11.56),
    1.0: ("251R14S1R0BV4T", 0.1459, 0.25, 10.00),
    1.2: ("251R14S1R2BV4T", 0.1461, 0.28, 8.70),
    2.7: ("251R14S2R7BV4T", 0.1434, 0.19, 7.03),
}
PUBLICATIONS = Path(r"D:\PhD_Fahdilou_Leat")


def test_four_rows_with_source():
    rows = {r.value_pf: r for r in JOHANSON_R14S_RF_1GHZ}
    assert set(rows) == set(EXPECTED)
    for c, (pn, esr, esl, srf) in EXPECTED.items():
        r = rows[c]
        assert r.part_number == pn and r.f_hz == 1e9
        assert abs(r.esr_ohm - esr) < 1e-3 and r.esl_nh == esl and r.srf_ghz == srf
        assert r.source.endswith(f"{c:.1f}pF - MLC Soft _ Johanson Technology.pdf")
        assert "Fiche_johanson_technologie" in r.source and "1000 MHz" in r.conditions


@pytest.mark.skipif(not PUBLICATIONS.is_dir(), reason="PUBLICATIONS folder not on this machine")
def test_source_pdfs_exist():
    for r in JOHANSON_R14S_RF_1GHZ:
        assert (PUBLICATIONS / r.source).is_file(), r.source


def test_series_model_reproduces_jtisoft():
    """|Z| at 1 GHz and SRF = 1 / (2 pi sqrt(L C)) of the series model match the jtisoft pages."""
    m = johanson_r14s_model()
    for r in JOHANSON_R14S_RF_1GHZ:
        c = r.value_pf * 1e-12
        z = capacitor_impedance(c, 1e9, m)
        assert abs(abs(z) / r.z_mag_ohm - 1) < 2e-3
        assert abs(z.real - r.esr_ohm) < 1e-12
        srf = 1 / (2 * np.pi * np.sqrt(r.esl_nh * 1e-9 * c)) / 1e9
        assert abs(srf / r.srf_ghz - 1) < 0.01


def test_lookup_keeps_nominal_parasitics_within_tolerance():
    m = johanson_r14s_model()
    assert isinstance(m, TabulatedCapacitorModel)
    for r in JOHANSON_R14S_RF_1GHZ:
        c = r.value_pf * 1e-12
        cs = c + np.array([-0.0999, 0.0, 0.0999]) * 1e-12                 # +- 0.1 pF kit tolerance
        esr, esl = m.parasitics(cs)
        assert np.allclose(esr, r.esr_ohm) and np.allclose(esl, r.esl_nh * 1e-9)
    # a kit value without a jtisoft page -> extrapolated values
    esr, esl = m.parasitics(np.array([0.8e-12, 1.5e-12, 2.2e-12, 33e-12]))
    assert np.allclose(esr, JOHANSON_R14S_EXTRAPOLATED["esr_ohm"])
    assert np.allclose(esl, JOHANSON_R14S_EXTRAPOLATED["esl_nh"] * 1e-9)
    assert "extrapolated" in JOHANSON_R14S_EXTRAPOLATED["status"]
    assert set(m.kit_values) == {c.value for c in CAPACITOR_KITS}


def test_scalar_model_unchanged():
    z = capacitor_impedance(1e-12, 1e9, ComponentModel(0.1, 0.5e-9))
    w = 2 * np.pi * 1e9
    assert z == pytest.approx(0.1 + 1j * w * 0.5e-9 + 1 / (1j * w * 1e-12))
