"""
Step 6b: lumped matching network (solver.optim.matching) and the 0402 kits / inductor
model of solver.optim.components.

  - lossless network: power conserved through the network (all topologies);
  - lossy network: the network loss is the ESR dissipation;
  - series C alone: analytic reflection coefficient and realised gain;
  - a known L section (25 Ohm -> 50 Ohm at 1 GHz) is found by the search;
  - -10 dB band against the analytic band of a series-resonant load;
  - inductor model: Q is mandatory, ESR = w L / Q, self-resonance;
  - kits, electrical merging, interpolation, Monte-Carlo reproducibility.
"""

import numpy as np
import pytest

from solver.optim.components import (
    CAPACITOR_KITS, IDEAL, INDUCTOR_KIT, JOHANSON_CHIP_L0402, JOHANSON_L402W, JOHANSON_S402,
    JOHANSON_S603DS, ComponentModel, Inductor, InductorModel, inductor_impedance, series_impedance,
)
from solver.optim.matching import (
    TOPOLOGIES, AntennaOnePort, Network, abcd_cascade, band_10db, electrical_parts, evaluate_solution,
    network_response, search, tolerance_draws, two_port_response,
)

F0 = 1.0e9
LOSSLESS_L = InductorModel(q=np.inf, c_par=0.0)


def flat_antenna(z, gain=1.0, span=0.02, n=9):
    f = F0 * (1 + np.linspace(-span, span, n))
    return AntennaOnePort(f, np.full(n, z, complex), np.full(n, gain), F0)


def _part(parts, label):
    return next(p for p in parts if p.label == label)


# ----------------------------------------------------------------------------- kits / models
def test_kits_and_electrical_merging():
    assert (len(JOHANSON_S603DS), len(JOHANSON_S402), len(JOHANSON_CHIP_L0402), len(JOHANSON_L402W)) == (28, 14, 14, 28)
    assert all(c.size == "0402" for c in JOHANSON_S402) and all(c.size == "0603" for c in JOHANSON_S603DS)
    eparts = electrical_parts(list(CAPACITOR_KITS) + list(INDUCTOR_KIT))
    one_pf = [e for e in eparts if not e.is_inductor and abs(e.value - 1e-12) < 1e-18]
    assert len(one_pf) == 1 and {a.size for a in one_pf[0].alternatives} == {"0402", "0603"}
    ten_nh = [e for e in eparts if e.is_inductor and abs(e.value - 10e-9) < 1e-15]
    assert len(ten_nh) == 1 and len(ten_nh[0].alternatives) == 2          # chip and L402W, both +-5 %
    one_nh = [e for e in eparts if e.is_inductor and abs(e.value - 1e-9) < 1e-15]
    assert len(one_nh) == 2                                                # +-0.3 nH (chip) vs +-0.2 nH (L402W)
    six8 = [e for e in eparts if not e.is_inductor and abs(e.value - 6.8e-12) < 1e-18]
    assert len(six8) == 2                                                  # +-0.25 pF (0603) vs +-5 % (0402)


def test_inductor_model_is_explicit():
    with pytest.raises(TypeError):
        InductorModel()                                                    # no default Q / C_par
    with pytest.raises(TypeError):
        series_impedance(_part(JOHANSON_L402W, "10 nH"), F0, IDEAL)       # a ComponentModel is not enough
    with pytest.raises(ValueError):
        InductorModel(q=0.0, c_par=0.0)
    L, w = 10e-9, 2 * np.pi * F0
    z = inductor_impedance(L, F0, InductorModel(q=40.0, c_par=0.0))
    assert z.real == pytest.approx(w * L / 40) and z.imag == pytest.approx(w * L)
    c = 0.1e-12                                                            # lossless: L_eff = L / (1 - w^2 L C)
    z = inductor_impedance(L, F0, InductorModel(q=np.inf, c_par=c))
    assert z.imag / w == pytest.approx(L / (1 - w ** 2 * L * c), rel=1e-12)


# ----------------------------------------------------------------------------- power balance
@pytest.mark.parametrize("topology", list(TOPOLOGIES))
def test_lossless_network_conserves_power(topology):
    rng = np.random.default_rng(3)
    eparts = electrical_parts(list(CAPACITOR_KITS) + list(INDUCTOR_KIT))
    caps = [e for e in eparts if not e.is_inductor]
    ant = flat_antenna(23.0 + 180j)
    for _ in range(20):
        parts = [eparts[rng.integers(len(eparts))] for _ in TOPOLOGIES[topology]]
        parts[-1] = caps[rng.integers(len(caps))] if topology in ("a", "c_sp", "c_ps") else parts[-1]
        r = network_response(Network(topology, tuple(parts)), F0, ant, IDEAL, LOSSLESS_L)
        assert r["p_load"] == pytest.approx(r["p_in"], rel=1e-10)
        assert r["p_in"] == pytest.approx(1 - abs(r["s11"]) ** 2, rel=1e-10)
        assert r["v2"] / r["i2"] == pytest.approx(23.0 + 180j, rel=1e-10)


def test_lossy_network_loss_is_esr_dissipation():
    # shunt C (ESR) + series L (Q): P_in - P_load = ESR |I_shunt|^2 / 2 + R_L |I_series|^2 / 2
    esr, q = 0.3, 30.0
    cm, lm = ComponentModel(esr, 0.0), InductorModel(q, 0.0)
    ep = electrical_parts([_part(JOHANSON_S603DS, "3.3 pF"), _part(JOHANSON_L402W, "3.9 nH")])
    net = Network("b_sp", (ep[0], ep[1]))
    ant = flat_antenna(25.0 + 0j)
    r = network_response(net, F0, ant, cm, lm)
    w = 2 * np.pi * F0
    v1 = r["z_port"] / (50 + r["z_port"])                    # V_s = 1
    i_shunt = v1 / series_impedance(_part(JOHANSON_S603DS, "3.3 pF"), F0, cm)
    p_avail = 1 / (8 * 50)
    p_loss = (0.5 * esr * abs(i_shunt) ** 2 + 0.5 * (w * 3.9e-9 / q) * abs(r["i2"]) ** 2) / p_avail
    assert r["p_in"] - r["p_load"] == pytest.approx(p_loss, rel=1e-10)
    assert r["eta_net"] < 1


# ----------------------------------------------------------------------------- series C alone
def test_series_capacitor_analytic_gamma():
    za, ga = 26.8 + 298.8j, 10 ** 0.976
    ant = flat_antenna(za, gain=ga)
    for label in ("0.5 pF", "0.8 pF", "1 pF"):
        c = _part(JOHANSON_S603DS, label)
        net = Network("a", tuple(electrical_parts([c])))
        r = network_response(net, F0, ant, IDEAL, LOSSLESS_L)
        z = za + 1 / (1j * 2 * np.pi * F0 * c.value)
        gamma = (z - 50) / (z + 50)
        assert complex(r["s11"]) == pytest.approx(gamma, rel=1e-12)
        assert float(r["G_r"]) == pytest.approx(ga * (1 - abs(gamma) ** 2), rel=1e-12)


def test_no_network_limit_matches_design_record():
    # G_r(50 Ohm) of the B3 1.00 GHz record (-2.73 dBi): a tiny series C -> infinite C limit
    za, g_db = 26.813059017820226 + 298.821418475942j, 9.761124898286589
    ant = flat_antenna(za, gain=10 ** (g_db / 10))
    huge = Network("a", tuple(electrical_parts([JOHANSON_S603DS[0].__class__(1.0, 0.0, "1 F")])))
    g = float(network_response(huge, F0, ant, IDEAL, LOSSLESS_L)["G_r"])
    assert 10 * np.log10(g) == pytest.approx(-2.7312294445031524, abs=1e-6)


# ----------------------------------------------------------------------------- search
def test_known_l_section_25_to_50_ohm_is_found():
    # analytic L section at 1 GHz: series X = +25 Ohm (3.98 nH), shunt B = 0.02 S (3.18 pF)
    ant = flat_antenna(25.0 + 0j)
    res = search(ant, IDEAL, LOSSLESS_L, topologies=("b_sp",), s11_max_db=-20.0, n_pre=5000, n_mc=50,
                 n_keep=None)
    assert res.counts["b_sp"] > 0
    ep = electrical_parts(list(CAPACITOR_KITS) + list(INDUCTOR_KIT))
    c33 = next(e for e in ep if not e.is_inductor and abs(e.value - 3.3e-12) < 1e-18)
    l39 = [e for e in ep if e.is_inductor and abs(e.value - 3.9e-9) < 1e-15]
    keys = {net.key() for net, _, _ in res.ranked}
    assert any(Network("b_sp", (c33, l)).key() in keys for l in l39)
    r = network_response(Network("b_sp", (c33, l39[0])), F0, ant, IDEAL, LOSSLESS_L)
    assert 20 * np.log10(abs(r["s11"])) < -25


def test_band_against_series_resonance():
    # load 50 + j w L_A, series C resonating at f0: |S11| = |X| / |100 + jX|, -10 dB at |X| = 100/3
    la = 50e-9
    f = F0 * (1 + np.linspace(-0.1, 0.1, 81))
    ant = AntennaOnePort(f, 50 + 1j * 2 * np.pi * f * la, np.ones_like(f), F0)
    c = 1 / ((2 * np.pi * F0) ** 2 * la)
    from solver.optim.components import Capacitor
    net = Network("a", tuple(electrical_parts([Capacitor(c, 0.0, "c0")])))
    b = band_10db(net, ant, IDEAL, LOSSLESS_L, n=20001)
    x = 100 / 3
    # w L - 1/(w C) = +-x  ->  w = (+-x + sqrt(x^2 + 4 L/C)) / (2 L)
    w_hi = (x + np.sqrt(x ** 2 + 4 * la / c)) / (2 * la)
    w_lo = (-x + np.sqrt(x ** 2 + 4 * la / c)) / (2 * la)
    assert b["f_lo_GHz"] == pytest.approx(w_lo / (2 * np.pi) / 1e9, rel=1e-5)
    assert b["f_hi_GHz"] == pytest.approx(w_hi / (2 * np.pi) / 1e9, rel=1e-5)
    assert not b["clipped"]


# ----------------------------------------------------------------------------- data / MC
def test_antenna_interpolation_and_range():
    f = F0 * (1 + np.linspace(-0.02, 0.02, 9))
    z = 27 + 1j * (300 + 3000 * (f / F0 - 1))
    ant = AntennaOnePort(f, z, 9 + np.arange(9) * 0.01, F0)
    assert np.allclose(ant.zin(f), z) and np.allclose(ant.gain(f), 9 + np.arange(9) * 0.01)
    assert ant.shifted(1 + 2j).zin(F0) == pytest.approx(ant.zin(F0) + 1 + 2j)
    with pytest.raises(ValueError):
        ant.zin(1.03 * F0)


def test_monte_carlo_reproducible():
    ant = flat_antenna(26.8 + 298.8j, gain=9.0)
    ep = electrical_parts(list(CAPACITOR_KITS))
    pick = lambda v: next(e for e in ep if abs(e.value - v) < 1e-18)
    net = Network("c_sp", (pick(2.7e-12), pick(1.2e-12), pick(1.2e-12)))
    cm = ComponentModel(0.1, 0.0)
    a = evaluate_solution(net, ant, cm, LOSSLESS_L, tolerance_draws(200, 3, 7))
    b = evaluate_solution(net, ant, cm, LOSSLESS_L, tolerance_draws(200, 3, 7))
    assert a == b
    u = tolerance_draws(1000, 3, 0)
    assert u.min() >= -1 and u.max() <= 1
    assert a["G_r_p10_dBi"] <= a["G_r_mean_dBi"]
