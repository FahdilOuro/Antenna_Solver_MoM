"""
Tests for backend.optim.inner (reactive-load optimisation) and the geometry
gain bound of backend.optim.objectives.

Geometry for all tests: two parallel copper strips (lambda/2 along x, width
lambda/100), spacing d = lambda/10 along y, 300 MHz, zs_scale = 0.5; element 1
('feed', y = -d/2) driven, element 2 ('parasite', y = +d/2) loaded. Target:
+y, phi polarisation (end-fire towards the parasite: a director).

test_two_element_endfire also writes the MoM reference for the CST sheet V6
to fiches_cst/resultats_V6/mom_reference.json (when that folder's parent exists).
"""

import datetime as _dt
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from backend.optim import ReducedMultiport
from backend.optim.farfield import RadiationOperator
from backend.optim.geometries import parallel_strips
from backend.optim.inner import optimize_loads
from backend.optim.objectives import gain_from_currents, geometry_gain_bound, loss_matrix
from backend.optim.threads import limit_blas_threads
from conftest import COPPER_SIGMA

F0 = 3e8
LAM = 1.0                       # m at F0 (c ~ 3e8 with the rwg3 constants: 2.9975e8)
DEG = np.pi / 180
PLUS_Y, MINUS_Y = (90 * DEG, 90 * DEG), (90 * DEG, 270 * DEG)
ZS_SCALE = 0.5
GEOM = dict(length=0.5 * LAM, width=0.01 * LAM, spacing=0.1 * LAM, n_len=24, n_width=1)
V6_JSON = Path(r"D:\PhD_Fahdilou_Leat\PUBLICATIONS\7-EurAAP Small Antenna Super-Gain Challenge"
               r"\fiches_cst\resultats_V6\mom_reference.json")


@pytest.fixture(scope="module")
def pair():
    mesh = parallel_strips(**GEOM)
    model = ReducedMultiport.from_mesh(mesh.points, mesh.triangles, mesh.edges, F0, mesh.ports(),
                                       conductivity=COPPER_SIGMA, zs_scale=ZS_SCALE, use_gpu=False)
    op = RadiationOperator(mesh.points, mesh.triangles, mesh.edges, F0)
    bound = geometry_gain_bound(model, op, *PLUS_Y, component="phi")
    return dict(mesh=mesh, model=model, op=op, bound=bound, M=loss_matrix(model)[0])


@pytest.fixture(scope="module")
def optimum(pair):
    return optimize_loads(pair["model"], PLUS_Y, "phi", objective="G", operator=pair["op"],
                          n_starts=32, seed=0)


def test_bound_dominates(pair, optimum):
    model, op, M, g_max = pair["model"], pair["op"], pair["M"], pair["bound"].g_max
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(200):
        v = rng.standard_normal(model.n_ports) + 1j * rng.standard_normal(model.n_ports)
        g = gain_from_currents(model.edge_currents(v), op, M, *PLUS_Y)
        worst = max(worst, g)
        assert g <= g_max * (1 + 1e-12)
    assert optimum.G <= g_max * (1 + 1e-12)
    print(f"\n[bound] G_max = {g_max:.4f} ({10 * np.log10(g_max):.3f} dBi), cond(M) = "
          f"{pair['bound'].cond:.3e}; best of 200 random port excitations: {worst:.4f}; "
          f"optimum: {optimum.G:.4f} (G/G_max = {optimum.g_ratio:.4f})")


def test_bound_attained(pair):
    b = pair["bound"]
    g = gain_from_currents(b.I_opt, pair["op"], pair["M"], *PLUS_Y)
    assert abs(g - b.g_max) / b.g_max < 1e-9
    g_scaled = gain_from_currents((0.3 - 2.1j) * b.I_opt, pair["op"], pair["M"], *PLUS_Y)
    assert abs(g_scaled - b.g_max) / b.g_max < 1e-9


def test_bound_lossless_warns(pair):
    model = pair["model"]
    pec = ReducedMultiport.from_matrices(model.Z, model.A, F0)
    with pytest.warns(UserWarning, match="no conductor loss"):
        assert geometry_gain_bound(pec, pair["op"], *PLUS_Y) is None


def test_single_load_bruteforce(pair):
    model, op = pair["model"], pair["op"]
    res = optimize_loads(model, PLUS_Y, "phi", objective="G", operator=op, n_starts=8, seed=1)
    name = op.attach_functional(model, *PLUS_Y, component="phi")
    xs = np.linspace(-2000.0, 2000.0, 20001)            # 0.2 Ohm resolution
    step = xs[1] - xs[0]
    G = np.empty_like(xs)
    with limit_blas_threads(1):
        for k, x in enumerate(xs):
            sol = model.solve([1.0, 0.0], [50.0, 1j * x])
            G[k] = 4 * np.pi * op.intensity_constant * abs(sol.functionals[name]) ** 2 / sol.p_in_total
    kb = int(np.argmax(G))
    print(f"\n[brute force] X_brute = {xs[kb]:+.2f} Ohm, G = {G[kb]:.6f};  "
          f"L-BFGS-B X = {res.x_opt[0]:+.4f} Ohm, G = {res.G:.6f}  "
          f"({res.n_evaluations} solves vs 20001)")
    assert abs(res.x_opt[0] - xs[kb]) <= step
    assert res.G >= G[kb] * (1 - 1e-12)                 # never worse than the grid


def _git():
    try:
        root = Path(__file__).resolve().parents[2]
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                             text=True, timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root,
                                    capture_output=True, text=True, timeout=10).stdout.strip())
        return dict(commit=out, dirty=dirty)
    except Exception:
        return None


def test_two_element_endfire(pair, optimum):
    model, op, res = pair["model"], pair["op"], optimum
    u_fwd = op.intensity(res.edge_currents, *PLUS_Y)[1][0]
    u_back = op.intensity(res.edge_currents, *MINUS_Y)[1][0]
    fb_db = 10 * np.log10(u_fwd / u_back)
    g_dbi = 10 * np.log10(res.G)
    print(f"\n[two-element end-fire, d = lambda/10, Cu, zs_scale = {ZS_SCALE}]"
          f"\n  {res.summary()}"
          f"\n  front/back (+y / -y, phi pol.) = {fb_db:.2f} dB"
          f"\n  G_max (geometry bound) = {10 * np.log10(res.g_max):.3f} dBi")
    assert g_dbi > 2.15                                # beats a lone half-wave dipole
    assert fb_db > 3.0

    if V6_JSON.parent.parent.is_dir():
        V6_JSON.parent.mkdir(exist_ok=True)
        ref = dict(
            description="MoM reference for CST sheet V6: two parallel copper strips, element 1 "
                        "fed (delta gap across the full width at x = 0), element 2 loaded by an "
                        "ideal reactance (delta gap at x = 0). Optimised for G (ideal matching) "
                        "towards +y, phi polarisation.",
            generated=_dt.datetime.now().isoformat(timespec="seconds"), git=_git(),
            frequency_Hz=F0,
            geometry=dict(strip_length_m=GEOM["length"], strip_width_m=GEOM["width"],
                          centre_spacing_m=GEOM["spacing"], plane="z = 0, strips along x",
                          feed_strip_y_m=-GEOM["spacing"] / 2, loaded_strip_y_m=GEOM["spacing"] / 2,
                          mesh=dict(n_len=GEOM["n_len"], n_width=GEOM["n_width"],
                                    n_edges=model.n_edges)),
            material=dict(sigma_S_per_m=COPPER_SIGMA, zs_scale=ZS_SCALE,
                          Zs_ohm=[model.zs.real, model.zs.imag]),
            constants=dict(epsilon0=8.854e-12, mu0=1.257e-6, note="as rwg3.calculate_z_matrice"),
            target=dict(theta_deg=90.0, phi_deg=90.0, polarization="phi", objective="G"),
            load=dict(reactance_ohm=float(res.x_opt[0]), kind=res.element_kind[0],
                      value=float(res.element_value[0]), label=res.element_label[0]),
            results=dict(G_dBi=g_dbi, G_r_50ohm_dBi=float(10 * np.log10(res.G_r)),
                         D_dBi=float(10 * np.log10(res.D)), eta_rad=res.eta_rad,
                         Zin_ohm=[res.z_in.real, res.z_in.imag],
                         front_to_back_dB=float(fb_db),
                         G_max_dBi=float(10 * np.log10(res.g_max)), G_over_G_max=res.g_ratio),
        )
        V6_JSON.write_text(json.dumps(ref, indent=2), encoding="utf-8")
        print(f"  wrote {V6_JSON}")
