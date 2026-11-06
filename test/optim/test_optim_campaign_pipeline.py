"""
Integration test of the step-6 campaign pipeline (campaign_b.py) on a COARSE mesh
and temporary folders: geometry evaluation (all fed positions), JSONL/CSV saving,
post-checks (frozen parts over f0 +- 2 %, input Q, mesh check) and CST export.
The campaigns themselves are not run here.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "campaign_b", ROOT / "solver/antenna_simulation/euraap_challenge/campaign_b.py")
cb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cb)


@pytest.fixture
def coarse(monkeypatch, tmp_path):
    monkeypatch.setattr(cb, "MESH_RULE", dict(n_width=2, edge_grading="cosine"))
    monkeypatch.setattr(cb, "CELL_LENGTH", 0.0225)
    monkeypatch.setattr(cb, "N_STARTS", 6)
    monkeypatch.setattr(cb, "CST_DIR", tmp_path / "cst")
    return tmp_path


def test_three_element_pipeline(coarse):
    tmp = coarse
    recs = cb.evaluate_geometry([0.025, 0.035], [0.9e9], r_in_min=10.0, n_starts=6, log=lambda *a: None)
    assert len(recs) == 3 and sum(r["best_fed"] for r in recs) == 1
    assert {r["fed"] for r in recs} == {"el1", "el2", "el3"}
    jl = tmp / "B3_designs.jsonl"
    cb.append_jsonl(jl, recs)
    back = cb.read_jsonl(jl)
    assert len(back) == 3 and back[0]["spacings_mm"] == [25.0, 35.0]
    cb.write_summary(tmp / "B3_summary.csv", back)
    assert (tmp / "B3_summary.csv").read_text(encoding="utf-8").count("\n") == 4

    best = cb.top_records(back, n=1)[0]
    checks = cb.post_checks(best, log=lambda *a: None)
    offs = [round(r["offset_pct"], 3) for r in checks["freq_rows"]]
    assert offs == [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    g0 = next(r for r in checks["freq_rows"] if r["offset_pct"] == 0.0)
    assert g0["G_dBi"] == pytest.approx(best["G_kit_dBi"], abs=1e-9)        # frozen parts reproduce f0
    assert checks["Q_in"] > 0
    assert len(checks["cut_xy"]) == 360 and len(checks["cut_yz"]) == 181
    gy = next(c for c in checks["cut_xy"] if c["phi_deg"] == 90.0)["G_phi_dBi"]
    assert gy == pytest.approx(best["G_kit_dBi"], abs=1e-9)                  # the cut passes through +y
    assert checks["mesh_check"]["N_fine"] > checks["mesh_check"]["N_coarse"]

    cb.export_cst(best, checks, "TEST", log=lambda *a: None)
    doc = json.loads((tmp / "cst" / "V6_parametres_TEST.json").read_text(encoding="utf-8"))
    assert doc["spacings_mm"]["reference"].startswith("CENTRE")
    assert len(doc["strips"]) == 3 and sum(s["role"] == "fed" for s in doc["strips"]) == 1
    for s in doc["strips"]:
        assert "EDGES" in s["y_edges_mm"]["reference"] and "CENTRE" in s["y_center_mm"]["reference"]
        if s["role"] == "loaded":
            assert s["component"]["esr_ohm"] == 0.0 and s["component"]["label"]
    for name in ("freq", "cut_theta90_xy", "cut_phi90_yz"):
        assert (tmp / "cst" / f"V6_expected_TEST_{name}.csv").is_file()
    print(f"\n[pipeline] best {best['spacings_mm']} mm fed {best['fed']}: G kit {best['G_kit_dBi']:.2f} dBi, "
          f"Q_in = {checks['Q_in']:.1f}, mesh check {np.round(checks['mesh_check']['G_kit_dBi'], 3)}")
