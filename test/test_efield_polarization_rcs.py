"""
Checks of the circular-polarisation split and of the RCS normalisation.

- compute_circular_components: with exp(+jwt), E_phi = -j E_theta is pure RHCP
  and E_phi = +j E_theta pure LHCP (IEEE convention).
- compute_rcs: RCS = 4 pi r^2 |E_s|^2 / |E_0|^2 does not depend on the amplitude
  of the incident field vector given to the scattering solve.
"""
import numpy as np
import pytest

from solver.efield.efield1 import compute_rcs
from solver.utils.dipole_parameters import compute_circular_components


def test_rhcp_is_e_phi_minus_j_e_theta():
    rhcp, lhcp = compute_circular_components(1.0 + 0j, -1j)
    assert abs(lhcp) < 1e-15
    assert abs(rhcp) == pytest.approx(np.sqrt(2))


def test_lhcp_is_e_phi_plus_j_e_theta():
    rhcp, lhcp = compute_circular_components(1.0 + 0j, 1j)
    assert abs(rhcp) < 1e-15
    assert abs(lhcp) == pytest.approx(np.sqrt(2))


def test_rhcp_rotation_sense_at_theta_zero():
    """E = x - j y at theta = 0 rotates from x to y: clockwise looking along +z."""
    wt = np.linspace(0, np.pi / 2, 3)
    e = np.array([1.0, -1j])
    real_field = np.real(e[:, None] * np.exp(1j * wt)[None, :])
    assert np.allclose(real_field[:, 0], [1, 0])      # along x at t = 0
    assert np.allclose(real_field[:, -1], [0, 1])     # along y a quarter period later


def test_rcs_unit_incident_field():
    e_s = np.array([1e-3, 2e-3j, 0.0])
    r = 100.0
    expected = 4 * np.pi * r ** 2 * (1e-6 + 4e-6)
    assert compute_rcs(e_s, r, [1.0, 0.0, 0.0]) == pytest.approx(expected)


def test_rcs_independent_of_incident_amplitude():
    """The scattered field is linear in E_0: scaling both leaves the RCS unchanged."""
    e_s = np.array([1e-3, 2e-3j, 0.0])
    reference = compute_rcs(e_s, 100.0, [1.0, 0.0, 0.0])
    for scale in (0.5, 3.0, 2.0 - 1.0j):
        assert compute_rcs(scale * e_s, 100.0, scale * np.array([1.0, 0.0, 0.0])) == pytest.approx(reference)


def test_rcs_zero_incident_field_raises():
    with pytest.raises(ValueError):
        compute_rcs(np.ones(3), 1.0, [0.0, 0.0, 0.0])


def _printed_rcs(text):
    """Last 'RCS = x' or 'RCS : x m²' value printed by efield1."""
    values = [line.split("=" if "=" in line else ":")[-1].replace("m²", "").strip()
              for line in text.splitlines() if line.strip().startswith("RCS")]
    return float(values[-1])


def test_efield1_rcs_does_not_depend_on_the_incident_amplitude(tmp_path, capsys):
    """End to end: a plate solved with E_0 = 1 V/m and 2 V/m along x gives the same RCS."""
    from scipy.io import savemat
    from test_scattering_algorithm import FREQUENCY, DIRECTION, _plate_mesh, _path_namespace
    from solver.efield.efield1 import (analyze_point_fields_polarization,
                                       calculate_electric_magnetic_field_at_point)
    from solver.src.scattering_algorithm.scattering_algorithm import scattering_algorithm

    observation = np.array([0.0, 0.0, 100.0])          # back-scattering direction
    rcs = {}
    for amplitude in (1.0, 2.0):
        folder = tmp_path / f"E0_{amplitude:g}"
        folder.mkdir()
        path = _path_namespace(folder)
        p, t = _plate_mesh()
        savemat(path.mat, {"p": p, "t": t})
        scattering_algorithm(path, FREQUENCY, DIRECTION, [amplitude, 0.0, 0.0], show=False, use_gpu=False)
        capsys.readouterr()
        calculate_electric_magnetic_field_at_point(path, observation, mode="scattering")
        rcs_point = _printed_rcs(capsys.readouterr().out)
        analyze_point_fields_polarization(path, observation, mode="scattering")
        rcs_polar = _printed_rcs(capsys.readouterr().out)
        assert rcs_point == pytest.approx(rcs_polar, rel=1e-5)   # printed with 6 significant digits
        rcs[amplitude] = rcs_point

    assert rcs[1.0] > 0
    assert rcs[2.0] == pytest.approx(rcs[1.0], rel=1e-6)
