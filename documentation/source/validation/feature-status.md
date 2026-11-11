# Feature status

{.mom-status}
{bdg-info}`Draft` Established by reading the code at commit `7c348a7` (2026-11-11); test suite re-run at commit `a0d0b21` (2026-11-11).

The state of every feature of the solver, including what does not work. A feature is
{bdg-success-line}`Stable` when it is covered by automated tests, {bdg-warning-line}`Experimental`
when it works but is not covered, {bdg-danger-line}`Known issue` when a result is known to be
wrong, {bdg-danger}`Not functional` when the code does not run.

## Features

```{list-table}
:header-rows: 1
:widths: 24 14 30 32

* - Feature
  - Status
  - Code
  - Notes
* - Mesh topology, RWG pairs, junctions
  - {bdg-success-line}`Stable`
  - `solver/rwg/rwg1.py`, `rwg2.py`
  - See {doc}`../theory/rwg-basis-functions`.
* - Impedance matrix (EFIE, singular quadrature)
  - {bdg-success-line}`Stable`
  - `solver/utils/impmet.py`, `solver/utils/quadrature/`, `solver/rwg/rwg3.py`
  - Regression, normalisation, reciprocity and CPU/GPU tests.
* - GPU assembly (CuPy)
  - {bdg-success-line}`Stable`
  - `solver/utils/compute_backend.py`, `impmet.impedance_matrix_z_sweep_gpu`
  - Automatic above 100 edges when a CUDA GPU is usable.
* - Radiation with gap sources (one or several ports)
  - {bdg-success-line}`Stable`
  - `solver/src/radiation_algorithm/`, `solver/utils/gap_source.py`, `rwg4.py`
  - Multi-port power budget: see known issues.
* - Frequency sweep, port impedance and reflection
  - {bdg-success-line}`Stable`
  - `solver/utils/frequency_sweep.py`
  - Reflection coefficients are active values (all ports driven together), not the full S-matrix.
* - Lumped R, L, C loads
  - {bdg-warning-line}`Experimental`
  - `rwg3.calculate_z_matrice_lumped_elements`
  - Only the CPU/GPU consistency is tested (`test_rwg3_gpu_dispatch.py`); no test checks the effect of a load on the result. Sweeps with loads fall back to one full simulation per frequency.
* - Lossy conductors (surface impedance)
  - {bdg-danger-line}`Known issue`
  - `solver/utils/lossy_electric_conductor.py`, `impmet.rwg_gram_matrix`
  - Radiation efficiency incorrect (`lossy_antenna/copper_antenna.ipynb`).
* - Scattering by a plane wave
  - {bdg-success-line}`Stable`
  - `solver/src/scattering_algorithm/scattering_algorithm.py`
  - Restored on 2026-11-11 (it raised a `TypeError` before), covered by `test/test_scattering_algorithm.py`.
* - Scattering from the field of another antenna
  - {bdg-danger}`Not functional`
  - `solver/src/scattering_algorithm/scattering_from_E_field_antenna.py`
  - Calls `find_feed_edges`, which does not exist, and an outdated file API.
* - Fields, patterns, polarisation, RCS
  - {bdg-warning-line}`Experimental`
  - `solver/efield/`, `solver/utils/dipole_parameters.py`
  - Used by the example notebooks; the old tests (`test_efield1-3`) no longer run. The printed "Gain" of `efield2` is a directivity.
* - Characteristic modes (one frequency and sweep, tracking, reports)
  - {bdg-success-line}`Stable`
  - `solver/cma/`
  - `TECACorrection` is registered but deliberately not implemented.
* - Reduced multiport and optimisation
  - {bdg-success-line}`Stable`
  - `solver/optim/`
  - 86 tests in `test/optim/`. Limits listed in `solver/optim/README.md`.
* - Adaptive mesh refinement
  - {bdg-warning-line}`Experimental`
  - `solver/utils/error_estimator.py`, `h_refinement_func.py`, `h_refinement_loop.py`
  - Not covered by tests.
* - Graphical interface
  - {bdg-secondary-line}`Planned`
  - `interface/`
  - Empty for now; `main.py` only prints how to use the Python API.
```

## Known issues

**Multi-port gap sources: power budget.** With several ports excited by the enhanced gap source
(`multiple_gap_sources`), the radiated power does not equal the input power. Reference notebook:
`solver/antenna_simulation/gap_voltage_implementation/strip_gap_radiation.ipynb`.

**Lossy conductors: efficiency.** The radiation efficiency computed for lossy antennas is wrong.
Reference notebook: `solver/antenna_simulation/lossy_antenna/copper_antenna.ipynb`.

**Free-space constant.** $\mu_0$ is taken as $1.257\times10^{-6}$ H/m instead of
$4\pi\times10^{-7}$ H/m, a relative difference of about $3\times10^{-4}$, the same value in every
module.

## Test suite

At commit `a0d0b21` (`python -m pytest test/ solver/`): 559 tests pass, 2 are skipped, 2 fail.

- `test_impedance_matrix_matches_golden` (2 cases): the frozen values were computed with the former
  `fill_mode="mirror"`; the default is now `"full_sym"`, checked by `test_impedance_fill_mode.py`.

`test_cma_frequency_sweep_tracks_a_real_structure` used to fail once in a while: its gmsh mesh changes
from run to run when gmsh uses several threads (see {doc}`../theory/numerical-limits`), and it compared
two numbers that can be equal to the last digit with a strict inequality. The comparison now allows
equality up to rounding (commit `5738844`).

Nine early scripts in `test/` are excluded from collection (`test/conftest.py`): they need `.mat`
files that are not in the repository.
