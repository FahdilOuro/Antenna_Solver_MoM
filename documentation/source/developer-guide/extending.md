# Extending the solver

{.mom-status}
{bdg-secondary-line}`Planned` Outline of a future page.

**This page will cover**, with worked examples:

- adding a Gauss–Dunavant order or another triangle rule (`solver/utils/quadrature/gauss_dunavant.py`);
- the three impedance assemblers that must stay consistent (`impedance_matrix_z`,
  `impedance_matrix_z_sweep`, `impedance_matrix_z_sweep_gpu`) and the regression tests that
  cross-check them;
- adding a mode-tracking correction (`ModeTrackingCorrection` in `solver/cma/mode_tracking.py`);
- adding a result type with its `DataManager` and its plot.
