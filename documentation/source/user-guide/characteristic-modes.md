# Characteristic mode analysis

{.mom-status}
{bdg-secondary-line}`Planned` Outline of a future page.

**This page will cover**

- One frequency: `cma_algorithm(path, frequency, n_modes, cma_solver, ...)`, modal currents with
  direction arrows, modal far fields, the Markdown report.
- A band: `cma_frequency_sweep(...)`, tracking (`sweep.tracked()`, `sweep.tracking.summary()`),
  and the plots of `ModalVisualizer` (eigenvalues, modal significance, characteristic angle).
- Choosing between the `'regularized'` and `'rank_truncated'` solvers and reading the mode
  diagnostics.
- Meshes coming from MATLAB without a `.msh` file (`cma_mesh.prepare_cma_mesh`).
