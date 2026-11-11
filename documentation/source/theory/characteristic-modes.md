# Characteristic modes

{.mom-status}
{bdg-secondary-line}`Planned` Outline of a future page.

The currents a structure "naturally" supports, computed from its impedance matrix alone, before any
feed is chosen.

**This page will cover**

- $Z = R + jX$ and the generalised eigenvalue problem $X\,\mathbf J_n = \lambda_n R\,\mathbf J_n$;
  physical meaning of $\lambda_n$ (net reactive power of a mode relative to its radiated power),
  resonance at $\lambda_n=0$.
- Characteristic angle $\delta_n = 180° - \arctan\lambda_n$ and modal significance
  $\mathrm{MS}_n = 1/\sqrt{1+\lambda_n^2}$ (`characteristic_angle.py`, `modal_significance.py`).
- Why $R$ is rank-deficient, and the two solvers of `solver.cma.eigen_solver`: `'regularized'` and
  `'rank_truncated'`; spurious modes and how `mode_diagnostics` flags them.
- Mode tracking across frequency by $R$-weighted correlation and linear assignment
  (`mode_tracking.py`), and what tracking cannot decide (degeneracies, avoided crossings).
- Modal currents and modal far fields; validation against AToM on a rectangular loop.
