# Load and matching optimisation

{.mom-status}
{bdg-secondary-line}`Planned` Outline of a future page.

**This page will cover**

- Defining ports with `port_from_cut` / `port_from_edge`, building a `ReducedMultiport`, solving
  for any generator and terminations.
- Far-field functionals and gain objectives; the geometry gain bound.
- `optimize_loads` (inner loop) and `BayesianOptimizer` (outer loop), with BLAS thread control.
- Snapping to real component kits and designing a matching network.
- The worked example in `solver/antenna_simulation/euraap_challenge/`.
