# Reduced multiport model and load optimisation

{.mom-status}
{bdg-secondary-line}`Planned` Outline of a future page.

When only the loads change and the geometry stays fixed, the full MoM system can be reduced exactly
to its ports, so that each new set of loads costs a few small matrix operations.

**This page will cover**

- Ports as cuts across the conductor, incidence vectors and the matrix $A$
  (`solver/optim/ports.py`).
- The exact reduction $B = Z_t^{-1}A$, $Y = A^\mathsf{T}B$, and the port equations with
  terminations (`solver/optim/reduced_model.py`); powers $P_\mathrm{rad}$, $P_\mathrm{cond}$,
  $P_\mathrm{loads}$.
- Far-field functionals and the geometry gain bound (`farfield.py`, `objectives.py`).
- The inner loop (optimal reactive loads, `inner.py`) and the outer Bayesian loop (`outer.py`).
- Real components and matching networks (`components.py`, `matching.py`).
- Limits stated in `solver/optim/README.md`: no open circuit, free space only, one frequency per
  instance.
