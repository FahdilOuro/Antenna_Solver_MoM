# Architecture

{.mom-status}
{bdg-secondary-line}`Planned` Outline of a future page.

**This page will cover**

- The packages of `solver/`: `rwg`, `utils` (with `quadrature`), `efield`, `cma`, `optim`, `src`
  (high-level algorithms) and the example notebooks in `antenna_simulation`.
- `solver/` as a namespace package (no `__init__.py`), installed with `pip install -e .`.
- Dependencies between packages, and the rule that the high-level functions call the lower levels
  and never the reverse.
- Other folders: `interface/` (future front end), `test/`, `documentation/`.
