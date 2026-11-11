# Testing

{.mom-status}
{bdg-secondary-line}`Planned` Outline of a future page.

**This page will cover**

- Running the suites: `python -m pytest test/`, `python -m pytest solver/utils/quadrature/tests/`,
  `python -m pytest test/optim -v -s`.
- What each group of tests protects (golden values, physical invariants such as scale invariance and
  positive semi-definiteness of $R$, CPU/GPU agreement, reduced-model exactness).
- The legacy scripts excluded from collection (`test/conftest.py`), the golden test that still
  asserts the former `fill_mode="mirror"` values, and the CMA tracking test that depends on a
  non-deterministic gmsh mesh.
- Writing a new test: build the mesh inside the test, write files under `tmp_path`.
