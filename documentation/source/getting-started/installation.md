# Installation

{.mom-status}
{bdg-secondary-line}`Planned` Outline of a future page.

**This page will cover**

- Python 3.12 in a Conda environment, `pip install -r requirements.txt`, then `pip install -e .`
  so that `import solver` works from any folder (the example notebooks need it).
- Required and optional dependencies: gmsh, Plotly with kaleido for image export, CuPy for GPU
  assembly, threadpoolctl for the optimisation loops.
- Checking the installation with the test suite (`python -m pytest test/`).
- Platform notes (the solver is developed on Windows).
