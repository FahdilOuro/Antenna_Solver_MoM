# Installation

{.mom-status}
{bdg-info}`Draft` Checked on 2026-11-11 by a fresh install in a new virtual environment (Python 3.12, Windows 11, no GPU), at solver commit `4282bf8`. Awaiting review.

The solver is a Python package, `solver`, installed from a clone of the repository. It needs Python
3.12; a new virtual environment with its dependencies takes about 750 MB on disk. A GPU is optional.

## Install

With [Conda](https://docs.conda.io/) (as the developers do):

```bash
git clone https://github.com/FahdilOuro/Antenna_Solver_MoM.git
cd Antenna_Solver_MoM

conda create --name antenna_solver python=3.12
conda activate antenna_solver

pip install -r requirements.txt
pip install -e .
```

Or with a plain virtual environment, from any Python 3.12:

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows; on Linux or macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The last line installs the `solver` package itself in *editable* mode: `import solver` then works
from any folder (the example notebooks, which live in subfolders, need it), and changes to the code in
the repository take effect without reinstalling. On the checking machine the two `pip` commands took
about two minutes.

`requirements.txt` includes Jupyter (`notebook`), to run the example notebooks of
`solver/antenna_simulation/` and the {doc}`first-simulation`: start it with `jupyter notebook` from the
folder you want to work in.

## What is installed

```{list-table}
:header-rows: 1
:widths: 22 78

* - Package
  - Used for
* - `numpy`, `scipy`
  - everything numerical: assembly, linear algebra, quadrature, optimisation
* - `gmsh`
  - drawing geometries and meshing them (`solver/utils/gmsh_function.py`)
* - `plotly`, `kaleido`
  - interactive 3D views (surface currents, patterns); `kaleido` exports them as images
    (`save_image=True`, rotating GIFs). To do so, `kaleido` 1.x drives a Chromium-based browser
    installed on the machine (Chrome, Chromium, Edge, Brave or Vivaldi) and raises
    `ChromeNotFoundError` if it finds none
* - `matplotlib`
  - 2D plots: polar cuts, sweeps, Smith chart, CMA plots
* - `scikit-learn`
  - the Bayesian outer loop of the optimisation module (`solver/optim/outer.py`)
* - `notebook`
  - Jupyter, for the example notebooks
* - `pytest`
  - the test suite
* - `numba`, `scikit-rf`
  - listed for planned work; no code of the solver imports them at the moment
```

Optional, not in `requirements.txt`:

- **CuPy**, for the GPU assembly of the impedance matrix (see {doc}`../user-guide/gpu`). Install the
  wheel that matches the CUDA version of your driver (shown by `nvidia-smi`), for example
  `pip install cupy-cuda12x`. Without it, or without a usable NVIDIA GPU, everything runs on the CPU.
- **threadpoolctl**, used by the optimisation loops to run the BLAS on one thread
  (`solver/optim/threads.py`); without it they work and print a warning once.

## Check the installation

```bash
python -c "import solver.utils.compute_backend as cb; print(cb.describe_backend())"
python -m pytest test/
```

The first line prints the compute backend the solver found:
`GPU (NVIDIA ..., CuPy ...)`, or `CPU (no usable CUDA GPU detected: ...)` followed by the reason. The
second runs the test suite (a few minutes; meshes are built inside the tests, no data files needed).
On the fresh install of 2026-11-11, without CuPy, 218 tests passed and 2 were skipped.

## Platforms

The solver is developed and tested on Windows 11. Nothing in it is specific to Windows, but no other
platform has been tested by the maintainers.

**Long paths on Windows.** Windows limits file paths to 260 characters unless long-path support is
enabled, and some files of JupyterLab (installed with `notebook`) are deep. With an environment created
in a deeply nested folder, `pip install -r requirements.txt` then stops with
`OSError: [Errno 2] No such file or directory` and a hint about long paths; this happened on the
checking machine with a virtual environment in a temporary folder. Create the environment in a short
path (the Conda default, or `.venv` in the clone), or enable long-path support as the hint explains.

On a Windows console with a legacy code page, printing some results can fail with a
`UnicodeEncodeError` on Greek letters (λ, Ω); running Python with `-X utf8`, or setting the environment
variable `PYTHONUTF8=1`, avoids it (see {doc}`../user-guide/troubleshooting`).

## Next

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {fas}`play` First simulation
:link: first-simulation
:link-type: doc

A half-wave dipole in four steps
:::

:::{grid-item-card} {fas}`microchip` GPU
:link: ../user-guide/gpu
:link-type: doc

Installing CuPy and choosing `use_gpu`
:::
::::
