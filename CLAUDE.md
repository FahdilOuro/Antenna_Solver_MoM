# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Python-based **Method of Moments (MoM) electromagnetic solver** for antenna simulation and analysis, translated and extended from Sergey Makarov's 2002 MATLAB book *Antenna and EM Modeling with Matlab*. The solver handles radiation (transmitting) and scattering (receiving) modes for arbitrary triangular-mesh antennas.

## Setup

```bash
# Create Conda environment (Python 3.12)
conda create --name antenna_solver python=3.12
conda activate antenna_solver

# Install dependencies
pip install -r requirements.txt

# Compile Cython extensions (optional performance layer)
python setup.py build_ext --inplace
```

Dependencies: `numpy`, `scipy`, `plotly`, `matplotlib`, `kaleido`, `gmsh`, `scikit-rf`, `numba`.

## Running Tests

```bash
# Run all tests
python -m pytest test/

# Run a single test file
python -m pytest test/test_rwg3.py -v

# MATLAB comparison tests (require .mat reference files in test/)
python test/compare_z_python_matlab.py
python test/compare_current_python_matlab.py
```

## Running Simulations

Simulations are driven via Jupyter notebooks in `backend/antenna_simulation/` or by calling the high-level API directly:

```python
from backend.src.radiation_algorithm.radiation_algorithm import radiation_algorithm
from backend.utils.file_path import FilePath

path = FilePath("my_antenna")
radiation_algorithm(path, frequency=2.4e9, feed_point=[[0, 0, 0]], voltage_amplitude=1, excitation_unit_vector='z')
```

## Architecture

### MoM Pipeline (Data Flow)

The solver is a sequential pipeline where each stage saves its output to disk for downstream reuse:

```
Gmsh (.msh) → MAT file
     ↓ rwg1: Points / Triangles / Edges  → _mesh1.mat
     ↓ rwg2: Barycentric centers, ρ vectors  → _mesh2.mat
     ↓ rwg3: Impedance matrix Z  → _impedance.mat
     ↓ gap_source.py / impmet.py: Excitation vector V
     ↓ rwg4: Solve Z·I = V → current I  → _current.mat
     ↓ rwg5: Surface current density J per triangle
     ↓ efield1–4: Far/near fields, gain, RCS
```

### Module Map

| Module | Responsibility |
|--------|---------------|
| `backend/rwg/rwg1.py` | `Points`, `Triangles`, `Edges` classes; mesh loading from MAT; T-junction filtering |
| `backend/rwg/rwg2.py` | Barycentric sub-triangle centers (9 per triangle); `Vecteurs_Rho` (ρ± vectors) |
| `backend/rwg/rwg3.py` | `calculate_z_matrice()` — core MoM impedance matrix; lumped-element variant |
| `backend/rwg/rwg4.py` | `calculate_current_radiation()` / `calculate_current_scattering()` — solve Z·I=V |
| `backend/rwg/rwg5.py` | Convert edge currents → surface current density; Plotly visualization |
| `backend/utils/gap_source.py` | Gap voltage source model — `multiple_gap_sources()` implements Ding et al. (2013) eq. 7 |
| `backend/utils/impmet.py` | RWG Gram matrix; impedance matrix assembly with surface impedance |
| `backend/utils/lossy_electric_conductor.py` | Surface impedance Zs for non-PEC conductors |
| `backend/utils/gmsh_function.py` | Gmsh mesh generation / `.msh` → `.mat` conversion |
| `backend/utils/frequency_sweep.py` | Multi-frequency loop: S11, port impedance vs. frequency |
| `backend/utils/dipole_parameters.py` | Equivalent dipole moment from current distribution |
| `backend/efield/efield1.py` | E/H fields at arbitrary observation points; Poynting vector; RCS |
| `backend/efield/efield2.py` | Radiation patterns, directivity, gain |
| `backend/src/radiation_algorithm/radiation_algorithm.py` | High-level radiation API (feed → Z, S11, pattern) |
| `backend/src/scattering_algorithm/scattering_algorithm.py` | High-level scattering API (incident wave → induced current, RCS) |

### Key Data Classes

- `Points(p)` — wraps (3, N) array; `.total_of_points`
- `Triangles(t)` — wraps (3, M) index array; after `calculate_triangles_area_and_center()`: `.triangles_area`, `.triangles_center`, `.triangles_plus`, `.triangles_minus` (RWG ± triangle indices per edge)
- `Edges` — obtained via `triangles.get_edges()`; after `compute_edges_length()`: `.edges_length`, `.total_number_of_edges`
- `Barycentric_triangle` — 9 sub-triangle centers per triangle for accurate integration
- `Vecteurs_Rho` — ρ± vectors (3, num_edges) used in Z-matrix and gap-source assembly

### Physical Constants (SI throughout)

| Symbol | Value | Role |
|--------|-------|------|
| ε₀ | 8.854 × 10⁻¹² F/m | Free-space permittivity |
| μ₀ | 1.257 × 10⁻⁶ H/m | Free-space permeability |
| c | 3 × 10⁸ m/s | Speed of light |
| η | ≈ 377 Ω | Wave impedance |
| k | 2πf/c | Free-space wave number |

All lengths in metres, frequencies in Hz.

## Known Issues and Active Work

### `multiple_gap_sources` power budget mismatch
In `backend/utils/gap_source.py`, the multi-port excitation using the enhanced gap source model (Ding et al. 2013, eq. 7) does **not** conserve power: radiated power ≠ input power. The reference notebook is `backend/antenna_simulation/gap_voltage_implementation/strip_gap_radiation.ipynb`. Investigation should focus on:
- Whether the port impedance calculation in `rwg4.calculate_current_radiation()` correctly accounts for the distributed gap voltage
- Whether the Gram matrix normalization in `impmet.rwg_gram_matrix()` is consistent with the multi-gap excitation convention
- Whether `gap_width` relative to mesh cell size causes under-sampling of the excitation window

### Lossy conductor efficiency
`backend/antenna_simulation/lossy_antenna/copper_antenna.ipynb` — radiation efficiency is incorrect (see recent commit `c13af63`).

## Planned Extensions

### Characteristic Modes (CMs)
To add CM analysis, the key extension point is `rwg3.calculate_z_matrice()`, which must be split into real (radiation matrix R) and imaginary (reactance matrix X) parts. The generalized eigenvalue problem **R·Jₙ = λₙ X·Jₙ** requires `scipy.linalg.eigh` with the current Z matrix. The DataManager pattern in `rwg3.py` / `rwg4.py` should be extended to cache eigenmode results.

### Optimization algorithms
Performance bottleneck is the O(N²) Z-matrix fill in `rwg3.calculate_z_matrice()`. Before adding optimization loops, this should be parallelized (e.g., `numba.prange` or `concurrent.futures`) and optionally compressed via `scipy.sparse` for sparse geometries. The existing `Cython` layer in `rwg/cython_rwg/` is the right place for the inner integration kernel.

## File Management

`backend/utils/file_path.py` provides a `FilePath` object that centralizes all intermediate file paths. Pass a single `FilePath` instance through the pipeline rather than building paths manually. Data is organized under `data/`:

```
data/antennas_mesh/       ← input .msh / .mat files
data/antennas_mesh1/      ← rwg1 output
data/antennas_mesh2/      ← rwg2 output
data/antennas_impedance/  ← rwg3 output
data/antennas_current/    ← rwg4 output
data/antennas_gain_power/ ← efield output
data/fig_image/           ← saved plots
```
