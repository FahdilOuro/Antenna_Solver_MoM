# Data and files

{.mom-status}
{bdg-secondary-line}`Planned` Outline of a future page.

**This page will cover**

- The `data/` tree (relative to the working directory) and which stage writes which file:
  `gmsh_files/`, `antennas_mesh/`, `antennas_mesh1/`, `antennas_mesh2/`, `antennas_impedance/`,
  `antennas_current/`, `antennas_gain_power/`, `antennas_sweep/`, `antennas_cma/`, `fig_image/`.
- The content of each `.mat` file, and the mesh fingerprint that prevents mixing a mesh with the
  impedance matrix of another one (`solver/utils/mesh_io.py`).
- Importing a mesh from MATLAB (`p`, `t` arrays, 1-based indices).
- Running notebooks from their own folder: each one uses the `data/` folder next to it.
