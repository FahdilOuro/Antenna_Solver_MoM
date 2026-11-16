# Data and files

{.mom-status}
{bdg-info}`Draft` Written from `solver/utils/file_path.py`, `solver/rwg/rwg1.py` to `rwg4.py`, `solver/utils/gmsh_function.py` and `solver/utils/mesh_io.py` at commit `4282bf8`; the file listing and the reloading example are outputs of `documentation/figure_notebooks/guide_first_simulation.ipynb`. Awaiting review.

The solver is a pipeline of stages, and each stage saves its result to a `.mat` file (MATLAB format,
read and written with `scipy.io`) before the next one reads it. This page says where those files are,
what they contain, and how to use them.

## Where: `data/`, under the working directory

Every path is **relative to the working directory**: a notebook run from its own folder reads and
writes the `data/` folder next to it, a script reads and writes the `data/` folder of the directory it
is started from. `setup_save_file_paths(name)` creates the folders below if they are missing and
returns a `path` object with the name of every file of the antenna `name`:

```{list-table}
:header-rows: 1
:widths: 26 30 44

* - Folder
  - File, `path.` attribute
  - Written by
* - `data/gmsh_files/`
  - `name.msh` (`msh`), `name.brep` (`geo`)
  - `gmsh_function.generate_and_save_mesh`: the mesh, and the geometry it was made from
* - `data/antennas_mesh/`
  - `name.mat` (`mat`)
  - `gmsh_function.extract_msh_to_mat`, called by `radiation_algorithm`: the mesh as arrays `p`, `t`
* - `data/antennas_mesh1/`
  - `name_mesh1.mat` (`mat_mesh1`)
  - stage `rwg1`: points, triangles, RWG edges
* - `data/antennas_mesh2/`
  - `name_mesh2.mat` (`mat_mesh2`)
  - stage `rwg2`: the same, plus the $\boldsymbol\rho$ vectors
* - `data/antennas_impedance/`
  - `name_impedance.mat` (`mat_impedance`)
  - stage `rwg3`: the impedance matrix
* - `data/antennas_current/`
  - `name_current.mat` (`mat_current`)
  - stage `rwg4`: excitation, currents, port results
* - `data/antennas_gain_power/`
  - `name_gain_power.mat` (`mat_gain_power`), and the RHCP / LHCP variants
  - `efield2`: radiated power, directivity, efficiency
* - `data/antennas_sweep/`
  - `name_freq_sweep.mat` (`mat_freq_sweep`)
  - `frequency_sweep`
* - `data/antennas_cma/`
  - `name_cma.mat`, `name_cma_sweep.mat`, Markdown reports
  - `solver/cma/` (this folder is created by the CMA functions themselves)
* - `data/fig_image/`
  - PDF and GIF figures
  - the plotting functions, with `save_image=True` or `save_gif=True`
```

The first use of `efield2` also writes the observation sphere, `data/gmsh_files/hollow_sphere.msh` and
`data/antennas_mesh/hollow_sphere.mat`, and reuses them afterwards.

## What: the content of each file

For the dipole of {doc}`first-simulation` ($N$ = 144 RWG edges, 116 triangles, 89 points):

```{list-table}
:header-rows: 1
:widths: 24 76

* - File
  - Variables (shape)
* - `name.mat`
  - `p` (3, 89) point coordinates in metres; `t` (4, 116) triangles: three vertex indices **starting
    at 1**, and in the 4th row the gmsh surface each triangle belongs to
* - `name_mesh1.mat`
  - `points`, `triangles`; per edge: `edge_first_points`, `edge_second_points`, `triangles_plus`,
    `triangles_minus`, `edges_length` (1, 144); per triangle: `triangles_area` (1, 116),
    `triangles_center` (3, 116); `mesh_fingerprint`
* - `name_mesh2.mat`
  - the same, plus `vecteur_rho_plus`, `vecteur_rho_minus` (3, 144)
* - `name_impedance.mat`
  - `frequency`, `omega`, `mu`, `epsilon`, `light_speed_c`, `eta`; `matrice_z` (144, 144) complex;
    `mesh_fingerprint`
* - `name_current.mat`
  - the same constants; `voltage`, `current` (1, 144) complex; per port: `gap_currents`,
    `gap_voltages`, `impedances`, `feed_powers`
* - `name_gain_power.mat`
  - `totalPower` (W), `gainLinear` and `gainLogarithmic` (the directivity, see
    {doc}`../theory/radiated-fields`), `efficiencyTotal`
```

Units are SI throughout; amplitudes are peak values.

## Reloading results

Each stage has a `DataManager` that reads its file back, so a result can be used again without
recomputing it. Continuing {doc}`first-simulation` (where `path` and `np` are defined):

```python
from solver.rwg.rwg2 import DataManager_rwg2
from solver.rwg.rwg3 import DataManager_rwg3
from solver.rwg.rwg4 import DataManager_rwg4

points, triangles, edges, vecteurs_rho = DataManager_rwg2.load_data(path.mat_mesh2)
frequency, omega, mu, epsilon, light_speed_c, eta, Z = DataManager_rwg3.load_data(path.mat_impedance)
(frequency, omega, mu, epsilon, light_speed_c, eta,
 V, I, gap_currents, gap_voltages, impedances, feed_powers) = DataManager_rwg4.load_data(path.mat_current, radiation=True)

print(edges.total_number_of_edges, "edges;", Z.shape, "matrix;", "Z_in =", np.round(impedances, 2), "Ohm")
```

```text
144 edges; (144, 144) matrix; Z_in = (72.74+4.09j) Ohm
```

`DataManager_rwg1.load_data(path.mat_mesh1)` returns `points, triangles, edges`. For a scattering
result, call `DataManager_rwg4.load_data(path.mat_current, scattering=True)`, which returns the
incident direction and polarisation instead of the port results.

:::{note}
The arrays read back are squeezed: with **one port**, `impedances` comes back as a 0-dimensional
array (printed `(72.74+4.09j)` above), whereas `radiation_algorithm` returned an array of length 1.
`np.atleast_1d(impedances)[0]` works in both cases.
:::

Missing or unreadable files raise `MeshDataError` with the name of the file, instead of failing later
on an unrelated error.

## Mixing files from two meshes

The mesh and impedance files carry a `mesh_fingerprint`, a short hash of the mesh arrays. Before
solving, `rwg4` checks that the `_mesh2` and `_impedance` files it is given have the same one, and
raises `MeshConsistencyError` otherwise: an impedance matrix left over from an earlier mesh cannot be
combined with a new one by mistake. Files written before this check existed carry no fingerprint and
pass silently. The current file has no fingerprint.

## Meshes from MATLAB

The `p`, `t` format is the one of Makarov's MATLAB code {footcite:p}`makarov2002`: `p` (3, N)
coordinates, `t` (4, M) 1-based vertex indices with a domain label in the 4th row. Such a file can be
used directly by `scattering_algorithm` (give the file name; see {doc}`../user-guide/scattering`).
The CMA solver falls back on `path.mat` when there is no `path.msh` (`cma_mesh.prepare_cma_mesh`;
see {doc}`../user-guide/characteristic-modes`).

`radiation_algorithm` cannot take one: it always starts by regenerating `path.mat` from the gmsh file
`path.msh`, which must therefore exist.

## References

```{footbibliography}
```
