# First simulation

{.mom-status}
{bdg-info}`Draft` Every code block below is run, unchanged, by `documentation/figure_notebooks/guide_first_simulation.ipynb` (solver commit `4282bf8`); the outputs shown are its outputs. Awaiting review.

A half-wave strip dipole at 300 MHz, from the geometry to the input impedance and the radiation
pattern, in four steps. Run it in a notebook or a script from an empty folder: the solver writes its
files in a `data/` folder under the working directory (see {doc}`data-and-files`).

## 1. Imports and file paths

```python
import gmsh
import numpy as np

from solver.utils.file_path import setup_save_file_paths
from solver.utils.gmsh_function import (mesh_and_gap_parameters, setup_performance_config,
                                        generate_and_save_mesh, check_simulation_stability)
from solver.src.radiation_algorithm.radiation_algorithm import radiation_algorithm
from solver.efield.efield2 import radiation_intensity_distribution_over_sphere_surface
from solver.efield.efield3 import antenna_directivity_pattern

path = setup_save_file_paths("strip_dipole")
```

`setup_save_file_paths(name)` creates the folders of `data/` if needed and returns a `path` object
holding the name of every file the simulation will read or write: `path.msh` for the gmsh mesh,
`path.mat_impedance` for the impedance matrix, `path.mat_current` for the currents, and so on. Every
function of the pipeline takes this one object.

## 2. Geometry and mesh

```python
frequency = 300e6                 # Hz, wavelength 1 m
length, width = 0.47, 0.01        # m: a strip along y, centred at the origin

# Element size and gap width from the frequency: gap = 0.05 wavelength,
# elements of wavelength/60, so that 3 elements fit in the gap.
mesh_size, gap_width, wavelength = mesh_and_gap_parameters(frequency, resolution=60, gap_fraction=0.05)

gmsh.initialize()
gmsh.model.add(path.name)
setup_performance_config(n_threads=1)          # one thread: the same mesh every run
strip = gmsh.model.occ.addRectangle(-width / 2, -length / 2, 0, width, length)
gap = gmsh.model.occ.addRectangle(-width / 2, -gap_width / 2, 0, width, gap_width)
gmsh.model.occ.fragment([(2, strip)], [(2, gap)])   # the gap edges become mesh lines
gmsh.model.occ.synchronize()
generate_and_save_mesh(path, mesh_size)
gmsh.finalize()
```

```text
λ = 1.0000 m  |  mesh_size = λ/60 = 0.0167 m  |  gap = 0.05λ = 0.0500 m  |  N_edges_in_gap ≈ 3.0
[PERFORMANCE] Gmsh configured to utilize 1 threads.
Geometry file saved in data/gmsh_files/strip_dipole.brep successfully
--- Starting Mesh Optimization (Dim: 2) ---
--- Optimization Complete ---
Mesh file saved in data/gmsh_files/strip_dipole.msh successfully
```

The antenna is a surface: a rectangle of zero thickness in the plane $z = 0$, drawn with gmsh's
OpenCASCADE kernel. `mesh_and_gap_parameters` chooses the element size and the width of the feed gap
together, so that at least three elements fit in the gap (its rules are explained in
{doc}`../theory/excitations`).

The second rectangle is the feed gap, glued to the strip by `fragment`: its two edges become lines of
the mesh. **Do not skip it.** Drawn as a single rectangle, the same dipole gives an input resistance
that jumps between 63 and 182 Ω from one element size to the next, because the gap edges then cut
through triangles ({doc}`../user-guide/mesh-size`). With one gmsh thread the mesh is the same at every
run; with several it is not ({doc}`../theory/numerical-limits`).

## 3. Solve

```python
feed_point = np.array([0.0, 0.0, 0.0])        # centre of the strip

(Z, V, I, surface_current_density,
 gap_currents, gap_voltages, impedances, feed_powers) = radiation_algorithm(
    path, frequency, feed_point,
    voltage_amplitude=1,                       # 1 V peak across the gap
    excitation_unit_vector="y",                # the gap field points along the strip
    gap_width=gap_width,
)

print(f"Input impedance: {impedances[0]:.2f} Ohm")
print(f"Input power:     {feed_powers[0] * 1e3:.3f} mW")
check_simulation_stability(impedances, frequency, mesh_size, gap_width)
```

```text
Number of points: 89
Number of triangles: 116
Number of edges: 144
[calculate_z_matrice] Using GPU (NVIDIA T600 Laptop GPU, CuPy 14.1.1) for the impedance matrix assembly (144x144 edges).

Total input power P_in = 0.5*Re(V·I*) = 6.852393e-03 W
Port 0: P_in = 6.8524e-03 W  |  Q_in = 3.8490e-04 VAR  |  Z_in = 72.7377+4.0857j Ohm

Input impedance: 72.74+4.09j Ohm
Input power:     6.852 mW

[Stability Check]
  Port 0: Z_in = 72.74+4.09j Ω  — ✓ OK
  Gap check: gap/mesh = 3.0 ≥ 3 — ✓ OK
  Mesh check: λ/60 — ✓ in recommended range [λ/10, λ/100]
  Overall: ✓ PASS
```

`radiation_algorithm` runs the whole pipeline of {doc}`../theory/galerkin-mom`: it converts the
mesh, builds the RWG functions, assembles $Z$ (on the GPU here, because one was found and the mesh has
more than 100 edges; without one it runs on the CPU), builds the excitation of the gap, solves
$ZI = V$ and computes the input impedance of each port. Every stage is saved in `data/`. The values
returned are the matrix, the excitation and the currents, then one entry per feed point for the gap
current, the voltage, the input impedance and the input power.

**The result**: $Z_\mathrm{in}$ = 72.7 + 4.1j Ω: a small positive reactance, so this strip resonates
a little below 300 MHz. How fine the mesh must be for this number to be trustworthy is the subject of
{doc}`../user-guide/mesh-size`. The line about the GPU depends on the machine. The function also opens an
interactive Plotly view of the surface current, reproduced in {numref}`fig-first-current`.

```{raw} html
<script src="../_static/plotly/plotly.min.js"></script>
```

```{raw} html
:file: ../figures/generated/first_simulation_current.html
```

```{figure} ../figures/generated/first_simulation_current.png
:name: fig-first-current
:class: mom-generated
:width: 85%
:target: ../_images/first_simulation_current.png
:alt: A thin strip along y, coloured by the magnitude of the surface current, red at the centre where the feed point is marked, blue at both ends.

Magnitude of the surface current density (A/m) on the strip, as drawn by `radiation_algorithm`: maximum
at the feed, zero at the ends, the half-sine of a half-wave dipole. Drag to rotate, scroll to zoom.
[Generated by `documentation/figure_notebooks/guide_first_simulation.ipynb` on solver commit `4282bf8`.]{.mom-provenance}
```

## 4. Radiation pattern

```python
radiation_intensity_distribution_over_sphere_surface(path)   # 3D pattern, "Gain" (a directivity), efficiency
antenna_directivity_pattern(path)                            # two polar cuts
```

```text
Hollow sphere loaded: 5150 points, 10296 triangles.

[Antenna Results]
  Total Radiated Power : 0.0068561612963706445 W
  Gain : 1.6341 (2.13 dBi)

[Power Budget]
  P_in  (0.5*Re(V·I*)) = 6.852393e-03 W
  P_rad (sphere integ.) = 6.856161e-03 W
  Raw efficiency       = 100.0550 %
  Port 0: Z_in = 72.7377+4.0857j Ω  |  R_in = 72.7377 Ω
  Total Efficiency     = 100.00 %
```

The first call creates, on its first use, the observation sphere in `data/` (a few seconds), computes
the radiation intensity on it and draws the 3D pattern ({numref}`fig-first-pattern`). The value it
prints as **"Gain" is the directivity**, 2.13 dBi here (the exact value for an infinitely thin
half-wave dipole is 2.15 dBi). The efficiency compares the radiated power, integrated over the sphere,
with the input power; it reads slightly above 100 % before being capped, because of the small error
of that integration ({doc}`../theory/radiated-fields` explains both points). The second call draws
two cuts of the pattern, in the $yz$ and $xy$ planes.

```{raw} html
:file: ../figures/generated/first_simulation_pattern.html
```

```{figure} ../figures/generated/first_simulation_pattern.png
:name: fig-first-pattern
:class: mom-generated
:width: 85%
:target: ../_images/first_simulation_pattern.png
:alt: A doughnut-shaped radiation pattern whose axis is the y axis, red over most of its surface, with a hole along y where the colour falls to green.

3D radiation pattern drawn by `radiation_intensity_distribution_over_sphere_surface`, in dBi over
20 dB of dynamic range: the doughnut of a dipole, with its null along the strip ($y$ axis). Drag to
rotate, scroll to zoom.
[Generated by `documentation/figure_notebooks/guide_first_simulation.ipynb` on solver commit `4282bf8`.]{.mom-provenance}
```

```{figure} ../figures/generated/first_simulation_polar.png
:name: fig-first-polar
:class: mom-generated
:width: 70%
:target: ../_images/first_simulation_polar.png
:alt: Polar plot with two figure-of-eight curves over 40 dB: the yz cut, maximum at 90 and 270 degrees, and the xy cut, maximum at 0 and 180 degrees, both with nulls along the y axis.

The two cuts drawn by `antenna_directivity_pattern`, in dBi over 40 dB below the maximum (its
`dynamic_range` argument). Solid: the $yz$ plane, against the angle from $+y$, so 90° is $+z$.
Dashed: the $xy$ plane, against $\phi$. Both have their nulls along the strip ($\pm y$).
[Generated by `documentation/figure_notebooks/guide_first_simulation.ipynb` on solver commit `4282bf8`. Click to enlarge.]{.mom-provenance}
```

## What was written

```python
for folder in sorted(Path("data").iterdir()):
    print(folder.as_posix() + "/", [f.name for f in sorted(folder.iterdir())])
```

```text
data/antennas_current/ ['strip_dipole_current.mat']
data/antennas_gain_power/ ['strip_dipole_gain_power.mat']
data/antennas_impedance/ ['strip_dipole_impedance.mat']
data/antennas_mesh/ ['hollow_sphere.mat', 'strip_dipole.mat']
data/antennas_mesh1/ ['strip_dipole_mesh1.mat']
data/antennas_mesh2/ ['strip_dipole_mesh2.mat']
data/antennas_sweep/ []
data/gmsh_files/ ['hollow_sphere.msh', 'strip_dipole.brep', 'strip_dipole.msh']
```

(This last block needs `from pathlib import Path`.) What each file contains, and how to reload it
without recomputing, is on {doc}`data-and-files`.

## Next

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} {fas}`folder-open` Data and files
:link: data-and-files
:link-type: doc

What `data/` holds, file by file
:::

:::{grid-item-card} {fas}`tower-broadcast` Radiation, in detail
:link: ../user-guide/radiation
:link-type: doc

Every parameter of `radiation_algorithm`
:::

:::{grid-item-card} {fas}`wave-square` Frequency sweep
:link: ../user-guide/frequency-sweep
:link-type: doc

Input impedance and $S_{11}$ over a band
:::
::::
