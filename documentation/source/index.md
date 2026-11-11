---
html_theme.sidebar_secondary.remove: true
---

# Antenna Solver MoM

```{raw} html
<div class="mom-hero">
  <p class="mom-tagline">
    A Method of Moments solver for antennas on triangular meshes, written in Python.
    RWG basis functions, an EFIE impedance matrix with singularity treatment,
    radiation and scattering, frequency sweeps, characteristic modes and load optimisation.
  </p>
  <div class="mom-actions">
    <a class="mom-primary" href="getting-started/index.html">Get started</a>
    <a class="mom-secondary" href="theory/index.html">Read the theory</a>
    <a class="mom-secondary" href="https://github.com/FahdilOuro/Antenna_Solver_MoM">Source on GitHub</a>
  </div>
</div>
```

The solver began as a translation of the MATLAB code of S. N. Makarov's *Antenna and EM Modeling
with MATLAB* {footcite:p}`makarov2002` and has since been extended: higher-order and singular
quadrature for the impedance matrix, GPU assembly, multi-port gap sources, lossy conductors,
characteristic mode analysis and a reduced multiport model for optimisation.

## How a simulation flows

Every stage writes its result to disk, so a later stage can be re-run without recomputing the earlier
ones. Click a box to open its page.

```{raw} html
<div class="mom-pipeline" role="img" aria-label="Pipeline of the solver: geometry and mesh, RWG basis functions, impedance matrix Z, excitation V, solution of Z I = V, results. Characteristic modes and the reduced multiport model branch off the impedance matrix.">
<svg viewBox="0 0 1200 320" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="mom-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path class="head" d="M0,0 L10,5 L0,10 z"/>
    </marker>
    <marker id="mom-arrow-branch" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path class="head branch" d="M0,0 L10,5 L0,10 z"/>
    </marker>
  </defs>

  <!-- main chain: six 180-wide boxes, 24 apart -->
  <path class="flow" d="M181 58 H203" marker-end="url(#mom-arrow)"/>
  <path class="flow" d="M385 58 H407" marker-end="url(#mom-arrow)"/>
  <path class="flow" d="M589 58 H611" marker-end="url(#mom-arrow)"/>
  <path class="flow" d="M793 58 H815" marker-end="url(#mom-arrow)"/>
  <path class="flow" d="M997 58 H1019" marker-end="url(#mom-arrow)"/>

  <!-- branches leaving the impedance matrix -->
  <path class="flow branch" d="M498 100 V165 H330 V222" marker-end="url(#mom-arrow-branch)"/>
  <path class="flow branch" d="M498 165 H720 V222" marker-end="url(#mom-arrow-branch)"/>
  <text class="caption" x="320" y="205" text-anchor="end">no excitation needed</text>

  <a href="user-guide/geometry-and-meshing.html"><g class="box core">
    <rect x="1" y="16" width="180" height="84" rx="10"/>
    <text class="title" x="91" y="52" text-anchor="middle">Geometry &amp; mesh</text>
    <text class="sub" x="91" y="80" text-anchor="middle">gmsh · .msh → .mat</text>
  </g></a>

  <a href="theory/rwg-basis-functions.html"><g class="box core">
    <rect x="205" y="16" width="180" height="84" rx="10"/>
    <text class="title" x="295" y="52" text-anchor="middle">RWG functions</text>
    <text class="sub" x="295" y="80" text-anchor="middle">rwg1 · rwg2</text>
  </g></a>

  <a href="theory/galerkin-mom.html"><g class="box core">
    <rect x="409" y="16" width="180" height="84" rx="10"/>
    <text class="title" x="499" y="52" text-anchor="middle">Impedance Z</text>
    <text class="sub" x="499" y="80" text-anchor="middle">rwg3 · impmet</text>
  </g></a>

  <a href="theory/excitations.html"><g class="box core">
    <rect x="613" y="16" width="180" height="84" rx="10"/>
    <text class="title" x="703" y="52" text-anchor="middle">Excitation V</text>
    <text class="sub" x="703" y="80" text-anchor="middle">gap source · wave</text>
  </g></a>

  <a href="user-guide/radiation.html"><g class="box core">
    <rect x="817" y="16" width="180" height="84" rx="10"/>
    <text class="title" x="907" y="52" text-anchor="middle">Solve Z I = V</text>
    <text class="sub" x="907" y="80" text-anchor="middle">rwg4</text>
  </g></a>

  <a href="user-guide/fields-and-patterns.html"><g class="box core">
    <rect x="1021" y="16" width="178" height="84" rx="10"/>
    <text class="title" x="1110" y="52" text-anchor="middle">Results</text>
    <text class="sub" x="1110" y="80" text-anchor="middle">rwg5 · efield</text>
  </g></a>

  <a href="theory/characteristic-modes.html"><g class="box branch">
    <rect x="200" y="226" width="260" height="84" rx="10"/>
    <text class="title" x="330" y="262" text-anchor="middle">Characteristic modes</text>
    <text class="sub" x="330" y="290" text-anchor="middle">X Jₙ = λₙ R Jₙ · cma/</text>
  </g></a>

  <a href="theory/reduced-multiport.html"><g class="box branch">
    <rect x="590" y="226" width="260" height="84" rx="10"/>
    <text class="title" x="720" y="262" text-anchor="middle">Reduced multiport</text>
    <text class="sub" x="720" y="290" text-anchor="middle">loads, matching · optim/</text>
  </g></a>
</svg>
</div>
```

## Find your way

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} {fas}`book-open` Understand
:link: theory/index
:link-type: doc

How the solver works: from Maxwell's equations to the EFIE, RWG basis functions, the impedance
matrix and its singular integrals, characteristic modes.
:::

:::{grid-item-card} {fas}`play` Use
:link: getting-started/index
:link-type: doc

Install, run a first antenna, then the user guide: meshing, feeds, frequency sweeps, port
impedance, patterns, characteristic modes, GPU.
:::

:::{grid-item-card} {fas}`code` Develop
:link: developer-guide/index
:link-type: doc

Architecture, data flow, which file to change for which evolution, the tests to run, and the
API reference generated from the code.
:::
::::

## What the solver computes

::::{grid} 1 2 3 3
:gutter: 2

:::{grid-item-card} Radiation
:link: user-guide/radiation
:link-type: doc

Feed one or several ports (gap sources, amplitude and phase per port) and get the currents, the
input impedance and the input power. Lumped R, L, C loads and lossy conductors are supported.
:::

:::{grid-item-card} Scattering
:link: user-guide/scattering
:link-type: doc

Illuminate a structure with a plane wave of given direction and polarisation and get the induced
currents.
:::

:::{grid-item-card} Frequency sweeps
:link: user-guide/frequency-sweep
:link-type: doc

Impedance matrices for a whole band assembled in one pass (CPU or CUDA GPU); input impedance and
reflection coefficient of every port (all ports driven together), Smith charts, bandwidth.
:::

:::{grid-item-card} Fields and patterns
:link: user-guide/fields-and-patterns
:link-type: doc

Near and far fields, 3D directivity patterns, polar cuts, circular polarisation (RHCP/LHCP).
:::

:::{grid-item-card} Characteristic modes
:link: user-guide/characteristic-modes
:link-type: doc

Eigenvalues, characteristic angles, modal significance, modal currents and far fields, mode
tracking across frequency, numerical health reports.
:::

:::{grid-item-card} Optimisation
:link: user-guide/optimisation
:link-type: doc

Exact reduction of the MoM system to its ports, gain bounds, optimal reactive loads, Bayesian
search, matching networks with real components.
:::
::::

```{admonition} About this documentation
:class: note

This is version 0.1 of the site: the structure is in place and pages are being written one by one.
Each page shows its status at the top, {bdg-secondary-line}`Planned` pages contain the outline of
what they will cover. The state of every solver feature, including the known issues, is listed in
{doc}`validation/feature-status`.
```

```{footbibliography}
```

```{toctree}
:hidden:
:maxdepth: 2

getting-started/index
theory/index
user-guide/index
developer-guide/index
api/index
validation/index
examples/index
bibliography
```
