# From Maxwell's equations to the EFIE

{.mom-status}
{bdg-info}`Draft` Written from `solver/utils/impmet.py`, `solver/rwg/rwg4.py` and `solver/utils/gap_source.py` at commit `181e5ad`, awaiting review.

Why the solver only needs a mesh of the metal surface: the field everywhere follows from the current
on that surface, and the current is fixed by a single condition on the metal. That condition, written
as an integral equation, is the electric field integral equation (EFIE) that the Method of Moments
solves.

## The idea

An antenna is a piece of metal in free space. When it is fed, or illuminated by a wave, currents
flow on its surface. These currents radiate a field, the **scattered field** $\mathbf E^s$, which
adds to the field that excites the structure, the **incident field** $\mathbf E^i$.

On a perfect conductor the total tangential electric field must vanish. That gives one equation,
on the surface only, whose unknown is the surface current $\mathbf J$. Once $\mathbf J$ is known,
the field anywhere in space is an integral over the surface. The volume around the antenna never
needs to be meshed.

## Maxwell's equations in the frequency domain

With the time convention $e^{j\omega t}$ (see {doc}`notation`), in a homogeneous medium of
permittivity $\varepsilon$ and permeability $\mu$:

```{math}
:label: eq-maxwell
\nabla\times\mathbf E = -j\omega\mu\,\mathbf H,
\qquad
\nabla\times\mathbf H = j\omega\varepsilon\,\mathbf E + \mathbf J,
\qquad
\nabla\cdot\mathbf J = -j\omega\,\rho .
```

The last relation, the continuity equation, links the current to the charge it accumulates. On a
surface it reads $\nabla_s\cdot\mathbf J = -j\omega\,\rho_s$.

## Potentials and the radiated field

The field radiated by surface currents is conveniently written with two potentials. In the Lorenz
gauge,

```{math}
:label: eq-field-potentials
\mathbf E^s = -j\omega\,\mathbf A - \nabla\Phi,
```

```{math}
:label: eq-potentials
\begin{aligned}
\mathbf A(\mathbf r) &= \mu\int_S \mathbf J(\mathbf r')\,G(\mathbf r,\mathbf r')\,dS',\\
\Phi(\mathbf r) &= \frac{1}{\varepsilon}\int_S \rho_s(\mathbf r')\,G(\mathbf r,\mathbf r')\,dS',
\end{aligned}
```

with the surface charge $\rho_s = -\nabla_s'\cdot\mathbf J/(j\omega)$ given by the continuity equation
and the free-space Green's function $G = e^{-jkR}/(4\pi R)$ of {eq}`eq-green`. The vector potential
comes from the current, the scalar potential from the charge. Keeping both, instead of combining
them into a single operator acting on $\mathbf J$, is the **mixed-potential** form: it only involves
first derivatives of $\mathbf J$, which is what RWG functions are built for.

Substituting $\rho_s$, the same field in terms of $\mathbf J$ alone is

```{math}
:label: eq-field-of-current
\mathbf E^s(\mathbf r) = -\frac{j\eta}{k}\left[\,k^2\!\int_S \mathbf J\,G\,dS'
+ \nabla\!\int_S \left(\nabla_s'\cdot\mathbf J\right) G\,dS'\right],
\qquad \eta = \sqrt{\mu/\varepsilon}.
```

## The boundary condition and the EFIE

On the surface $S$ of a perfect electric conductor (PEC), the tangential component of the total field
is zero:

```{math}
:label: eq-pec
\hat{\mathbf n}\times\left(\mathbf E^i + \mathbf E^s\right) = \mathbf 0
\qquad \text{on } S .
```

With {eq}`eq-field-potentials` this becomes the electric field integral equation in mixed-potential
form {footcite:p}`rao1982`:

```{math}
:label: eq-efie
\hat{\mathbf n}\times\Big[\,j\omega\,\mathbf A(\mathbf r) + \nabla\Phi(\mathbf r)\Big]
= \hat{\mathbf n}\times\mathbf E^i(\mathbf r),
\qquad \mathbf r \in S .
```

The unknown $\mathbf J$ is inside $\mathbf A$ and $\Phi$; the right-hand side is known. The Method of
Moments turns {eq}`eq-efie` into a linear system by expanding $\mathbf J$ in RWG functions and
testing the equation with the same functions (see {doc}`rwg-basis-functions` and
{doc}`galerkin-mom`). Textbook derivations of the integral equation and of the moment method are in
{footcite:t}`harrington1993`, chapter 12 of {footcite:t}`balanis2023` and {footcite:t}`gibson2021`.

## Two kinds of problems, two right-hand sides

The operator on the left of {eq}`eq-efie` depends on the geometry and the frequency only. What
distinguishes a transmitting antenna from a scatterer is the incident field.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Radiation: a voltage across a gap
The antenna is fed by a voltage $V$ across a narrow gap of width $W$. The source is modelled as an
impressed field confined to the gap,

$$
\mathbf E^i =
\begin{cases}
(V/W)\,\hat{\mathbf e} & \text{in the gap},\\
\mathbf 0 & \text{elsewhere},
\end{cases}
$$

along the feed direction $\hat{\mathbf e}$. See {doc}`excitations`.
:::

:::{grid-item-card} Scattering: an incident plane wave
The structure is illuminated by a plane wave travelling along $\hat{\mathbf k}$, with field vector
$\mathbf E_0 \perp \hat{\mathbf k}$:

$$
\mathbf E^i(\mathbf r) = \mathbf E_0\,e^{-jk\,\hat{\mathbf k}\cdot\mathbf r}.
$$

The induced current re-radiates the scattered field, from which the radar cross section follows.
:::
::::

## In this solver

:::{admonition} Theory vs. implementation
:class: theory

Everything above is general. This section states what the code implements and the conventions it
fixes.
:::

- **Medium.** Free space everywhere, with the constants of {doc}`notation`. No dielectric, no
  substrate, no infinite ground plane and no image theory is implemented in `solver/rwg/` or
  `solver/utils/`: a ground plane, if needed, must be meshed as metal.
- **Conductors.** The surface $S$ is a triangulated sheet of zero thickness, treated as a perfect
  conductor. Losses can be added through a surface impedance (see {doc}`lossy-conductors`).
- **Green's function and potentials.** `solver.utils.impmet` assembles the two terms of
  {eq}`eq-efie` with the prefactors `pref_A = jωμ/(4π)` (vector potential) and
  `pref_Phi = 1/(jωε·4π)` (scalar potential) and the kernel $e^{-jkR}/R$.
- **Plane wave.** `rwg4.calculate_current_scattering` evaluates
  $\mathbf E^i = \mathbf E_0\,e^{-jk\,\hat{\mathbf k}\cdot\mathbf r}$ with `polarization` as
  $\mathbf E_0$ (its magnitude is the amplitude, in V/m) and `wave_incident_direction` as
  $\hat{\mathbf k}$. **Neither vector is normalised by the code:** pass a unit vector for
  $\hat{\mathbf k}$ and a polarisation perpendicular to it.
- **Gap source.** `gap_source.multiple_gap_sources` feeds every RWG function that has at least one
  of its two triangles with its centroid inside a cube of side $W$ (`gap_width`) centred on the feed
  point; only the triangles inside the cube contribute. The field is along the axis
  `excitation_unit_vector` (`'x'`, `'y'` or `'z'`). The code refers to it as the enhanced gap
  source model of {footcite:t}`ding2013` (eq. 7). Several ports can be fed at once, each with its
  own amplitude and phase.
- **Fields from the currents.** Once $\mathbf J$ is known, the fields in space are not computed with
  {eq}`eq-field-of-current` directly: `solver/efield/` replaces each RWG function by a small dipole
  (see {doc}`radiated-fields`).

## Assumptions and limits

- Perfect (or surface-impedance) conductors only, in free space.
- Zero-thickness surfaces: the model has no notion of the metal thickness.
- One frequency at a time; a frequency sweep repeats the solution at each frequency.
- The gap model represents a feed of finite width $W$; the input impedance depends on $W$ and on
  the mesh inside the gap (see {doc}`excitations`).

## Continue

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} {fas}`arrow-right` Next: RWG basis functions
:link: rwg-basis-functions
:link-type: doc

The functions in which $\mathbf J$ is expanded.
:::

:::{grid-item-card} {fas}`table` Notation and conventions
:link: notation
:link-type: doc

Symbols, units, constants and their names in the code.
:::

:::{grid-item-card} {fas}`bolt` Excitations
:link: excitations
:link-type: doc

How the gap source and the plane wave enter the linear system.
:::
::::

## References

```{footbibliography}
```
