# Radiated fields, directivity and gain

{.mom-status}
{bdg-info}`Draft` Written from `solver/utils/dipole_parameters.py`, `solver/utils/point_field.py`, `solver/efield/efield1.py` to `efield3.py`, `solver/optim/farfield.py` and `solver/cma/modal_farfield.py` at commit `5e8d892`, awaiting review.

Once the edge currents $I$ are known, every field quantity follows from them: the field at any point,
the radiated power, the pattern, the directivity, the polarisation and the radar cross section. This
page defines those quantities and says how the solver computes them. It has two ways of doing it,
which agree closely but not exactly.

## The idea

The surface current $\mathbf J = \sum_n I_n\mathbf f_n$ radiates in free space. Far from the antenna
the field is a spherical wave: its amplitude decreases as $1/r$, it is transverse to the direction
$\hat{\mathbf r}$, and its dependence on the direction is the **pattern**. Integrating the power
density of that wave over a large sphere gives the radiated power; comparing the power in one
direction with the average gives the directivity; comparing it with the power taken from the
generator gives the gain.

## The far field of the currents

With the time convention $e^{j\omega t}$ and the Green's function $e^{-jkR}/(4\pi R)$ of
{doc}`efie`, the field of the currents at a distance $r$ large compared with the antenna and with the
wavelength is

```{math}
:label: eq-rf-farfield
\mathbf E(r\hat{\mathbf r}) \simeq -\frac{jk\eta_0}{4\pi r}\,e^{-jkr}
\Big[\mathbf N - (\mathbf N\cdot\hat{\mathbf r})\,\hat{\mathbf r}\Big],
\qquad
\mathbf N(\hat{\mathbf r}) = \int_S \mathbf J(\mathbf r')\,e^{jk\hat{\mathbf r}\cdot\mathbf r'}\,dS' .
```

$\mathbf N$ is the **radiation vector**. With RWG functions it is a sum over the edges,
$\mathbf N = \sum_n I_n\,\mathbf n_n(\hat{\mathbf r})$, where $\mathbf n_n$ is the same integral for
$\mathbf f_n$ alone. Only its components across $\hat{\mathbf r}$, $N_\theta$ and $N_\phi$, radiate.

## Power, intensity, directivity and gain

The **radiation intensity** is the power radiated per unit solid angle (peak phasors, hence the
factor $\tfrac12$ in the power density $\tfrac12\mathrm{Re}\{\mathbf E\times\mathbf H^*\}$):

```{math}
:label: eq-rf-intensity
U(\hat{\mathbf r}) = \frac{r^2\,|\mathbf E|^2}{2\eta_0}
= \frac{k^2\eta_0}{32\pi^2}\Big(|N_\theta|^2 + |N_\phi|^2\Big),
\qquad
P_\mathrm{rad} = \oint U\,d\Omega .
```

For a perfect conductor in free space, the radiated power is also the quadratic form of the real part
of the impedance matrix {footcite:p}`harrington1971theory`:

```{math}
:label: eq-rf-prad-r
P_\mathrm{rad} = \tfrac12\,I^\mathsf{H} R\,I, \qquad R = \mathrm{Re}\,Z .
```

The standard antenna quantities are then {footcite:p}`ieee2014terms`:

```{list-table}
:header-rows: 1
:widths: 22 38 40

* - Quantity
  - Definition
  - Normalised by
* - Directivity
  - $D(\hat{\mathbf r}) = 4\pi\,U(\hat{\mathbf r})/P_\mathrm{rad}$
  - the power that leaves the antenna
* - Radiation efficiency
  - $\eta_\mathrm{rad} = P_\mathrm{rad}/P_\mathrm{in}$
  - the power accepted at the port
* - Gain
  - $G(\hat{\mathbf r}) = 4\pi\,U(\hat{\mathbf r})/P_\mathrm{in} = \eta_\mathrm{rad}\,D(\hat{\mathbf r})$
  - the power accepted at the port
* - Realised gain
  - $G_\mathrm{r} = (1 - |\Gamma|^2)\,G$
  - the power available from the generator
```

$\Gamma$ is the reflection coefficient at the port. For a lossless perfect conductor $\eta_\mathrm{rad} = 1$ and gain
and directivity are the same number; they differ only when power is dissipated in the metal
({doc}`lossy-conductors`) or in loads.

## Polarisation

In the far field $\mathbf E = E_\theta\hat{\boldsymbol\theta} + E_\phi\hat{\boldsymbol\phi}$. With
$e^{j\omega t}$, the wave is **right-hand circularly polarised** when the field rotates clockwise for an
observer looking in the direction of propagation {footcite:p}`ieee2014terms`; that is
$E_\phi = -jE_\theta$. The field splits into a right-hand and a left-hand part:

```{math}
:label: eq-rf-cp
E_\mathrm{R} = \frac{E_\theta + jE_\phi}{\sqrt 2},
\qquad
E_\mathrm{L} = \frac{E_\theta - jE_\phi}{\sqrt 2},
\qquad
\mathrm{AR} = \frac{|E_\mathrm{R}| + |E_\mathrm{L}|}{\big||E_\mathrm{R}| - |E_\mathrm{L}|\big|} .
```

A purely right-hand wave gives $E_\mathrm{L} = 0$, a linear one $|E_\mathrm{R}| = |E_\mathrm{L}|$
(infinite axial ratio AR). The partial intensities $U_\mathrm{R,L} = r^2|E_\mathrm{R,L}|^2/(2\eta_0)$
add up to $U$, and $4\pi U_\mathrm{R,L}/P_\mathrm{rad}$ are the partial directivities, in dBic.

## Radar cross section

In scattering, the field radiated by the induced currents is the scattered field $\mathbf E^s$. The
radar cross section compares the power scattered towards a direction with the power density of the
incident wave:

```{math}
:label: eq-rf-rcs
\sigma = \lim_{r\to\infty} 4\pi r^2\,\frac{|\mathbf E^s|^2}{|\mathbf E^i|^2} .
```

## In this solver

:::{admonition} Theory vs. implementation
:class: theory

The post-processing in `solver/efield/` does not evaluate {eq}`eq-rf-farfield`: it uses a **dipole
model** of the currents and integrates the power over a meshed sphere. The exact radiation vector is
implemented separately, in `solver/optim/farfield.py`.
:::

### The dipole model (`solver/efield/`)

This part of the solver comes from the MATLAB code of {footcite:t}`makarov2002`.
`dipole_parameters.compute_dipole_center_moment` replaces each RWG
function by a point dipole, placed at the middle of the two centroids, with a moment equal to the
integral of $I_n\mathbf f_n$:

```{math}
:label: eq-rf-dipole
\mathbf m_n = \ell_n\,I_n\left(\mathbf r_n^{c-} - \mathbf r_n^{c+}\right),
\qquad
\mathbf r_n = \tfrac12\left(\mathbf r_n^{c+} + \mathbf r_n^{c-}\right).
```

The moment is exact: $\int\mathbf f_n\,dS = \ell_n(\mathbf r_n^{c-} - \mathbf r_n^{c+})$. What the model
approximates is the phase: all the current of $\mathbf f_n$ radiates from the single point
$\mathbf r_n$.

`point_field.radiated_scattered_field_at_a_point` then sums the **complete** fields of these dipoles
(near, intermediate and far terms) at the observation point $\mathbf r$. With
$\mathbf R = \mathbf r - \mathbf r_n$, $R = |\mathbf R|$,
$C = \frac{1}{R^2}\big(1 + \frac{1}{jkR}\big)$ and
$\mathbf M = \frac{\mathbf R\cdot\mathbf m_n}{R^2}\,\mathbf R$:

```{math}
:label: eq-rf-dipole-fields
\begin{aligned}
\mathbf H_n &= \frac{jk}{4\pi}\,(\mathbf m_n\times\mathbf R)\,C\,e^{-jkR},\\
\mathbf E_n &= \frac{\eta_0}{4\pi}\Big[(\mathbf M - \mathbf m_n)\Big(\frac{jk}{R} + C\Big) + 2\,\mathbf M\,C\Big]e^{-jkR}.
\end{aligned}
```

Far away only the $jk/R$ term survives and $\mathbf E_n$ tends to {eq}`eq-rf-farfield` with
$\mathbf N = \mathbf m_n\,e^{jk\hat{\mathbf r}\cdot\mathbf r_n}$. `dipole_parameters.compute_e_h_field`
adds the contributions of all the edges and returns $\mathbf E$, $\mathbf H$, the time-averaged
Poynting vector $\mathbf S = \tfrac12\mathrm{Re}\{\mathbf E\times\mathbf H^*\}$, $w = |\mathbf S|$ and
$u = |\mathbf r|^2 w$.

**`efield1`** prints these quantities at one point (`calculate_electric_magnetic_field_at_point`), and
the $\theta$, $\phi$, RHCP and LHCP components with the axial ratio
(`analyze_point_fields_polarization`). In scattering mode it also prints the RCS,
`efield1.compute_rcs`: $\sigma = 4\pi|\mathbf r|^2|\mathbf E^s|^2/|\mathbf E_0|^2$, i.e.
{eq}`eq-rf-rcs` at a finite distance, with $\mathbf E_0$ the `polarization` vector given to
`scattering_algorithm` (which uses it as is, without normalising it). The result therefore does not
depend on the amplitude chosen for the incident wave.

**`efield2`** (`radiation_intensity_distribution_over_sphere_surface`) works on a sphere of **radius
100 m** centred at the origin: the unit sphere of `gmsh_function.create_hollow_sphere` (10 296
triangles), scaled by 100 in `load_or_create_hollow_sphere`. With $\mathbf c_t$ and $A_t$ the
centroid and area of sphere triangle $t$:

```{math}
:label: eq-rf-efield2
P_\mathrm{rad} \approx \sum_t |\mathbf S(\mathbf c_t)|\,A_t,
\qquad
D_\mathrm{max} \approx \frac{4\pi\,\max_t |\mathbf c_t|^2\,|\mathbf S(\mathbf c_t)|}{P_\mathrm{rad}} .
```

The value is printed and saved (`gainLinear` in `path.mat_gain_power`) as **"Gain"**, but it is
normalised by $P_\mathrm{rad}$: it is the **directivity**. The 3D pattern is drawn from the same
quantity at the sphere vertices, in dB.

**The efficiency** printed by `efield2.compute_antenna_efficiency_metrics` is
$\min(P_\mathrm{rad}/P_\mathrm{in},\,1)$, with $P_\mathrm{rad}$ from {eq}`eq-rf-efield2` and
$P_\mathrm{in} = \tfrac12\mathrm{Re}\{V^\mathsf T I^*\}$ from the port results ({doc}`excitations`). A
warning is printed above 101 %. The consequences for lossy antennas are discussed in
{doc}`lossy-conductors`.

**Circular polarisation.** `efield2.polar_circular_gain_distribution_over_sphere` (aliases
`display_rhcp_gain`, `display_lhcp_gain`) and `efield3.antenna_circular_directivity_pattern` compute
$4\pi U_\mathrm{R,L}/P_\mathrm{rad}$ with {eq}`eq-rf-cp` (`dipole_parameters.compute_circular_components`),
labelled "Gain (dBic)" or "Directivity (dBic)": they are partial directivities. The formulas follow
the IEEE convention: a pure RHCP wave has $\arg E_\phi - \arg E_\theta = -90°$.

**`efield3`** draws two cuts at $r$ = 100 m, normalised by the $P_\mathrm{rad}$ that `efield2` saved
(so `efield2` must run first). `antenna_directivity_pattern`: the $yz$ plane ($\phi = 90°$), against
the angle from the $+y$ axis, and the $xy$ plane ($\theta = 90°$), against $\phi$.
`antenna_circular_directivity_pattern`: the $yz$ plane ($\phi = 90°$) and the $xz$ plane
($\phi = 0°$), against the angle from the $+y$ and $+x$ axis respectively.

**Characteristic modes.** `cma.modal_farfield.compute_modal_farfield` uses the same dipole model and
the same sphere for the directivity of one mode.

### The exact radiation integral (`solver/optim/farfield.py`)

`RadiationOperator(points, triangles, edges, frequency, quad_order=7)` evaluates $\mathbf n_n$ in
{eq}`eq-rf-farfield` with a Gauss–Dunavant rule on each triangle ({doc}`numerical-integration`), so
the phase is integrated instead of being sampled at one point. `intensity` gives $U_\theta$ and
$U_\phi$ of {eq}`eq-rf-intensity`. `radiated_power_sphere` integrates $U$ with Gauss–Legendre nodes in
$\cos\theta$ and the trapezoid rule in $\phi$, on a grid sized from $ka$ ($a$ the radius of the mesh
about its centroid): enough to integrate $|\mathbf N|^2$ to rounding error. It uses the constants
$\varepsilon_0$, $\mu_0$ of `calculate_z_matrice`, so its $P_\mathrm{rad}$ can be compared with
{eq}`eq-rf-prad-r` to machine precision. The reduced multiport model uses it for its far-field
objectives ({doc}`reduced-multiport`).

## How close are the two? A check

The centre-fed strip of {doc}`excitations` (75 MHz, 2 m × 5 cm, gap 50 mm) was solved on five meshes
and its radiated power computed four ways: {eq}`eq-rf-prad-r` as the reference, the exact radiation
integral, and the dipole model integrated either as `efield2` does or on an accurate Gauss–Legendre
sphere.

```{figure} ../figures/generated/radiated_fields_models.svg
:name: fig-rf-models
:class: mom-generated
:width: 100%
:target: ../_images/radiated_fields_models.svg
:alt: Left, two directivity curves of a strip dipole against the angle from the z axis; they overlap down to about minus 20 dBi. Right, relative difference of four radiated-power estimates against the number of triangles, on log axes: the port power and the exact integral stay between 1e-16 and 1e-13, the dipole model levels off near 2e-4 on a Gauss sphere and near 6e-4 on the efield2 sphere.

(a) Directivity in the $xz$ plane (the plane of the strip axis), 164 triangles: exact radiation
integral and dipole model at $r$ = 100 m normalised as `efield2` does. Peak 1.6573 (2.19 dBi) exact
against 1.6560; within 20 dB of the peak the curves differ by at most 0.004 dB, more in the null along
the strip. (b) Relative difference of each estimate of $P_\mathrm{rad}$ from
$\tfrac12 I^\mathsf H R I$.
[Generated by `documentation/figure_notebooks/theory_radiated_fields.ipynb` on solver commit `5e8d892`; all values in `radiated_fields.json`. Click to enlarge.]{.mom-provenance}
```

What the figure shows:

- **Port power and $\tfrac12 I^\mathsf H R I$ agree to $10^{-15}$.** This is an algebraic identity, not
  a physical check: with $V = ZI$ and $Z$ symmetric,
  $\tfrac12\mathrm{Re}\{V^\mathsf T I^*\} = \tfrac12\mathrm{Re}\{I^\mathsf H Z I\} = \tfrac12 I^\mathsf H R I$.
  It holds because the impedance matrix is assembled symmetric (`fill_mode="full_sym"`).
- **The exact radiation integral agrees to $10^{-13}$.** Two independent computations, one through
  the Green's function in $Z$, the other through the far field, give the same radiated power.
- **The dipole model is high by about $2\times10^{-4}$ of its own.** On the Gauss–Legendre sphere its
  error goes from $1.5\times10^{-3}$ (24 triangles) to $2.2\times10^{-4}$ (324 triangles). On the
  finest mesh the value does not change when the sphere grid is doubled or the radius raised from
  100 m to 1000 m: it is the error of the model, not of the integration.
- **The `efield2` sphere adds about $3.6\times10^{-4}$**: one point per facet on 10 296 flat facets.
  The total error of `efield2` levels off at $5.8\times10^{-4}$.
- The printed efficiency is 1.0000 on every mesh: the dipole-model $P_\mathrm{rad}$ exceeds
  $P_\mathrm{in}$ by up to 0.19 %, which the cap at 100 % hides.

For a pattern or a directivity these differences do not matter. They matter when a small difference
of powers is the result: an efficiency close to 1, the loss of a good conductor, the residual field
of nearly cancelling currents (the reason `optim/farfield.py` was written, as its docstring explains).
Use {eq}`eq-rf-prad-r` or `RadiationOperator` there.

## Assumptions and limits

- **Point dipoles.** Each RWG function radiates from one point. The fields are good where the
  observation point is far from the triangles compared with their size, so `efield1` near the surface
  of the antenna is not reliable.
- **Finite radius.** `efield2` and `efield3` evaluate the complete dipole fields at 100 m, not the
  limit $r\to\infty$. For this strip, 100 m and 1000 m give the same power; for an antenna whose size
  is not small compared with 100 m, the observation sphere is no longer in the far field.
- **Origin.** The sphere is centred at the coordinate origin: the antenna should be meshed around it.
- **Naming.** "Gain" in `efield2` and "Gain (dBic)" are directivities; neither realised gain nor
  $G = \eta_\mathrm{rad}D$ with a computed $P_\mathrm{cond}$ is reported by `solver/efield/`.
- **RCS** is evaluated at a finite distance (the observation point given to `efield1`), not in the
  limit $r\to\infty$.

## Where to look in the code

```{list-table}
:header-rows: 1
:widths: 45 55

* - Function
  - Computes
* - `utils/dipole_parameters.compute_dipole_center_moment`
  - $\mathbf m_n$, $\mathbf r_n$ of {eq}`eq-rf-dipole`
* - `utils/point_field.radiated_scattered_field_at_a_point`
  - fields of every dipole, {eq}`eq-rf-dipole-fields`
* - `utils/dipole_parameters.compute_e_h_field`
  - $\mathbf E$, $\mathbf H$, $\mathbf S$, $w$, $u$ at one point
* - `utils/dipole_parameters.compute_spherical_e_field`, `compute_circular_components`
  - $E_\theta$, $E_\phi$; $E_\mathrm R$, $E_\mathrm L$ of {eq}`eq-rf-cp`
* - `efield/efield1.calculate_electric_magnetic_field_at_point`, `analyze_point_fields_polarization`
  - fields, polarisation, axial ratio and RCS at one point
* - `efield/efield1.compute_rcs`
  - $4\pi r^2|\mathbf E^s|^2/|\mathbf E_0|^2$
* - `efield/efield2.compute_total_radiated_power`
  - $P_\mathrm{rad}$ and $u$ on the sphere, {eq}`eq-rf-efield2`
* - `efield/efield2.radiation_intensity_distribution_over_sphere_surface`
  - $D_\mathrm{max}$ (printed as "Gain"), efficiency, 3D pattern
* - `efield/efield3.antenna_directivity_pattern`, `antenna_circular_directivity_pattern`
  - two pattern cuts each
* - `optim/farfield.RadiationOperator`
  - $\mathbf N$, $U$, $P_\mathrm{rad}$ from the exact radiation integral
```

## Continue

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} {fas}`arrow-right` Next: characteristic modes
:link: characteristic-modes
:link-type: doc

Radiating currents of the shape alone
:::

:::{grid-item-card} {fas}`sliders` Reduced multiport model
:link: reduced-multiport
:link-type: doc

Exact power balance and far-field objectives
:::

:::{grid-item-card} {fas}`clipboard-check` Feature status
:link: ../validation/feature-status
:link-type: doc

What is validated, and how
:::
::::

## References

```{footbibliography}
```
