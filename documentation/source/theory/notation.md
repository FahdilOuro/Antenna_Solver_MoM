# Notation and conventions

{.mom-status}
{bdg-info}`Draft` Written from `solver/rwg/rwg3.py`, `solver/utils/impmet.py`, `solver/rwg/rwg4.py` and `solver/utils/dipole_parameters.py` at commit `181e5ad`, awaiting review.

The symbols, units and sign conventions used on every theory page, and how they map to the code.
When two pages seem to disagree, this page is the reference.

## Conventions

**Time-harmonic fields, $e^{j\omega t}$.** All fields and currents are complex amplitudes (phasors)
at a single angular frequency $\omega = 2\pi f$; the physical field is
$\mathrm{Re}\{\mathbf E(\mathbf r)\,e^{j\omega t}\}$. With this convention an outgoing spherical
wave is $e^{-jkr}/r$, which is the form the solver uses in its Green's function
{eq}`eq-green`.

**Peak amplitudes, not RMS.** A voltage $V$ or a current $I$ is the amplitude of the sinusoid. The
time-averaged powers therefore carry a factor ½:

```{math}
:label: eq-power-convention
P = \tfrac12\,\mathrm{Re}\{V I^*\},
\qquad
\langle\mathbf S\rangle = \tfrac12\,\mathrm{Re}\{\mathbf E\times\mathbf H^*\}.
```

The code uses exactly these expressions: the input power of each port in
`rwg4.compute_port_results`, and the Poynting vector in `dipole_parameters.compute_e_h_field`.

**Free-space Green's function.**

```{math}
:label: eq-green
G(\mathbf r,\mathbf r') = \frac{e^{-jkR}}{4\pi R},
\qquad
R = |\mathbf r - \mathbf r'|,
\qquad
k = \frac{\omega}{c}.
```

In `solver/utils/impmet.py` the factor $1/(4\pi)$ is kept in the prefactors
(`pref_A = 1j*omega*mu/(4*pi)`, `pref_Phi = 1/(1j*omega*epsilon*4*pi)`) and the kernel evaluated is
`exp(-k_complex*R)/R` with `k_complex = 1j*k`.

**Units.** SI throughout: metres, seconds, hertz, volts, amperes, ohms. Mesh coordinates are in
metres; a geometry drawn in millimetres must be scaled before it is solved.

## Physical constants

The constants are defined in the code, not imported from a library. All numerical results depend
on these values.

```{list-table}
:header-rows: 1
:widths: 14 24 26 36

* - Symbol
  - Value in the solver
  - Exact value
  - Where
* - $\varepsilon_0$
  - $8.854\times10^{-12}$ F/m
  - $8.8541878128\times10^{-12}$ F/m (CODATA), relative difference $-2.1\times10^{-5}$
  - `rwg3.calculate_z_matrice` (and its sweep and lumped variants)
* - $\mu_0$
  - $1.257\times10^{-6}$ H/m
  - $\approx 4\pi\times10^{-7}$ H/m, relative difference $+2.9\times10^{-4}$
  - same, and `lossy_electric_conductor.MU_0`, `optim/farfield.MU_0`
* - $c$
  - $1/\sqrt{\varepsilon_0\mu_0} = 2.997524\times10^{8}$ m/s
  - $299\,792\,458$ m/s, relative difference $-1.3\times10^{-4}$
  - derived in `rwg3`
* - $\eta_0$
  - $\sqrt{\mu_0/\varepsilon_0} = 376.79\ \Omega$
  - $376.73\ \Omega$ (from CODATA $\mu_0$, $\varepsilon_0$)
  - derived in `rwg3`
* - $k$
  - $\omega/c$
  - –
  - derived in `rwg3`
```

Two helper modules use other values of $c$, for estimates only: $3\times10^8$ m/s in
`gmsh_function.mesh_and_gap_parameters` and `check_simulation_stability` (mesh and gap sizing), and
the exact $299\,792\,458$ m/s in `cma.cma_report` (electrical size $ka$ in the reports). Neither enters
the impedance matrix.

## Symbols

Vectors are bold ($\mathbf E$), unit vectors carry a hat ($\hat{\mathbf n}$), matrices are upright
capitals ($Z$). In a matrix entry $Z_{mn}$, $m$ is the **testing** function and $n$ the **basis**
(source) function.

### Fields and sources

```{list-table}
:header-rows: 1
:widths: 18 52 30
:class: mom-symbols

* - Symbol
  - Meaning
  - Unit
* - $\mathbf E,\ \mathbf H$
  - electric and magnetic fields
  - V/m, A/m
* - $\mathbf E^{i},\ \mathbf E^{s}$
  - incident (impressed) and scattered electric fields
  - V/m
* - $\mathbf J$
  - surface current density on the conductor
  - A/m
* - $\rho_s$
  - surface charge density
  - C/m²
* - $\mathbf A,\ \Phi$
  - magnetic vector potential and electric scalar potential
  - Wb/m, V
* - $G(\mathbf r,\mathbf r')$
  - free-space Green's function {eq}`eq-green`
  - 1/m
* - $S,\ \hat{\mathbf n}$
  - conducting surface and its unit normal
  - –
```

### Mesh and basis functions

See {doc}`rwg-basis-functions` for the full table.

```{list-table}
:header-rows: 1
:widths: 18 52 30
:class: mom-symbols

* - Symbol
  - Meaning
  - Unit
* - $T_n^\pm,\ A_n^\pm$
  - the two triangles of basis function $n$ and their areas
  - –, m²
* - $\ell_n$
  - length of edge $n$
  - m
* - $\mathbf f_n$
  - RWG basis function of edge $n$
  - dimensionless
* - $N$
  - number of basis functions (unknowns)
  - –
```

### Linear system and derived quantities

```{list-table}
:header-rows: 1
:widths: 18 52 30
:class: mom-symbols

* - Symbol
  - Meaning
  - Unit
* - $Z = R + jX$
  - impedance matrix ($N\times N$), its real and imaginary parts
  - Ω
* - $I = (I_n)$
  - coefficients of the current expansion
  - A/m
* - $V = (V_m)$
  - excitation vector (tested incident field)
  - V
* - $\mathsf G$
  - Gram matrix of the RWG functions, $\mathsf G_{mn} = \int_S \mathbf f_m\cdot\mathbf f_n\,dS$
  - m²
* - $Z_s$
  - surface impedance of a lossy conductor
  - Ω
* - $Z_\mathrm{in}$
  - input impedance of a port
  - Ω
* - $\lambda_n,\ \mathbf J_n$
  - characteristic eigenvalue and characteristic current of mode $n$
  - –, A/m
```

Two letters have two uses, always distinguished by their form: $A_n^\pm$ (an area, indexed) and
$\mathbf A$ (the vector potential, bold); $G(\mathbf r, \mathbf r')$ (Green's function, with
arguments) and $\mathsf G$ (Gram matrix, sans-serif).

:::{admonition} Three different impedances
:class: note

- $\eta_0 \approx 377\ \Omega$, the **wave impedance** of free space, ratio of $|\mathbf E|$ to
  $|\mathbf H|$ in a plane wave.
- $Z_s$, the **surface impedance** of a real metal, ratio of tangential $\mathbf E$ to surface
  current (see {doc}`lossy-conductors`).
- $Z_\mathrm{in}$, the **input impedance** seen at a port. The solver obtains it from the complex
  power delivered to the port (see {doc}`excitations`).

The matrix $Z$ has units of ohms too, but it is none of these: it couples basis functions.
:::

## Names in the code

```{list-table}
:header-rows: 1
:widths: 28 30 42

* - Quantity
  - Name in the code
  - Defined in
* - $Z$
  - `matrice_z`
  - `rwg3.calculate_z_matrice`
* - $I$, $V$
  - `current`, `voltage`
  - `rwg4.calculate_current_radiation`, `calculate_current_scattering`
* - $\boldsymbol\rho_n^{c\pm}$
  - `vecteurs_rho.vecteur_rho_plus / _minus`
  - `rwg2.Vecteurs_Rho`
* - $\omega,\ \mu_0,\ \varepsilon_0,\ c,\ \eta_0$
  - `omega`, `mu`, `epsilon`, `light_speed_c`, `eta`
  - `rwg3`; saved in the impedance and current `.mat` files
* - $jk$
  - `k_complex` (`complex_k` in some modules)
  - `rwg3`, `efield`
* - $\mathsf G$
  - `Gram_rwg`
  - `impmet.rwg_gram_matrix`
* - $Z_s$
  - `impedance_zs`
  - `lossy_electric_conductor.calculate_normal_surface_impedance`
```

```{note}
`k_complex` is $jk$, not a complex wavenumber $k' - jk''$: the code writes $e^{-jkR}$ as
`exp(-k_complex * R)`. The medium is lossless free space everywhere.
```
