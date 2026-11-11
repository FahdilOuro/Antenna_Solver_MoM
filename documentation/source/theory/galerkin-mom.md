# Method of Moments and the impedance matrix

{.mom-status}
{bdg-info}`Draft` Written from `solver/utils/impmet.py`, `solver/rwg/rwg3.py` and `solver/rwg/rwg4.py` at commit `1b6a6e2`, awaiting review.

How the integral equation of {doc}`efie` becomes a linear system $Z\,I = V$ of size $N\times N$,
what each entry of $Z$ contains, and how the solver fills and solves it.

## The idea

The EFIE must hold at every point of the surface, for an unknown current that is a function. The
Method of Moments trades both for finite objects {footcite:p}`harrington1993`:

1. the current is written as a combination of $N$ known functions, with $N$ unknown coefficients;
2. instead of imposing the equation at every point, it is imposed "on average" against $N$
   **testing functions**, one equation per testing function.

That gives $N$ linear equations for $N$ unknowns. With the same RWG functions for expansion and
testing, the scheme is called **Galerkin**, and it is what the solver does.

## From the EFIE to $Z\,I = V$

Expand the current in RWG functions, $\mathbf J = \sum_n I_n \mathbf f_n$ (see
{doc}`rwg-basis-functions`), and take the inner product of the EFIE {eq}`eq-efie` with each
$\mathbf f_m$:

```{math}
:label: eq-tested-efie
\big\langle \mathbf f_m,\ j\omega\,\mathbf A + \nabla\Phi \big\rangle
= \big\langle \mathbf f_m,\ \mathbf E^i \big\rangle,
\qquad
\langle \mathbf a, \mathbf b\rangle = \int_S \mathbf a\cdot\mathbf b\,dS .
```

The gradient is moved from $\Phi$ onto the testing function by integration by parts,
$\langle \mathbf f_m, \nabla\Phi\rangle = -\langle \nabla_s\cdot\mathbf f_m, \Phi\rangle$. There is
no boundary term because $\mathbf f_m$ has no normal component on the outer edges of its two
triangles. Only first derivatives of the RWG functions remain, which are the constants
$\pm\ell_m/A_m^\pm$. The result is

```{math}
:label: eq-zmn
\begin{aligned}
Z_{mn} ={}& j\omega\mu\int_{S}\!\int_{S} \mathbf f_m(\mathbf r)\cdot\mathbf f_n(\mathbf r')\,G\,dS'\,dS\\
&+ \frac{1}{j\omega\varepsilon}\int_{S}\!\int_{S}
\big(\nabla_s\cdot\mathbf f_m\big)\big(\nabla_s'\cdot\mathbf f_n\big)\,G\,dS'\,dS,
\end{aligned}
```

```{math}
:label: eq-vm
V_m = \int_S \mathbf f_m\cdot\mathbf E^i\,dS,
\qquad
\sum_{n=1}^{N} Z_{mn}\,I_n = V_m,\quad m = 1,\dots,N,
```

with $G = e^{-jkR}/(4\pi R)$. The first term of {eq}`eq-zmn` comes from the vector potential (the
current), the second from the scalar potential (the charge).

## What the solver computes

Each RWG function lives on two triangles, so each $Z_{mn}$ is a sum of **four** double integrals,
one per pair $(T_m^t, T_n^s)$ with $t, s \in \{+,-\}$. On $T^\pm$, the solver writes
$\mathbf f = \pm\frac{\ell}{2A^\pm}(\mathbf r - \mathbf v^\pm)$ and
$\nabla_s\cdot\mathbf f = \pm\ell/A^\pm$, and integrates over each triangle with a quadrature rule
whose weights sum to ½ (the area of the reference triangle), so that
$\int_T g\,dS \approx 2A\sum_p w_p\,g(\mathbf r_p)$. The factors $2A$ then cancel the
$\ell/(2A)$ of the RWG normalisation, and {eq}`eq-zmn` becomes

```{math}
:label: eq-zmn-discrete
Z_{mn} = \ell_m\ell_n \sum_{t,s\,\in\,\{+,-\}} s_t\,s_s \sum_{p}\sum_{q} w_p\,w_q
\left[\frac{j\omega\mu}{4\pi}\,\boldsymbol\rho_t(\mathbf r_p)\cdot\boldsymbol\rho_s(\mathbf r_q)
+ \frac{4}{4\pi\, j\omega\varepsilon}\right]\frac{e^{-jkR_{pq}}}{R_{pq}},
```

where $s_\pm = \pm1$, $\boldsymbol\rho_t(\mathbf r) = \mathbf r - \mathbf v_m^t$ is measured from the
free vertex of the test triangle, $\mathbf r_p$ and $\mathbf r_q$ are the quadrature nodes on the test
and source triangles and $R_{pq} = |\mathbf r_p - \mathbf r_q|$. The factor 4 of the charge term is
$(\ell/A)^2(2A)^2/\ell^2$. This is the formula of the docstring of
`impmet.impedance_matrix_z`; in the code the two brackets are `pref_A * rho_dot` and
`4.0 * pref_Phi`.

{eq}`eq-zmn-discrete` is used as written only for pairs of triangles that do not touch (FAR pairs).
When the two triangles touch, $1/R$ is singular and the same rule on both sides is inaccurate or
fails: the SELF, EDGE and VERTEX pairs are integrated differently, see {doc}`numerical-integration`.

## Symmetry and fill modes

Because $G(\mathbf r,\mathbf r') = G(\mathbf r',\mathbf r)$ and the same functions are used for
expansion and testing, the exact matrix is **complex symmetric**: $Z_{mn} = Z_{nm}$ (not Hermitian).
Its real and imaginary parts $R = \mathrm{Re}\,Z$ and $X = \mathrm{Im}\,Z$ are real symmetric
matrices. This is reciprocity, and characteristic modes rely on it (see {doc}`characteristic-modes`).

The assembled matrix is symmetric only up to quadrature error, because the singular pairs are not
integrated the same way on the test and source sides. The argument `fill_mode` of the assemblers
decides what to do about it:

```{list-table}
:header-rows: 1
:widths: 18 42 40

* - `fill_mode`
  - What is computed
  - Use
* - `"full_sym"` (default)
  - every $(m,n)$ entry, then $Z \leftarrow (Z + Z^\mathsf{T})/2$
  - production. Exactly symmetric, and the charge term stays exactly neutral on loop currents.
* - `"mirror"`
  - entries $m \le n$ only, lower half copied
  - comparison only. Half the cost, but not charge-neutral.
* - `"full"`
  - every entry, no symmetrisation
  - diagnostic: $\max|Z - Z^\mathsf T|/\max|Z|$ measures the quadrature error.
```

The reason for `"full_sym"` is documented in `impmet._mirror_upper_triangle`. The charge term has the
structure $Z_\Phi = D^\mathsf{T} K D$, where $D$ maps edge currents to triangle charges, so it
vanishes **exactly** on any current that deposits no charge (a loop), whatever the quadrature,
provided every entry uses the same triangle kernel $K$. Mirroring breaks that structure. On a
20 mm loop the developers measured $|u^\mathsf{H} Z_\Phi u| / |u^\mathsf{H} Z_A u| = 0.10$ with the
mirrored fill against $5\times10^{-13}$ with the full fill, and the loop reactance did not converge
with mesh refinement. Symmetrising the full fill keeps the structure, since
$(D^\mathsf T K D + D^\mathsf T K^\mathsf T D)/2 = D^\mathsf T\,\tfrac{K+K^\mathsf T}{2}\,D$.

## The right-hand side

$V_m$ in {eq}`eq-vm` is the incident field tested by $\mathbf f_m$. The solver evaluates it with one
point per triangle, the centroid, where
$\mathbf f_m(\mathbf r_m^{c\pm}) = \ell_m\boldsymbol\rho_m^{c\pm}/(2A_m^\pm)$ and the factor $A_m^\pm$
of the one-point rule cancels:

```{math}
:label: eq-vm-centroid
V_m \approx \frac{\ell_m}{2}\Big[\mathbf E^i(\mathbf r_m^{c+})\cdot\boldsymbol\rho_m^{c+}
+ \mathbf E^i(\mathbf r_m^{c-})\cdot\boldsymbol\rho_m^{c-}\Big].
```

- **plane wave** (`rwg4.calculate_current_scattering`): {eq}`eq-vm-centroid` with the plane wave of
  {doc}`efie`;
- **gap source** (`gap_source.multiple_gap_sources`): {eq}`eq-vm-centroid` with the field $V/W$ of
  the gap, keeping only the triangles whose centroid lies in the gap window.

Both are detailed in {doc}`excitations`. Two modifications of $Z$ also belong to the system solved:
a lumped impedance $\Delta Z$ on edge $m$ adds $\ell_m^2\,\Delta Z$ to $Z_{mm}$
(`rwg3.calculate_z_matrice_lumped_elements`), and a lossy conductor replaces $Z$ by $Z + Z_s\mathsf G$
(see {doc}`lossy-conductors`).

## Solving the system

`rwg4.calculate_current_radiation` and `calculate_current_scattering` call `numpy.linalg.solve`: a
dense LU factorisation with partial pivoting (LAPACK). The matrix is never inverted.

```{list-table}
:header-rows: 1
:widths: 30 30 40

* - Step
  - Cost
  - Notes
* - Fill of $Z$
  - $O(N^2)$ entries, each 4 double integrals
  - the dominant cost for most meshes; CPU or GPU (see {doc}`../user-guide/gpu`)
* - Storage of $Z$
  - $16\,N^2$ bytes (complex128)
  - $N = 760$: 9.2 MB; $N = 5\,000$: 400 MB; $N = 10\,000$: 1.6 GB
* - LU solve
  - $O(N^3)$ operations
  - one factorisation per frequency
```

A frequency sweep keeps one matrix per frequency (`calculate_z_matrice_sweep` returns an array
$N_f\times N\times N$), so its memory grows with the number of frequencies.

## Conditioning

Two properties of $Z$ matter in practice.

- **The charge term dominates at small cell size.** Relative to the current term, the charge term of
  {eq}`eq-zmn` is larger by about $4/(kh)^2$ for a cell of size $h$ (comment in `impmet.py`). As the
  frequency drops or the mesh is refined, the two terms separate by orders of magnitude and $Z$
  becomes ill-conditioned. This is the low-frequency breakdown of the EFIE, cured by splitting the
  current into loop and star (or tree) parts {footcite:p}`zhao2000`. **The solver has no such
  decomposition.** Its meshing helper `gmsh_function.mesh_and_gap_parameters` warns when the cell
  falls below $\lambda/100$.
- **$R = \mathrm{Re}\,Z$ is only positive semi-definite.** The radiated power
  $\tfrac12 I^\mathsf{H} R\, I$ is never negative, but many current distributions radiate almost
  nothing, so $R$ has many tiny eigenvalues on an electrically small structure. This does not affect
  $Z\,I = V$, but it is central to characteristic modes, which solve a problem in $R$ (see
  {doc}`characteristic-modes`).

## In this solver

:::{admonition} Theory vs. implementation
:class: theory

What the code implements, its entry points and the parameters you can change.
:::

```{list-table}
:header-rows: 1
:widths: 38 62

* - Function
  - Role
* - `rwg3.calculate_z_matrice(points, triangles, edges, frequency, use_gpu=None, fill_mode=None)`
  - sets the constants, chooses CPU or GPU, returns $\omega, \mu, \varepsilon, c, \eta$ and $Z$
* - `rwg3.calculate_z_matrice_sweep(..., frequencies, ...)`
  - same for many frequencies in one pass (shared geometry precomputation)
* - `rwg3.calculate_z_matrice_lumped_elements(...)`
  - $Z$ plus lumped R, L, C loads
* - `impmet.impedance_matrix_z`, `impedance_matrix_z_sweep`, `impedance_matrix_z_sweep_gpu`
  - the three assemblers. They implement the same scheme and their regression tests cross-check
    them to $10^{-10}$
* - `rwg4.calculate_current_radiation`, `calculate_current_scattering`
  - build $V$ and solve $Z\,I = V$
```

**Parameters.** `fill_mode` and `use_gpu` are exposed by the `rwg3` functions. The quadrature orders
(`quad_order`, `near_order`, `self_outer_order`, `duffy_order`) are arguments of the `impmet`
assemblers only: `calculate_z_matrice` always uses their defaults. See {doc}`numerical-integration`.

**Tests.** `solver/utils/quadrature/tests/test_impedance_matrix_regression.py`,
`test_impedance_sweep_regression.py`, `test_impedance_sweep_gpu_regression.py`,
`test_impedance_fill_mode.py`, `test_impedance_quadrature_normalisation.py`, and
`test/test_rwg3_gpu_dispatch.py`.

## Assumptions and limits

- Dense matrix and dense LU: memory $O(N^2)$, time $O(N^3)$. No fast method (multilevel fast
  multipole, compression) is implemented.
- No loop–star or loop–tree decomposition: avoid cells much smaller than needed, especially at low
  frequency.
- The right-hand side uses one point per triangle (centroid); the matrix uses full quadrature.
- Quadrature orders are not exposed by the high-level functions.

## Continue

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} {fas}`arrow-right` Next: numerical integration
:link: numerical-integration
:link-type: doc

How the singular pairs of triangles are integrated, and how accurate it is.
:::

:::{grid-item-card} {fas}`shapes` RWG basis functions
:link: rwg-basis-functions
:link-type: doc

The functions $\mathbf f_n$ used for expansion and testing.
:::

:::{grid-item-card} {fas}`bolt` Excitations
:link: excitations
:link-type: doc

How the vector $V$ is built for a gap source and for a plane wave
:::
::::

## References

```{footbibliography}
```
