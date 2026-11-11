# Characteristic modes

{.mom-status}
{bdg-info}`Draft` Written from `solver/cma/` at commit `2106e70`, awaiting review.

Characteristic modes are the currents a conducting structure "naturally" supports, computed from its
impedance matrix alone, before any feed is chosen. They tell how a shape wants to radiate, which is
what makes characteristic mode analysis (CMA) a design tool rather than only an analysis tool.

## The idea

A driven simulation answers "what current flows for this feed?". CMA asks a question about the
geometry only: which current distributions radiate well at a given frequency, and which ones mostly
store energy? It answers with a set of real current patterns, the **characteristic currents**
$\mathbf J_n$, each with a number $\lambda_n$ saying how far it is from resonance. Any current the
structure can carry is a combination of them, so placing a feed comes down to choosing which modes
to excite.

## Origin and definition

The modes were introduced by {footcite:t}`garbacz1971` as the eigenfunctions that diagonalise the
scattering operator of a body. {footcite:t}`harrington1971theory` reformulated them from the
impedance operator of the EFIE, which is the form every MoM code uses
{footcite:p}`harrington1971computation`. Split the impedance matrix of {doc}`galerkin-mom` into its
real and imaginary parts,

```{math}
:label: eq-cma-split
Z = R + jX, \qquad R = \mathrm{Re}\,Z, \quad X = \mathrm{Im}\,Z,
```

both real and symmetric. The characteristic modes are the solutions of the **generalised eigenvalue
problem**

```{math}
:label: eq-cma-gep
X\,\mathbf J_n = \lambda_n\,R\,\mathbf J_n .
```

Because $R$ and $X$ are real symmetric and $R$ is positive semi-definite, the eigenvalues
$\lambda_n$ are real and the eigenvectors $\mathbf J_n$ can be chosen real: each characteristic
current has the same phase everywhere on the structure. They are orthogonal in the following sense:

```{math}
:label: eq-cma-orthogonality
\mathbf J_m^\mathsf{T} R\,\mathbf J_n = \delta_{mn},
\qquad
\mathbf J_m^\mathsf{T} X\,\mathbf J_n = \lambda_n\,\delta_{mn},
\qquad
\mathbf J_m^\mathsf{T} Z\,\mathbf J_n = (1 + j\lambda_n)\,\delta_{mn}.
```

## What the eigenvalue means

For a current $\mathbf J$, $\tfrac12\mathbf J^\mathsf{H} R\,\mathbf J$ is the radiated power and
$\tfrac12\mathbf J^\mathsf{H} X\,\mathbf J$ the reactive power, $2\omega(W_m - W_e)$ (magnetic minus
electric stored energy). For a characteristic current their ratio is the eigenvalue:

```{math}
:label: eq-cma-lambda
\lambda_n = \frac{\mathbf J_n^\mathsf{T} X\,\mathbf J_n}{\mathbf J_n^\mathsf{T} R\,\mathbf J_n}
= \frac{2\omega\,(W_m - W_e)}{P_\mathrm{rad}} .
```

- $\lambda_n = 0$: the mode is **resonant**, it radiates without net stored energy;
- $\lambda_n > 0$: magnetic energy dominates, the mode is **inductive**;
- $\lambda_n < 0$: electric energy dominates, the mode is **capacitive**.

Since $\lambda_n$ spans many orders of magnitude, two bounded quantities are read instead
{footcite:p}`cabedo2007`:

```{math}
:label: eq-cma-ms
\mathrm{MS}_n = \left|\frac{1}{1 + j\lambda_n}\right| = \frac{1}{\sqrt{1+\lambda_n^2}},
\qquad
\delta_n = 180^\circ - \arctan\lambda_n .
```

The **modal significance** $\mathrm{MS}_n \in (0, 1]$ is 1 at resonance; $\mathrm{MS}_n \ge 1/\sqrt2$
(that is $|\lambda_n| \le 1$) is the usual criterion for a mode that radiates efficiently. The
**characteristic angle** $\delta_n \in (90^\circ, 270^\circ)$ crosses $180^\circ$ at resonance.

**Why "significance".** Expand a driven current in the modes, $\mathbf J = \sum_n a_n \mathbf J_n$,
insert it in $Z\,I = V$ and use {eq}`eq-cma-orthogonality`: each coefficient is
$a_n = \mathbf J_n^\mathsf{T} V / (1 + j\lambda_n)$. The factor $1/(1+j\lambda_n)$, whose modulus is
$\mathrm{MS}_n$, is the part that depends on the structure only; $\mathbf J_n^\mathsf{T} V$ (the modal
excitation coefficient) depends on the feed.

## An example: a rectangular plate

```{figure} ../figures/generated/cma_plate_sweep.svg
:name: fig-cma-sweep
:class: mom-generated
:width: 100%
:target: ../_images/cma_plate_sweep.svg
:alt: Three plots for a 100 by 50 millimetre plate between 0.5 and 3.5 gigahertz. Left, eigenvalues on a symmetric logarithmic axis, with mode 1 crossing zero at 1.32 gigahertz and grey dots for the untracked values. Middle, modal significance, mode 1 reaching 1 at 1.32 gigahertz. Right, eigenvalues of R falling by twelve orders of magnitude within the first fifty indices.

A 100 mm × 50 mm plate (the plate of `cma_implementation/plate_cma.ipynb`, 403 RWG functions) from
0.5 to 3.5 GHz, with the default `CMAParameters` (regularised solver). (a) Eigenvalues: lines are
the four first modes after tracking across frequency, grey dots the raw values sorted by $|\lambda|$
at each frequency. Mode 1 crosses zero at 1.324 GHz. (b) Modal significance of the same modes.
(c) Eigenvalues of $R$ normalised to the largest, at three frequencies: above $10^{-6}$ (the default
`rank_tol`) only 12, 27 and 44 of the 403 directions remain. `mode_report` flags modes 3 and 4 as
spurious at 3 of the 61 frequencies each; modes 1 and 2 are never flagged.
[Generated by `documentation/figure_notebooks/theory_characteristic_modes.ipynb` on solver commit `2106e70`; numbers in `cma_plate.json`. Click to enlarge.]{.mom-provenance}
```

```{figure} ../figures/generated/cma_plate_currents.svg
:name: fig-cma-currents
:class: mom-generated
:width: 100%
:target: ../_images/cma_plate_currents.svg
:alt: Three current maps on the plate. Mode 1 flows along the length, strongest on the two long edges. Mode 2 flows across the width, strongest on the two short edges. Mode 3 circulates around the edge of the plate like a loop.

Characteristic currents of the three modes with the smallest $|\lambda|$ at the first resonance
(1.324 GHz): colour is $|\mathbf J_n|$ per triangle, arrows its direction. Mode 1 ($\lambda \approx 0$)
is the longitudinal dipole-like mode, carried by the two long edges. Mode 2 ($\lambda = -4.24$,
capacitive) is the transverse one. Mode 3 ($\lambda = +9.68$, inductive) circulates around the plate
like a loop. All three are classified `PHYSICAL` by `mode_report`.
[Same notebook. Click to enlarge.]{.mom-provenance}
```

## The practical difficulty: $R$ is rank-deficient

$R$ is the radiation operator. On an electrically small or moderate structure, most current
distributions radiate almost nothing, so most eigenvalues of $R$ are at the level of numerical noise
({numref}`fig-cma-sweep` c). Then {eq}`eq-cma-gep` is ill-posed on those directions: when
$R\,\mathbf J = 0$, no finite $\lambda$ satisfies it. The solver offers two ways to handle this,
chosen with `CMAParameters.solver`:

`'regularized'` (default)
: Solve with $R + \varepsilon I$ on the full space, where
  $\varepsilon = \max(\varepsilon_0,\ \varepsilon_0 - \mu_{\min}(R))$ and
  $\varepsilon_0 = $ `regularization_epsilon` $\times \|R\|_F / N$, just enough to make the matrix
  positive definite. Every basis function then yields a mode, including those that radiate nothing:
  for them the equation becomes $X\mathbf J = \lambda\varepsilon\mathbf J$, so their $\lambda$ is
  set by $\varepsilon$, not by the structure, and can even show a large modal significance. A warning
  is issued when $\varepsilon$ exceeds $10^{-4}\|R\|_F/N$.

`'rank_truncated'`
: Project onto the subspace that $R$ actually spans. With $R = U\,\mathrm{diag}(\mu)\,U^\mathsf{T}$,
  keep the directions where $\mu > $ `rank_tol` $\times \mu_{\max}$, set
  $W = U_r\,\mathrm{diag}(\mu_r)^{-1/2}$ (so that $W^\mathsf{T} R W = I$), solve the ordinary
  symmetric problem $W^\mathsf{T} X W\,\mathbf y = \lambda\,\mathbf y$ and return
  $\mathbf J = W\mathbf y$. No regularisation enters, and at most $\mathrm{rank}(R)$ modes are
  returned.

Neither is free of judgement: the truncation threshold `rank_tol` is relative, and the spectrum of $R$
decays smoothly, without a gap where to cut ({numref}`fig-cma-sweep` c).

**Telling physical modes from artefacts.** `cma.mode_diagnostics.mode_report` computes, for each mode,
two quantities that do not depend on $\lambda$:

- the radiated power per unit current, the Rayleigh quotient
  $P_\mathrm{rad} = \mathbf J^\mathsf{T} R\mathbf J / \mathbf J^\mathsf{T}\mathbf J$, relative to the
  strongest mode of the set (the raw $\mathbf J^\mathsf{T} R\mathbf J$ is useless here: the eigensolver
  normalises it to 1 for every mode);
- the leakage, the fraction of $\mathbf J$ lying in the null space of $R$.

A mode is `SPURIOUS` when its relative power is below $10^{-6}$ or its leakage above 0.5. The report
also gives the residual of {eq}`eq-cma-gep` projected on the range of $R$.

## Following modes across frequency

At each frequency the solver sorts the modes by $|\lambda|$. That index is a label, not an identity:
where two curves cross, "mode 2" before and after the crossing are different currents. The grey
dots of {numref}`fig-cma-sweep` a are those raw labels.

`cma.mode_tracking` restores the identity by comparing currents at adjacent frequencies with the
$R$-weighted correlation

```{math}
:label: eq-cma-correlation
C_{mn} = \frac{|\mathbf J_m^\mathsf{H} R\,\mathbf J_n|}
{\sqrt{(\mathbf J_m^\mathsf{H} R\,\mathbf J_m)(\mathbf J_n^\mathsf{H} R\,\mathbf J_n)}},
```

$R$ being the metric in which the modes are orthonormal (averaged between the two frequencies by
default). The one-to-one assignment that maximises the total correlation is solved exactly
(`scipy.optimize.linear_sum_assignment`), and the sign of each tracked mode is kept consistent.

Tracking reports what it cannot decide instead of hiding it: swaps, quasi-degenerate pairs, avoided
crossings and steps with a weak accepted correlation (below 0.7) are returned as events. Two practical
rules follow from the code and from the example above:

- a mode can be followed only while it stays among the `n_modes` kept at each frequency. With 6 modes
  kept, one step of the plate sweep had a best correlation of 0.0098; with 12 modes kept (and 4
  plotted) the minimum rose to 0.76. **Keep more modes than you plot.**
- where two eigenvalues are nearly equal, any rotation of the two currents is an equally valid pair of
  eigenvectors and no correlation can decide. The code reserves an extension point,
  `ModeTrackingCorrection`, for an eigenvalue-based correction; its `TECACorrection` entry cites
  {footcite:t}`chen2021` and is deliberately not implemented.

## In this solver

:::{admonition} Theory vs. implementation
:class: theory

What `solver/cma/` implements, and what it does not.
:::

```{list-table}
:header-rows: 1
:widths: 40 60

* - Function
  - Role
* - `cma_algorithm(path, frequency, n_modes, cma_solver, rank_tol, ...)`
  - one frequency: rebuilds the mesh and $Z$, solves {eq}`eq-cma-gep`, saves `ModalData`, writes a
    Markdown report, plots the modal currents
* - `cma_frequency_sweep(path, freq_start, freq_stop, n_points, n_modes, ..., track=True)`
  - a band: $Z(f)$ in one pass, eigenvalues **and** eigenvectors at every frequency, tracking;
    `sweep.tracked()` applies it, `sweep.tracking.summary()` describes it
* - `GeneralizedEigenSolver.solve(R, X, params)`
  - the two solvers above, sorting (`sort_by='magnitude'` or `'value'`) and a sign convention (largest
    component positive)
* - `modal_significance`, `characteristic_angle`
  - {eq}`eq-cma-ms`
* - `ModalCurrent.reconstruct`, `compute_modal_farfield`, `ModalVisualizer`
  - modal currents, modal far field and plots ($\lambda$, MS, $\delta$ versus frequency)
* - `mode_report`, `z_matrix_diagnostics.diagnose_z_matrix`
  - the diagnostics above; reciprocity and positive semi-definiteness of $R$
```

```{list-table} CMAParameters
:header-rows: 1
:widths: 26 20 54

* - Field
  - Default
  - Meaning
* - `n_modes`
  - 10
  - modes returned per frequency (the smallest $|\lambda|$ first with `sort_by='magnitude'`)
* - `solver`
  - `'regularized'`
  - `'regularized'` or `'rank_truncated'`
* - `regularization_epsilon`
  - `1e-10`
  - relative floor of $\varepsilon$ (regularised solver)
* - `rank_tol`
  - `1e-6`
  - relative threshold on the eigenvalues of $R$ (truncated solver)
* - `sort_by`, `normalize`
  - `'magnitude'`, `True`
  - ordering, sign convention
```

**Checks done by the code.** Before symmetrising $R$ and $X$, `GeneralizedEigenSolver.solve` refuses a
matrix whose asymmetry $\max|M - M^\mathsf T|/\max|M|$ exceeds `RECIPROCITY_TOL` $= 10^{-3}$: a Galerkin
matrix is symmetric, so a larger asymmetry is an assembly defect that symmetrising would hide.

**Not implemented.** The modal excitation coefficients $\mathbf J_n^\mathsf{T} V$ and the modal
decomposition of a driven current are not computed by `solver/cma/`; they follow from
{eq}`eq-cma-orthogonality` and the eigenvectors it returns.

**Side effects to know.** `cma_algorithm` and `cma_frequency_sweep` write their results and reports
under `data/antennas_cma/` (relative to the working directory). `compute_modal_farfield` evaluates
the field with the dipole model of {doc}`radiated-fields` and returns a **directivity** (a mode has no
input power); it reads, and creates if missing, the sphere mesh `data/gmsh_files/hollow_sphere.msh`.

## Validation

- **Rectangular loop against AToM.** As reported in the repository's README, on a 480 mm × 240 mm
  loop the matrix $R$ matches a MATLAB/AToM computation of the same mesh to 0.02 %, and the two
  resonances fall at 247.0 and 266.7 MHz against 247.5 and 266.1 MHz.
- **PEC sphere against the analytic modes.** The characteristic numbers of a spherical shell are
  known in closed form {footcite:p}`capek2017`. The study in
  `solver/antenna_simulation/cma_implementation/sphere_validation/` (report of 2026-08-26, commit
  `be8036f`, before `fill_mode="full_sym"` became the default) found:
  - on a regular (geodesic) mesh of 750 RWG functions, the degeneracies $2p+1$ of the TM and TE
    families are reproduced exactly (multiplets of 3, 3, 5, 5, internal spread $3\times10^{-5}$ to
    $1.3\times10^{-3}$), with the right signs; on the gmsh mesh of the same sphere, whose triangles are
    less regular, they are not;
  - without truncation, the eigenvalues of the first multiplets are within 0.3 % to 3.9 % of the
    analytic values at $ka$ = 0.5, 1 and 1.5, independently of $ka$, and the error decreases with
    mesh refinement (TM2: 2.95 %, 1.72 %, 0.75 % for 750, 1470, 3000 unknowns);
  - with truncation, a relative `rank_tol` biases the eigenvalues at small $ka$ (TE2: 39 % at
    $ka = 0.5$ with `rank_tol = 1e-4`, 0.73 % at $ka = 1.5$), because the spectrum of $R$ decays faster
    at small $ka$. The report concludes that `rank_tol` cannot be fixed once for all and that the
    default is safe for $ka \gtrsim 1.5$.

## Assumptions and limits

- CMA here is defined for perfect conductors in free space, from the PEC impedance matrix.
- The weakly radiating modes are sensitive to the numerical treatment of $R$ (regularisation or
  truncation threshold); trust modes with $|\lambda|$ up to a few units, and check the others with
  `mode_report`.
- Tracking is reliable only with enough modes kept and enough frequency points; its events say where
  it was not.
- The modal far field uses the dipole model, and the sphere mesh file it needs is written to
  `data/`.

## Continue

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} {fas}`play` Use it: CMA in practice
:link: ../user-guide/characteristic-modes
:link-type: doc

Running `cma_algorithm` and `cma_frequency_sweep`, reading the reports
:::

:::{grid-item-card} {fas}`calculator` The impedance matrix
:link: galerkin-mom
:link-type: doc

Where $R$ and $X$ come from, and why $Z$ is symmetric
:::

:::{grid-item-card} {fas}`code` API of `solver.cma`
:link: ../api/cma
:link-type: doc

Every function and class of the CMA package
:::
::::

## References

```{footbibliography}
```
