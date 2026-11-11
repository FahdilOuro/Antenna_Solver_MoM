# Lossy conductors

{.mom-status}
{bdg-info}`Draft` Written from `solver/utils/lossy_electric_conductor.py`, `solver/utils/impmet.py`, `solver/rwg/rwg4.py`, `solver/efield/efield2.py` and `solver/optim/reduced_model.py` at commit `935470d`, awaiting review.

How the solver models a real metal (copper, aluminium, ...) instead of a perfect conductor, and what
it can and cannot say about the losses.

## The idea

In a good conductor the current does not flow on the surface only: it penetrates over a few skin
depths $\delta = \sqrt{2/(\omega\mu\sigma)}$, where it meets the finite conductivity $\sigma$ and
dissipates power. When $\delta$ is much smaller than the thickness of the metal and its radii of
curvature, the inside of the metal does not need to be modelled: its effect on the outside is a
**surface impedance** $Z_s$ relating the tangential electric field on the surface to the surface
current. The perfect conductor is the limit $Z_s \to 0$.

## The surface impedance boundary condition

For a good conductor of conductivity $\sigma$ and permeability $\mu = \mu_0\mu_r$
{footcite:p}`balanis2023`:

```{math}
:label: eq-zs
Z_s = (1 + j)\sqrt{\frac{\omega\mu}{2\sigma}} = \frac{1 + j}{\sigma\,\delta},
\qquad
R_s = \mathrm{Re}\,Z_s .
```

The PEC condition {eq}`eq-pec` becomes: the total tangential field equals $Z_s$ times the current,

```{math}
:label: eq-sibc
\hat{\mathbf n}\times\left(\mathbf E^i + \mathbf E^s\right) = Z_s\,\hat{\mathbf n}\times\mathbf J
\qquad\text{on } S .
```

Testing it with the RWG functions, as in {doc}`galerkin-mom`, adds one term to the system:

```{math}
:label: eq-lossy-system
\big(Z + Z_s\,\mathsf G\big)\,I = V,
\qquad
\mathsf G_{mn} = \int_S \mathbf f_m\cdot\mathbf f_n\,dS ,
```

where $\mathsf G$ is the **Gram matrix** of the RWG functions. Unlike $Z$, it has no Green's function:
$\mathsf G_{mn}$ is non-zero only when $\mathbf f_m$ and $\mathbf f_n$ share a triangle.

## The power balance

With losses, the input power splits into radiated and dissipated parts:

```{math}
:label: eq-power-balance
P_\mathrm{in} = P_\mathrm{rad} + P_\mathrm{cond},
\qquad
P_\mathrm{rad} = \tfrac12 I^\mathsf{H} R\, I,
\qquad
P_\mathrm{cond} = \tfrac12\,R_s\, I^\mathsf{H}\mathsf G\, I,
```

with $R = \mathrm{Re}\,Z$ the PEC radiation matrix. The **radiation efficiency** is
$\eta = P_\mathrm{rad}/P_\mathrm{in}$.

## In this solver

:::{admonition} Theory vs. implementation
:class: theory

Where the surface impedance enters, and which powers are actually computed.
:::

**The surface impedance.** `lossy_electric_conductor.calculate_normal_surface_impedance(frequency,
conductivity, mu_r=1)` implements {eq}`eq-zs`, with $\mu_0 = 1.257\times10^{-6}$ H/m. The same module
has `calculate_superconductor_surface_impedance` ($Z_s = j\omega\mu\lambda_L$, with $\lambda_L$ the
London depth), which no high-level function calls.

**The Gram matrix.** `impmet.rwg_gram_matrix(edges, triangles, points, quad_order=4)` assembles
$\mathsf G$ triangle by triangle. On a triangle shared by $\mathbf f_m$ and $\mathbf f_n$ the integrand
$\boldsymbol\rho_m\cdot\boldsymbol\rho_n$ is a polynomial of degree 2, so the degree-4 rule (6 points)
integrates it exactly. It is pure geometry: in a frequency sweep it is built once.

**The system.** With `lossy_conductor=True` and `conductivity_conductor=σ`, `radiation_algorithm` and
`frequency_sweep` solve {eq}`eq-lossy-system` (`rwg4.calculate_current_radiation` forms
`Z + impedance_zs * Gram_rwg`). In a sweep $Z_s$ is recomputed at every frequency. The same
$Z_s$ is applied to the whole structure.

**The efficiency reported by `efield2`.** `efield2.compute_antenna_efficiency_metrics` computes
$\eta = P_\mathrm{rad}/P_\mathrm{in}$ with

- $P_\mathrm{in} = \tfrac12\mathrm{Re}\{V^\mathsf T I^*\}$, from the port results;
- $P_\mathrm{rad}$ obtained by integrating the far field of the **dipole model** over a sphere
  (see {doc}`radiated-fields`), not from $\tfrac12 I^\mathsf H R I$;

and then **caps the result at 100 %** (`min(efficiency, 1.0)`), printing a warning above 101 %. The
dissipated power $P_\mathrm{cond}$ is never computed on this path. Any error of the dipole model on
$P_\mathrm{rad}$ therefore goes straight into $\eta$, and an overestimate is hidden by the cap. This is
the context of the known issue on the efficiency of lossy antennas
(`lossy_antenna/copper_antenna.ipynb`, see {doc}`../validation/feature-status`).

**The reduced multiport model computes the balance exactly.** `optim.reduced_model.ReducedMultiport`
uses $Z_t = Z + \alpha Z_s\mathsf G$ and evaluates $P_\mathrm{rad} = \tfrac12 I^\mathsf H R I$ and
$P_\mathrm{cond} = \tfrac12\mathrm{Re}(\alpha Z_s)\,I^\mathsf H\mathsf G I$ directly from the currents,
with $\eta$ (`radiation_efficiency`) and the residual of {eq}`eq-power-balance`
(`power_balance_residual`) as outputs (see {doc}`reduced-multiport`).

**An open question: the factor $\alpha$.** The mesh is a sheet of zero thickness carrying the total
current of a real strip that has two faces. If the current splits equally between them, each face
carries $J/2$ and the loss is $2\times\tfrac12 R_s|J/2|^2 = \tfrac12 (R_s/2)|J|^2$: an effective
$Z_s/2$. If it crowds onto one face (above a ground plane, for instance), the factor is closer to 1.
`ReducedMultiport` exposes it as `zs_scale` (default 1.0, documented as a placeholder until a
comparison with CST settles it). `rwg4` has no such factor: it uses $Z_s$ as is, i.e. $\alpha = 1$.

**Tests.** `test/test_frequency_sweep.py::test_fast_sweep_lossy_conductor_matches_per_frequency_radiation_algorithm`
(lossy sweep against single-frequency runs), `solver/utils/quadrature/tests/test_gram_matrix_regression.py`
(Gram matrix), `test/optim/test_optim_loss_model_array.py` and the power-balance tests of
`test/optim/`.

## Assumptions and limits

- **Skin depth small** compared with the thickness and the radii of curvature of the metal; edges and
  corners, where the current crowds, are where the condition is least accurate.
- **Zero-thickness sheet** for a two-faced conductor: the effective factor $\alpha$ is not settled.
- **One conductivity** for the whole mesh.
- **Efficiency on the `efield2` path**: dipole-model $P_\mathrm{rad}$ and a cap at 100 %. For a
  quantitative efficiency use the power balance {eq}`eq-power-balance`, as the reduced multiport model
  does.

## Continue

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} {fas}`arrow-left` Excitations
:link: excitations
:link-type: doc

Input power and impedance of a port
:::

:::{grid-item-card} {fas}`satellite-dish` Radiated fields
:link: radiated-fields
:link-type: doc

How $P_\mathrm{rad}$ and the patterns are computed
:::

:::{grid-item-card} {fas}`play` Use it
:link: ../user-guide/lossy-and-lumped
:link-type: doc

`lossy_conductor=True` in practice
:::
::::

## References

```{footbibliography}
```
