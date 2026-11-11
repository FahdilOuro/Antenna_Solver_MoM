# Reduced multiport model and load optimisation

{.mom-status}
{bdg-info}`Draft` Written from `solver/optim/` (`ports.py`, `reduced_model.py`, `farfield.py`, `objectives.py`, `inner.py`, `outer.py`, `loss_model.py`, `components.py`, `matching.py`) and `solver/optim/README.md` at commit `5e8d892`, awaiting review.

When the geometry is fixed and only the loads change (parasitic elements closed on capacitors,
inductors, PIN diodes or varactors), the full MoM system does not need to be solved again for each
choice of loads. It can be reduced **exactly** to its ports, after which every new set of loads costs
a few operations on matrices of the size of the number of ports.

## The idea

A MoM solve costs $O(N^2)$ to fill $Z$ and $O(N^3)$ to factorise it, with $N$ in the hundreds or
thousands. A load connected across a small gap only touches the few edges of that gap. If the
response of the structure to a unit voltage at each gap is computed once, the response to any
combination of generators and loads at those gaps follows from circuit theory on a $P$-port, with
$P$ of the order of ten. Nothing is approximated: the reduced model gives the same currents as the
full solve.

## Ports as cuts

A **port** $p$ is a cut across a conductor: a set $S_p$ of RWG edges lying on one line across a
strip, and a unit vector $\mathbf u_p$ giving the positive direction of the current across the cut.
Its **incidence vector** $\mathbf a_p \in \mathbb R^N$ is

```{math}
:label: eq-rm-incidence
(\mathbf a_p)_n =
\begin{cases}
s_n\,\ell_n & n \in S_p,\\
0 & \text{otherwise},
\end{cases}
\qquad
s_n = \operatorname{sign}\!\big((\mathbf r_n^{c-} - \mathbf r_n^{c+})\cdot\mathbf u_p\big).
```

An RWG function carries the current $\ell_n I_n$ across its edge, from $T_n^+$ to $T_n^-$
({doc}`rwg-basis-functions`), so the current across the cut is $i_p = \mathbf a_p^\mathsf T I$. A
delta-gap voltage $v_p$ across the cut, tested with $\mathbf f_m$, gives $V_m = s_m\ell_m v_p$, the
gap source of {footcite:t}`makarov2001` extended to several edges and either orientation. With the
$P$ ports side by side in $A = [\mathbf a_1 \cdots \mathbf a_P]$ ($N\times P$):

```{math}
:label: eq-rm-ports
V = A\,v, \qquad i = A^\mathsf T I .
```

These ports use this delta gap on a line of edges, not the enhanced gap source of {doc}`excitations`.

## Exact reduction

With the conductor loss of {doc}`lossy-conductors` included, the system matrix is
$Z_t = Z + \alpha Z_s\mathsf G$. One LU factorisation of $Z_t$ and $P$ solves give

```{math}
:label: eq-rm-reduction
B = Z_t^{-1}A, \qquad Y = A^\mathsf T B,
\qquad\text{so that}\qquad
I = B\,v, \qquad i = Y v .
```

$B$ ($N\times P$) holds the edge currents for 1 V at each port, the others short-circuited; $Y$
($P\times P$) is the **port admittance matrix**. It is symmetric because $Z_t$ is.

**Terminations.** Each port is closed on a Thévenin source of emf $e_p$ and internal impedance
$Z_{D,p}$; a passive load is $e_p = 0$, $Z_{D,p} = Z_{L,p}$. With $Z_D = \mathrm{diag}(Z_{D,p})$,
$v = e - Z_D\,i$, and eliminating $v$ without inverting $Y$:

```{math}
:label: eq-rm-terminations
(\mathbb 1_P + Y Z_D)\,i = Y e, \qquad v = e - Z_D\, i .
```

This is algebraically the same as solving the full system $(Z_t + A Z_D A^\mathsf T)\,I = A e$. A short
circuit ($Z_{D,p} = 0$) is exact and equivalent to not declaring the port: an undeclared cut is
continuous metal. An open circuit ($i_p = 0$) cannot be written in this form.

## Powers, gain and their balance

All the powers are quadratic forms in the port voltages, with $P\times P$ matrices computed once:

```{math}
:label: eq-rm-powers
\begin{aligned}
P_\mathrm{rad} &= v^\mathsf H M_\mathrm{rad}\, v, & M_\mathrm{rad} &= \tfrac12 B^\mathsf H R\, B, \quad R = \mathrm{Re}\,Z,\\
P_\mathrm{cond} &= v^\mathsf H M_\mathrm{cond}\, v, & M_\mathrm{cond} &= \tfrac12\,\mathrm{Re}(\alpha Z_s)\, B^\mathsf H \mathsf G\, B,\\
P_\mathrm{loads} &= \tfrac12\textstyle\sum_{p\ \text{loaded}} \mathrm{Re}(Z_{D,p})\,|i_p|^2, &
P_{\mathrm{in},p} &= \tfrac12\mathrm{Re}(v_p\, i_p^*),
\quad P_{\mathrm{avail},p} = \frac{|e_p|^2}{8\,\mathrm{Re}\,Z_{D,p}} .
\end{aligned}
```

$P_\mathrm{rad}$ is {eq}`eq-rf-prad-r` and $P_\mathrm{cond}$ the loss of {eq}`eq-power-balance`, both
computed from the currents, not from a field integration. The power entering the driven ports must
equal what is radiated, dissipated in the metal and dissipated in the loads:

```{math}
:label: eq-rm-balance
\sum_{p\ \text{driven}} P_{\mathrm{in},p} = P_\mathrm{rad} + P_\mathrm{cond} + P_\mathrm{loads} .
```

**Far field.** A quantity linear in the currents, $q = \mathbf c^\mathsf T I$, becomes
$q = \mathbf h^\mathsf T v$ with $\mathbf h = B^\mathsf T\mathbf c$. One far-field component
$N_\phi(\theta, \phi)$ of {eq}`eq-rf-farfield` is such a quantity, so the intensity $U$ in a given
direction also costs $O(P)$ per set of loads. Then, with the definitions of {doc}`radiated-fields`:

```{math}
:label: eq-rm-gains
D = \frac{4\pi U}{P_\mathrm{rad}}, \qquad
G = \frac{4\pi U}{P_\mathrm{in}}, \qquad
G_\mathrm r = \frac{4\pi U}{P_\mathrm{avail}} = (1 - |\Gamma|^2)\,\eta_\mathrm{rad}\,D,
```

the last form for a single driven port with a real generator impedance. $G$ is the gain behind an
ideal, lossless matching network; $G_\mathrm r$ the realised gain on the bare generator.

**A bound for the geometry.** With reactive loads,
$P_\mathrm{in} = P_\mathrm{rad} + P_\mathrm{cond} = I^\mathsf H M I$ with
$M = \tfrac12\big(R + \mathrm{Re}(\alpha Z_s)\mathsf G\big)$, and
$U = \frac{k^2\eta_0}{32\pi^2}|\mathbf a^\mathsf H I|^2$ for the far-field vector $\mathbf a$ of the
chosen direction and component. The Cauchy–Schwarz inequality in the $M$ inner product then bounds
the gain of **any** current the geometry can carry, whatever feeds and loads produce it:

```{math}
:label: eq-rm-bound
G \le G_\mathrm{max} = \frac{k^2\eta_0}{8\pi}\,\mathbf a^\mathsf H M^{-1}\mathbf a,
\qquad\text{reached for } I \propto M^{-1}\mathbf a .
```

$M$ is positive definite only with conductor loss. Without it, $M = R/2$ is close to singular for a
small antenna and the bound is not meaningful.

## In this solver

:::{admonition} Theory vs. implementation
:class: theory

Everything on this page is in `solver/optim/`, and is separate from the `radiation_algorithm` path:
different ports, different excitation, exact powers.
:::

```{list-table}
:header-rows: 1
:widths: 38 62

* - Module, object
  - Role
* - `ports.port_from_cut`, `port_from_edge`, `incidence_matrix`
  - {eq}`eq-rm-incidence`: a port from all the edges on a segment, or from one edge (the edge
    `rwg3.calculate_z_matrice_lumped_elements` would pick); checks that the cut is covered and
    oriented, and raises `PortDefinitionError` saying what to change
* - `reduced_model.ReducedMultiport.from_mesh` / `from_matrices`
  - {eq}`eq-rm-reduction`: assembles $Z$ (and $\mathsf G$, $Z_s$ if a `conductivity` is given), one LU,
    $B$, $Y$, $M_\mathrm{rad}$, $M_\mathrm{cond}$; `timings` records each step
* - `ReducedMultiport.solve(emf, z_term)` → `PortSolution`
  - {eq}`eq-rm-terminations` and {eq}`eq-rm-powers`: `v`, `i`, `z_in`, `gamma`, `p_rad`, `p_cond`,
    `p_loads`, `p_in`, `p_avail`, `radiation_efficiency`, `power_balance_residual`
* - `ReducedMultiport.add_linear_functional`, `farfield.RadiationOperator.attach_functional`
  - $\mathbf h = B^\mathsf T\mathbf c$ for a far-field component
* - `objectives.gain`, `realized_gain`, `directivity_from_R`, `geometry_gain_bound`
  - {eq}`eq-rm-gains`, {eq}`eq-rm-bound`
* - `inner.optimize_loads`
  - best reactances $jX_p$ of the loaded ports for $G$ or $G_\mathrm r$ in one direction: Latin
    hypercube starts, then L-BFGS-B on $-\log G$, one BLAS thread
* - `outer.BayesianOptimizer`
  - geometry optimisation by Gaussian-process regression and expected improvement, with a
    reparametrisation of $K$-strip arrays that makes every point of the box a valid geometry
* - `loss_model.alpha_calibrated`, `alpha_band`
  - an empirical $\alpha$ for one meshing rule (see below)
* - `components`, `matching`
  - catalogue capacitors and inductors with their parasitics, snapping of the optimal reactances
    to catalogue parts, and lumped matching networks evaluated over the part tolerances
* - `geometries.planar`
  - structured meshes built without Gmsh (strip dipole, square loop, parallel strips, $K$-strip
    arrays) on which every grid line is a line of edges, so port cuts fall on RWG edges
```

**The factor $\alpha$** (`zs_scale`, default 1.0) is the open question of {doc}`lossy-conductors`.
`loss_model` provides an **empirical** value for one meshing rule (4 cells across each strip, cosine
grading towards the edges): $\alpha$ calibrated against CST on a tuned loop, from 0.71 to 1.09 for
strip widths of 1 to 4 mm, copper 35 µm thick, about 0.9 GHz, with an uncertainty of ±15 %. Its
docstring explains why no mesh-independent value exists: on a zero-thickness sheet the current is
singular at the strip edges and the computed loss keeps growing as the edge cells shrink. Outside
that domain `alpha_calibrated` raises an error rather than extrapolate.

**Dependencies.** `outer.BayesianOptimizer` imports scikit-learn (listed in `requirements.txt`); the
rest of `solver/optim/` needs only NumPy and SciPy (`threadpoolctl` is optional).

## A check: one parasite, one load

Two parallel copper strips, 140 mm × 2 mm, 30 mm apart, at 1 GHz (396 RWG edges): one fed by a 1 V,
50 Ω generator, the other closed on a reactance $jX$. The reduced model sweeps $X$; nine full solves
of $(Z_t + A Z_D A^\mathsf T) I = A e$, read with the exact far field, check it.

```{figure} ../figures/generated/reduced_multiport_load_sweep.svg
:name: fig-rm-sweep
:class: mom-generated
:width: 100%
:target: ../_images/reduced_multiport_load_sweep.svg
:alt: Left, gain towards plus y and minus y against the load reactance from minus 400 to plus 400 ohms; the two curves swap near zero ohms, the full-solve circles lie on them, a star marks the optimum at 7.1 dBi below a dashed bound at 8.2 dBi. Right, relative differences between the reduced and full solutions and the power balance residual, all between 1e-16 and 1e-13.

(a) Gain $G_\phi$ towards the parasite ($+y$) and towards the fed strip ($-y$) against the load
reactance, from 801 solves of the reduced model; circles: full MoM solves; star: optimum found by
`optimize_loads` for $+y$, $X$ = −11.5 Ω (13.84 pF), $G$ = 7.09 dBi; dashed: bound $G_\mathrm{max}$ =
8.21 dBi of {eq}`eq-rm-bound`. (b) Relative difference between the reduced model and the full solve
(gains, and $P_\mathrm{rad}$ from $v^\mathsf H M_\mathrm{rad} v$ against the far-field sphere
integral), and residual of the power balance {eq}`eq-rm-balance`.
[Generated by `documentation/figure_notebooks/theory_reduced_multiport.ipynb` on solver commit `5e8d892`; all values in `reduced_multiport.json`. Click to enlarge.]{.mom-provenance}
```

- **The reduction is exact**: gains differ from the full solve by at most $1.1\times10^{-13}$, and
  the radiated power of the reduced model ($v^\mathsf H M_\mathrm{rad} v$, from $\mathrm{Re}\,Z$)
  matches the far-field sphere integral of the full-solve currents to $4\times10^{-14}$.
- **The balance closes**: {eq}`eq-rm-balance` holds to $1.1\times10^{-14}$ of $P_\mathrm{in}$ for
  every load.
- **The cost**: on the machine used, the $Z$ fill took 12 s (GPU) and the LU 0.18 s, once. After that,
  one load value took 69 µs on the reduced model, against 117 ms for a full solve with the same
  far-field evaluation, about 1700 times more.
- **The optimum**: with $X$ = −11.5 Ω the parasite acts as a director (the beam points towards it).
  $G$ = 7.09 dBi with
  $\eta_\mathrm{rad}$ = 0.963, a front-to-back ratio of 7.3 dB, and $Z_\mathrm{in}$ = 14.2 − 20.6j Ω.
  On the bare 50 Ω generator the realised gain is 5.05 dBi. The search took 135 evaluations;
  $G/G_\mathrm{max}$ = 0.77: with one parasite and one load, the geometry is not used to its bound.

## Assumptions and limits

These are stated in `solver/optim/README.md` and the module docstrings:

- **No open circuit** at a port. To model an open gap, cut it in the mesh, or use a large finite
  impedance.
- **$\alpha$ is not settled** in general; the calibrated value applies only to its meshing rule and
  its domain.
- **Free space only**: no substrate, no ground plane unless meshed.
- **One frequency per instance**. The constructor takes assembled matrices so that a sweep can build
  one instance per frequency from `calculate_z_matrice_sweep`; no such sweep class exists yet.
- $Y$ is exactly symmetric because $Z$ is assembled with `fill_mode="full_sym"`; the reciprocity error
  of the underlying integrals, measured with a full unsymmetrised fill, is about $10^{-4}$
  (`test_admittance_symmetry`).

**Tests**: `python -m pytest test/optim -v -s`. The meshes are built inside the tests, with neither
`.mat` files nor Gmsh. The comparisons with CST are scripts and notebooks in
`solver/antenna_simulation/euraap_challenge/`.

## Continue

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} {fas}`arrow-left` Radiated fields
:link: radiated-fields
:link-type: doc

$P_\mathrm{rad}$, directivity and gain
:::

:::{grid-item-card} {fas}`gauge-high` Numerical limits
:link: numerical-limits
:link-type: doc

Accuracy, mesh density, conditioning
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
