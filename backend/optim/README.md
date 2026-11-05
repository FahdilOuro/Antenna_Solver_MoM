# backend/optim — ports and reduced multiport model

Building blocks for antenna optimisation where **the geometry is fixed and the
loads change**: parasitic elements with lumped loads, PIN diodes or varactors.
The expensive MoM work (O(N²) fill, O(N³) factorisation) is done once per
geometry and frequency. After that, every choice of generator and load costs
O(P³), with P ≤ ~10 ports.

```
mesh ──► Z (rwg3), G (impmet), Zs ──► LU(Z_t) once ──► B, Y, M_rad, M_cond   (P×P)
                                                         │
                         any emf e, terminations Z_D ────┴──► v, i, Zin, Γ, P_rad, P_cond, P_loads
```

## Ports (`ports.py`)

A port is a cut across a conductor: a set of RWG edges `S_p` and a unit vector
`u_p` giving the positive current direction across the cut.

    (a_p)_n = s_n l_n  (n ∈ S_p),   s_n = sign((c_n⁻ − c_n⁺)·u_p)

* `c_n±` are the centroids of `T_n±`. An RWG function carries current
  `T⁺ → T⁻` and has unit normal component on its edge, so `s_n = +1` means
  that the edge's reference direction is along `u_p`.
* Delta-gap voltage `v_p` → right-hand side `V = a_p v_p` (Makarov's `V_m = l_m V`).
* Current across the port: `i_p = a_p^T I`. With all ports, `A = [a_1 … a_P]`.

Constructors: `port_from_cut` (all edges on a segment, with checks on coverage
and orientation) and `port_from_edge` (one edge, using the same selection rule as
`rwg3.calculate_z_matrice_lumped_elements`). Errors raise `PortDefinitionError`
and the message says what to fix.

## Reduced model (`reduced_model.py`)

    Z_t = Z + α Zs G          B = Z_t⁻¹ A          Y = Aᵀ B
    v = e − Z_D i     ⇒     (1 + Y Z_D) i = Y e      (Y is never inverted)
    M_rad  = ½ Bᴴ Re(Z) B                    P_rad  = vᴴ M_rad v
    M_cond = ½ Re(α Zs) Bᴴ G B               P_cond = vᴴ M_cond v
    P_loads = ½ Σ_loaded Re(Z_D,p)|i_p|²     P_in,p = ½ Re(v_p i_p*)
    P_avail,p = |e_p|²/(8 Re Z_D,p)          Γ_p = (Zin − Z_D,p*)/(Zin + Z_D,p)
    linear functional q = cᵀI = (Bᵀc)ᵀ v     (hook for the far field)

The reduction is exact. It gives the same result as a full solve of
`(Z_t + A Z_D Aᵀ) I = A e`, and the tests check this to 1e-10.

```python
from backend.optim import ReducedMultiport, port_from_cut

ports = [port_from_cut(points, triangles, edges, p0, p1, direction=[1, 0, 0], name="feed"), ...]
model = ReducedMultiport.from_mesh(points, triangles, edges, f, ports, conductivity=5.8e7)
sol = model.solve(emf=[1, 0, 0], z_term=[50, 1j * 30, 10 - 80j])
sol.z_in[0], sol.gamma[0], sol.p_rad, sol.p_cond, sol.p_loads, sol.radiation_efficiency
```

`model.timings` records the time taken by the Z assembly, the Gram assembly,
the LU factorisation and the precomputation.

## Far field (`farfield.py`) and figures of merit (`objectives.py`)

`RadiationOperator` evaluates the radiation integral of the RWG currents with a
Gauss–Dunavant rule on every triangle (`quad_order`, default 7), instead of the
point-dipole model of `dipole_parameters`:

    N(r̂) = Σ_n I_n ∫ f_n e^{jk r̂·r'} dS',   ∫_{T±} f_n e^{…} = ± l_n Σ_q w_q (r_q − v±) e^{jk r̂·r_q}
    U_θ,φ = k² η0 / (32π²) |N_θ,φ|²   (W/sr),   φ̂ = (−sin φ, cos φ, 0)

* `vectors(θ, φ)` returns `f_θ, f_φ` with `N_θ = f_θ @ I`. `attach_functional(model, θ, φ)`
  registers `f_φ` on a `ReducedMultiport`, so that `N_φ = hᵀ v` is O(P) per load set.
* `radiated_power_sphere(I)` is Gauss–Legendre in cos θ times a trapezoid in φ, with a
  grid sized from the electrical radius. It matches `½ Iᴴ Re(Z) I` to 1e-11 (test V2).
* `objectives`: `directivity_from_R`, `gain` (4πU/P_in), `realized_gain` (4πU/P_avail),
  with `G_r = (1−|Γ|²) η_rad D`.
* **Constants:** ε0 = 8.854e-12 and μ0 = 1.257e-6, exactly as in `rwg3`, so that k, η0
  and Z are consistent. μ0 is 3e-4 away from CODATA; the constants will be harmonised
  project-wide later.

## Geometry gain bound (`objectives.geometry_gain_bound`)

`M = ½ (R + Re(α Zs) G)` gives `P_rad + P_cond = Iᴴ M I` for any currents (when
the loads are reactive). Since `G(I) = (k²η0/8π) |aᴴI|² / (Iᴴ M I)` with `a = f*`,
the Cauchy–Schwarz inequality gives

    G_max = (k² η0 / 8π) aᴴ M⁻¹ a,   attained for I ∝ M⁻¹ a.

The bound uses a Cholesky factorisation and reports cond(M). Without loss
(`Re(α Zs) = 0`), `M = R/2` is close to singular, so the function warns and
returns `None`.

## Inner loop (`inner.py`)

`optimize_loads(model, (θ, φ), "phi", objective="G" | "G_r", x_bounds, n_starts, seed)`
searches for the reactances `jX_p` of the loaded ports. It draws starting points
by Latin hypercube, runs L-BFGS-B on `−log(objective)` from each, and runs under
`limit_blas_threads(1)`. `InnerResult` holds:
- the optimal X in Ω, with the equivalent C or L at f0;
- G (ideal matching), G_r (50 Ω generator), D, η_rad and Z_in;
- G/G_max and the history of every start.

## Geometries (`geometries/planar.py`)

Structured meshes built without Gmsh. Each one is the set of kept cells of a
tensor grid, so it is conformal by construction and every grid line is a line
of edges, which lets port cuts fall exactly on RWG edges.

* `strip_dipole(length, width, n_len, n_width, edge_grading)`: feed cut at x = 0, +x.
* `square_loop(side_centerline, width, n_side, n_width, edge_grading)`: feed cut at
  x = 0 on the side y = −s/2, +x, and a `load` cut on the opposite side (used by
  V5b′). The corners are n_width × n_width blocks.
* `parallel_strips(length, width, spacing, n_len, n_width)`: cuts `feed` and `parasite`.
* `edge_grading='cosine'` clusters cells towards the strip edges, where J ~ 1/√d
  and therefore the loss is concentrated.

## Threads (`threads.py`)

On small matrices, a multithreaded BLAS is slower than a single thread. For
example, a 157×157 LU takes 170–330 ms with 16 OpenBLAS threads and 0.5–1 ms with
one. Wrap inner loops in `with limit_blas_threads(1):`. This needs
`threadpoolctl`; without it the call does nothing and warns once.

## Known limits

* **No open circuit.** `Z_D = ∞` is rejected. A short circuit (`Z_D = 0`) is
  exact, and it is the same as not declaring the port, because an undeclared cut
  is continuous metal. To model an open gap, cut it in the mesh, or use a large
  finite impedance as an approximation.
* **`zs_scale` (α) is unresolved.** On a zero-thickness sheet, a strip of
  thickness ≫ δ may need `Zs/2` (current split equally between the two faces) or
  something closer to `Zs` (current crowding onto one face or the edges). The
  default of 1.0 is a placeholder until a comparison with CST settles it.
* **Free space only.** There is no substrate and no ground plane unless it is
  meshed.
* **One frequency per instance.** The constructor takes already-assembled
  matrices so that a future `ReducedMultiportSweep` can wrap one instance per
  slice of `calculate_z_matrice_sweep`.
* `Y` is exactly symmetric because the default `Z` is symmetrised
  (`fill_mode="full_sym"`: every pair filled, then (Z + Zᵀ)/2). That mode also
  keeps the scalar-potential term exactly charge-neutral on loop currents. The
  former mirrored fill (`fill_mode="mirror"`) leaked about 10% of Z_A into the
  reactance of small loops (see `backend/utils/impmet.py`). The real reciprocity
  error, measured with `impedance_matrix_z(..., full_fill=True)`, is about 1e-4
  (see `test_admittance_symmetry`).

Tests: `python -m pytest test/optim -v -s` (32 tests). The meshes are built
inside the tests, with no `.mat` files and no Gmsh. The V5 scripts (CST
comparison) are in `backend/antenna_simulation/euraap_challenge/`.
