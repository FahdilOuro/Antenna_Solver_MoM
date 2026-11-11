# Method of Moments and the impedance matrix

{.mom-status}
{bdg-secondary-line}`Planned` Outline of a future page.

How the EFIE becomes the linear system $Z\,I = V$.

**This page will cover**

- Expansion of the current in RWG functions and Galerkin testing (test functions = basis functions).
- The impedance matrix entry $Z_{mn}$ as implemented in `solver.utils.impmet.impedance_matrix_z`:
  a vector-potential term and a scalar-potential term, summed over the four triangle pairs
  $(T_m^\pm, T_n^\pm)$ with the sign product $s_{ts}$.
- Why $Z$ is complex-symmetric (reciprocity) and how `fill_mode` (`"full_sym"` by default,
  `"mirror"`, `"full"`) keeps it exactly symmetric and charge-neutral on loop currents.
- The entry points: `rwg3.calculate_z_matrice`, `calculate_z_matrice_sweep`,
  `calculate_z_matrice_lumped_elements`, and the automatic CPU / GPU choice.
- Solving $Z I = V$ (dense LU, `numpy.linalg.solve`, in `solver/rwg/rwg4.py`).
