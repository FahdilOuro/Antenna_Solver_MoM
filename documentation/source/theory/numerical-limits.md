# Numerical limits and conditioning

{.mom-status}
{bdg-secondary-line}`Planned` Outline of a future page.

What limits the accuracy and the size of the problems the solver can handle.

**This page will cover**

- Cost: $O(N^2)$ memory and fill time, $O(N^3)$ dense solve; what the GPU path changes and the
  memory check it performs (`solver/utils/compute_backend.py`).
- Mesh density versus wavelength, and the guardrails in `mesh_and_gap_parameters`.
- Conditioning of $Z$ and of $R = \mathrm{Re}(Z)$ for small electrical sizes.
- Reproducibility: gmsh meshes are not deterministic when meshed with several threads.
- Health checks available in the code: reciprocity and positive semi-definiteness of $R$
  (`solver/cma/z_matrix_diagnostics.py`).
