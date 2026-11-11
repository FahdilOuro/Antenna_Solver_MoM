# Troubleshooting

{.mom-status}
{bdg-secondary-line}`Planned` Outline of a future page.

**This page will cover** the errors the solver raises on purpose and what to do about them:

- `GapSourceError`: no RWG edge inside the gap window.
- `MeshDataError` / `MeshConsistencyError`: missing or stale `.mat` files, mesh and impedance
  matrix from different meshes.
- `PortDefinitionError` in the optimisation module.
- CMA warnings about regularisation and the reciprocity check of $Z$.
- GPU errors (out of memory, CUDA failure during a run).
- `UnicodeEncodeError` with Greek letters on a Windows console.
