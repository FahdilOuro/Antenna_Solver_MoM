# Scattering: illuminating a structure

{.mom-status}
{bdg-secondary-line}`Planned` Outline of a future page.

**This page will cover**

- `scattering_algorithm(mesh, frequency, wave_incident_direction, polarization, ...)` with a `path`
  object or a `.mat` file name, `load_from_matlab`, `use_gpu`.
- Direction and polarisation conventions of the incident plane wave.
- Induced currents, fields and radar cross section from the saved current (`efield1`, mode
  `'scattering'`).
- Limits: triangles with a label greater than 1 in `t[3]` are discarded (AEMM convention);
  `scattering_from_E_field_antenna.py` is not functional (see {doc}`../validation/feature-status`).
