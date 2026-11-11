# Frequency sweeps and port parameters

{.mom-status}
{bdg-secondary-line}`Planned` Outline of a future page.

**This page will cover**

- `frequency_sweep(path, freq_start, freq_stop, feed_point, n_points, ...)`: what is computed once
  for the whole band and what is computed per frequency.
- Results per port: input impedance and reflection coefficient against `Z0`, with all ports driven
  together (active values, not the full S-matrix), and input power.
- `plot_s_parameters`, `plot_radiation_efficiency`, and the tools of `solver/efield/efield4.py`:
  Smith chart, $|S_{11}|$ curve, bandwidth, quality factor, comparison with CST or measurements.
- Lumped loads in a sweep (slower per-frequency path).
