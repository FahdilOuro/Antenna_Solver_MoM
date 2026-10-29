# Characteristic Mode Analysis — plate

Frequency sweep: **300.000 MHz to 2.0000 GHz**, 11 points, 4 modes  
Generated 2026-10-10 20:39

## Structure and discretisation

| Quantity | Value |
|---|---|
| Points | 166 |
| Triangles | 284 |
| RWG edges (unknowns) | 403 |
| Bounding box | 0.1000 x 0.0500 x 0.0000 m |
| Edge length min / mean / max | 4.50 / 6.40 / 8.08 mm |
| Frequency (band start) | 300.000 MHz |
| Wavelength | 0.9993 m |
| Largest dimension / wavelength | 0.100 |
| Electrical size ka (enclosing sphere) | 0.351 |
| RWG edges per wavelength (mean) | 156.3 |
| Frequency (band stop) | 2.0000 GHz |
| Wavelength | 0.1499 m |
| Largest dimension / wavelength | 0.667 |
| Electrical size ka (enclosing sphere) | 2.343 |
| RWG edges per wavelength (mean) | 23.4 |

## Solver settings

| Setting | Value |
|---|---|
| Modes tracked | 4 |
| Mode ordering | `magnitude` (closest to resonance first) |
| Regularization epsilon | 1.0e-10 |

## Mode tracking

| Setting | Value |
|---|---|
| Correlation metric | `R` (R-weighted, `C_mn = \|J_m^H R J_n\| / sqrt(J_m^H R J_m . J_n^H R J_n)`) |
| Assignment | Hungarian (linear sum assignment), maximising the total `C` |
| Correction stage | `none` |
| Applied to the tables below | yes |
| Accepted correlation, min / median | 0.0009 / 0.9999 |
| Assignment margin, min | -0.0420 |

The saved `.mat` keeps the raw per-frequency order -- the permutation is
stored beside it as `track_order`, so both views are recoverable. The tables
in this report are the tracked one.

| Event | Count | Meaning |
|---|---|---|
| `swap` | 1 | two curves crossed and the solver's labels inverted; tracking followed the modes through |
| `degeneracy` | 0 | `lambda_m ~ lambda_n`, so the eigenvector pair is set by roundoff and correlation cannot separate them |
| `avoided_crossing` | 0 | quasi-degenerate with no label crossing and weak correlation: the currents exchanged character |
| `low_correlation` | 1 | the best match fell below the threshold, usually a sweep step too coarse for that band |

### Events

| Frequency | Kind | Mode(s) | Measurement |
|---|---|---|---|
| 2.0000 GHz | `swap` | 3, 4 | solver labels 3/4 -> 4/3; tracked through with C = 0.0009/0.9980 |
| 2.0000 GHz | `low_correlation` | 3 | best C = 0.0009 < 0.7; the mode changed more between these two frequencies than a mode should, usually a sweep step too coarse for this band |

### Accepted correlation per step

`C` between each mode and its match at the previous frequency. Values near
1 mean the current barely changed over the step; a dip is where the
identity of that mode is least certain.

| Frequency | mode 1 | mode 2 | mode 3 | mode 4 |
|---|---|---|---|---|
| 470.000 MHz | 1.0000 | 1.0000 | 1.0000 | 0.9999 |
| 640.000 MHz | 1.0000 | 1.0000 | 0.9999 | 0.9997 |
| 810.000 MHz | 1.0000 | 1.0000 | 0.9999 | 0.9995 |
| 980.000 MHz | 1.0000 | 1.0000 | 0.9998 | 0.9993 |
| 1.1500 GHz | 1.0000 | 1.0000 | 0.9997 | 0.9990 |
| 1.3200 GHz | 1.0000 | 1.0000 | 0.9996 | 0.9987 |
| 1.4900 GHz | 1.0000 | 1.0000 | 0.9996 | 0.9984 |
| 1.6600 GHz | 1.0000 | 0.9999 | 0.9995 | 0.9981 |
| 1.8300 GHz | 1.0000 | 0.9999 | 0.9996 | 0.9981 |
| 2.0000 GHz | 1.0000 | 0.9999 | 0.0009 | 0.9980 |


## Characteristic values lambda

0 marks a resonance; the sign gives the reactive character.

| Frequency | mode 1 | mode 2 | mode 3 | mode 4 |
|---|---|---|---|---|
| 300.000 MHz | -115.412 | -345.895 | +563.855 | -54261.1 |
| 470.000 MHz | -27.9936 | -90.0884 | +151.247 | -5717 |
| 640.000 MHz | -9.48527 | -35.8093 | +63.0663 | -1210.3 |
| 810.000 MHz | -3.86704 | -17.7667 | +33.14 | -367.989 |
| 980.000 MHz | -1.53572 | -10.1325 | +20.1654 | -139.475 |
| 1.1500 GHz | -0.477707 | -6.34464 | +13.5239 | -61.202 |
| 1.3200 GHz | -0.0262079 | -4.27155 | +9.74969 | -29.7981 |
| 1.4900 GHz | +0.22179 | -3.03681 | +7.40548 | -15.6587 |
| 1.6600 GHz | +0.349762 | -2.24293 | +5.8666 | -8.70555 |
| 1.8300 GHz | +0.415222 | -1.7141 | +4.78968 | -5.06039 |
| 2.0000 GHz | +0.431406 | -1.33504 | -3.74509 | -3.03925 |

## Modal significance MS

1 at resonance, 0 for an unexcitable mode.

| Frequency | mode 1 | mode 2 | mode 3 | mode 4 |
|---|---|---|---|---|
| 300.000 MHz | 0.0087 | 0.0029 | 0.0018 | 0.0000 |
| 470.000 MHz | 0.0357 | 0.0111 | 0.0066 | 0.0002 |
| 640.000 MHz | 0.1048 | 0.0279 | 0.0159 | 0.0008 |
| 810.000 MHz | 0.2504 | 0.0562 | 0.0302 | 0.0027 |
| 980.000 MHz | 0.5457 | 0.0982 | 0.0495 | 0.0072 |
| 1.1500 GHz | 0.9023 | 0.1557 | 0.0737 | 0.0163 |
| 1.3200 GHz | 0.9997 | 0.2279 | 0.1020 | 0.0335 |
| 1.4900 GHz | 0.9763 | 0.3128 | 0.1338 | 0.0637 |
| 1.6600 GHz | 0.9439 | 0.4072 | 0.1680 | 0.1141 |
| 1.8300 GHz | 0.9236 | 0.5039 | 0.2044 | 0.1939 |
| 2.0000 GHz | 0.9182 | 0.5995 | 0.2580 | 0.3125 |

## Characteristic angle delta (deg)

180 deg marks a resonance.

| Frequency | mode 1 | mode 2 | mode 3 | mode 4 |
|---|---|---|---|---|
| 300.000 MHz | 269.50 | 269.83 | 90.10 | 270.00 |
| 470.000 MHz | 267.95 | 269.36 | 90.38 | 269.99 |
| 640.000 MHz | 263.98 | 268.40 | 90.91 | 269.95 |
| 810.000 MHz | 255.50 | 266.78 | 91.73 | 269.84 |
| 980.000 MHz | 236.93 | 264.36 | 92.84 | 269.59 |
| 1.1500 GHz | 205.53 | 261.04 | 94.23 | 269.06 |
| 1.3200 GHz | 181.50 | 256.82 | 95.86 | 268.08 |
| 1.4900 GHz | 167.49 | 251.77 | 97.69 | 266.35 |
| 1.6600 GHz | 160.72 | 245.97 | 99.67 | 263.45 |
| 1.8300 GHz | 157.45 | 239.74 | 101.79 | 258.82 |
| 2.0000 GHz | 156.66 | 233.17 | 255.05 | 251.79 |

## Impedance matrix (at 1.1500 GHz, mid-band)

| Measurement | Value | Reference |
|---|---|---|
| Reciprocity, max\|Z-Z^T\|/max\|Z\| | 0.000e+00 | expected ~1e-4; above 1e-03 is suspect |
| min eig(R) | -3.2803e-17 Ohm | should be >= 0 (R is the radiation operator) |
| max eig(R) | 1.1990e-03 Ohm | - |
| min eig(R) / max eig(R) | -2.736e-14 | negative only at roundoff level |
| Numerical rank of R (> 1e-8 x max) | 27 of 403 | expected << N: only a few current distributions radiate |

Reciprocity reads exactly `0.000e+00` here because the assembler computes
only the `m <= n` half of Z and mirrors it, so `Z = Z^T` holds by
construction rather than by agreement. It is therefore a check that the
mirroring ran, not a check that the kernel is right -- a wrong kernel would
read `0` too. The independent version (both halves computed separately,
`impedance_matrix_z(..., full_fill=True)`) is what still detects a broken
integration scheme; on a correct assembly it lands near 1e-4.

## Caveat on mode numbering

The modes in the tables below have been tracked across the sweep: at each step
the eigenvectors are correlated with the previous frequency in the `R`-weighted
metric and matched one-to-one by the Hungarian algorithm, so a column follows
one physical current rather than the solver's per-frequency `|lambda|` rank. A
`swap` row in the tracking events is therefore a crossing that was followed
through, not an error.

The tracking section above lists what could NOT be decided that way. Inside a
quasi-degenerate pair the eigenvectors are set by roundoff and no correlation
metric can separate them, so a `degeneracy` or `avoided_crossing` row marks a
band where the mode labels may still be interchanged.

A separate trap when reading `lambda(f)`: a sign change is a resonance when
`lambda` passes through **zero**, but a **pole** when it runs off through
infinity, and the second means the mode is at its least excitable, not its most.
`MS` distinguishes them at a glance -- it goes to 1 at a resonance and to 0 at a
pole.

## Definitions

Characteristic Mode Analysis decomposes what a structure can support on its
own, independently of any feed. Each mode is a current pattern; an excitation
only decides how much of each one is stirred up.

Three quantities describe a mode, all carrying the same information:

| Symbol | Definition | Reference values |
|---|---|---|
| `lambda_n` | characteristic value, from `X J = lambda R J` | `0` at resonance; `> 0` inductive (stores magnetic energy); `< 0` capacitive (stores electric energy); unbounded |
| `MS_n` | modal significance, `1 / sqrt(1 + lambda^2)` | `1` at resonance, `0` for an unexcitable mode; `>= 0.707` (i.e. `\|lambda\| <= 1`) is the usual "significant" threshold |
| `delta_n` | characteristic angle, `180 deg - atan(lambda)` | `180 deg` at resonance; `-> 90 deg` strongly inductive; `-> 270 deg` strongly capacitive |

Other reference values used above:

- **RWG edges per wavelength**: below roughly 10, the discretisation rather
  than the geometry starts to set the higher-order results.
- **ka**: electrical size of the enclosing sphere. Well below 1 the structure
  is electrically small, and every `|lambda|` is large.

## Source data

- Eigenvalues and eigenvectors: `data/antennas_cma/plate_cma_sweep.mat`
- Plots: `ModalVisualizer.plot_characteristic_angle_vs_frequency`, `plot_eigenvalues_vs_frequency`, `plot_modal_significance`.

