# Characteristic Mode Analysis — Nice_Cube

Frequency sweep: **100.000 MHz to 500.000 MHz**, 7 points, 4 modes  
Generated 2026-10-30 14:13

## Structure and discretisation

| Quantity | Value |
|---|---|
| Points | 307 |
| Triangles | 526 |
| RWG edges (unknowns) | 751 |
| Bounding box | 0.7000 x 0.7000 x 0.3000 m |
| Edge length min / mean / max | 5.59 / 33.04 / 46.69 mm |
| Frequency (band start) | 100.000 MHz |
| Wavelength | 2.9979 m |
| Largest dimension / wavelength | 0.233 |
| Electrical size ka (enclosing sphere) | 1.084 |
| RWG edges per wavelength (mean) | 90.7 |
| Frequency (band stop) | 500.000 MHz |
| Wavelength | 0.5996 m |
| Largest dimension / wavelength | 1.167 |
| Electrical size ka (enclosing sphere) | 5.420 |
| RWG edges per wavelength (mean) | 18.1 |

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
| Accepted correlation, min / median | 0.0003 / 0.9977 |
| Assignment margin, min | -0.0058 |

The saved `.mat` keeps the raw per-frequency order -- the permutation is
stored beside it as `track_order`, so both views are recoverable. The tables
in this report are the tracked one.

| Event | Count | Meaning |
|---|---|---|
| `swap` | 9 | two curves crossed and the solver's labels inverted; tracking followed the modes through |
| `degeneracy` | 10 | `lambda_m ~ lambda_n`, so the eigenvector pair is set by roundoff and correlation cannot separate them |
| `avoided_crossing` | 3 | quasi-degenerate with no label crossing and weak correlation: the currents exchanged character |
| `low_correlation` | 2 | the best match fell below the threshold, usually a sweep step too coarse for that band |

### Events

| Frequency | Kind | Mode(s) | Measurement |
|---|---|---|---|
| 166.667 MHz | `degeneracy` | 1, 2 | lambda = -1.33333 vs -1.35156 (delta 233.13 vs 233.50 deg); relative gap 1.349e-02 < 0.05. The eigenvectors inside a degenerate pair are set by roundoff, so this assignment is not decidable from correlation alone |
| 166.667 MHz | `low_correlation` | 4 | best C = 0.0004 < 0.7; the mode changed more between these two frequencies than a mode should, usually a sweep step too coarse for this band |
| 233.333 MHz | `swap` | 1, 2 | solver labels 1/2 -> 3/2; tracked through with C = 0.9981/0.9998 |
| 233.333 MHz | `swap` | 1, 3 | solver labels 1/3 -> 3/1; tracked through with C = 0.9981/0.9992 |
| 233.333 MHz | `swap` | 2, 3 | solver labels 2/3 -> 2/1; tracked through with C = 0.9998/0.9992 |
| 233.333 MHz | `degeneracy` | 1, 2 | lambda = +0.601038 vs +0.575031 (delta 148.99 vs 150.10 deg); relative gap 2.601e-02 < 0.05. The eigenvectors inside a degenerate pair are set by roundoff, so this assignment is not decidable from correlation alone |
| 300.000 MHz | `swap` | 1, 3 | solver labels 3/1 -> 2/3; tracked through with C = 0.7217/0.7085 |
| 300.000 MHz | `swap` | 2, 3 | solver labels 2/1 -> 1/3; tracked through with C = 0.9994/0.7085 |
| 300.000 MHz | `degeneracy` | 1, 2 | lambda = +1.03232 vs +1.02015 (delta 134.09 vs 134.43 deg); relative gap 1.178e-02 < 0.05. The eigenvectors inside a degenerate pair are set by roundoff, so this assignment is not decidable from correlation alone |
| 300.000 MHz | `avoided_crossing` | 1, 2 | quasi-degenerate without a label crossing, C = 0.7217/0.9994, margin = 0.0188/0.9909; the eigenvalue curves veer apart while the currents exchange character, so mode identity across this step is ambiguous |
| 300.000 MHz | `degeneracy` | 1, 3 | lambda = +1.03232 vs +1.04184 (delta 134.09 vs 133.83 deg); relative gap 9.143e-03 < 0.05. The eigenvectors inside a degenerate pair are set by roundoff, so this assignment is not decidable from correlation alone |
| 300.000 MHz | `degeneracy` | 2, 3 | lambda = +1.02015 vs +1.04184 (delta 134.43 vs 133.83 deg); relative gap 2.082e-02 < 0.05. The eigenvectors inside a degenerate pair are set by roundoff, so this assignment is not decidable from correlation alone |
| 366.667 MHz | `swap` | 1, 3 | solver labels 2/3 -> 3/2; tracked through with C = 0.7360/0.7147 |
| 366.667 MHz | `degeneracy` | 1, 2 | lambda = +1.23763 vs +1.1306 (delta 128.94 vs 131.49 deg); relative gap 8.648e-02 < 0.05. The eigenvectors inside a degenerate pair are set by roundoff, so this assignment is not decidable from correlation alone |
| 366.667 MHz | `avoided_crossing` | 1, 2 | quasi-degenerate without a label crossing, C = 0.7360/0.9977, margin = 0.0610/0.9429; the eigenvalue curves veer apart while the currents exchange character, so mode identity across this step is ambiguous |
| 366.667 MHz | `degeneracy` | 1, 3 | lambda = +1.23763 vs +1.13167 (delta 128.94 vs 131.47 deg); relative gap 8.561e-02 < 0.05. The eigenvectors inside a degenerate pair are set by roundoff, so this assignment is not decidable from correlation alone |
| 366.667 MHz | `degeneracy` | 2, 3 | lambda = +1.1306 vs +1.13167 (delta 131.49 vs 131.47 deg); relative gap 9.456e-04 < 0.05. The eigenvectors inside a degenerate pair are set by roundoff, so this assignment is not decidable from correlation alone |
| 366.667 MHz | `avoided_crossing` | 2, 3 | quasi-degenerate without a label crossing, C = 0.9977/0.7147, margin = 0.9429/0.0182; the eigenvalue curves veer apart while the currents exchange character, so mode identity across this step is ambiguous |
| 433.333 MHz | `swap` | 2, 3 | solver labels 1/2 -> 2/1; tracked through with C = 0.9780/0.9978 |
| 433.333 MHz | `degeneracy` | 2, 3 | lambda = +1.11657 vs +1.11003 (delta 131.85 vs 132.01 deg); relative gap 5.850e-03 < 0.05. The eigenvectors inside a degenerate pair are set by roundoff, so this assignment is not decidable from correlation alone |
| 500.000 MHz | `swap` | 1, 4 | solver labels 3/4 -> 4/3; tracked through with C = 0.9736/0.0003 |
| 500.000 MHz | `swap` | 2, 3 | solver labels 2/1 -> 1/2; tracked through with C = 0.9893/0.9969 |
| 500.000 MHz | `degeneracy` | 2, 3 | lambda = +1.03575 vs +1.04039 (delta 133.99 vs 133.87 deg); relative gap 4.460e-03 < 0.05. The eigenvectors inside a degenerate pair are set by roundoff, so this assignment is not decidable from correlation alone |
| 500.000 MHz | `low_correlation` | 4 | best C = 0.0003 < 0.7; the mode changed more between these two frequencies than a mode should, usually a sweep step too coarse for this band |

### Accepted correlation per step

`C` between each mode and its match at the previous frequency. Values near
1 mean the current barely changed over the step; a dip is where the
identity of that mode is least certain.

| Frequency | mode 1 | mode 2 | mode 3 | mode 4 |
|---|---|---|---|---|
| 166.667 MHz | 0.9999 | 1.0000 | 0.9999 | 0.0004 |
| 233.333 MHz | 0.9981 | 0.9998 | 0.9992 | 0.9981 |
| 300.000 MHz | 0.7217 | 0.9994 | 0.7085 | 0.9989 |
| 366.667 MHz | 0.7360 | 0.9977 | 0.7147 | 0.9999 |
| 433.333 MHz | 0.9743 | 0.9780 | 0.9978 | 0.9997 |
| 500.000 MHz | 0.9736 | 0.9893 | 0.9969 | 0.0003 |


## Characteristic values lambda

0 marks a resonance; the sign gives the reactive character.

| Frequency | mode 1 | mode 2 | mode 3 | mode 4 |
|---|---|---|---|---|
| 100.000 MHz | -14.0892 | -14.6127 | -30.2372 | +198.914 |
| 166.667 MHz | -1.33333 | -1.35156 | -3.61311 | -28.8768 |
| 233.333 MHz | +0.601038 | +0.575031 | +0.142283 | +0.809667 |
| 300.000 MHz | +1.03232 | +1.02015 | +1.04184 | +3.4195 |
| 366.667 MHz | +1.23763 | +1.1306 | +1.13167 | +3.28024 |
| 433.333 MHz | +1.28635 | +1.11657 | +1.11003 | +2.93631 |
| 500.000 MHz | +1.39972 | +1.03575 | +1.04039 | -1.21396 |

## Modal significance MS

1 at resonance, 0 for an unexcitable mode.

| Frequency | mode 1 | mode 2 | mode 3 | mode 4 |
|---|---|---|---|---|
| 100.000 MHz | 0.0708 | 0.0683 | 0.0331 | 0.0050 |
| 166.667 MHz | 0.6000 | 0.5948 | 0.2667 | 0.0346 |
| 233.333 MHz | 0.8571 | 0.8669 | 0.9900 | 0.7772 |
| 300.000 MHz | 0.6958 | 0.7000 | 0.6925 | 0.2807 |
| 366.667 MHz | 0.6285 | 0.6625 | 0.6622 | 0.2916 |
| 433.333 MHz | 0.6138 | 0.6672 | 0.6693 | 0.3224 |
| 500.000 MHz | 0.5813 | 0.6946 | 0.6930 | 0.6358 |

## Characteristic angle delta (deg)

180 deg marks a resonance.

| Frequency | mode 1 | mode 2 | mode 3 | mode 4 |
|---|---|---|---|---|
| 100.000 MHz | 265.94 | 266.09 | 268.11 | 90.29 |
| 166.667 MHz | 233.13 | 233.50 | 254.53 | 268.02 |
| 233.333 MHz | 148.99 | 150.10 | 171.90 | 141.00 |
| 300.000 MHz | 134.09 | 134.43 | 133.83 | 106.30 |
| 366.667 MHz | 128.94 | 131.49 | 131.47 | 106.95 |
| 433.333 MHz | 127.86 | 131.85 | 132.01 | 108.81 |
| 500.000 MHz | 125.54 | 133.99 | 133.87 | 230.52 |

## Impedance matrix (at 300.000 MHz, mid-band)

| Measurement | Value | Reference |
|---|---|---|
| Reciprocity, max\|Z-Z^T\|/max\|Z\| | 0.000e+00 | expected ~1e-4; above 1e-03 is suspect |
| min eig(R) | -1.1415e-15 Ohm | should be >= 0 (R is the radiation operator) |
| max eig(R) | 6.7683e-02 Ohm | - |
| min eig(R) / max eig(R) | -1.687e-14 | negative only at roundoff level |
| Numerical rank of R (> 1e-8 x max) | 69 of 751 | expected << N: only a few current distributions radiate |

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

- Eigenvalues and eigenvectors: `data/antennas_cma/Nice_Cube_cma_sweep.mat`
- Plots: `ModalVisualizer.plot_characteristic_angle_vs_frequency`, `plot_eigenvalues_vs_frequency`, `plot_modal_significance`.

