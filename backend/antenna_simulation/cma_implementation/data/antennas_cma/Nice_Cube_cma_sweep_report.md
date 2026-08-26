# Characteristic Mode Analysis — Nice_Cube

Frequency sweep: **100.000 MHz to 500.000 MHz**, 7 points, 4 modes  
Generated 2026-08-24 15:34

## Structure and discretisation

| Quantity | Value |
|---|---|
| Points | 309 |
| Triangles | 530 |
| RWG edges (unknowns) | 757 |
| Bounding box | 0.7100 x 0.7100 x 0.3000 m |
| Edge length min / mean / max | 5.59 / 33.01 / 46.69 mm |
| Frequency (band start) | 100.000 MHz |
| Wavelength | 2.9979 m |
| Largest dimension / wavelength | 0.237 |
| Electrical size ka (enclosing sphere) | 1.098 |
| RWG edges per wavelength (mean) | 90.8 |
| Frequency (band stop) | 500.000 MHz |
| Wavelength | 0.5996 m |
| Largest dimension / wavelength | 1.184 |
| Electrical size ka (enclosing sphere) | 5.491 |
| RWG edges per wavelength (mean) | 18.2 |

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
| Accepted correlation, min / median | 0.0004 / 0.9993 |
| Assignment margin, min | -0.0004 |

The saved `.mat` keeps the raw per-frequency order -- the permutation is
stored beside it as `track_order`, so both views are recoverable. The tables
in this report are the tracked one.

| Event | Count | Meaning |
|---|---|---|
| `swap` | 11 | two curves crossed and the solver's labels inverted; tracking followed the modes through |
| `degeneracy` | 10 | `lambda_m ~ lambda_n`, so the eigenvector pair is set by roundoff and correlation cannot separate them |
| `avoided_crossing` | 0 | quasi-degenerate with no label crossing and weak correlation: the currents exchanged character |
| `low_correlation` | 2 | the best match fell below the threshold, usually a sweep step too coarse for that band |

### Events

| Frequency | Kind | Mode(s) | Measurement |
|---|---|---|---|
| 166.667 MHz | `degeneracy` | 1, 2 | lambda = -1.2076 vs -1.21093 (delta 230.37 vs 230.45 deg); relative gap 2.755e-03 < 0.05. The eigenvectors inside a degenerate pair are set by roundoff, so this assignment is not decidable from correlation alone |
| 166.667 MHz | `low_correlation` | 4 | best C = 0.0005 < 0.7; the mode changed more between these two frequencies than a mode should, usually a sweep step too coarse for this band |
| 233.333 MHz | `swap` | 1, 2 | solver labels 1/2 -> 3/2; tracked through with C = 0.9998/0.9997 |
| 233.333 MHz | `swap` | 1, 3 | solver labels 1/3 -> 3/1; tracked through with C = 0.9998/0.9994 |
| 233.333 MHz | `swap` | 2, 3 | solver labels 2/3 -> 2/1; tracked through with C = 0.9997/0.9994 |
| 233.333 MHz | `degeneracy` | 1, 2 | lambda = +0.639779 vs +0.609133 (delta 147.39 vs 148.65 deg); relative gap 3.065e-02 < 0.05. The eigenvectors inside a degenerate pair are set by roundoff, so this assignment is not decidable from correlation alone |
| 300.000 MHz | `swap` | 1, 3 | solver labels 3/1 -> 2/3; tracked through with C = 0.9994/0.9991 |
| 300.000 MHz | `swap` | 2, 3 | solver labels 2/1 -> 1/3; tracked through with C = 0.9994/0.9991 |
| 300.000 MHz | `degeneracy` | 1, 2 | lambda = +1.04814 vs +1.03967 (delta 133.65 vs 133.89 deg); relative gap 8.087e-03 < 0.05. The eigenvectors inside a degenerate pair are set by roundoff, so this assignment is not decidable from correlation alone |
| 300.000 MHz | `degeneracy` | 1, 3 | lambda = +1.04814 vs +1.06912 (delta 133.65 vs 133.09 deg); relative gap 1.962e-02 < 0.05. The eigenvectors inside a degenerate pair are set by roundoff, so this assignment is not decidable from correlation alone |
| 300.000 MHz | `degeneracy` | 2, 3 | lambda = +1.03967 vs +1.06912 (delta 133.89 vs 133.09 deg); relative gap 2.754e-02 < 0.05. The eigenvectors inside a degenerate pair are set by roundoff, so this assignment is not decidable from correlation alone |
| 366.667 MHz | `swap` | 1, 2 | solver labels 2/1 -> 1/2; tracked through with C = 0.9977/0.9890 |
| 366.667 MHz | `degeneracy` | 1, 2 | lambda = +1.14379 vs +1.14408 (delta 131.16 vs 131.16 deg); relative gap 2.583e-04 < 0.05. The eigenvectors inside a degenerate pair are set by roundoff, so this assignment is not decidable from correlation alone |
| 366.667 MHz | `degeneracy` | 1, 3 | lambda = +1.14379 vs +1.25767 (delta 131.16 vs 128.49 deg); relative gap 9.055e-02 < 0.05. The eigenvectors inside a degenerate pair are set by roundoff, so this assignment is not decidable from correlation alone |
| 366.667 MHz | `degeneracy` | 2, 3 | lambda = +1.14408 vs +1.25767 (delta 131.16 vs 128.49 deg); relative gap 9.032e-02 < 0.05. The eigenvectors inside a degenerate pair are set by roundoff, so this assignment is not decidable from correlation alone |
| 433.333 MHz | `swap` | 1, 2 | solver labels 1/2 -> 2/1; tracked through with C = 0.9782/0.9874 |
| 433.333 MHz | `degeneracy` | 1, 2 | lambda = +1.11914 vs +1.11569 (delta 131.78 vs 131.87 deg); relative gap 3.082e-03 < 0.05. The eigenvectors inside a degenerate pair are set by roundoff, so this assignment is not decidable from correlation alone |
| 500.000 MHz | `swap` | 1, 2 | solver labels 2/1 -> 2/3; tracked through with C = 0.9896/0.9991 |
| 500.000 MHz | `swap` | 1, 4 | solver labels 2/4 -> 2/1; tracked through with C = 0.9896/0.0004 |
| 500.000 MHz | `swap` | 2, 4 | solver labels 1/4 -> 3/1; tracked through with C = 0.9991/0.0004 |
| 500.000 MHz | `swap` | 3, 4 | solver labels 3/4 -> 4/1; tracked through with C = 0.9742/0.0004 |
| 500.000 MHz | `degeneracy` | 1, 2 | lambda = +1.03805 vs +1.03822 (delta 133.93 vs 133.93 deg); relative gap 1.633e-04 < 0.05. The eigenvectors inside a degenerate pair are set by roundoff, so this assignment is not decidable from correlation alone |
| 500.000 MHz | `low_correlation` | 4 | best C = 0.0004 < 0.7; the mode changed more between these two frequencies than a mode should, usually a sweep step too coarse for this band |

### Accepted correlation per step

`C` between each mode and its match at the previous frequency. Values near
1 mean the current barely changed over the step; a dip is where the
identity of that mode is least certain.

| Frequency | mode 1 | mode 2 | mode 3 | mode 4 |
|---|---|---|---|---|
| 166.667 MHz | 1.0000 | 0.9999 | 0.9999 | 0.0005 |
| 233.333 MHz | 0.9998 | 0.9997 | 0.9994 | 0.9996 |
| 300.000 MHz | 0.9994 | 0.9994 | 0.9991 | 0.9998 |
| 366.667 MHz | 0.9977 | 0.9890 | 0.9865 | 1.0000 |
| 433.333 MHz | 0.9782 | 0.9874 | 0.9652 | 0.9998 |
| 500.000 MHz | 0.9896 | 0.9991 | 0.9742 | 0.0004 |


## Characteristic values lambda

0 marks a resonance; the sign gives the reactive character.

| Frequency | mode 1 | mode 2 | mode 3 | mode 4 |
|---|---|---|---|---|
| 100.000 MHz | -13.906 | -14.1523 | -29.9699 | +198.32 |
| 166.667 MHz | -1.2076 | -1.21093 | -3.52143 | -26.019 |
| 233.333 MHz | +0.639779 | +0.609133 | +0.244856 | +1.21857 |
| 300.000 MHz | +1.04814 | +1.03967 | +1.06912 | +3.48826 |
| 366.667 MHz | +1.14379 | +1.14408 | +1.25767 | +3.29512 |
| 433.333 MHz | +1.11914 | +1.11569 | +1.30456 | +2.94477 |
| 500.000 MHz | +1.03805 | +1.03822 | +1.4239 | -1.03548 |

## Modal significance MS

1 at resonance, 0 for an unexcitable mode.

| Frequency | mode 1 | mode 2 | mode 3 | mode 4 |
|---|---|---|---|---|
| 100.000 MHz | 0.0717 | 0.0705 | 0.0333 | 0.0050 |
| 166.667 MHz | 0.6378 | 0.6368 | 0.2732 | 0.0384 |
| 233.333 MHz | 0.8424 | 0.8540 | 0.9713 | 0.6344 |
| 300.000 MHz | 0.6903 | 0.6932 | 0.6831 | 0.2756 |
| 366.667 MHz | 0.6582 | 0.6581 | 0.6224 | 0.2904 |
| 433.333 MHz | 0.6663 | 0.6674 | 0.6084 | 0.3216 |
| 500.000 MHz | 0.6938 | 0.6937 | 0.5747 | 0.6947 |

## Characteristic angle delta (deg)

180 deg marks a resonance.

| Frequency | mode 1 | mode 2 | mode 3 | mode 4 |
|---|---|---|---|---|
| 100.000 MHz | 265.89 | 265.96 | 268.09 | 90.29 |
| 166.667 MHz | 230.37 | 230.45 | 254.15 | 267.80 |
| 233.333 MHz | 147.39 | 148.65 | 166.24 | 129.37 |
| 300.000 MHz | 133.65 | 133.89 | 133.09 | 106.00 |
| 366.667 MHz | 131.16 | 131.16 | 128.49 | 106.88 |
| 433.333 MHz | 131.78 | 131.87 | 127.47 | 108.76 |
| 500.000 MHz | 133.93 | 133.93 | 125.08 | 226.00 |

## Impedance matrix (at 300.000 MHz, mid-band)

| Measurement | Value | Reference |
|---|---|---|
| Reciprocity, max\|Z-Z^T\|/max\|Z\| | 0.000e+00 | expected ~1e-4; above 1e-03 is suspect |
| min eig(R) | -1.1954e-15 Ohm | should be >= 0 (R is the radiation operator) |
| max eig(R) | 6.7961e-02 Ohm | - |
| min eig(R) / max eig(R) | -1.759e-14 | negative only at roundoff level |
| Numerical rank of R (> 1e-8 x max) | 69 of 757 | expected << N: only a few current distributions radiate |

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

