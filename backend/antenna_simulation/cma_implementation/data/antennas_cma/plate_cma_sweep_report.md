# Characteristic Mode Analysis — plate

Frequency sweep: **300.000 MHz to 2.0000 GHz**, 9 points, 4 modes  
Generated 2026-08-18 15:01

## Structure and discretisation

| Quantity | Value |
|---|---|
| Points | 216 |
| Triangles | 380 |
| RWG edges (unknowns) | 545 |
| Bounding box | 0.0800 x 0.0900 x 0.0000 m |
| Edge length min / mean / max | 4.63 / 6.63 / 8.43 mm |
| Frequency (band start) | 300.000 MHz |
| Wavelength | 0.9993 m |
| Largest dimension / wavelength | 0.090 |
| Electrical size ka (enclosing sphere) | 0.379 |
| RWG edges per wavelength (mean) | 150.8 |
| Frequency (band stop) | 2.0000 GHz |
| Wavelength | 0.1499 m |
| Largest dimension / wavelength | 0.600 |
| Electrical size ka (enclosing sphere) | 2.524 |
| RWG edges per wavelength (mean) | 22.6 |

## Solver settings

| Setting | Value |
|---|---|
| Modes tracked | 4 |
| Mode ordering | `magnitude` (closest to resonance first) |
| Regularization epsilon | 1.0e-10 |


## Characteristic values lambda

0 marks a resonance; the sign gives the reactive character.

| Frequency | mode 1 | mode 2 | mode 3 | mode 4 |
|---|---|---|---|---|
| 300.000 MHz | -108.287 | -130.691 | +295.076 | -22790.2 |
| 512.500 MHz | -20.0357 | -24.757 | +62.8426 | -1550.36 |
| 725.000 MHz | -6.14362 | -8.11102 | +23.9047 | -268.142 |
| 937.500 MHz | -2.32863 | -3.31906 | +12.1558 | -71.3739 |
| 1.1500 GHz | -0.927201 | -1.55166 | +7.32795 | -24.0873 |
| 1.3625 GHz | -0.350049 | -0.773026 | +4.89272 | -9.29399 |
| 1.5750 GHz | -0.0903046 | -0.392698 | +3.52777 | -3.78407 |
| 1.7875 GHz | +0.0353019 | -0.185838 | -1.49744 | +2.6636 |
| 2.0000 GHz | -0.089161 | +0.0902834 | -0.475703 | +2.09136 |

## Modal significance MS

1 at resonance, 0 for an unexcitable mode.

| Frequency | mode 1 | mode 2 | mode 3 | mode 4 |
|---|---|---|---|---|
| 300.000 MHz | 0.0092 | 0.0077 | 0.0034 | 0.0000 |
| 512.500 MHz | 0.0498 | 0.0404 | 0.0159 | 0.0006 |
| 725.000 MHz | 0.1607 | 0.1224 | 0.0418 | 0.0037 |
| 937.500 MHz | 0.3946 | 0.2885 | 0.0820 | 0.0140 |
| 1.1500 GHz | 0.7333 | 0.5417 | 0.1352 | 0.0415 |
| 1.3625 GHz | 0.9438 | 0.7912 | 0.2002 | 0.1070 |
| 1.5750 GHz | 0.9959 | 0.9308 | 0.2727 | 0.2555 |
| 1.7875 GHz | 0.9994 | 0.9832 | 0.5554 | 0.3515 |
| 2.0000 GHz | 0.9960 | 0.9959 | 0.9030 | 0.4314 |

## Characteristic angle delta (deg)

180 deg marks a resonance.

| Frequency | mode 1 | mode 2 | mode 3 | mode 4 |
|---|---|---|---|---|
| 300.000 MHz | 269.47 | 269.56 | 90.19 | 270.00 |
| 512.500 MHz | 267.14 | 267.69 | 90.91 | 269.96 |
| 725.000 MHz | 260.76 | 262.97 | 92.40 | 269.79 |
| 937.500 MHz | 246.76 | 253.23 | 94.70 | 269.20 |
| 1.1500 GHz | 222.84 | 237.20 | 97.77 | 267.62 |
| 1.3625 GHz | 199.29 | 217.70 | 101.55 | 263.86 |
| 1.5750 GHz | 185.16 | 201.44 | 105.83 | 255.20 |
| 1.7875 GHz | 177.98 | 190.53 | 236.26 | 110.58 |
| 2.0000 GHz | 185.10 | 174.84 | 205.44 | 115.56 |

## Impedance matrix (at 1.1500 GHz, mid-band)

| Measurement | Value | Reference |
|---|---|---|
| Reciprocity, max\|Z-Z^T\|/max\|Z\| | 1.765e-04 | expected ~1e-4; above 1e-03 is suspect |
| min eig(R) | -3.9629e-17 Ohm | should be >= 0 (R is the radiation operator) |
| max eig(R) | 1.7119e-03 Ohm | - |
| min eig(R) / max eig(R) | -2.315e-14 | negative only at roundoff level |
| Numerical rank of R (> 1e-8 x max) | 31 of 545 | expected << N: only a few current distributions radiate |

## Caveat on mode numbering

At each frequency the modes are sorted independently by `|lambda|`; no attempt
is made to follow a given physical mode across the sweep. Where two eigenvalue
curves cross, the labels "mode 1" and "mode 2" can therefore swap between
adjacent rows of the tables below.

A related trap when reading `lambda(f)`: a sign change is a resonance when
`lambda` passes through **zero**, but a **pole** when it runs off through
infinity, and the second means the mode is at its least excitable, not its
most. `MS` distinguishes them at a glance -- it goes to 1 at a resonance and to
0 at a pole.

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

