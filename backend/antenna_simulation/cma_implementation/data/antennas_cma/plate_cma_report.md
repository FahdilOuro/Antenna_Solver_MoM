# Characteristic Mode Analysis — plate

Single frequency: **1.0000 GHz**  
Generated 2026-08-18 15:01

## Structure and discretisation

| Quantity | Value |
|---|---|
| Points | 216 |
| Triangles | 380 |
| RWG edges (unknowns) | 545 |
| Bounding box | 0.0800 x 0.0900 x 0.0000 m |
| Edge length min / mean / max | 4.63 / 6.63 / 8.43 mm |
| Frequency | 1.0000 GHz |
| Wavelength | 0.2998 m |
| Largest dimension / wavelength | 0.300 |
| Electrical size ka (enclosing sphere) | 1.262 |
| RWG edges per wavelength (mean) | 45.2 |

## Solver settings

| Setting | Value |
|---|---|
| Modes extracted | 4 |
| Mode ordering | `magnitude` (closest to resonance first) |
| Regularization epsilon | 1.0e-10 |

## Modes

| Mode | lambda | MS | delta (deg) |
|---|---|---|---|
| 1 | -1.76215 | 0.4936 | 240.43 |
| 2 | -2.61553 | 0.3571 | 249.08 |
| 3 | +10.2868 | 0.0968 | 95.55 |
| 4 | -50.8531 | 0.0197 | 268.87 |

## Impedance matrix

| Measurement | Value | Reference |
|---|---|---|
| Reciprocity, max\|Z-Z^T\|/max\|Z\| | 1.764e-04 | expected ~1e-4; above 1e-03 is suspect |
| min eig(R) | -4.1057e-17 Ohm | should be >= 0 (R is the radiation operator) |
| max eig(R) | 1.3536e-03 Ohm | - |
| min eig(R) / max eig(R) | -3.033e-14 | negative only at roundoff level |
| Numerical rank of R (> 1e-8 x max) | 27 of 545 | expected << N: only a few current distributions radiate |

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

- Eigenvalues and eigenvectors: `data/antennas_cma/plate_cma.mat`
- Per-mode surface currents and far fields: `cma_algorithm(..., save_image=True)` and `compute_modal_farfield`.

