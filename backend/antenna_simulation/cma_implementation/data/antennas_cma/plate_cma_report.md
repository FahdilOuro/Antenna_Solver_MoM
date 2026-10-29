# Characteristic Mode Analysis — plate

Single frequency: **1.0000 GHz**  
Generated 2026-10-10 20:40

## Structure and discretisation

| Quantity | Value |
|---|---|
| Points | 166 |
| Triangles | 284 |
| RWG edges (unknowns) | 403 |
| Bounding box | 0.1000 x 0.0500 x 0.0000 m |
| Edge length min / mean / max | 4.50 / 6.40 / 8.08 mm |
| Frequency | 1.0000 GHz |
| Wavelength | 0.2998 m |
| Largest dimension / wavelength | 0.334 |
| Electrical size ka (enclosing sphere) | 1.172 |
| RWG edges per wavelength (mean) | 46.9 |

## Solver settings

| Setting | Value |
|---|---|
| Modes extracted | 4 |
| Mode ordering | `magnitude` (closest to resonance first) |
| Regularization epsilon | 1.0e-10 |

## Modes

| Mode | lambda | MS | delta (deg) |
|---|---|---|---|
| 1 | -1.33145 | 0.6005 | 233.09 |
| 2 | -9.54503 | 0.1042 | 264.02 |
| 3 | +19.1287 | 0.0522 | 92.99 |
| 4 | -125.745 | 0.0080 | 269.54 |

## Impedance matrix

| Measurement | Value | Reference |
|---|---|---|
| Reciprocity, max\|Z-Z^T\|/max\|Z\| | 0.000e+00 | expected ~1e-4; above 1e-03 is suspect |
| min eig(R) | -3.4571e-17 Ohm | should be >= 0 (R is the radiation operator) |
| max eig(R) | 9.3525e-04 Ohm | - |
| min eig(R) / max eig(R) | -3.696e-14 | negative only at roundoff level |
| Numerical rank of R (> 1e-8 x max) | 26 of 403 | expected << N: only a few current distributions radiate |

Reciprocity reads exactly `0.000e+00` here because the assembler computes
only the `m <= n` half of Z and mirrors it, so `Z = Z^T` holds by
construction rather than by agreement. It is therefore a check that the
mirroring ran, not a check that the kernel is right -- a wrong kernel would
read `0` too. The independent version (both halves computed separately,
`impedance_matrix_z(..., full_fill=True)`) is what still detects a broken
integration scheme; on a correct assembly it lands near 1e-4.

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

