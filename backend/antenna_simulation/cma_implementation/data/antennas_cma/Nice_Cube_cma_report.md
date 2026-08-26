# Characteristic Mode Analysis — Nice_Cube

Single frequency: **436.500 MHz**  
Generated 2026-08-24 15:35

## Structure and discretisation

| Quantity | Value |
|---|---|
| Points | 309 |
| Triangles | 530 |
| RWG edges (unknowns) | 757 |
| Bounding box | 0.7100 x 0.7100 x 0.3000 m |
| Edge length min / mean / max | 5.59 / 33.01 / 46.69 mm |
| Frequency | 436.500 MHz |
| Wavelength | 0.6868 m |
| Largest dimension / wavelength | 1.034 |
| Electrical size ka (enclosing sphere) | 4.794 |
| RWG edges per wavelength (mean) | 20.8 |

## Solver settings

| Setting | Value |
|---|---|
| Modes extracted | 4 |
| Mode ordering | `magnitude` (closest to resonance first) |
| Regularization epsilon | 1.0e-10 |

## Modes

| Mode | lambda | MS | delta (deg) |
|---|---|---|---|
| 1 | +1.11137 | 0.6689 | 131.98 |
| 2 | +1.11667 | 0.6671 | 131.84 |
| 3 | +1.30753 | 0.6075 | 127.41 |
| 4 | +2.93321 | 0.3227 | 108.83 |

## Impedance matrix

| Measurement | Value | Reference |
|---|---|---|
| Reciprocity, max\|Z-Z^T\|/max\|Z\| | 0.000e+00 | expected ~1e-4; above 1e-03 is suspect |
| min eig(R) | -1.4880e-14 Ohm | should be >= 0 (R is the radiation operator) |
| max eig(R) | 1.2478e-01 Ohm | - |
| min eig(R) / max eig(R) | -1.193e-13 | negative only at roundoff level |
| Numerical rank of R (> 1e-8 x max) | 93 of 757 | expected << N: only a few current distributions radiate |

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

- Eigenvalues and eigenvectors: `data/antennas_cma/Nice_Cube_cma.mat`
- Per-mode surface currents and far fields: `cma_algorithm(..., save_image=True)` and `compute_modal_farfield`.

