# Lossy conductors

{.mom-status}
{bdg-secondary-line}`Planned` Outline of a future page.

Modelling a real metal (copper, ...) instead of a perfect conductor with a surface impedance.

**This page will cover**

- The surface impedance $Z_s = (1+j)\sqrt{\omega\mu/(2\sigma)}$ of a good conductor, and the
  superconductor form $Z_s = j\omega\mu\lambda_L$ (`solver/utils/lossy_electric_conductor.py`).
- The RWG Gram matrix $G_{mn} = \int \mathbf f_m\cdot\mathbf f_n\,dS$ (`impmet.rwg_gram_matrix`) and
  the modified system $(Z + Z_s G)\,I = V$ (`rwg4.calculate_current_radiation`).
- The open question of the scaling factor on a zero-thickness sheet (`zs_scale` in
  `solver/optim/reduced_model.py`).
- Known issue: radiation efficiency of lossy antennas (see {doc}`../validation/feature-status`).
