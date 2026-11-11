# Notation and conventions

{.mom-status}
{bdg-secondary-line}`Planned` Outline of a future page.

One place for every symbol, unit and sign convention used in the theory pages, so that the
equations of different pages can be combined without translation.

**This page will cover**

- The time convention $e^{j\omega t}$ and the Green's function $e^{-jkR}/(4\pi R)$ used by the
  impedance assembly.
- The physical constants as the code defines them (`solver/rwg/rwg3.py`): $\varepsilon_0 = 8.854\times10^{-12}$ F/m,
  $\mu_0 = 1.257\times10^{-6}$ H/m (a rounding of $4\pi\times10^{-7}$), and $c$, $\eta$ derived from
  them.
- Units: SI throughout, lengths in metres, frequencies in hertz.
- The symbol table: mesh quantities ($\ell_n$, $A_n^\pm$, $T_n^\pm$), operators, matrices
  ($Z = R + jX$, $G$), modal quantities ($\lambda_n$, $\mathbf J_n$).
- Naming correspondence between the equations and the code (`matrice_z`, `vecteurs_rho`, ...).
