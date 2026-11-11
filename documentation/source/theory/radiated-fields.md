# Radiated fields, directivity and gain

{.mom-status}
{bdg-secondary-line}`Planned` Outline of a future page.

From the edge currents to the fields, the radiated power and the patterns.

**This page will cover**

- The dipole model used by `solver/efield/` (`solver.utils.dipole_parameters`): each RWG function
  as a small dipole of moment $\ell_n I_n(\mathbf r_n^{c-}-\mathbf r_n^{c+})$, and the fields it
  radiates at any distance.
- The exact radiation integral with Gauss–Dunavant quadrature used by `solver.optim.farfield.RadiationOperator`,
  and when the two differ.
- Radiated power by integration of the Poynting vector over a sphere; radiation intensity.
- Directivity, gain, realised gain and radiation efficiency, precisely defined. Note that the value
  printed as "Gain" by `efield2` is $4\pi U_\mathrm{max}/P_\mathrm{rad}$, i.e. a directivity.
- Circular polarisation components (RHCP / LHCP) and RCS.
