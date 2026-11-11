# Excitations and input impedance

{.mom-status}
{bdg-secondary-line}`Planned` Outline of a future page.

The right-hand side $V$ of $Z I = V$: how a feed or an incident wave enters the system, and how
the input impedance and power are read from the solution.

**This page will cover**

- The gap source of `solver.utils.gap_source.multiple_gap_sources` (enhanced gap model, Ding et al.
  2013, eq. 7, as cited in the code): window of width `gap_width` around each feed point,
  amplitude and phase per port.
- Choosing `gap_width` relative to the wavelength and to the mesh
  (`gmsh_function.mesh_and_gap_parameters`).
- Port results (`rwg4.compute_port_results`): $P_\mathrm{in} = \tfrac12\mathrm{Re}(V\cdot I^*)$ per
  port and the input impedance derived from it.
- The plane-wave right-hand side for scattering (`rwg4.calculate_current_scattering`).
- Lumped R, L, C loads (`rwg3.calculate_z_matrice_lumped_elements`) and ports defined by a cut
  (`solver/optim/ports.py`).
- Known issue: the power budget of multi-port gap excitation (see
  {doc}`../validation/feature-status`).
