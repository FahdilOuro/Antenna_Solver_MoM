# Geometry and meshing

{.mom-status}
{bdg-secondary-line}`Planned` Outline of a future page.

**This page will cover**

- Drawing a geometry with the gmsh Python API and the helpers of `solver/utils/gmsh_function.py`
  (`rectangle_surface`, `generate_and_save_mesh`, `setup_performance_config`, ...).
- Ready-made parametric antennas: plate, slot, bowtie, IFA, strip
  (`solver/utils/create_antenna_for_scattering.py`), and structured meshes without gmsh
  (`solver/optim/geometries/planar.py`).
- Choosing the element size and the gap width (`mesh_and_gap_parameters`, default λ/20).
- Conversion `.msh` → `.mat` (`extract_msh_to_mat`) and importing MATLAB meshes.
- Junctions between surfaces and what makes a mesh unusable (non-conforming edges, duplicated nodes),
  see {doc}`../theory/rwg-basis-functions`.
- Adaptive h-refinement ({bdg-warning-line}`Experimental`, `solver/utils/h_refinement_*.py`).
