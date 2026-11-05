"""Structured meshes for the optimiser (no Gmsh): see planar.py."""

from backend.optim.geometries.planar import (
    PlanarMesh, grading_nodes, tensor_grid_mesh, strip_dipole, square_loop, parallel_strips,
)

__all__ = ["PlanarMesh", "grading_nodes", "tensor_grid_mesh", "strip_dipole", "square_loop",
           "parallel_strips"]
