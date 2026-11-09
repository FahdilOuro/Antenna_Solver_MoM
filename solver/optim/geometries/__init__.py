"""Structured meshes for the optimiser (no Gmsh): see planar.py."""

from solver.optim.geometries.planar import (
    PlanarMesh, grading_nodes, tensor_grid_mesh, strip_dipole, square_loop, parallel_strips,
    parallel_strip_array, array_ports, BoxError, BOX_LENGTH, BOX_WIDTH,
)

__all__ = ["PlanarMesh", "grading_nodes", "tensor_grid_mesh", "strip_dipole", "square_loop",
           "parallel_strips", "parallel_strip_array", "array_ports", "BoxError",
           "BOX_LENGTH", "BOX_WIDTH"]
