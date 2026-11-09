"""
Antenna optimisation building blocks.

ports          -- ports as cuts of RWG edges (incidence vectors a_p)
reduced_model  -- exact P-port reduction of the MoM problem (ReducedMultiport)
farfield       -- exact radiation integral by triangle quadrature (RadiationOperator)
objectives     -- directivity, gain, realized gain
geometries     -- structured planar meshes without Gmsh (strip, loop, parallel strips)
threads        -- BLAS thread limiting for small-matrix loops

See solver/optim/README.md.
"""

from solver.optim.ports import (
    Port, PortDefinitionError, port_from_cut, port_from_edge, incidence_matrix,
)
from solver.optim.reduced_model import ReducedMultiport, PortSolution
from solver.optim.farfield import RadiationOperator

__all__ = [
    "Port", "PortDefinitionError", "port_from_cut", "port_from_edge", "incidence_matrix",
    "ReducedMultiport", "PortSolution", "RadiationOperator",
]
