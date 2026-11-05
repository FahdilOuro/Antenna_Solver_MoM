"""
Shared fixtures for the backend.optim tests.

Z assembly dominates the run time (~5 s for the wide strip), so each mesh's
model is built once per session and the tests derive what they need from it
(ReducedMultiport.from_matrices reuses the stored Z and G).
"""

import numpy as np
import pytest

from backend.optim import ReducedMultiport, port_from_cut
from optim_strip_mesh import make_strip_mesh

FREQUENCY = 3e8           # Hz -- the 0.5 m strip is a half-wave dipole
COPPER_SIGMA = 5.8e7      # S/m
X_HAT = np.array([1.0, 0.0, 0.0])


@pytest.fixture(scope="session")
def wide_strip():
    """0.5 m x 20 mm strip, 16 x 3 cells: 125 RWG edges, 3 edges per transverse cut."""
    return make_strip_mesh(length=0.5, width=0.02, nx=16, ny=3)


@pytest.fixture(scope="session")
def narrow_strip():
    """0.5 m x 10 mm strip, 50 x 1 cells: 99 RWG edges, 1 edge per transverse cut."""
    return make_strip_mesh(length=0.5, width=0.01, nx=50, ny=1)


@pytest.fixture(scope="session")
def wide_ports(wide_strip):
    """Driven port at the centre, loaded ports on the grid lines nearest x = -0.100 m
    (-0.09375 m) and x = +0.125 m."""
    m = wide_strip
    return [port_from_cut(m.points, m.triangles, m.edges, *m.cut_at(m.nearest_grid_x(x)),
                          direction=X_HAT, name=name)
            for name, x in (("feed", 0.0), ("load_left", -0.100), ("load_right", 0.125))]


@pytest.fixture(scope="session")
def wide_model(wide_strip, wide_ports):
    """Copper wide strip, 3 ports, built through the full from_mesh path."""
    m = wide_strip
    return ReducedMultiport.from_mesh(m.points, m.triangles, m.edges, FREQUENCY, wide_ports,
                                      conductivity=COPPER_SIGMA, use_gpu=False)


def rel_err(a, b):
    """||a - b|| / ||b|| (2-norm, works for scalars and arrays)."""
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.linalg.norm(np.ravel(a - b)) / np.linalg.norm(np.ravel(b)))
