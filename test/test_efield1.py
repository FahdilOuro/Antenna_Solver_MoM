from test_setup import setup_project_root
setup_project_root()

import numpy as np

from efield.efield1 import calculate_electric_magnetic_field_at_point

filename_mesh2_to_load = 'data/antennas_mesh2/dipole_mesh2.mat'
filename_current_to_load = 'data/antennas_current/dipole_current.mat'

observation_point = np.array([5, 0, 0])

calculate_electric_magnetic_field_at_point(filename_mesh2_to_load, filename_current_to_load, observation_point)