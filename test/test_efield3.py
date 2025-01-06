from test_setup import setup_project_root
setup_project_root()

from efield.efield3 import antenna_directivity_pattern


filename_mesh2_to_load = 'data/antennas_mesh2/dipole_mesh2.mat'
filename_current_to_load = 'data/antennas_current/dipole_current.mat'
filename_gain_power_to_load = 'data/antennas_gain_power/dipole_gain_power.mat'

antenna_directivity_pattern(filename_mesh2_to_load, filename_current_to_load, filename_gain_power_to_load)