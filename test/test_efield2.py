from test_setup import setup_project_root
setup_project_root()

from efield.efield2 import radiation_intensity_distribution_over_sphere_surface

filename_mesh2_to_load = 'data/antennas_mesh2/dipole_mesh2.mat'
filename_current_to_load = 'data/antennas_current/dipole_current.mat'
filename_sphere_to_load = 'data/sphere_mesh/sphere.mat'
filename_sphere_dense_to_load = 'data/sphere_mesh/sphere_dense.mat'

# calcul et affichage soit avec sphere ou dense sphere
radiation_intensity_distribution_over_sphere_surface(filename_mesh2_to_load, filename_current_to_load, filename_sphere_dense_to_load)