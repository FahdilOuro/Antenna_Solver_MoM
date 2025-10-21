import os

from test_setup import setup_project_root
setup_project_root()

from rwg.rwg2 import DataManager_rwg2
from rwg.rwg5 import *

filename_plate_to_load = 'data/antennas_mesh2/plate_mesh2.mat'
filename_current_plate = 'data/antennas_current/plate_current.mat'

points_plate, triangles_plate, edges_plate, barycentric_triangles_plate, vecteurs_rho_plate = DataManager_rwg2.load_data(filename_plate_to_load)

print("Calcul des courants de surfaces")
surface_current_density_plate = calculate_current_density(filename_current_plate, triangles_plate, edges_plate, vecteurs_rho_plate)

antennas_name = os.path.splitext(os.path.basename(filename_plate_to_load))[0].replace('_mesh2', ' antennas surface in receiving mode')
print(f"{antennas_name} is successfully created")
fig = visualize_surface_current(points_plate, triangles_plate, surface_current_density_plate, antennas_name)
fig.show()