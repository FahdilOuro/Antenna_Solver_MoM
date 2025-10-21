from test_setup import setup_project_root
setup_project_root()

from rwg.rwg2 import *
from rwg.rwg3 import *

filename_plate_to_load = 'data/antennas_mesh2/plate_mesh2.mat'
points_plate, triangles_plate, edges_plate, barycentric_triangles_plate, vecteurs_rho_plate = DataManager_rwg2.load_data(filename_plate_to_load)

frequency = 3e8 # La fréquence du champ électromagnétique (300 MHz)
omega, mu, epsilon, light_speed_c, eta, matrice_z_plate = calculate_z_matrice(triangles_plate, edges_plate, barycentric_triangles_plate, vecteurs_rho_plate, frequency)

save_folder_name = 'data/antennas_impedance/'
DataManager_rwg3.save_data(filename_plate_to_load, save_folder_name, frequency, omega, mu, epsilon, light_speed_c, eta, matrice_z_plate)