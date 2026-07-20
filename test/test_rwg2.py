from test_setup import setup_project_root
setup_project_root()

from rwg.rwg1 import *
from rwg.rwg2 import *

# Chargement des données sauvegardées avec le fichier precedent
filename_plate_to_load = 'data/antennas_mesh1/plate_mesh1.mat'
points_plate, triangles_plate, edges_plate = DataManager_rwg1.load_data(filename_plate_to_load)
print(f"Taille tableau point = {points_plate.points.shape}")
print(f"On a {points_plate.total_of_points} points au total")
print("point = ")
print(points_plate.points)
print(f"Taille tableau triangles = {triangles_plate.triangles.shape}")
print(f"On a {triangles_plate.total_of_triangles} triangle au total")
print("triangle =")
print(triangles_plate.triangles)
triangles_plate.filter_triangles()
print(f"Taille tableau Aire triangle = {triangles_plate.triangles_area.shape}")
print("Aire des triangles =")
print(triangles_plate.triangles_area)
print(f"Taille tableau centre triangle = {triangles_plate.triangles_center.shape}")
print("Centre des triangles =")
print(triangles_plate.triangles_center)
print("triangle Plus =")
print(triangles_plate.triangles_plus)
print("triangle Minus =")
print(triangles_plate.triangles_minus)
print("Edge first point list =")
print(edges_plate.first_points)
print("Edge second point list =")
print(edges_plate.second_points)
print(f"Total number of edge = {edges_plate.total_number_of_edges}")
edges_plate.compute_edges_length(points_plate)
print("Edge length list =")
print(edges_plate.edges_length)
# Travail sur les vecteurs RHO
vecteurs_rho_plate = Vecteurs_Rho()
vecteurs_rho_plate.calculate_vecteurs_rho(points_plate, triangles_plate, edges_plate)
print(f"Taille du vecteur RHO plus {vecteurs_rho_plate.vecteur_rho_plus.shape}")
print("vecteur RHO plus =")
print(vecteurs_rho_plate.vecteur_rho_plus)
print(f"Taille du vecteur RHO Minus {vecteurs_rho_plate.vecteur_rho_minus.shape}")
print("vecteur RHO Minus =")
print(vecteurs_rho_plate.vecteur_rho_minus)
# Sauvegarde des données
save_folder_name = 'data/antennas_mesh2/'
DataManager_rwg2.save_data(filename_plate_to_load, save_folder_name, vecteurs_rho_plate)