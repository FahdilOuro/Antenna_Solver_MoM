import json
import os
import gmsh
import numpy as np

from utils.gmsh_function import *
from utils.refinement_function import *


def plate_gmsh(longueur, hauteur, mesh_name, feed_point, length_feed_edge, angle, save_mesh_folder, high_current_points_list=np.array([3, 0]), iteration=0):
    # Initialisation de Gmsh
    gmsh.initialize()
    gmsh.model.add(os.path.splitext(os.path.basename(mesh_name))[0])

    filename = create_pos_file(mesh_name)

    # Vérifier si le dossier existe, sinon le créer
    if not os.path.exists(save_mesh_folder):
        os.makedirs(save_mesh_folder)

    # Utilisation d'Open Cascade pour créer un rectangle (x, y, z, largeur, hauteur)
    plate = gmsh.model.occ.addRectangle(0, 0, 0, longueur, hauteur)
    gmsh.model.occ.synchronize()

    # feed_edge(plate, feed_point, length_feed_edge, mesh_name, angle)

    output_path = finalize_antenna_model_gmsh(filename, high_current_points_list, save_mesh_folder, mesh_name, iteration)

    return output_path



    '''feed_edge(bowtie, feed_point, length_feed_edge, mesh_name, angle)

    # Définir le chemin du fichier JSON
    json_path = f'data/json/feed_edge_info_{os.path.splitext(mesh_name)[0]}.json'

    # Charger l'information sur l'arête insérée
    with open(json_path, "r") as f:
        data = json.load(f)

    # Extraire les coordonnées depuis le fichier JSON
    coordinates = data["coordinates"]
    
    # Convertir les coordonnées en tableau Numpy
    p_feed = np.array([[point["x"], point["y"], point["z"]] for point in coordinates]).T

    output_path = finalize_antenna_model_gmsh(filename, high_current_points_list, save_mesh_folder, mesh_name, iteration)

    return output_path'''

def strip_gmsh(Longueur, largeur, mesh_name, feed_point, length_feed_edge, angle, save_mesh_folder, high_current_points_list=np.array([3, 0]), iteration=0):
    # Initialisation de Gmsh
    gmsh.initialize()
    gmsh.model.add(os.path.splitext(os.path.basename(mesh_name))[0])

    filename = create_pos_file(mesh_name)

    # Utilisation d'Open Cascade pour créer un rectangle (x, y, z, largeur, hauteur)
    strip_antenna = gmsh.model.occ.addRectangle(-Longueur/2, -largeur/2, 0, Longueur, largeur)
    gmsh.model.occ.synchronize()

    gmsh.model.occ.synchronize()

    # feed_edge(strip_antenna, feed_point, length_feed_edge, mesh_name, angle)

    output_path = finalize_antenna_model_gmsh(filename, high_current_points_list, save_mesh_folder, mesh_name, iteration)

    return output_path