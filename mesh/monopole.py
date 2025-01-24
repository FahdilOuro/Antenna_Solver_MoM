import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

from scipy.spatial import Delaunay
from scipy.io import savemat

def compute_aspect_ratios(points):
    """Calcule les rapports d'échelle pour l'affichage 3D."""
    x_, y_, z_ = points
    fig_scale = max(max(x_) - min(x_), max(y_) - min(y_), max(z_) - min(z_))
    return {
        "x": (max(x_) - min(x_)) / fig_scale,
        "y": (max(y_) - min(y_)) / fig_scale,
        "z": (max(z_) - min(z_)) / fig_scale,
    }

def create_figure(points, triangles, title="Antennas Mesh"):
    """Crée une figure 3D Plotly à partir des points et triangles."""
    x_, y_, z_ = points
    simplices = triangles[:3, :].T

    aspect_ratios = compute_aspect_ratios(points)

    fig = ff.create_trisurf(
        x=x_,
        y=y_,
        z=z_,
        simplices=simplices,
        color_func=np.arange(len(simplices)),  # Couleurs des triangles
        show_colorbar=False,
        title=title,
        aspectratio=aspect_ratios,
    )

    return fig

def data_save(filename, save_folder_name, points, triangle):
    # Crée un dictionnaire contenant toutes les données à sauvegarder
    data = {
        'p' : points,
        't' : triangle
    }
    # Génère le nom du fichier de sauvegarde à partir du nom d'origine
    save_file_name = filename + '.mat'
    full_save_path = os.path.join(save_folder_name, save_file_name)  # Chemin complet pour la sauvegarde

    # Vérifie si le dossier existe, sinon crée le dossier
    if not os.path.exists(save_folder_name): # Vérification et création du dossier si nécessaire
        os.makedirs(save_folder_name)
        print(f"Directory '{save_folder_name}' created.")

    # Sauvegarde les données dans le fichier MAT
    savemat(full_save_path, data)
    print(f"Data saved successfully to {full_save_path}")

    # Retourne le nom du fichier sauvegardé
    return save_file_name

# Paramètres
L = 2.0       # Longueur de la plaque
W = 2.0       # Largeur de la plaque
Nx = 11       # Discrétisation en x
Ny = 11       # Discrétisation en y
h = 1.0       # Hauteur du monopôle
Number = 7    # Nombre de rectangles du monopôle

# Créer les sommets du plan de masse
epsilon = 1e-6
coordonnees_x = []
coordonnees_y = []

for i in range(Nx + 1):
    for j in range(Ny + 1): 
        x_val = -L/2 + (i / Nx) * L
        y_val = -W/2 + (j / Ny) * W - (epsilon * x_val)
        coordonnees_x.append(x_val)
        coordonnees_y.append(y_val)

# Convertir en numpy array
coordonnees_x = np.array(coordonnees_x)
coordonnees_y = np.array(coordonnees_y)

# Ajouter les points d'alimentation
x_feed = np.array([-0.02, 0.02])
y_feed = np.array([0, 0])
coordonnees_x = np.append(coordonnees_x, x_feed)
coordonnees_y = np.append(coordonnees_y, y_feed)

# Calcul et ajout des nouveaux points
C = np.mean(x_feed)
x1 = np.array([C, C])
y1 = np.mean(y_feed) + 2 * np.array([np.max(x_feed) - C, np.min(x_feed) - C])
coordonnees_x = np.append(coordonnees_x, x1)
coordonnees_y = np.append(coordonnees_y, y1)

print("longueur de la taille coordonnees_x =", coordonnees_x.shape)
print("longueur de la taille coordonnees_y =", coordonnees_y.shape)

points_base = np.column_stack((coordonnees_x, coordonnees_y))

print("points shape =", points_base.shape)

# Triangulation de Delaunay; Peut etre replacer eventuellement ici par la fonction triangle
triangulation = Delaunay(points_base)
print("triangulation simplices shape =", triangulation.simplices.shape)
t = np.zeros((4, triangulation.simplices.shape[0]), dtype=int)
t[:3, :] = triangulation.simplices.T
print("t shape =", t.shape)
p = np.zeros((3, points_base.shape[0]))
p[:2, :] = points_base.T
print("p shape =", p.shape)
fig = create_figure(p, t, "pate ground")
fig.show()
"""
filename = "plate_ground"
save_folder_name = 'data/antennas_mesh/'
data_save(filename, save_folder_name, p, t)
"""

# Paramètres
h = 1.0       # Hauteur du monopôle
W_h = 0.04     # Largeur du monopole
Nx_h = 7       # Discrétisation en x
Ny_h = 1       # Discrétisation en y

# Créer les sommets du plan de masse
epsilon = 1e-6
coordonnees_x = []
coordonnees_y = []

for i in range(Nx_h + 1):
    for j in range(Ny_h + 1):
        x_val = 0 + (i / Nx_h) * h
        y_val = -W_h/2 + (j / Ny_h) * W_h - (epsilon * x_val)
        coordonnees_x.append(x_val)
        coordonnees_y.append(y_val)

# Convertir en numpy array
coordonnees_x = np.array(coordonnees_x)
coordonnees_y = np.array(coordonnees_y)

points_base_monopole = np.column_stack((coordonnees_x, coordonnees_y))

print("points shape =", points_base_monopole.shape)

# Triangulation de Delaunay; Peut etre replacer eventuellement ici par la fonction triangle
triangulation_monopole = Delaunay(points_base_monopole)
print("triangulation simplices shape =", triangulation_monopole.simplices.shape)
t_monopole = np.zeros((4, triangulation_monopole.simplices.shape[0]), dtype=int)
t_monopole[:3, :] = triangulation_monopole.simplices.T
print("t shape =", t_monopole.shape)
p_monopole = np.zeros((3, points_base_monopole.shape[0]))
p_monopole[:2, :] = points_base_monopole.T
print("p shape =", p_monopole.shape)
fig = create_figure(p_monopole, t_monopole, "Monopole")
fig.show()

