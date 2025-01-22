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

points = np.column_stack((coordonnees_x, coordonnees_y))

print("points shape =", points.shape)

# Triangulation de Delaunay; Peut etre replacer eventuellement ici par la fonction triangle
triangulation = Delaunay(points)
print("triangulation simplices shape =", triangulation.simplices.shape)
t = np.zeros((4, triangulation.simplices.shape[0]), dtype=int)
t[:3, :] = triangulation.simplices.T
print("t shape =", t.shape)
p = np.zeros((3, points.shape[0]))
p[:2, :] = points.T
print("p shape =", p.shape)
"""fig = create_figure(p, t, "pate ground")
fig.show()"""

filename = "plate_ground"
save_folder_name = 'data/antennas_mesh/'
data_save(filename, save_folder_name, p, t)

# Affichage de la grille 2D pour sélectionner les points d’alimentation
plt.figure()
plt.triplot(coordonnees_x, coordonnees_y, triangulation.simplices, color="gray")
plt.scatter(coordonnees_x, coordonnees_y, color="red", marker="o", label="Points du maillage")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Cliquez sur deux points d’alimentation, puis appuyez sur ENTRÉE")
plt.legend()
plt.show()

# Sélection des points avec la souris
FeedingTriangle = []
while len(FeedingTriangle) < 2:
    points_selected = plt.ginput(1, timeout=0)
    if not points_selected:
        break  # Si aucun point sélectionné, arrêter
    xi, yi = points_selected[0]
    TriangleNumber = triangulation.find_simplex([xi, yi])
    
    if TriangleNumber < 0:
        print("Point hors du maillage, veuillez réessayer.")
        continue
    
    FeedingTriangle.append(TriangleNumber)

# Création du monopôle
for n in range(len(FeedingTriangle) // 2):
    FT = [FeedingTriangle[2 * n], FeedingTriangle[2 * n + 1]]
    N, M = t[:3, FT[0]], t[:3, FT[1]]
    
    # Trouver le bord d'alimentation
    a = np.isin(N, M)
    Edge_B = M[a]

    # Créer les points du haut du monopôle
    p = np.hstack((p, p[:, Edge_B[0]].reshape(3, 1) + [[0], [0], [h]]))
    p = np.hstack((p, p[:, Edge_B[1]].reshape(3, 1) + [[0], [0], [h]]))
    
    Edge_T = [p.shape[1] - 2, p.shape[1] - 1]
    
    # Construire les segments intermédiaires
    Edge_MM = Edge_B
    for k in range(1, Number):
        new_point_1 = k / Number * (p[:, Edge_T[0]] - p[:, Edge_B[0]]) + p[:, Edge_B[0]]
        new_point_2 = k / Number * (p[:, Edge_T[1]] - p[:, Edge_B[1]]) + p[:, Edge_B[1]]
        p = np.hstack((p, new_point_1.reshape(3, 1), new_point_2.reshape(3, 1)))

        Edge_M = [p.shape[1] - 2, p.shape[1] - 1]
        t = np.hstack((t, np.array([[Edge_MM[0]], [Edge_MM[1]], [Edge_M[1]], [1]])))
        t = np.hstack((t, np.array([[Edge_MM[0]], [Edge_M[0]], [Edge_M[1]], [1]])))
        
        Edge_MM = Edge_M

    # Dernière couche du monopôle
    t = np.hstack((t, np.array([[Edge_M[0]], [Edge_M[1]], [Edge_T[1]], [1]])))
    t = np.hstack((t, np.array([[Edge_M[0]], [Edge_T[0]], [Edge_T[1]], [1]])))

# Affichage final du monopôle
fig = create_figure(p, t, "Monopole Antenna")
fig.show()

# Affichage pour vérifier la triangulation
"""
plt.triplot(points[:, 0], points[:, 1], triangulation.simplices, color='gray')
plt.scatter(points[:, 0], points[:, 1], marker='o', color='red')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Triangulation de Delaunay")
plt.grid(True)
plt.show()
"""

"""
# Sélection des triangles d'alimentation avec clics de souris
feeding_triangles = []
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.triplot(points[:, 0], points[:, 1], triangulation.simplices, color='gray')

def on_click(event):
    if event.inaxes != ax:
        return

    xi, yi = event.xdata, event.ydata
    simplex = triangulation.find_simplex([[xi, yi]])
    
    if simplex >= 0:  # Si le point appartient à un triangle
        feeding_triangles.append(simplex)
        simplex_points = points[triangulation.simplices[simplex]]
        ax.fill(simplex_points[:, 0], simplex_points[:, 1], 'w', edgecolor='r')

    plt.draw()

fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()

# Créer le monopôle (ajouter les segments verticaux)
p = points.copy()  # Copie des points initiaux

for n in range(len(feeding_triangles) // 2):
    simplex_idx = feeding_triangles[2 * n: 2 * n + 2]
    
    for simplex in simplex_idx:
        simplex_points = points[triangulation.simplices[simplex]]
        Edge_B = simplex_points[:2]  # Prendre les deux premiers points

        # Création des arêtes supérieures
        Edge_T = Edge_B + np.array([[0, 0, h], [0, 0, h]])
        p = np.vstack([p, Edge_T])

        for k in range(1, Number):
            new_edge = k / Number * (Edge_T - Edge_B) + Edge_B
            p = np.vstack([p, new_edge])

# Sauvegarde des données
np.save('monopole.npy', p)

# Visualisation du monopôle en 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(p[:, 0], p[:, 1], np.zeros_like(p[:, 0]), c='b', marker='o')  # Base
ax.scatter(p[:, 0], p[:, 1], p[:, 2], c='r', marker='^')  # Partie élevée

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
"""