import os
import numpy as np
import plotly.figure_factory as ff
import scipy.io as sio


def load_mesh_file(filename):
    """Charge le fichier .mat et retourne les points et triangles du maillage."""
    try:
        mesh = sio.loadmat(filename)
    except FileNotFoundError:
        raise FileNotFoundError(f"Le fichier '{filename}' est introuvable.")
    except Exception as error:
        raise RuntimeError(f"Erreur lors du chargement du fichier : {error}")

    # Validation des variables requises
    if 'p' not in mesh or 't' not in mesh:
        raise ValueError("Le fichier doit contenir les variables 'p' (points) et 't' (triangles).")

    points = mesh['p']  # Coordonnées des points du maillage (3 x N) → Chaque point sur le maillage a une coordonnée
    triangles = mesh['t']  # Indices des triangles (4 x M) → Il y a M triangles et le vecteur triangle[:, i] est le i-eme triangle et triangle[0, i] correspond au premier sommet de ce i-eme triangle et ainsi de suite
    return points, triangles


def filter_triangles(triangles):
    """Filtre les triangles dont la quatrième ligne est > 1."""
    valid_indices = np.where(triangles[3, :] <= 1)[0]
    return triangles[:, valid_indices].astype(int)  # Conversion explicite pour éviter les erreurs


def compute_aspect_ratios(points):
    """Calcule les rapports d'échelle pour l'affichage 3D."""
    x_, y_, z_ = points
    fig_scale = max(max(x_) - min(x_), max(y_) - min(y_), max(z_) - min(z_))
    return {
        "x": (max(x_) - min(x_)) / fig_scale,
        "y": (max(y_) - min(y_)) / fig_scale,
        "z": (max(z_) - min(z_)) / fig_scale,
    }


def calculate_mesh_dimension(points):
    point_x_min = min(points[0])
    point_x_max = max(points[0])

    point_y_min = min(points[1])
    point_y_max = max(points[1])

    point_z_min = min(points[2])
    point_z_max = max(points[2])

    longueur_mesh_x = point_x_max - point_x_min
    largeur_mesh_y = point_y_max - point_y_min
    hauteur_mesh_y = point_z_max - point_z_min

    return longueur_mesh_x, largeur_mesh_y, hauteur_mesh_y


def create_figure(points, triangles, create_from_matlab, title="Antennas Mesh"):
    """Crée une figure 3D Plotly à partir des points et triangles."""
    x_, y_, z_ = points
    if create_from_matlab:
        simplices = triangles[:3, :].T - 1
    else:
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


def viewer(filename, apply_filter_triangles=True, create_from_matlab=True):

    """Charge, filtre et visualise un fichier de maillage."""
    print(f"Chargement du fichier : {filename}")
    points, triangles = load_mesh_file(filename)

    print(f"Points shape: {points.shape}")
    print(f"Triangles shape: {triangles.shape}")

    if apply_filter_triangles:
        # Filtrer les triangles invalides --------- A commenter. N'est pas tres utile
        triangles = filter_triangles(triangles)
        print(f"Filtered Triangles shape: {triangles.shape}")
    else:
        print("Aucun triangle n'as ete filtrer")

    # Calcul des dimensions du mesh
    longueur, largeur, hauteur = calculate_mesh_dimension(points)
    print(f"Votre mesh a une dimension de {longueur} * {largeur} * {hauteur} metre")     # Nous déduisons que le mesh doit avoir une dimension en metre
    print(f"Longueur suivant l'axe x = {longueur} metre\nlargeur suivant l'axe y = {largeur} metre\nHauteur suivant l'axe z = {hauteur} metre")

    # Créer et afficher la figure
    antennas_file_name = os.path.splitext(os.path.basename(filename))[0]  + ' antenna mesh'
    fig = create_figure(points, triangles, create_from_matlab, antennas_file_name)
    fig.show()