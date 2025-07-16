import numpy as np

import plotly.figure_factory as ff
import plotly.graph_objects as go


def calculate_current_density(current, triangles, edges, vecteurs_rho):
    """
        Calcule la densité de courant surfacique pour chaque triangle d'un maillage.

        Paramètres :
            * current : n-d-array, vecteur des courants calculés pour chaque arête (A).
            * triangles : objet contenant des informations sur les triangles du maillage :
                % total_of_triangles : int, nombre total de triangles.
                % triangles_plus : n-d-array, indices des triangles associés au côté "plus" des arêtes.
                % triangles_minus : n-d-array, indices des triangles associés au côté "moins" des arêtes.
                % triangles_area : n-d-array, aires des triangles (m²).
            * edges : objet contenant des informations sur les arêtes :
                % total_number_of_edges : int, nombre total d'arêtes.
                % edges_length : n-d-array, longueurs des arêtes (m).
            * vecteurs_rho : objet contenant les vecteurs ρ associés aux triangles et aux arêtes :
                % vecteur_rho_plus : n-d-array, vecteurs ρ pour le côté "plus" des arêtes.
                % vecteur_rho_minus : n-d-array, vecteurs ρ pour le côté "moins" des arêtes.

        Comportement :
        1. Initialise un tableau `surface_current_density` pour stocker la norme de la densité de courant pour chaque triangle.
        2. Parcourt chaque triangle du maillage.
        3. Pour chaque triangle, accumule la contribution des courants sur les arêtes associées à ce triangle :
            % Multiplie le courant par la longueur de l'arête pour obtenir une contribution pondérée.
            % Ajoute cette contribution en fonction de l'association de l'arête au triangle (côté "plus" ou "moins").
        4. Normalise cette contribution par l'aire du triangle correspondant.
        5. Calcule la norme de la densité de courant pour ce triangle.
        6. Détermine la valeur maximale de la densité de courant sur tous les triangles.
        7. Affiche la densité de courant maximale en ampères par mètre (A/m).

        Retour :
        surface_current_density : n-d-array, normes de la densité de courant pour chaque triangle (A/m).

        Exemple :
        Pour un maillage donné, cette fonction permet d'analyser la répartition du courant sur la surface des triangles.

        Note :
        La densité de courant surfacique est une mesure de l'intensité de courant par unité de surface, utile pour étudier les antennes ou les surfaces conductrices.
    """

    # Initialisation du tableau pour stocker la densité de courant surfacique
    surface_current_density_abs_norm = np.zeros(triangles.total_of_triangles)  # Norme du courant pour chaque triangle
    surface_current_density_norm = np.zeros(triangles.total_of_triangles)
    surface_current_density_vector = np.zeros((3, triangles.total_of_triangles), dtype=complex)

    # Parcours de chaque triangle pour calculer la densité de courant
    for triangle in range(triangles.total_of_triangles):
        current_density_for_triangle = np.array([0.0, 0.0, 0.0], dtype=complex)  # Initialisation en complexe  # Initialisation du vecteur densité de courant pour ce triangle
        for edge in range(edges.total_number_of_edges):
            current_times_edge = current[edge] * edges.edges_length[edge]   # I(m) * EdgeLength(m)

            # Contribution si l'arête est associée au triangle côté "plus"
            if triangles.triangles_plus[edge] == triangle:
                current_density_for_triangle += current_times_edge * vecteurs_rho.vecteur_rho_plus[:, edge] / (2 * triangles.triangles_area[triangles.triangles_plus[edge]])

            # Contribution si l'arête est associée au triangle côté "moins"
            elif triangles.triangles_minus[edge] == triangle:
                current_density_for_triangle += current_times_edge * vecteurs_rho.vecteur_rho_minus[:, edge] / (2 * triangles.triangles_area[triangles.triangles_minus[edge]])

        # Calcul de la norme de la densité de courant pour ce triangle
        # surface_current_density[triangle] = np.abs(np.linalg.norm(current_density_for_triangle))
        surface_current_density_abs_norm[triangle] = np.abs(np.linalg.norm(current_density_for_triangle))  # abs(norm(i))
        surface_current_density_norm[triangle] = np.linalg.norm(np.abs(current_density_for_triangle))  # norm(abs(i))
        surface_current_density_vector[:, triangle] = current_density_for_triangle  # Stockage du vecteur courant

    # Densité de courant maximale
    j_max_surface_current_abs_norm = max(surface_current_density_abs_norm)
    # print(f"Max Current value = {j_max_surface_current_abs_norm} [A/m]")

    # Trouver la valeur maximale et son indice
    j_max_index = np.argmax(surface_current_density_norm)  # Renvoie l'indice du max

    # Récupérer le vecteur correspondant
    """
    surface_currentMax = surface_current_density_vector[:, j_max_index]
    print(f"CurrentMax in complex form = {surface_currentMax[0] :4f} [A/m]")
    print(f"                             {surface_currentMax[1] :4f} [A/m]")
    print(f"                             {surface_currentMax[2] :4f} [A/m]")
    """

    return surface_current_density_abs_norm


def compute_aspect_ratios(points):
    """
        Calcule les rapports d'échelle pour l'affichage 3D des données.

        Paramètres :
        points : tuple ou n-d-array, coordonnées des points dans l'espace 3D sous forme (x, y, z).

        Retour :
        dict : Dictionnaire contenant les rapports d'aspect normalisés pour les axes 'x', 'y', et 'z'.

        Fonctionnement :
            1. Extrait les coordonnées x, y et z des points.
            2. Calcule la plage (max — min) pour chaque axe.
            3. Détermine l'échelle globale comme étant la plus grande plage parmi les trois axes.
            4. Normalise chaque plage par l'échelle générale pour obtenir les rapports d'aspect.

        Exemple :
        Si les données couvrent différentes échelles sur les axes, cette fonction ajuste les proportions
        pour une visualisation 3D cohérente.
    """

    # Extraction des coordonnées x, y et z à partir de points
    x_, y_, z_ = points

    # Calcul de l'échelle globale (figure scale) en prenant la plus grande différence entre les axes
    fig_scale = max(max(x_) - min(x_), max(y_) - min(y_), max(z_) - min(z_))

    # Calcul des rapports d'échelle pour chaque axe par rapport à l'échelle globale
    return {
        "x": (max(x_) - min(x_)) / fig_scale,
        "y": (max(y_) - min(y_)) / fig_scale,
        "z": 0.3,
    }

def visualize_surface_current(points_data, triangles_data, surface_current_density, feed_point, title="Antennas Surface Current"):
    """
        Visualise la densité de courant surfacique sur une surface triangulée en 3D à l'aide de Plotly.

        Paramètres :
        * points_data : objet contenant les coordonnées des points de la surface, sous forme de tableau 2D (3, n_points),
                        où les lignes correspondent aux coordonnées X, Y et Z.
        * triangles_data : objet contenant les indices des sommets des triangles de la surface, sous forme de tableau 2D (3, n_triangles),
                           où chaque colonne correspond à un triangle défini par trois indices de sommets.
        * surface_current_density : n-d-array, densité de courant surfacique normalisée ou brute associée à chaque triangle.
        * title : str, titre de la visualisation (par défaut "Antennas Surface Current").

        Retour :
        fig : objet Plotly représentant la figure 3D.

        Fonctionnement :
            1. Extrait les coordonnées X, Y, Z des points à partir de 'points_data'.
            2. Prépare les indices des triangles à partir de `triangles_data` pour la compatibilité avec Plotly.
            3. Calcule les rapports d'aspect pour un rendu visuel cohérent à l'aide de 'compute_aspect_ratios'.
            4. Crée une figure de type "trisurf" avec Plotly, colorée selon la densité de courant.
            5. Affiche une barre de couleur pour indiquer les niveaux de densité de courant.
            6. Retourne l'objet figure pour affichage ou sauvegarde.

        Exemple d'application :
        Cette fonction permet de visualiser la répartition de la densité de courant sur une surface triangulée,
        utile pour l'analyse de modèles d'antennes ou de conducteurs.

        Notes :
            * La densité de courant surfacique (surface_current_density) doit être une valeur par triangle, correspondant
              à 'triangles_data'.
            * Assurez-vous que la bibliothèque `plotly` est installée et que `ff.create_trisurf` est disponible.
    """
    # Extraction des coordonnées des sommets
    x_, y_, z_ = points_data.points  # Coordonnées X, Y, Z des points

    # Création des simplices pour plotly (les indices des sommets de chaque triangle)
    simplices = triangles_data.triangles[:3, :].T  # Transpose pour passer de [3, n_triangles] à [n_triangles, 3]

    # Visualisation avec plotly
    aspect_ratios = compute_aspect_ratios(points_data.points)

    # Création de la figure avec trisurf
    fig = ff.create_trisurf(
        x=x_,
        y=y_,
        z=z_,
        simplices=simplices,
        colormap="Rainbow",
        color_func=surface_current_density,  # Utilisation de la densité de courant pour colorer
        show_colorbar=True,
        # title=title,
        title='',
        aspectratio=aspect_ratios,
    )
    # Ajout du/des feed-point(s) en rouge et en évidence
    feed_point = np.atleast_2d(feed_point)  # S'assure que feed_point est de forme (n, 3)
    fig.add_trace(go.Scatter3d(
        x=feed_point[:, 0],
        y=feed_point[:, 1],
        z=feed_point[:, 2],
        mode='markers+text',
        marker=dict(size=6, color='red', symbol='circle'),
        name='Feed Point(s)'
    ))
    # Configuration de la légende
    fig.update_layout(
        scene=dict(
            camera=dict(
                eye=dict(x=0.65, y=0.65, z=0.65)  # Valeurs plus grandes = zoom arrière
            )
        ),
        legend=dict(
            x=0.2,  # Position horizontale (0=left, 1=right)
            y=0.9,  # Position verticale (0=bottom, 1=top)
            xanchor='left',  # Ancrage horizontal ('auto', 'left', 'center', 'right')
            yanchor='top',   # Ancrage vertical ('auto', 'top', 'middle', 'bottom')
            bgcolor='rgba(255,255,255,0.7)',  # Fond semi-transparent
            bordercolor='lightgray',
            borderwidth=1
        )
    )

    return fig

def calculate_seuil_surface_current_density(surface_current_density):

    # print(f"\nNombre de triangles = {surface_current_density.shape}\n")

    # Calculer la valeur maximale
    max_value = np.max(surface_current_density)
    seuil = 0.7 * max_value
    
    # Identifier les indices des éléments inférieurs au seuil
    indices_below_seuil = np.where(surface_current_density < seuil)[0]
    
    # Afficher les résultats
    '''print("Seuil choisi basé sur la médiane et l'écart-type:", seuil)
    print("Valeur maximale de la densité de courant de surface:", max_value)'''
    # print(f"\nNombre d'éléments inférieurs au seuil: {len(indices_below_seuil)}")
    '''print("Indices des éléments inférieurs au seuil:", indices_below_seuil)'''

    return indices_below_seuil