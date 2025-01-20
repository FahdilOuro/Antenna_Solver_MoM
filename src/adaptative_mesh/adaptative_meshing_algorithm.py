import triangle as tr
import numpy as np
import plotly.figure_factory as ff

# Définir les points (sommets)
vertices = np.array([
    [0, 0], [1, 0], [1, 1], [0, 1], [0.5, 1.5]
])

# Définir les segments (arêtes) par les indices des points
segments = np.array([
    [0, 1], [1, 2], [2, 3], [3, 0], [2, 4], [4, 3]
])

# Préparer les données pour la bibliothèque Triangle
antenna_data = {
    'vertices': vertices,
    'segments': segments
}

# Générer le maillage initial
mesh = tr.triangulate(antenna_data, 'pq30a0.000001Yj')

# Extraire les sommets et les triangles
points = np.array(mesh['vertices']).T  # Convertir en numpy array pour faciliter le traitement
triangles = np.array(mesh['triangles']).T

# Exemple de courants de surface (à remplacer par les valeurs réelles)
currents = np.random.rand(triangles.shape[1])

# Calculer le courant moyen
average_current = np.mean(currents)

# Fonction pour estimer l'erreur basée sur les courants de surface
def estimate_error(triangles, vertices, currents, average_current):
    errors = np.zeros(len(triangles.T))
    for i, tri in enumerate(triangles.T):
        # Calculer l'erreur pour les courants supérieurs au courant moyen
        current_value = currents[i]
        if current_value > average_current:
            errors[i] = current_value - average_current
    return errors

# Fonction pour raffiner le maillage
def refine_mesh(antenna_data, errors, threshold):
    refine_indices = np.where(errors > threshold)[0]
    for idx in refine_indices:
        triangle = antenna_data['triangles'][idx]
        midpoints = (vertices[triangle] + vertices[np.roll(triangle, -1)]) / 2
        vertices = np.vstack([vertices, midpoints])
    antenna_data['vertices'] = vertices
    return tr.triangulate(antenna_data, 'pq30a0.000001Yj')

# Boucle adaptative
threshold = 0.0  # Seuil d'erreur pour raffinement
for _ in range(5):  # Nombre d'itérations de raffinement
    mesh = tr.triangulate(antenna_data, 'pq30a0.000001Yj')
    points = np.array(mesh['vertices']).T  # Convertir en numpy array pour faciliter le traitement
    triangles = np.array(mesh['triangles']).T
    errors = estimate_error(triangles, points, currents, average_current)
    antenna_data = refine_mesh(antenna_data, errors, threshold)

# Affichage du maillage final
fig = ff.create_trisurf(x=points[0], y=points[1], simplices=triangles.T)
fig.show()