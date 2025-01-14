import numpy as np
import triangle as tr

def is_in_refinement_zone(point, center, radius):
    return np.linalg.norm(point - center) < radius

refinement_center = np.array([0.5, 0.5])
refinement_radius = 0.5

# Fonction pour estimer l'erreur basée sur la position des triangles
def estimate_error(triangles, vertices, center, radius):
    errors = np.zeros(len(triangles.T))
    for i, tri in enumerate(triangles.T):
        triangle_points = vertices[:, tri].T
        if any(is_in_refinement_zone(point, center, radius) for point in triangle_points):
            errors[i] = 1  # Marquer les triangles dans la zone de raffinement
    return errors

# Fonction pour raffiner le maillage
def refine_mesh(antenna_data, errors, threshold):
    refine_indices = np.where(errors > threshold)[0]
    for idx in refine_indices:
        triangle = antenna_data['triangles'][idx]
        midpoints = (vertices[triangle] + vertices[np.roll(triangle, -1)]) / 2
        vertices = np.vstack([vertices, midpoints])
    antenna_data['vertices'] = vertices
    return tr.triangulate(antenna_data, 'pq30a0.0001Dj')

# Boucle adaptative
threshold = 0.5
for _ in range(5):  # Nombre d'itérations de raffinement
    mesh = tr.triangulate(antenna_data, 'pq30a0.000001Yj')
    points = np.array(mesh['vertices']).T  # Convertir en numpy array pour faciliter le traitement
    triangles = np.array(mesh['triangles']).T
    errors = estimate_error(triangles, points, refinement_center, refinement_radius)
    antenna_data = refine_mesh(antenna_data, errors, threshold)