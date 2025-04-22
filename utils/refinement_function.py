import numpy as np

def compute_size_from_current(nodes, triangles, current_values, mesh_size, feed_point, mesh_dividend, r_threshold):
    print("\nshapes nodes =", nodes.shape)
    print("shapes triangles =", triangles.shape)
    pts = nodes[triangles]  # forme (T, 3, 3) -> T triangles, 3 sommets, 3 coords

    # Calculer le centre de chaque triangle : moyenne des trois sommets
    centers = pts.mean(axis=1)  # forme (T, 3)

    distances = np.hypot(centers[:, 0] - feed_point[0], centers[:, 1] - feed_point[1], centers[:, 2] - feed_point[2])
    print("\nshapes distances =", distances.shape)
    print("shapes current =", current_values.shape)

    # 1. Crée un champ de taille uniforme
    size_field = np.full_like(current_values, mesh_size)

    # 2. Masque : éléments en dehors du seuil de distance
    outside_threshold = distances > r_threshold

    # 3. Masque combiné : à la fois en dehors du seuil ET courant fort
    high_current_mask = (current_values > 0.7 * np.max(current_values)) & outside_threshold

    # 4. Réduire la taille de maillage là où le courant est fort ET loin
    size_field[high_current_mask] /= mesh_dividend

    print("Taille de size_field", size_field.shape)

    print("size_field = ", size_field)
    
    # Retourner le champ de taille
    return size_field