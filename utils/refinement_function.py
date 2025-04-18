import numpy as np

def compute_size_from_current(nodes, triangles, current_values, mesh_size, feed_point, mesh_dividend, r_threshold=0.1):
    print("\nshapes nodes =", nodes.shape)
    print("shapes triangles =", triangles.shape)
    pts = nodes[triangles]  # forme (T, 3, 3) -> T triangles, 3 sommets, 3 coords

    # Calculer le centre de chaque triangle : moyenne des trois sommets
    centers = pts.mean(axis=1)  # forme (T, 3)

    distances = np.hypot(centers[:, 0] - feed_point[0], centers[:, 1] - feed_point[1], centers[:, 2] - feed_point[2])
    print("\nshapes distances =", distances.shape)
    print("shapes current =", current_values.shape)
    # Appliquer la condition du rayon `r_threshold` : les points dans ce rayon gardent la taille de maillage par défaut
    size_field = np.full_like(current_values, mesh_size)

    # print("Taille de size_field", size_field.shape)
    
    # Pour les points à l'extérieur du rayon, on applique la logique de taille en fonction du champ de courant
    outside_threshold = distances > r_threshold

    new_mesh_field = size_field[outside_threshold]

    count = 0
    for current in current_values[outside_threshold]:
        if current > 0.7 * np.max(current_values):
            new_mesh_field[count] = size_field[count] / mesh_dividend
        count += 1

    size_field[outside_threshold] = new_mesh_field

    '''print("Taille de size_field", size_field.shape)'''

    print("size_field = ", size_field)
    
    # Retourner le champ de taille
    return size_field