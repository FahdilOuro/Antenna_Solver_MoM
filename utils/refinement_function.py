import numpy as np

def compute_size_from_current(nodes, triangles, current_values, mesh_size, feed_point, mesh_dividend, r_threshold):
    print("\nshapes nodes =", nodes.shape)
    print("shapes triangles =", triangles.shape)
    pts = nodes[triangles]  # shape (T, 3, 3) -> T triangles, 3 vertices, 3 coords

    # Compute the center of each triangle: average of the three vertices
    centers = pts.mean(axis=1)  # shape (T, 3)

    distances = np.hypot(centers[:, 0] - feed_point[0], centers[:, 1] - feed_point[1], centers[:, 2] - feed_point[2])
    print("\nshapes distances =", distances.shape)
    print("shapes current =", current_values.shape)

    # 1. Create a uniform size field
    size_field = np.full_like(current_values, mesh_size)

    # 2. Mask: elements outside the distance threshold
    outside_threshold = distances > r_threshold

    # 3. Combined mask: both outside threshold AND high current
    high_current_mask = (current_values > 0.5 * np.max(current_values)) & outside_threshold

    # 4. Reduce mesh size where current is high AND far away
    size_field[high_current_mask] /= mesh_dividend

    print("Shape of size_field", size_field.shape)

    print("size_field = ", size_field)
    
    # Return the size field
    return size_field