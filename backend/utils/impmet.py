"""
    This module implements the calculation of the impedance matrix Z for a system of edges and triangles,
    based on the method of moments (MoM) in electromagnetics.

    Main functionalities:
        1. Construction of the Z matrix, representing impedance interactions between edges.
        2. Use of barycentric centers, rho vectors, and contributions from adjacent triangles.
        3. Calculation based on a function g_mn(r') and scalar coupling terms.

    Main inputs:
        * edges_data : Data on the edges, including total number and their lengths.
        * triangles_data : Data on the triangles, including their centers and indices of triangles adjacent to edges.
        * barycentric_triangles_data : Barycentric centers of the triangles.
        * vecteurs_rho_data : Rho vectors associated with edges and barycentric triangles.
        * parameter_k : Complex wavenumber of the medium.
        * factor_a : Weighting factor for vector contributions A_{mn}.
        * factor_fi : Weighting factor for scalar contributions Phi_{mn}.

    Output:
        matrice_z : Impedance matrix Z (complex), dimension [total number of edges, total number of edges].
"""
import numpy as np

def impedance_matrix_z(edges_data, triangles_data, barycentric_triangles_data, vecteurs_rho_data, parameter_k, factor_a, factor_fi):
    """
    Calculates the impedance matrix Z for the interactions between the edges of the mesh.

    Parameters:
        * edges_data : Object containing edge data (lengths, total number, etc.).
        * triangles_data : Object containing triangle data (centers, adjacent triangles, etc.).
        * barycentric_triangles_data : Barycentric centers of the triangles.
        * vecteurs_rho_data : Data on the rho vectors.
        * parameter_k : Complex wavenumber of the medium.
        * factor_a : Factor for vector contributions.
        * factor_fi : Factor for scalar contributions.

    Returns:
        matrice_z : Complex impedance matrix Z.
    """
    # Initialization of global variables and necessary data
    total_number_of_edges = edges_data.total_number_of_edges
    total_of_triangles = triangles_data.total_of_triangles
    triangles_plus = triangles_data.triangles_plus
    triangles_minus = triangles_data.triangles_minus
    triangles_center = triangles_data.triangles_center
    edges_length = edges_data.edges_length
    barycentric_triangle_center = barycentric_triangles_data.barycentric_triangle_center
    vecteur_rho_plus = vecteurs_rho_data.vecteur_rho_plus
    vecteur_rho_minus = vecteurs_rho_data.vecteur_rho_minus
    vecteur_rho_barycentric_plus = vecteurs_rho_data.vecteur_rho_barycentric_plus
    vecteur_rho_barycentric_minus = vecteurs_rho_data.vecteur_rho_barycentric_minus

    # Preparation of rho vectors for calculations
    vecteur_rho_plus_tiled = np.tile(vecteur_rho_plus[:, None, :], (1, 9, 1))    # Dimension [3, 9, total_number_of_edges]
    vecteur_rho_minus_tiled = np.tile(vecteur_rho_minus[:, None, :], (1, 9, 1))  # Dimension [3, 9, total_number_of_edges]

    # Initialization of the impedance matrix Z
    matrice_z = np.zeros((total_number_of_edges, total_number_of_edges), dtype=complex)   # Dimension [total_number_of_edges, total_number_of_edges]

    # Loop over triangles to calculate interactions
    for triangle in range(total_of_triangles):
        # Identification of the contributions of the plus and minus triangles
        positions_plus = triangles_plus == triangle
        positions_minus = triangles_minus == triangle

        # Calculation of the function g_min(r'); the index m corresponds to the triangle being processed by the program, and the indices n correspond to all triangles from 0 to the index total_of_triangles - 1.
        distances = barycentric_triangle_center - triangles_center[:, triangle][:, None, None]   # Dimension [3, 9, total_of_triangles]
        norm_of_distances = np.sqrt(np.sum(distances**2, axis=0, keepdims=True))                 # Dimension [1, 9, total_of_triangles]
        g_function = np.exp(-parameter_k * norm_of_distances) / norm_of_distances                # Dimension [1, 9, total_of_triangles]

        g_function_plus = g_function[:, :, triangles_plus]                                       # Dimension [1, 9, total_number_of_edges]
        g_function_minus = g_function[:, :, triangles_minus]                                     # Dimension [1, 9, total_number_of_edges]

        # Scalar contribution fi for Phi_mn
        fi = np.sum(g_function_plus, axis=1, keepdims=True) - np.sum(g_function_minus, axis=1, keepdims=True)      # Dimension [1, 1, total_number_of_edges]

        impedance_coupling_zf = factor_fi.reshape(-1, 1) * fi.squeeze().reshape(-1, 1)                             # Dimension [total_number_of_edges, 1]

        # Function to update Z based on rho and A_mn
        def update(the_position, vecteur_rho_barycentric_p_m, sign):
            vecteur_rho_barycentric = np.tile(vecteur_rho_barycentric_p_m[:, :, the_position][:, :, None], (1, 1, total_number_of_edges))   # Dimension [3, 9, total_number_of_edges]
            a_contribution = (np.sum(g_function_plus * np.sum(vecteur_rho_barycentric * vecteur_rho_plus_tiled, axis=0), axis=0)
                              +
                              np.sum(g_function_minus * np.sum(vecteur_rho_barycentric * vecteur_rho_minus_tiled, axis=0), axis=0))     # Dimension [9, total_number_of_edges]
            z1 = factor_a * a_contribution[:, None]
            z1_reshaped = z1.squeeze(axis=1).sum(axis=0)  # Remove unnecessary edges and reduce to match (total_number_of_edges)
            matrice_z[:, the_position] += edges_length[the_position] * (z1_reshaped + sign * impedance_coupling_zf.squeeze())

        # Calculating contributions for plus and minus triangles
        for position in np.where(positions_plus)[0]:
            update(position, vecteur_rho_barycentric_plus, +1)
        for position in np.where(positions_minus)[0]:
            update(position, vecteur_rho_barycentric_minus, -1)

    return matrice_z

def rwg_gram_matrix(edges_data, triangles_data, vecteurs_rho_data):
    """
    Assembles the global Gram matrix G using NumPy vectorization.
    
    This function pre-calculates triangle-edge connectivity and uses Einstein 
    summation to compute the 9-point quadrature integral efficiently.
    """
    num_edges = edges_data.total_number_of_edges
    num_tri = triangles_data.total_of_triangles
    G_global = np.zeros((num_edges, num_edges), dtype=np.float64)

    # --- Step 1: Pre-calculate connectivity (Triangle to Edges mapping) ---
    # This replaces the slow 'for e_idx in range(num_edges)' inside the triangle loop.
    tri_to_edges = [[] for _ in range(num_tri)]
    tri_edge_sides = [[] for _ in range(num_tri)]
    
    for e_idx in range(num_edges):
        tp = triangles_data.triangles_plus[e_idx]
        tm = triangles_data.triangles_minus[e_idx]
        if tp != -1:
            tri_to_edges[tp].append(e_idx)
            tri_edge_sides[tp].append('plus')
        if tm != -1:
            tri_to_edges[tm].append(e_idx)
            tri_edge_sides[tm].append('minus')

    print(f"Starting vectorized G calculation for {num_tri} triangles...")

    # --- Step 2: Main loop over triangles ---
    for tri_idx in range(num_tri):
        area = triangles_data.triangles_area[tri_idx]
        edges_in_tri = tri_to_edges[tri_idx]
        sides_in_tri = tri_edge_sides[tri_idx]
        n_local      = len(edges_in_tri)
        
        if not edges_in_tri:
            continue

        # Get lengths for the 3 edges of the current triangle
        l_local = edges_data.edges_length[edges_in_tri] # Shape: (3,)
        
        # --- Step 3: Vectorized Rho Extraction ---
        # We extract all rho vectors for the 3 edges at once.
        # Original shape is (3, 9, N_edges) -> We want (3_edges, 9_points, 3_coords)
        rho_local = np.zeros((n_local, 9, 3), dtype=np.float64)
        
        for i, (e_idx, side) in enumerate(zip(edges_in_tri, sides_in_tri)):
            if side == 'plus':
                # Slicing (3, 9, e_idx) and transposing to (9, 3)
                rho_local[i] = vecteurs_rho_data.vecteur_rho_barycentric_plus[:, :, e_idx].T
            else:
                rho_local[i] = vecteurs_rho_data.vecteur_rho_barycentric_minus[:, :, e_idx].T

        g_block = np.zeros((n_local, n_local), dtype=np.float64)
        
        for i in range(n_local):
            for j in range(i, n_local):          # j >= i  →  upper triangle + diagonal
                # Sum dot products over the 9 quadrature points
                # dot_ij = np.dot(rho_local[i].ravel(), rho_local[j].ravel()) # Equivalent to: 
                dot_ij = np.einsum('kd,kd->', rho_local[i], rho_local[j])

                val = (l_local[i] * l_local[j] / (36.0 * area)) * dot_ij

                g_block[i, j]  = val
                g_block[j, i]  = val   # symmetry: free copy, no recomputation

        # --- Step 5: Assemble into global matrix ---
        G_global[np.ix_(edges_in_tri, edges_in_tri)] += g_block

    return G_global