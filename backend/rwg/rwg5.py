import numpy as np

import plotly.figure_factory as ff
import plotly.graph_objects as go


def calculate_current_density(current, triangles, edges, vecteurs_rho):
    """
        Calculates the surface current density for each triangle in a mesh.

        Parameters:
            * current : n-d-array, vector of currents computed for each edge (A).
            * triangles : object containing triangle information of the mesh:
                % total_of_triangles : int, total number of triangles.
                % triangles_plus : n-d-array, indices of triangles associated with the "plus" side of edges.
                % triangles_minus : n-d-array, indices of triangles associated with the "minus" side of edges.
                % triangles_area : n-d-array, areas of triangles (m²).
            * edges : object containing edge information:
                % total_number_of_edges : int, total number of edges.
                % edges_length : n-d-array, edge lengths (m).
            * vecteurs_rho : object containing ρ vectors associated with triangles and edges:
                % vecteur_rho_plus : n-d-array, ρ vectors for the "plus" side of edges.
                % vecteur_rho_minus : n-d-array, ρ vectors for the "minus" side of edges.

        Behavior:
        1. Initializes an array `surface_current_density` to store the norm of the current density for each triangle.
        2. Iterates over each triangle in the mesh.
        3. For each triangle, accumulates contributions from currents on edges associated with the triangle:
            % Multiplies current by edge length to obtain a weighted contribution.
            % Adds this contribution depending on the edge's association with the triangle ("plus" or "minus").
        4. Normalizes this contribution by the area of the corresponding triangle.
        5. Computes the norm of the current density for this triangle.
        6. Determines the maximum current density over all triangles.
        7. Prints the maximum surface current density in amperes per meter (A/m).

        Returns:
        surface_current_density : n-d-array, norms of the current density for each triangle (A/m).

        Example:
        For a given mesh, this function allows analyzing the distribution of current over the surface of the triangles.

        Note:
        Surface current density is a measure of current intensity per unit area, useful for studying antennas or conductive surfaces.
    """

    # Initialize array to store surface current density
    surface_current_density_abs_norm = np.zeros(triangles.total_of_triangles)  # Norm of current for each triangle
    surface_current_density_norm = np.zeros(triangles.total_of_triangles)
    surface_current_density_vector = np.zeros((3, triangles.total_of_triangles), dtype=complex)

    # Iterate over each triangle to calculate current density
    for triangle in range(triangles.total_of_triangles):
        current_density_for_triangle = np.array([0.0, 0.0, 0.0], dtype=complex)  # Initialize vector for this triangle
        for edge in range(edges.total_number_of_edges):
            current_times_edge = current[edge] * edges.edges_length[edge]   # I(m) * EdgeLength(m)

            # Contribution if edge is associated with the triangle "plus" side
            if triangles.triangles_plus[edge] == triangle:
                current_density_for_triangle += current_times_edge * vecteurs_rho.vecteur_rho_plus[:, edge] / (2 * triangles.triangles_area[triangles.triangles_plus[edge]])

            # Contribution if edge is associated with the triangle "minus" side
            elif triangles.triangles_minus[edge] == triangle:
                current_density_for_triangle += current_times_edge * vecteurs_rho.vecteur_rho_minus[:, edge] / (2 * triangles.triangles_area[triangles.triangles_minus[edge]])

        # Compute the norm of the current density for this triangle
        surface_current_density_abs_norm[triangle] = np.abs(np.linalg.norm(current_density_for_triangle))  # abs(norm(i))
        surface_current_density_norm[triangle] = np.linalg.norm(np.abs(current_density_for_triangle))    # norm(abs(i))
        surface_current_density_vector[:, triangle] = current_density_for_triangle  # Store the current vector

    # Maximum surface current density
    j_max_surface_current_abs_norm = max(surface_current_density_abs_norm)
    # print(f"Maximum Surface Current Density : {j_max_surface_current_abs_norm} (A/m)")

    # Find the maximum value and its index
    j_max_index = np.argmax(surface_current_density_norm)

    return surface_current_density_abs_norm


def compute_aspect_ratios(points):
    """
        Computes aspect ratios for 3D visualization of data.

        Parameters:
        points : tuple or n-d-array, coordinates of points in 3D space as (x, y, z).

        Returns:
        dict : Dictionary containing normalized aspect ratios for 'x', 'y', and 'z' axes.

        Functionality:
            1. Extracts x, y, and z coordinates from points.
            2. Computes the range (max — min) for each axis.
            3. Determines the global scale as the largest range among the three axes.
            4. Normalizes each range by the global scale to obtain aspect ratios.

        Example:
        If the data spans different scales along the axes, this function adjusts the proportions
        for consistent 3D visualization.
    """

    # Extract x, y, z coordinates from points
    x_, y_, z_ = points

    # Compute global scale (figure scale) as the largest difference among axes
    fig_scale = max(max(x_) - min(x_), max(y_) - min(y_), max(z_) - min(z_))

    # Compute aspect ratios for each axis relative to global scale
    return {
        "x": (max(x_) - min(x_)) / fig_scale,
        "y": (max(y_) - min(y_)) / fig_scale,
        "z": 0.3,
    }

def visualize_surface_current(points_data, triangles_data, surface_current_density, feed_point=None, title="Antennas Surface Current"):
    """
    Visualizes the surface current density on a triangulated 3D surface using Plotly.

    This function renders a 3D mesh (trisurf) where each triangle's color represents 
    its surface current density. It automatically handles cases with uniform data 
    (to prevent Plotly errors) and adjusts the scene's aspect ratio for 
    accurate physical proportions.

    Parameters
    ----------
    points_data : object
        An object or dataclass containing the mesh vertices. 
        Must have a `.points` attribute with shape (3, N) or (N, 3).
    triangles_data : object
        An object or dataclass containing the connectivity matrix. 
        Must have a `.triangles` attribute where the first 3 rows contain 
        the vertex indices for each triangle.
    surface_current_density : array-like
        A 1D array of shape (n_triangles,) containing the magnitude of the 
        surface current density for each face.
    feed_point : array-like, optional
        Coordinates of the excitation point(s). Can be a single point (3,) 
        or multiple points (N, 3). If provided, they are rendered as red markers.
    title : str, optional
        The title of the generated Plotly figure. Defaults to "Antennas Surface Current".

    Returns
    -------
    plotly.graph_objects.Figure
        A Plotly Figure object containing the 3D trisurf plot and optional markers.

    Notes
    -----
    - The function includes a safety check for `vmin >= vmax`. If the current 
      density is uniform (e.g., all zeros), a micro-epsilon (1e-15) is added 
      to the data range to prevent a PlotlyError.
    - Aspect ratios are calculated based on the bounding box of the mesh to 
      ensure the antenna geometry is not distorted visually.

    Example
    -------
    >>> fig = visualize_surface_current(mesh_pts, mesh_tris, j_density, feed_pt=[0, 0, 0])
    >>> fig.show()
    """
    # 1. Extract vertex coordinates
    x_, y_, z_ = points_data.points  

    # 2. Create simplices for plotly
    simplices = triangles_data.triangles[:3, :].T  

    # --- CORRECTION POUR PLOTLY ERROR (vmin >= vmax) ---
    # We ensure surface_current_density is a numpy array
    scd = np.array(surface_current_density)
    
    v_min = np.min(scd)
    v_max = np.max(scd)

    # If all values are identical (e.g., all zeros), Plotly's trisurf fails.
    # We add a tiny epsilon to the max value to create a valid range.
    if v_min >= v_max:
        scd = scd.astype(float) # Ensure float type
        scd[0] += 1e-15
    # --------------------------------------------------

    # 3. Visualization logic
    aspect_ratios = compute_aspect_ratios(points_data.points)

    custom_colormap = [
        "rgb(0, 0, 180)", "rgb(0, 0, 255)", "rgb(0, 255, 255)", 
        "rgb(0, 255, 0)", "rgb(255, 255, 0)", "rgb(255, 140, 0)", "rgb(255, 0, 0)"
    ]

    # Create the trisurf figure
    fig = ff.create_trisurf(
        x=x_,
        y=y_,
        z=z_,
        simplices=simplices,
        colormap=custom_colormap,
        color_func=scd,  # Use the sanitized data
        show_colorbar=True,
        title=title,
        aspectratio=aspect_ratios,
    )

    # 4. Highlight feed point(s)
    if feed_point is not None:
        feed_point = np.atleast_2d(feed_point)
        fig.add_trace(go.Scatter3d(
            x=feed_point[:, 0],
            y=feed_point[:, 1],
            z=feed_point[:, 2],
            mode='markers+text',
            marker=dict(size=6, color='red', symbol='circle'),
            name='Feed Point(s)'
        ))

    # 5. Layout configuration
    fig.update_layout(
        scene=dict(
            camera=dict(eye=dict(x=0.65, y=0.65, z=0.65))
        ),
        legend=dict(
            x=0.2, y=0.9,
            xanchor='left', yanchor='top',
            bgcolor='rgba(255,255,255,0.7)',
            bordercolor='lightgray',
            borderwidth=1
        )
    )

    return fig