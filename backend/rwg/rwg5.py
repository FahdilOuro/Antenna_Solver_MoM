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

        Parameters:
        * points_data : object containing the surface point coordinates, as a 2D array (3, n_points),
                        where rows correspond to X, Y, and Z coordinates.
        * triangles_data : object containing the indices of the surface triangles, as a 2D array (3, n_triangles),
                           where each column corresponds to a triangle defined by three vertex indices.
        * surface_current_density : n-d-array, normalized or raw surface current density associated with each triangle.
        * title : str, title of the visualization (default "Antennas Surface Current").

        Returns:
        fig : Plotly figure object representing the 3D plot.

        Functionality:
            1. Extracts X, Y, Z coordinates from 'points_data'.
            2. Prepares triangle indices from `triangles_data` for Plotly compatibility.
            3. Computes aspect ratios for coherent visualization using 'compute_aspect_ratios'.
            4. Creates a "trisurf" figure with Plotly, colored according to current density.
            5. Displays a color bar indicating current density levels.
            6. Returns the figure object for display or saving.

        Example:
        This function helps visualize the distribution of surface current on a triangulated surface,
        useful for analyzing antenna or conductor models.

        Notes:
            * The surface current density (surface_current_density) should be one value per triangle, corresponding
              to 'triangles_data'.
            * Ensure the `plotly` library is installed and `ff.create_trisurf` is available.
    """
    # Extract vertex coordinates
    x_, y_, z_ = points_data.points  # X, Y, Z coordinates of points

    # Create simplices for plotly (vertex indices of each triangle)
    simplices = triangles_data.triangles[:3, :].T  # Transpose from [3, n_triangles] to [n_triangles, 3]

    # Visualization with plotly
    aspect_ratios = compute_aspect_ratios(points_data.points)

    # Create the trisurf figure
    fig = ff.create_trisurf(
        x=x_,
        y=y_,
        z=z_,
        simplices=simplices,
        colormap="Rainbow",
        color_func=surface_current_density,  # Color using surface current density
        show_colorbar=True,
        title='',
        aspectratio=aspect_ratios,
    )
    # Highlight feed point(s) in red if feed_point is provided
    if feed_point is not None:
        feed_point = np.atleast_2d(feed_point)  # Ensure shape (n, 3)
        fig.add_trace(go.Scatter3d(
            x=feed_point[:, 0],
            y=feed_point[:, 1],
            z=feed_point[:, 2],
            mode='markers+text',
            marker=dict(size=6, color='red', symbol='circle'),
            name='Feed Point(s)'
        ))
    # Configure the legend
    fig.update_layout(
        scene=dict(
            camera=dict(
                eye=dict(x=0.65, y=0.65, z=0.65)  # Larger values = zoom out
            )
        ),
        legend=dict(
            x=0.2,  # Horizontal position (0=left, 1=right)
            y=0.9,  # Vertical position (0=bottom, 1=top)
            xanchor='left',  # Horizontal anchor ('auto', 'left', 'center', 'right')
            yanchor='top',   # Vertical anchor ('auto', 'top', 'middle', 'bottom')
            bgcolor='rgba(255,255,255,0.7)',  # Semi-transparent background
            bordercolor='lightgray',
            borderwidth=1
        )
    )

    return fig

def calculate_threshold_surface_current_density(surface_current_density):
    """
        Calculates a threshold for the surface current density and identifies triangles below this threshold.

        Parameters:
        * surface_current_density : n-d-array, surface current density for each triangle.

        Returns:
        * indices_below_threshold : n-d-array, indices of triangles where the surface current density is below 70% of the maximum value.

        Functionality:
            1. Computes the maximum value of the surface current density.
            2. Defines a threshold as 70% of this maximum.
            3. Finds the indices of triangles whose density is below the threshold.
    """

    # Maximum value of the surface current density
    max_value = np.max(surface_current_density)
    threshold = 0.7 * max_value
    
    # Identify indices of elements below the threshold
    indices_below_threshold = np.where(surface_current_density < threshold)[0]

    return indices_below_threshold