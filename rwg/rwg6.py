import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from rwg.rwg2 import DataManager_rwg2
from rwg.rwg4 import DataManager_rwg4


def surface_calculate_current_density(current, triangles, edges, vecteurs_rho):
    # Initialize the array to store surface current density
    surface_current_density = np.zeros((2, triangles.total_of_triangles))  # Current magnitude for each triangle

    # Loop over each triangle to compute the current density
    for triangle in range(triangles.total_of_triangles):
        current_density_for_triangle = np.array([0.0, 0.0, 0.0], dtype=complex)  # Initialize as complex  # Initialize the current density vector for this triangle
        for edge in range(edges.total_number_of_edges):
            current_times_edge = current[edge] * edges.edges_length[edge]   # I(m) * EdgeLength(m)

            # Contribution if the edge is associated with the triangle on the "plus" side
            if triangles.triangles_plus[edge] == triangle:
                current_density_for_triangle += current_times_edge * vecteurs_rho.vecteur_rho_plus[:, edge] / (2 * triangles.triangles_area[triangles.triangles_plus[edge]])

            # Contribution if the edge is associated with the triangle on the "minus" side
            elif triangles.triangles_minus[edge] == triangle:
                current_density_for_triangle += current_times_edge * vecteurs_rho.vecteur_rho_minus[:, edge] / (2 * triangles.triangles_area[triangles.triangles_minus[edge]])

        # Compute the magnitude of the current density for this triangle
        surface_current_density[:, triangle] = np.abs(current_density_for_triangle[:2])

    return surface_current_density


def plot_surface_current_distribution(filename_mesh_2, filename_current, scattering=False, radiation=False):
    points, triangles, _, _, edges, _, vecteurs_rho = DataManager_rwg2.load_data(filename_mesh_2)
    
    if scattering:
        *_, current = DataManager_rwg4.load_data(filename_current, scattering=scattering)
    elif radiation:
        *_, current, _, _, _, _ = DataManager_rwg4.load_data(filename_current, radiation=radiation)

    total_of_triangles = triangles.total_of_triangles

    # Compute surface current density
    surface_current_density = surface_calculate_current_density(current, triangles, edges, vecteurs_rho)

    # Define sampling limits
    K = 20
    y0, y1 = np.min(points.points[1, :]), np.max(points.points[1, :])

    # Compute current densities at sampled points
    y = np.linspace(y0, y1, K + 1)
    X = np.zeros(K + 1)
    Y = np.zeros(K + 1)

    for n in range(K + 1):
        y_n = y[n]
        Dist = np.linalg.norm(
            np.vstack([np.zeros(total_of_triangles), np.full(total_of_triangles, y_n), np.zeros(total_of_triangles)]).T
            - triangles.triangles_center.T, axis=1
        )
        Index = np.argmin(Dist)
        X[n] = surface_current_density[0, Index]
        Y[n] = surface_current_density[1, Index]

    # Cubic interpolation
    yi = np.linspace(y0, y1, 100)
    interp_X = interp1d(y, X, kind='cubic')
    interp_Y = interp1d(y, Y, kind='cubic')

    Xi = interp_X(yi)
    Yi = interp_Y(yi)

    # Plot the curves
    plt.plot(yi, Xi, label='|Jx|')
    plt.plot(yi, Yi, '*', label='|Jy|')
    plt.xlabel('Dipole length, m')
    plt.ylabel('Surface current density, A/m')
    plt.grid(True)
    plt.legend()
    plt.show()
