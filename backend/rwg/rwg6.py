import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree

from backend.rwg.rwg2 import DataManager_rwg2
from backend.rwg.rwg4 import DataManager_rwg4


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

def plot_surface_current_distribution(filename_mesh, filename_current, mode='scattering'):
    """
    Plots the surface current distribution along the dipole length.
    
    Parameters:
    - mode: 'scattering' or 'radiation'
    """
    # 1. Parameter validation
    if mode not in ['scattering', 'radiation']:
        raise ValueError("mode must be either 'scattering' or 'radiation'.")
    
    print(f"MODE SELECTED: {mode}")

    # 2. Load geometry and current data
    # Note: Adapting to your existing DataManager structure
    points, triangles, edges, _, vecteurs_rho = DataManager_rwg2.load_data(filename_mesh)
    
    if mode == 'scattering':
        # Unpacking specifically for scattering format
        *_, current = DataManager_rwg4.load_data(filename_current, scattering=True)
    else:
        # Unpacking specifically for radiation format
        *_, current, _, _, _, _ = DataManager_rwg4.load_data(filename_current, radiation=True)

    # 3. Compute surface current density (J)
    # This assumes J is a vector (Jx, Jy, Jz) for each triangle
    surface_current_density = surface_calculate_current_density(current, triangles, edges, vecteurs_rho)
    
    # 4. Spatial sampling along the Y-axis
    K = 100 # Number of sampling points
    y_min, y_max = np.min(points.points[1, :]), np.max(points.points[1, :])
    y_samples = np.linspace(y_min, y_max, K)
    
    # Create the coordinates for sampling (assumed centered at X=0, Z=0)
    # Shape (K, 3)
    sampling_points = np.zeros((K, 3))
    sampling_points[:, 1] = y_samples 

    # 5. Efficient Nearest Neighbor Search using KDTree
    # Instead of a manual loop, KDTree finds the closest triangle center for all points at once
    tree = cKDTree(triangles.triangles_center.T)
    _, indices = tree.query(sampling_points)
    
    # Extract Jx and Jy at the found indices
    X_samples = np.abs(surface_current_density[0, indices])
    Y_samples = np.abs(surface_current_density[1, indices])

    # 6. Smooth Interpolation
    y_fine = np.linspace(y_min, y_max, 300)
    jx_interp = interp1d(y_samples, X_samples, kind='cubic')(y_fine)
    jy_interp = interp1d(y_samples, Y_samples, kind='cubic')(y_fine)

    # 7. Professional Plotting
    plt.style.use('seaborn-v0_8-whitegrid') # Modern clean style
    fig, ax = plt.subplots(figsize=(14, 8))
    
    ax.plot(y_fine, jx_interp, label=r'$|J_x|$', linewidth=2)
    ax.plot(y_fine, jy_interp, '--', label=r'$|J_y|$', linewidth=2)
    
    ax.set_title(f'Surface Current Distribution ({mode.capitalize()})', fontsize=14)
    ax.set_xlabel('Dipole length (m)', fontsize=12)
    ax.set_ylabel('Current Density (A/m)', fontsize=12)
    
    ax.legend(frameon=True)
    ax.spines[['top', 'right']].set_visible(False) # Remove unnecessary borders
    
    plt.tight_layout()
    plt.show()