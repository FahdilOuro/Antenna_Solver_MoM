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

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree

def plot_surface_current_distribution(path, mode='scattering'):
    """
    Plots the surface current distribution along the dipole length with 
    the aesthetic style of the S11 plot (Adelle font, clean grid).
    
    Args:
        path (object, optional): An object containing .mat_mesh2 and .mat_current attributes.
        mode (str): Mode of operation, either 'scattering' or 'radiation'.
    """
    # 1. Parameter validation
    if mode not in ['scattering', 'radiation']:
        raise ValueError("mode must be either 'scattering' or 'radiation'.")
    
    print(f"MODE SELECTED: {mode}")

    # 2. Load geometry and current data
    points, triangles, edges, _, vecteurs_rho = DataManager_rwg2.load_data(path.mat_mesh2)
    
    if mode == 'scattering':
        *_, current = DataManager_rwg4.load_data(path.mat_current, scattering=True)
    else:
        *_, current, _, _, _, _ = DataManager_rwg4.load_data(path.mat_current, radiation=True)

    # 3. Compute surface current density (J)
    surface_current_density = surface_calculate_current_density(current, triangles, edges, vecteurs_rho)
    
    # 4. Spatial sampling along the Y-axis
    K = 100 
    y_min, y_max = np.min(points.points[1, :]), np.max(points.points[1, :])
    y_samples = np.linspace(y_min, y_max, K)
    
    sampling_points = np.zeros((K, 3))
    sampling_points[:, 1] = y_samples 

    # 5. Efficient Nearest Neighbor Search
    tree = cKDTree(triangles.triangles_center.T)
    _, indices = tree.query(sampling_points)
    
    X_samples = np.abs(surface_current_density[0, indices])
    Y_samples = np.abs(surface_current_density[1, indices])

    # 6. Smooth Interpolation (Using 500 points for high-resolution as in S11 plot)
    y_fine = np.linspace(y_min, y_max, 500)
    jx_interp = interp1d(y_samples, X_samples, kind='cubic')(y_fine)
    jy_interp = interp1d(y_samples, Y_samples, kind='cubic')(y_fine)

    # --- 7. Professional Plotting (Aesthetic Match with plot_s11) ---
    
    # Font Configuration
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Adelle', 'DejaVu Serif', 'Times New Roman']
    
    fig, ax = plt.subplots(figsize=(14, 8), dpi=100)
    
    # Plotting lines with S11 color palette
    # Using #c90f0f (Red) and #2c3e50 (Dark Slate) from your S11 style
    ax.plot(y_fine, jx_interp, color="#c90f0f", linewidth=2.5, label=r'$|J_x|$', zorder=2)
    ax.plot(y_fine, jy_interp, color="#2c3e50", linewidth=2.0, linestyle='--', 
            label=r'$|J_y|$', alpha=0.8, zorder=2)
    
    # Labels and Title
    ax.set_title(f'Surface Current Distribution ({mode.capitalize()})', 
                 fontsize=15, pad=20, fontweight='bold')
    ax.set_xlabel('Dipole length (m)', fontsize=13, labelpad=10)
    ax.set_ylabel('Current Density (A/m)', fontsize=13, labelpad=10)
    
    # Subtle grid and spine management (Exact match with plot_s11)
    ax.grid(True, which='both', linestyle=':', alpha=0.5, color='#bdc3c7')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)

    # Legend style
    ax.legend(frameon=False, fontsize=11)
    
    plt.tight_layout()
    plt.show()