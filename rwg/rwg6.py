import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from rwg.rwg2 import DataManager_rwg2
from rwg.rwg4 import DataManager_rwg4
from rwg.rwg5 import calculate_current_density


def plot_surface_current_distribution(filename_mesh_2, filename_current):
    points, triangles, edges, _, vecteurs_rho = DataManager_rwg2.load_data(filename_mesh_2)
    *_, current = DataManager_rwg4.load_data(filename_current)

    surface_current_density = calculate_current_density(current, triangles, edges, vecteurs_rho)

    # Définir les limites de l'échantillonnage
    K = 20
    x0, x1 = np.min(points[0, :]), np.max(points[0, :])
    y0, y1 = np.min(points[1, :]), np.max(points[1, :])

    # Calculer les densités de courant à partir des échantillons
    y = np.linspace(y0, y1, K + 1)
    X = np.zeros(K + 1)
    Y = np.zeros(K + 1)

    for n in range(K + 1):
        y_n = y[n]
        Dist = np.linalg.norm(np.vstack([np.zeros(triangles), np.full(triangles, y_n), np.zeros(triangles)]).T - triangles.triangles_center, axis=1)
        Index = np.argmin(Dist)
        X[n] = surface_current_density[0, Index]
        Y[n] = surface_current_density[1, Index]

    # Interpolation cubique
    yi = np.linspace(y0, y1, 100)
    interp_X = interp1d(y, X, kind='cubic')
    interp_Y = interp1d(y, Y, kind='cubic')

    Xi = interp_X(yi)
    Yi = interp_Y(yi)

    # Tracer les courbes
    plt.plot(yi, Xi, label='Jx')
    plt.plot(yi, Yi, '*', label='Jy')
    plt.xlabel('Dipole length, m')
    plt.ylabel('Surface current density, A/m')
    plt.grid(True)
    plt.legend()
    plt.show()
