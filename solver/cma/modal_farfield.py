import numpy as np

from solver.efield.efield2 import load_or_create_hollow_sphere, compute_total_radiated_power
from solver.utils.dipole_parameters import compute_dipole_center_moment, compute_e_h_field


def compute_modal_farfield(triangles, edges, eigenvector, frequency, light_speed_c, eta):
    """
    Compute the far-field directivity pattern radiated by a single
    characteristic mode current (an eigenvector treated as a surface current
    distribution), over the standard far-field sphere.

    Why directivity, not gain
    -------------------------
    A characteristic mode has no associated physical excitation, so there is
    no input power P_in to normalize against (unlike
    solver.efield.efield2.radiation_intensity_distribution_over_sphere_surface,
    which also reports an efficiency against a real P_in -- meaningless for a
    mode in isolation). Directivity D(theta,phi) = 4*pi*U(theta,phi)/P_rad
    only needs the mode's own radiated power, and is invariant to the
    eigenvector's arbitrary normalization: scaling the "current" by any
    constant scales U and P_rad by the same factor, so D is unaffected.

    Parameters
    ----------
    triangles, edges : mesh objects (from DataManager_rwg2.load_data(path.mat_mesh2))
    eigenvector       : float64 (n_edges,) -- one column of
                        ModalData.eigenvectors / CMASweepData.eigenvectors[freq_idx]
    frequency         : float, Hz -- the frequency this eigenvector was solved at
    light_speed_c, eta : physical constants for that same solve
                        (DataManager_rwg3.load_data(path.mat_impedance), or the
                        omega/eta returned by calculate_z_matrice_sweep)

    Returns
    -------
    sphere_points, sphere_triangles : the far-field sphere mesh
    gain_db                : float64 (n_sphere_points,), directivity in dBi at each sphere vertex
    directivity_linear_max : float, peak linear directivity
    """
    sphere_points, sphere_triangles = load_or_create_hollow_sphere()

    omega = 2.0 * np.pi * frequency
    k = omega / light_speed_c
    complex_k = 1j * k

    dipole_center, dipole_moment = compute_dipole_center_moment(triangles, edges, eigenvector)

    total_power, _u_triangles = compute_total_radiated_power(
        sphere_points, sphere_triangles, eta, complex_k, dipole_moment, dipole_center)

    n_sphere_points = sphere_points.shape[1]
    u_points = np.zeros(n_sphere_points)
    for i in range(n_sphere_points):
        obs_point = sphere_points[:, i]
        _, _, _, _, u_points[i], _ = compute_e_h_field(obs_point, eta, complex_k, dipole_moment, dipole_center)

    gain_linear = 4 * np.pi * u_points / total_power
    gain_db = 10 * np.log10(np.maximum(gain_linear, 1e-12))
    directivity_linear_max = float(np.max(gain_linear))

    return sphere_points, sphere_triangles, gain_db, directivity_linear_max
