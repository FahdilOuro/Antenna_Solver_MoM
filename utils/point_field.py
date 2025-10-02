import numpy as np


def radiated_scattered_field_at_a_point(point, eta, parameter_k, dipole_moment, dipole_center):
    """
    Calculates the radiated and scattered electric (E) and magnetic (H) fields
    from a set of dipoles at a given observation point.

    Parameters:
        * point (np.ndarray) : Coordinates of the observation point, shape (3,).
        * eta (float) : Characteristic impedance of the medium.
        * parameter_k (complex float) : Complex wavenumber, where omega is angular frequency and c is the speed of light.
        * dipole_moment (np.ndarray) : Dipole moments, shape (3, N), N = number of dipoles.
        * dipole_center (np.ndarray) : Positions of dipoles, shape (3, N).

    Returns:
        * e_field (np.ndarray) : Electric field at the observation point, shape (3, N).
        * h_field (np.ndarray) : Magnetic field at the observation point, shape (3, N).

    Description:
        * Models the electromagnetic fields radiated and scattered by an array of dipoles.
        * Uses classical formulas derived from Maxwell's equations for oscillating dipoles.

    Main calculations:
        1. Distance vector between the observation point and each dipole.
        2. Geometric and attenuation factors based on distance.
        3. Contribution of dipole moments to the electric and magnetic fields.
    """
    # Normalization constants for the fields
    c = 4 * np.pi
    constant_h = parameter_k / c
    constant_e = eta / c

    # Vector from observation point to dipole centers
    r = point.reshape(3, 1) - dipole_center       # (3, N)
    r_norm = np.sqrt(np.sum(r ** 2, axis=0))      # Norm of r, shape (N,)
    r_norm2 = r_norm ** 2                         # Squared distance

    # Exponential attenuation factor
    exp = np.exp(-parameter_k * r_norm)           # Shape (N,)

    # Geometric factors
    parameter_c = (1 / r_norm2) * (1 + (1 / (parameter_k * r_norm)))  # Shape (N,)
    parameter_d = np.sum(r * dipole_moment, axis=0) / r_norm2          # Shape (N,)

    # Parameter M term
    parameter_m = parameter_d * r                 # Shape (3, N)

    # Magnetic field H
    h_field = constant_h * np.cross(dipole_moment, r, axis=0) * parameter_c * exp  # (3, N)

    # Electric field E
    e_field = constant_e * ((parameter_m - dipole_moment) * (parameter_k / r_norm + parameter_c) + 2 * parameter_m * parameter_c) * exp  # (3, N)

    return e_field, h_field