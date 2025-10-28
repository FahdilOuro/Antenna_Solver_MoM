"""
    This module implements the computation of electromagnetic fields and associated properties
    from triangle, edge, and current data for an antenna.

    Main functionalities:
        1. Calculation of centers and dipole moments associated with the edges of a triangular mesh.
        2. Determination of radiated and scattered electric (E) and magnetic (H) fields at an observation point.
        3. Calculation of power density (Poynting vector), radiation density, and radiation intensity.

    Main inputs:
        * triangles_data: Contains triangle data, including their centers and edge-related indices.
        * edges_data: Contains lengths and the total number of edges.
        * current_data: Array of electric currents associated with the mesh edges.
        * observation_point: Point in space where the fields will be computed (3D vector).
        * eta: Characteristic impedance of the medium.
        * complex_k: Complex wavenumber of the medium.
"""
import numpy as np

from backend.utils.point_field import radiated_scattered_field_at_a_point


def compute_dipole_center_moment(triangles_data, edges_data, current_data):
    """
    Computes the centers and dipole moments associated with the edges of a mesh.

    Parameters:
        * triangles_data: Object with .triangles_center (3xN), .triangles_plus, .triangles_minus (indices)
        * edges_data: Object with .edges_length (N), .total_number_of_edges (int)
        * current_data: 1D complex array of currents on each edge (N,)

    Returns:
        * dipole_center: np.ndarray (3 x N), centers of the dipoles
        * dipole_moment: np.ndarray (3 x N), complex dipole moments
    """
    # Retrieve the centers of the plus and minus triangles
    point_plus_center = triangles_data.triangles_center[:, triangles_data.triangles_plus]
    point_minus_center = triangles_data.triangles_center[:, triangles_data.triangles_minus]

    # Vectorized calculation of dipole centers
    dipole_center = 0.5 * (point_plus_center + point_minus_center)

    # Vectorized calculation of dipole moments
    delta = -point_plus_center + point_minus_center
    scaling = edges_data.edges_length * current_data  # (N,)
    dipole_moment = delta * scaling  # Broadcasting (3,N) * (N,) → (3,N)

    return dipole_center, dipole_moment

def compute_e_h_field(observation_point, eta, complex_k, dipole_moment, dipole_center):
    """
        Computes the radiated and scattered electric and magnetic fields at the observation point,
        along with associated quantities like the Poynting vector and radiation intensity.

        Parameters:
            * observation_point : Coordinates of the observation point (3D vector).
            * eta : Characteristic impedance of the medium.
            * complex_k : Complex wavenumber.
            * dipole_moment : Dipole moments associated with the edges (3xN matrix).
            * dipole_center : Dipole centers associated with the edges (3xN matrix).

        Returns:
         * e_field_total : Total electric field at the observation point (3D vector).
         * h_field_total : Total magnetic field at the observation point (3D vector).
         * poynting_vector : Poynting vector representing the transported power density (3D vector).
         * w : Radiation density (power per unit surface).
         * u : Radiation intensity (power per unit solid angle).
         * norm_observation_point : Distance between the observation point and the origin.
    """
    # Compute E and H fields at the observation point from the dipole moments
    e_field, h_field = radiated_scattered_field_at_a_point(observation_point, eta, complex_k, dipole_moment, dipole_center)

    # Sum the contributions of the dipoles to get the total fields
    e_field_total = np.sum(e_field, axis=1)
    h_field_total = np.sum(h_field, axis=1)

    # Compute the Poynting vector (power density carried by the EM waves)
    poynting_vector = np.real(0.5 * (np.cross(e_field_total.flatten(), np.conj(h_field_total).flatten())))

    # Norm of the observation point position
    norm_observation_point = np.linalg.norm(observation_point)

    # Radiation density: norm of the Poynting vector
    w = np.linalg.norm(poynting_vector)

    # Radiation intensity: radiation density scaled by the square of the distance
    u = (norm_observation_point ** 2) * w

    return e_field_total, h_field_total, poynting_vector, w, u, norm_observation_point