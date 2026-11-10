"""
    This algorithm simulates the current distribution on an antenna receiving an incident electromagnetic wave.
    It relies on the RWG (Rao-Wilton-Glisson) functions available in the "/rwg" folder.
    The main steps include:
        1. Loading and processing the antenna mesh.
        2. Building the impedance matrix and the required vectors.
        3. Computing the induced current by the incident wave.
        4. Visualizing the surface currents on the antenna.

    Main inputs:
        * mesh : either the SimpleNamespace returned by setup_save_file_paths() (same
                 convention as radiation_algorithm), or the path of a .mat mesh file.
        * frequency : Frequency of the incident wave (in Hz).
        * wave_incident_direction : Propagation direction of the incident wave (3D vector).
        * polarization : Polarization of the incident wave (3D vector).

    Main outputs:
        * Visualization of surface currents on the antenna.
        * Saving intermediate data into different folders for further processing.
"""
import os
from types import SimpleNamespace

import numpy as np

from solver.rwg.rwg1 import *
from solver.rwg.rwg2 import *
from solver.rwg.rwg3 import *
from solver.rwg.rwg4 import *
from solver.rwg.rwg5 import *
from solver.utils.file_path import setup_save_file_paths
from solver.utils.gmsh_function import extract_msh_to_mat


def _resolve_scattering_paths(mesh):
    """
    Return (path, mesh_file) for scattering_algorithm.

    * A SimpleNamespace from setup_save_file_paths() is used as is. If its .msh
      exists, the .mat is regenerated from it first, exactly as
      radiation_algorithm() does, so a re-meshed geometry is never solved with a
      stale .mat.
    * A string is the legacy calling convention (path of a .mat mesh file, e.g.
      a MATLAB/AEMM mesh). The intermediate files go to the standard data/
      folders under the file's base name, which is where this function always
      wrote them.
    """
    if isinstance(mesh, SimpleNamespace):
        if getattr(mesh, 'msh', None) and os.path.exists(mesh.msh):
            extract_msh_to_mat(mesh.msh, mesh.mat)
        return mesh, mesh.mat

    if isinstance(mesh, (str, os.PathLike)):
        name = os.path.splitext(os.path.basename(os.fspath(mesh)))[0]
        return setup_save_file_paths(name), os.fspath(mesh)

    raise TypeError(
        "scattering_algorithm: 'mesh' must be the SimpleNamespace returned by "
        "setup_save_file_paths() or the path of a .mat mesh file, got "
        f"{type(mesh).__name__}."
    )


def scattering_algorithm(mesh, frequency, wave_incident_direction, polarization, show=True,
                         show_current_arrows=True, load_from_matlab=True, use_gpu=None,
                         excitation_unit_vector=None, gap_width=None):
    """
        Implements the electromagnetic scattering algorithm for an antenna.

        Parameters:
            * mesh : SimpleNamespace from setup_save_file_paths(), or path of a .mat mesh file.
            * frequency : float, frequency of the incident plane wave (Hz).
            * wave_incident_direction : (3,) propagation direction (unit vector), e.g. [0, 0, -1].
            * polarization : (3,) incident electric-field vector, e.g. [1, 0, 0].
            * show : bool, display the surface current (Plotly).
            * show_current_arrows : bool, overlay the current direction on the plot.
            * load_from_matlab : bool, the mesh file uses 1-based triangle indices
              (MATLAB/AEMM and gmsh_function.extract_msh_to_mat both do). Default True.
            * use_gpu : bool or None, passed to calculate_z_matrice (None = auto).
            * excitation_unit_vector, gap_width : ignored. Accepted so that this function
              and radiation_algorithm() share the call signature used by
              solver.utils.h_refinement_loop.run_refinement_cycle.

        Returns:
            matrice_z, voltage, current, surface_current_density

        Notes:
            Triangles whose 4th row (domain label) is > 1 are discarded
            (Triangles.filter_triangles, the AEMM convention). This function has
            always done so; radiation_algorithm() does not. For a gmsh mesh the
            4th row is the surface tag, so a geometry made of several surfaces
            keeps only surface 1 here.
    """
    path, mesh_file = _resolve_scattering_paths(mesh)
    wave_incident_direction = np.asarray(wave_incident_direction, dtype=float)
    polarization = np.asarray(polarization)
    if not np.iscomplexobj(polarization):
        polarization = polarization.astype(float)

    # Load mesh file
    p, t = load_mesh_file(mesh_file, load_from_matlab)

    # Define points and triangles from the mesh
    points = Points(p)
    triangles = Triangles(t)

    # Filter invalid triangles and compute geometric properties (areas, centers)
    triangles.filter_triangles()
    triangles.calculate_triangles_area_and_center(points)

    # Define edges and compute their lengths
    edges = triangles.get_edges()
    edges.compute_edges_length(points)

    # Filter complex junctions to simplify the mesh structure
    filter_complexes_jonctions(points, triangles, edges)

    # Save processed mesh data
    DataManager_rwg1.save_data(path, points, triangles, edges)

    # Compute RHO vectors for the edges
    vecteurs_rho = Vecteurs_Rho()
    vecteurs_rho.calculate_vecteurs_rho(points, triangles, edges)

    # Save RHO vectors data
    DataManager_rwg2.save_data(path, vecteurs_rho)

    # Compute electromagnetic constants and impedance matrix Z
    omega, mu, epsilon, light_speed_c, eta, matrice_z = calculate_z_matrice(
        points, triangles, edges, frequency, use_gpu=use_gpu)

    # Save impedance data
    DataManager_rwg3.save_data(path, frequency, omega, mu, epsilon, light_speed_c, eta, matrice_z)

    # Compute the induced current on the antenna by the incident wave
    frequency, omega, mu, epsilon, light_speed_c, eta, voltage, current = calculate_current_scattering(
        path.mat_mesh2, path.mat_impedance, wave_incident_direction, polarization)

    # Save current data (written to path.mat_current: <name>_current.mat)
    DataManager_rwg4.save_data_for_scattering(path.mat_mesh2, os.path.dirname(path.mat_current), frequency,
                                              omega, mu, epsilon, light_speed_c, eta, wave_incident_direction,
                                              polarization, voltage, current)

    # Compute surface currents from the total current
    surface_current_density, surface_current_vector = calculate_current_density(current, triangles, edges, vecteurs_rho)

    # Visualize surface currents if requested
    if show:
        antennas_name = f"{path.name} antenna surface current in receiving mode"
        print(f"\n{antennas_name} view is successfully created at frequency {frequency} Hz")
        arrows = surface_current_vector if show_current_arrows else None
        fig = visualize_surface_current(points, triangles, surface_current_density, title=antennas_name, current_vector=arrows)
        fig.show()

    return matrice_z, voltage, current, surface_current_density
