"""
VIEWER Visualizes the structure - all Chapters

Usage:  python viewer.py data/antennas_mesh/bowtie      or
        python viewer.py data/antennas_mesh/bowtie.mat

Copyright 2002 AEMM. Revision 2002/03/05 Chapter 2
"""
import argparse
import os
import re

import numpy as np
import plotly.figure_factory as ff
import scipy.io as sio


def load_mesh_file(filename):
    """Load the .mat file and return mesh points and triangles."""
    try:
        mesh = sio.loadmat(filename)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{filename}' was not found.")
    except Exception as error:
        raise RuntimeError(f"Error loading the file: {error}")

    # Validate required variables
    if 'p' not in mesh or 't' not in mesh:
        raise ValueError("The file must contain the variables 'p' (points) and 't' (triangles).")

    points = mesh['p']  # Mesh point coordinates (3 x N) → Each point in the mesh has a coordinate
    triangles = mesh['t']  # Triangle indices (4 x M) → There are M triangles and triangle[:, i] is the i-th triangle; triangle[0, i] corresponds to the first vertex of this i-th triangle, and so on
    return points, triangles


def filter_triangles(triangles):
    """Filter triangles whose fourth row is > 1."""
    valid_indices = np.where(triangles[3, :] <= 1)[0]
    return triangles[:, valid_indices].astype(int)  # Explicit conversion to avoid errors


def compute_aspect_ratios(points):
    """Compute aspect ratios for 3D display."""
    x_, y_, z_ = points
    fig_scale = max(max(x_) - min(x_), max(y_) - min(y_), max(z_) - min(z_))
    return {
        "x": (max(x_) - min(x_)) / fig_scale,
        "y": (max(y_) - min(y_)) / fig_scale,
        "z": (max(z_) - min(z_)) / fig_scale,
    }


def calculate_mesh_dimension(points):
    point_x_min = min(points[0])
    point_x_max = max(points[0])
    point_y_min = min(points[1])
    point_y_max = max(points[1])
    longueur_mesh_x = point_x_max - point_x_min
    hauteur_mesh_y = point_y_max - point_y_min
    return longueur_mesh_x, hauteur_mesh_y


def create_figure(points, triangles, title="Antennas Mesh"):
    """Create a 3D Plotly figure from points and triangles."""
    x_, y_, z_ = points
    simplices = (triangles[:3, :].T - 1)  # Adjust for Python indexing

    aspect_ratios = compute_aspect_ratios(points)

    fig = ff.create_trisurf(
        x=x_,
        y=y_,
        z=z_,
        simplices=simplices,
        color_func=np.arange(len(simplices)),  # Triangle colors
        show_colorbar=False,
        title=title,
        aspectratio=aspect_ratios,
    )
    return fig


def viewer(filename):
    """Load, filter, and visualize a mesh file."""
    print(f"Loading file: {filename}")
    points, triangles = load_mesh_file(filename)

    print(f"Points shape: {points.shape}")
    print(f"Triangles shape: {triangles.shape}")

    # Filter invalid triangles
    triangles = filter_triangles(triangles)
    print(f"Filtered Triangles shape: {triangles.shape}")

    # Compute mesh dimensions
    longueur, hauteur = calculate_mesh_dimension(points)
    print(f"Your mesh has dimensions {longueur} * {hauteur} meters")  # Assuming mesh units are meters
    print(f"Length along x-axis = {longueur} meters \nHeight along y-axis = {hauteur} meters")

    # Create and display the figure
    antennas_file_name = os.path.splitext(os.path.basename(filename))[0] + ' antenna mesh'
    fig = create_figure(points, triangles, antennas_file_name)
    fig.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize a .mat file containing a 3D mesh.")
    parser.add_argument("filename", help="Name of the .mat file to visualize.")
    args = parser.parse_args()

    try:
        viewer(args.filename)
    except Exception as error:
        print(f"Error: {error}")