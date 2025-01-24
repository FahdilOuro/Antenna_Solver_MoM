import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from scipy.spatial import Delaunay
from scipy.io import savemat

def compute_aspect_ratios(points):
    x_, y_, z_ = points
    fig_scale = max(max(x_) - min(x_), max(y_) - min(y_), max(z_) - min(z_))
    return {"x": (max(x_) - min(x_)) / fig_scale, "y": (max(y_) - min(y_)) / fig_scale, "z": (max(z_) - min(z_)) / fig_scale}

def create_figure(points, triangles, title="Antennas Mesh"):
    x_, y_, z_ = points
    simplices = triangles[:3, :].T
    aspect_ratios = compute_aspect_ratios(points)
    fig = ff.create_trisurf(x=x_, y=y_, z=z_, simplices=simplices, color_func=np.arange(len(simplices)), show_colorbar=False, title=title, aspectratio=aspect_ratios)
    return fig

def data_save(filename, save_folder_name, points, triangle):
    data = {'p': points, 't': triangle}
    save_file_name = filename + '.mat'
    full_save_path = os.path.join(save_folder_name, save_file_name)
    if not os.path.exists(save_folder_name):
        os.makedirs(save_folder_name)
        print(f"Directory '{save_folder_name}' created.")
    savemat(full_save_path, data)
    print(f"Data saved successfully to {full_save_path}")
    return save_file_name

L, W, Nx, Ny, h, Number = 2.0, 2.0, 11, 11, 1.0, 7
epsilon = 1e-6
coordonnees_x, coordonnees_y = [], []
for i in range(Nx + 1):
    for j in range(Ny + 1): 
        x_val, y_val = -L/2 + (i / Nx) * L, -W/2 + (j / Ny) * W - (epsilon * (-L/2 + (i / Nx) * L))
        coordonnees_x.append(x_val)
        coordonnees_y.append(y_val)
coordonnees_x, coordonnees_y = np.array(coordonnees_x), np.array(coordonnees_y)
x_feed, y_feed = np.array([-0.02, 0.02]), np.array([0, 0])
coordonnees_x, coordonnees_y = np.append(coordonnees_x, x_feed), np.append(coordonnees_y, y_feed)
C = np.mean(x_feed)
x1 = np.array([C, C])
y1 = np.mean(y_feed) + 2 * np.array([np.max(x_feed) - C, np.min(x_feed) - C])
coordonnees_x, coordonnees_y = np.append(coordonnees_x, x1), np.append(coordonnees_y, y1)
points_base = np.column_stack((coordonnees_x, coordonnees_y))
triangulation = Delaunay(points_base)
t = np.zeros((4, triangulation.simplices.shape[0]), dtype=int)
t[:3, :] = triangulation.simplices.T
p = np.zeros((3, points_base.shape[0]))
p[:2, :] = points_base.T

feed_points = np.column_stack((x_feed, y_feed))
feed_indices = []
for fp in feed_points:
    idx = np.where((p[:2, :].T == fp).all(axis=1))[0]
    if len(idx) > 0:
        feed_indices.append(idx[0])
if len(feed_indices) != 2:
    raise ValueError("Les points d'alimentation ne sont pas correctement trouvés dans le maillage du plan de masse.")

monopole_x = np.tile(x_feed, Number + 1)
monopole_y = np.tile(y_feed, Number + 1)
monopole_z = np.linspace(0, h, Number + 1).repeat(2)
p_monopole_3D = np.vstack((monopole_x, monopole_y, monopole_z))

offset = p.shape[1]
p = np.hstack((p, p_monopole_3D))

t_monopole = []
for i in range(Number):
    t_monopole.append([offset + 2*i, offset + 2*i + 1, offset + 2*i + 3, 1])
    t_monopole.append([offset + 2*i, offset + 2*i + 2, offset + 2*i + 3, 1])
t_monopole = np.array(t_monopole).T
t = np.hstack((t, t_monopole))

t_bridge_1 = np.array([[feed_indices[0], feed_indices[1], offset + 1, 1]]).T
t_bridge_2 = np.array([[feed_indices[0], offset, offset + 1, 1]]).T
t = np.hstack((t, t_bridge_1, t_bridge_2))

fig = create_figure(p, t, "Antenna Monopole with Ground Plane")
fig.show()
data_save("monopole_antenna", "data/antennas_mesh/", p, t)

