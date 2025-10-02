from utils.gmsh_function import *

print("This code is written to simulate an antenna with adaptive meshing.")

# -------------------------------------- File names --------------------------------------

mesh_name = 'radiate_ifa1_Adapt.msh'
save_mesh_folder = 'data/gmsh_files/'
ifa1_Adapt_mat = 'data/antennas_mesh/radiate_ifa1_Adapt.mat'
ifa1_Adapt_msh = save_mesh_folder + mesh_name

# -------------------------------------- Antenna creation code --------------------------------------

gmsh.initialize()

model_name  = "ifa_1_adapt"
feed_point = [0.0025, 0.1, 0]
feed_length = 0.005

# Create the model
gmsh.model.add(model_name)

# Define points
p0 = gmsh.model.occ.addPoint(0, 0, 0)
p1 = gmsh.model.occ.addPoint(0, 0.1, 0)
p2 = gmsh.model.occ.addPoint(0, 0.155, 0)
p3 = gmsh.model.occ.addPoint(0.05, 0.155, 0)
p4 = gmsh.model.occ.addPoint(0.05, 0.13, 0)
p5 = gmsh.model.occ.addPoint(0.03, 0.13, 0)
p6 = gmsh.model.occ.addPoint(0.03, 0.135, 0)
p7 = gmsh.model.occ.addPoint(0.045, 0.135, 0)
p8 = gmsh.model.occ.addPoint(0.045, 0.15, 0)
p9 = gmsh.model.occ.addPoint(0.005, 0.15, 0)
p10 = gmsh.model.occ.addPoint(0.005, 0.1, 0)
p11 = gmsh.model.occ.addPoint(0.05, 0.1, 0)
p12 = gmsh.model.occ.addPoint(0.05, 0, 0)

# List points in order
points = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12]

# Create lines between consecutive points and close the loop (last to first)
lines = [gmsh.model.occ.addLine(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]
cl = gmsh.model.occ.addCurveLoop(lines)
ifa_1 = gmsh.model.occ.addPlaneSurface([cl])

apply_mesh_size(feed_length * 10)
generate_surface_mesh()

# -------------------------------------- Adaptive antenna code --------------------------------------

run()
refine_antenna(model_name, 6.059e+8, mesh_name, feed_point, feed_length, ifa1_Adapt_msh, ifa1_Adapt_mat, save_mesh_folder)

gmsh.finalize()