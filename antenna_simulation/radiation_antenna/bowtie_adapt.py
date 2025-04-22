from utils.gmsh_function import *

print("This code is write to simulate an antenna with adatative meshing code.")

# -------------------------------------- Files names --------------------------------------

mesh_name = 'radiate_bowtie_gmsh.msh'
save_mesh_folder = 'data/gmsh_files/'
radiate_bowtie_mat_gmsh = 'data/antennas_mesh/radiate_bowtie_gmsh.mat'
radiate_bowtie_msh_gmsh = save_mesh_folder + mesh_name

# -------------------------------------- antenna creation code --------------------------------------

width = 0.5   # Largeur totale
hight = 1.0   # Hauteur totale
width_finite = 0.1 # Largeur de la ligne d'alimentation
hight_up = 2.0
feed_point = [0, 0, 0]
model_name  = "bowtie_antenna"

gmsh.initialize()
gmsh.model.add(model_name)

# Définition des points
p0 = gmsh.model.occ.addPoint(-width/2, -hight/2, 0)
p1 = gmsh.model.occ.addPoint(-width_finite/2, 0, 0)
p2 = gmsh.model.occ.addPoint(-width/2, hight/2, 0)
p3 = gmsh.model.occ.addPoint(-width_finite/2, hight_up/2, 0)
p4 = gmsh.model.occ.addPoint(width_finite/2, hight_up/2, 0)
p5 = gmsh.model.occ.addPoint(width/2, hight/2, 0)
p6 = gmsh.model.occ.addPoint(width_finite/2, 0, 0)
p7 = gmsh.model.occ.addPoint(width/2, -hight/2, 0)
p8 = gmsh.model.occ.addPoint(width_finite/2, -hight_up/2, 0)
p9 = gmsh.model.occ.addPoint(-width_finite / 2, -hight_up / 2, 0)

# Création des segments du contour
l1 = gmsh.model.occ.addLine(p0, p1)
l2 = gmsh.model.occ.addLine(p1, p2)
l3 = gmsh.model.occ.addLine(p2, p3)
l4 = gmsh.model.occ.addLine(p3, p4)
l5 = gmsh.model.occ.addLine(p4, p5)
l6 = gmsh.model.occ.addLine(p5, p6)
l7 = gmsh.model.occ.addLine(p6, p7)
l8 = gmsh.model.occ.addLine(p7, p8)
l9 = gmsh.model.occ.addLine(p8, p9)
l10 = gmsh.model.occ.addLine(p9, p0)

cl = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4, l5, l6, l7, l8, l9, l10])
bowtie = gmsh.model.occ.addPlaneSurface([cl])

apply_mesh_size(width_finite)

gmsh.model.mesh.generate(2)

# -------------------------------------- Adapt antenna code --------------------------------------

run()
refine_antenna(model_name, mesh_name, feed_point, width_finite, radiate_bowtie_msh_gmsh, radiate_bowtie_mat_gmsh, save_mesh_folder)

gmsh.finalize()