import gmsh
from utils.gmsh_function import *

def antenna_ifa_meander(meander_x, meander_y, terminal_x, terminal_y, feed_x, feed_y, save_mesh_folder, mesh_name, mesh_size):
    gmsh.initialize()
    model_name  = "IFA_meander"
    gmsh.model.add(model_name)

    ifa_meander = rectangle_surface(meander_x, meander_y)
    # print("tag du ifa_meander =", ifa_meander)

    ifa_feed = rectangle_surface(feed_x, feed_y)
    # print("tag du ifa_meander =", ifa_meander)

    # Creation du terminal
    terminal = rectangle_surface(terminal_x, terminal_y)

    # Fusion du terminal et du meander
    antenna_ifa_meander, _ = gmsh.model.occ.fuse([(2, ifa_meander)], [(2, terminal), (2, ifa_feed)])

    # Synchronisation et sauvegarde
    gmsh.model.occ.synchronize()
    
    apply_mesh_size(mesh_size)

    # Afficher le modèle dans l’interface Gmsh
    gmsh.model.mesh.generate(2)

    # run()

    write(save_mesh_folder, mesh_name)

    gmsh.finalize()