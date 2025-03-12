import os
import gmsh

def plate_gmsh(longueur, hauteur, mesh_name, save_mesh_folder, mesh_size = 0.5):
    # Initialisation de Gmsh
    gmsh.initialize()
    gmsh.model.add("plate_gmsh")

    # Utilisation d'Open Cascade pour créer un rectangle (x, y, z, largeur, hauteur)
    plate = gmsh.model.occ.addRectangle(0, 0, 0, longueur, hauteur)
    gmsh.model.occ.synchronize()

    if mesh_size == 0:
        raise ValueError("mesh_size should not be null")
    elif mesh_size > 1:
        raise ValueError("mesh_size should not be greater than 1")
    elif longueur >= hauteur:
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size * longueur)
    else:
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size * hauteur)


    # Génération du maillage 2D sur la surface
    gmsh.model.mesh.generate(2)

    # Vérifier si le dossier existe, sinon le créer
    if not os.path.exists(save_mesh_folder):
        os.makedirs(save_mesh_folder)
    
    # Définir le chemin où enregistrer le fichier
    output_path = os.path.join(save_mesh_folder, mesh_name)

    gmsh.write(output_path)
    print(f"{mesh_name} saved in {output_path} successfully")

    # Fermeture de Gmsh
    gmsh.finalize()

    return output_path