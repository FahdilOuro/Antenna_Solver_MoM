import os
import gmsh

def plate_gmsh(longueur, hauteur, mesh_name, save_mesh_folder, mesh_size = 0.5):
    # Initialisation de Gmsh
    gmsh.initialize()
    gmsh.model.add("plate_gmsh")

    # Utilisation d'Open Cascade pour créer un rectangle (x, y, z, largeur, hauteur)
    plate = gmsh.model.occ.addRectangle(0, 0, 0, longueur, hauteur)
    gmsh.model.occ.synchronize()

    Automatic = 2
    gmsh.option.setNumber("Mesh.Algorithm", Automatic)   # To set The "Automatic" algorithm / Change if necessary

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

def plate_2_gmsh(longueur, hauteur, mesh_name, save_mesh_folder, refinement_order=4):
    # Initialisation de Gmsh
    gmsh.initialize()
    gmsh.model.add("plate_gmsh")

    # Utilisation d'Open Cascade pour créer un rectangle (x, y, z, largeur, hauteur)
    plate = gmsh.model.occ.addRectangle(0, 0, 0, longueur, hauteur)
    gmsh.model.occ.synchronize()

    Automatic = 2
    gmsh.option.setNumber("Mesh.Algorithm", Automatic)   # To set The "Automatic" algorithm / Change if necessary

    if longueur >= hauteur:
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), longueur)
    else:
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), hauteur)

    # Génération du maillage 2D sur la surface
    gmsh.model.mesh.generate(2)

    
    if refinement_order <= 0:
        raise ValueError("refinement_order should not be less or equal to zero")
    elif refinement_order > 4:
        print("refinement_order greater than 6 give an hyper meshing... prefer to be equal to 4")
        refinement_order = 4
    
    # réaffinement en 3 étapes
    for i in range(refinement_order):
        gmsh.model.mesh.refine()
        gmsh.model.mesh.optimize()

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

def slot_gmsh(longueur, hauteur, mesh_name, save_mesh_folder, refinement_order=4):
    # Initialisation de Gmsh
    gmsh.initialize()
    gmsh.model.add("plate_gmsh")

    # Utilisation d'Open Cascade pour créer un rectangle (x, y, z, largeur, hauteur)
    plate = gmsh.model.occ.addRectangle(-longueur/2, longueur/2, 0, longueur, hauteur)
    gmsh.model.occ.synchronize()

    Automatic = 2
    gmsh.option.setNumber("Mesh.Algorithm", Automatic)   # To set The "Automatic" algorithm / Change if necessary

    if longueur >= hauteur:
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), longueur)
    else:
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), hauteur)

    # Génération du maillage 2D sur la surface
    gmsh.model.mesh.generate(2)

    
    if refinement_order <= 0:
        raise ValueError("refinement_order should not be less or equal to zero")
    elif refinement_order > 4:
        print("refinement_order greater than 6 give an hyper meshing... prefer to be equal to 4")
        refinement_order = 4
    
    # réaffinement en 3 étapes
    for i in range(refinement_order):
        gmsh.model.mesh.refine()
        gmsh.model.mesh.optimize()

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