import os
import gmsh

def plate_gmsh(longueur, hauteur, mesh_name, save_mesh_folder, mesh_size=0.5):
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

    if refinement_order < 0:
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

def slot_gmsh(longueur, hauteur, slot_longueur, slot_largeur, espacement, mesh_name, save_mesh_folder, mesh_size=0.5):
    # Initialisation de Gmsh
    gmsh.initialize()
    gmsh.model.add("plate_gmsh")

    # Utilisation d'Open Cascade pour créer un rectangle (x, y, z, largeur, hauteur)
    plate = gmsh.model.occ.addRectangle(-longueur/2, -hauteur/2, 0, longueur, hauteur)
    gmsh.model.occ.synchronize()

    # Création des deux fentes alignées horizontalement
    slot_1 = gmsh.model.occ.addRectangle(-slot_longueur - espacement/2, -slot_largeur/2, 0, slot_longueur, slot_largeur)
    slot_2 = gmsh.model.occ.addRectangle(espacement/2, -slot_largeur/2, 0, slot_longueur, slot_largeur)
    gmsh.model.occ.synchronize()

    # Obtention du slot antenna
    slot, _ = gmsh.model.occ.cut([(2, plate)], [(2, slot_1), (2, slot_2)])
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

def bowtie_gmsh(width, hight, width_finite, mesh_name, save_mesh_folder, mesh_size=0.5):
    # Initialisation de Gmsh
    gmsh.initialize()
    gmsh.model.add("bowtie_antenna")

    # Définition des points
    p0 = gmsh.model.occ.addPoint(-width/2, -hight/2, 0)
    p1 = gmsh.model.occ.addPoint(-width_finite/2, 0, 0)
    p2 = gmsh.model.occ.addPoint(-width/2, hight/2, 0)
    p3 = gmsh.model.occ.addPoint(width/2, hight/2, 0)
    p4 = gmsh.model.occ.addPoint(width_finite/2, 0, 0)
    p5 = gmsh.model.occ.addPoint(width/2, -hight/2, 0)

    # Création des segments du contour
    l1 = gmsh.model.occ.addLine(p0, p1)
    l2 = gmsh.model.occ.addLine(p1, p2)
    l3 = gmsh.model.occ.addLine(p2, p3)
    l4 = gmsh.model.occ.addLine(p3, p4)
    l5 = gmsh.model.occ.addLine(p4, p5)
    l6 = gmsh.model.occ.addLine(p5, p0)

    # Création d'un wire (contour fermé)
    wire = gmsh.model.occ.addWire([l1, l2, l3, l4, l5, l6])

    # Création de la surface
    surface = gmsh.model.occ.addPlaneSurface([wire])
    gmsh.model.occ.synchronize()

    Automatic = 2
    gmsh.option.setNumber("Mesh.Algorithm", Automatic)   # To set The "Automatic" algorithm / Change if necessary

    if mesh_size == 0:
        raise ValueError("mesh_size should not be null")
    elif mesh_size > 1:
        raise ValueError("mesh_size should not be greater than 1")
    elif width >= hight:
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size * width)
    else:
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size * hight)

    # Génération du maillage
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

def ifa_gmsh(Lenght, Hight, Small_Lenght, Small_Hight, gap_F, Tight_F, mesh_name, save_mesh_folder, mesh_size):
    # Initialisation de Gmsh
    gmsh.initialize()
    gmsh.model.add("IFA_antenna")

    # Création du rectangle de base
    base = gmsh.model.occ.addRectangle(0, 0, 0, Lenght, Hight)

    # Création du "F" inversé
    # Barre 1 verticale du "F"
    f_vertical_1 = gmsh.model.occ.addRectangle(0, Hight, 0, Small_Lenght, Small_Hight)
    # Barre 1 verticale du "F"
    f_vertical_2 = gmsh.model.occ.addRectangle(gap_F, Hight, 0, Small_Lenght, Small_Hight)

    # Barre horizontale supérieure du "F"
    f_top = gmsh.model.occ.addRectangle(0, Hight + Small_Hight - Small_Lenght, 0, Lenght, Small_Lenght)

    # Fusion des formes
    antenna, _ = gmsh.model.occ.fuse([(2, base)], [(2, f_vertical_1), (2, f_vertical_2), (2, f_top)])
    gmsh.model.occ.synchronize()

    Automatic = 2
    gmsh.option.setNumber("Mesh.Algorithm", Automatic)   # To set The "Automatic" algorithm / Change if necessary

    if mesh_size == 0:
        raise ValueError("mesh_size should not be null")
    elif mesh_size > 1:
        raise ValueError("mesh_size should not be greater than 1")
    elif Lenght >= Hight:
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size * Lenght)
    else:
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size * Hight)

    # Génération du maillage
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

def strip_gmsh(Longueur, largeur, mesh_name, save_mesh_folder, mesh_size):
    # Initialisation de Gmsh
    gmsh.initialize()
    gmsh.model.add("Strip_Antenna")

    # Création des points
    p0 = gmsh.model.occ.addPoint(0, 0, 0)
    p1 = gmsh.model.occ.addPoint(Longueur, 0, 0)
    p2 = gmsh.model.occ.addPoint(Longueur, largeur, 0)
    p3 = gmsh.model.occ.addPoint(0, largeur, 0)

    # Création des segments
    l1 = gmsh.model.occ.addLine(p0, p1)
    l2 = gmsh.model.occ.addLine(p1, p2)
    l3 = gmsh.model.occ.addLine(p2, p3)
    l4 = gmsh.model.occ.addLine(p3, p0)

    # Création de la surface
    wire = gmsh.model.occ.addWire([l1, l2, l3, l4])  # Contour fermé
    strip_antenna = gmsh.model.occ.addPlaneSurface([wire])  # Surface
    gmsh.model.occ.synchronize()

    Automatic = 2
    gmsh.option.setNumber("Mesh.Algorithm", Automatic)   # To set The "Automatic" algorithm / Change if necessary

    if mesh_size == 0:
        raise ValueError("mesh_size should not be null")
    elif mesh_size > 1:
        raise ValueError("mesh_size should not be greater than 1")
    elif Longueur >= largeur:
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size * Longueur)
    else:
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size * largeur)

    # Génération du maillage
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
