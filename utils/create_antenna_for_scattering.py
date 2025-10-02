import os
import gmsh

def plate_gmsh(longueur, hauteur, mesh_name, save_mesh_folder, mesh_size=0.5):
    # Initialize Gmsh
    gmsh.initialize()
    gmsh.model.add("plate_gmsh")

    # Use Open Cascade to create a rectangle (x, y, z, width, height)
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

    # Generate 2D mesh on the surface
    gmsh.model.mesh.generate(2)

    # Check if folder exists, if not create it
    if not os.path.exists(save_mesh_folder):
        os.makedirs(save_mesh_folder)
    
    # Define the path to save the file
    output_path = os.path.join(save_mesh_folder, mesh_name)

    gmsh.write(output_path)
    print(f"{mesh_name} saved in {output_path} successfully")

    # Close Gmsh
    gmsh.finalize()

    return output_path

def plate_2_gmsh(longueur, hauteur, mesh_name, save_mesh_folder, refinement_order=4):
    # Initialize Gmsh
    gmsh.initialize()
    gmsh.model.add("plate_gmsh")

    # Use Open Cascade to create a rectangle (x, y, z, width, height)
    plate = gmsh.model.occ.addRectangle(0, 0, 0, longueur, hauteur)
    gmsh.model.occ.synchronize()

    Automatic = 2
    gmsh.option.setNumber("Mesh.Algorithm", Automatic)   # To set the "Automatic" algorithm / Change if necessary

    if longueur >= hauteur:
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), longueur)
    else:
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), hauteur)

    # Generate 2D mesh on the surface
    gmsh.model.mesh.generate(2)

    if refinement_order < 0:
        raise ValueError("refinement_order should not be less or equal to zero")
    elif refinement_order > 4:
        print("refinement_order greater than 6 gives hyper meshing... prefer to set it to 4")
        refinement_order = 4
    
    # Refinement in 3 steps
    for i in range(refinement_order):
        gmsh.model.mesh.refine()
        gmsh.model.mesh.optimize()

    # Check if folder exists, if not create it
    if not os.path.exists(save_mesh_folder):
        os.makedirs(save_mesh_folder)
    
    # Define the path to save the file
    output_path = os.path.join(save_mesh_folder, mesh_name)

    gmsh.write(output_path)
    print(f"{mesh_name} saved in {output_path} successfully")

    # Close Gmsh
    gmsh.finalize()

    return output_path

def slot_gmsh(longueur, hauteur, slot_longueur, slot_largeur, espacement, mesh_name, save_mesh_folder, mesh_size=0.5):
    # Initialize Gmsh
    gmsh.initialize()
    gmsh.model.add("plate_gmsh")

    # Use Open Cascade to create a rectangle (x, y, z, width, height)
    plate = gmsh.model.occ.addRectangle(-longueur/2, -hauteur/2, 0, longueur, hauteur)
    gmsh.model.occ.synchronize()

    # Create two horizontally aligned slots
    slot_1 = gmsh.model.occ.addRectangle(-slot_longueur - espacement/2, -slot_largeur/2, 0, slot_longueur, slot_largeur)
    slot_2 = gmsh.model.occ.addRectangle(espacement/2, -slot_largeur/2, 0, slot_longueur, slot_largeur)
    gmsh.model.occ.synchronize()

    # Obtain the slotted antenna
    slot, _ = gmsh.model.occ.cut([(2, plate)], [(2, slot_1), (2, slot_2)])
    gmsh.model.occ.synchronize()

    Automatic = 2
    gmsh.option.setNumber("Mesh.Algorithm", Automatic)   # To set the "Automatic" algorithm / Change if necessary

    if mesh_size == 0:
        raise ValueError("mesh_size should not be null")
    elif mesh_size > 1:
        raise ValueError("mesh_size should not be greater than 1")
    elif longueur >= hauteur:
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size * longueur)
    else:
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size * hauteur)

    # Generate 2D mesh on the surface
    gmsh.model.mesh.generate(2)

    # Check if folder exists, if not create it
    if not os.path.exists(save_mesh_folder):
        os.makedirs(save_mesh_folder)
    
    # Define the path to save the file
    output_path = os.path.join(save_mesh_folder, mesh_name)

    gmsh.write(output_path)
    print(f"{mesh_name} saved in {output_path} successfully")

    # Close Gmsh
    gmsh.finalize()

    return output_path

def bowtie_gmsh(width, hight, width_finite, mesh_name, save_mesh_folder, mesh_size=0.5):
    # Initialize Gmsh
    gmsh.initialize()
    gmsh.model.add("bowtie_antenna")

    # Define points
    p0 = gmsh.model.occ.addPoint(-width/2, -hight/2, 0)
    p1 = gmsh.model.occ.addPoint(-width_finite/2, 0, 0)
    p2 = gmsh.model.occ.addPoint(-width/2, hight/2, 0)
    p3 = gmsh.model.occ.addPoint(width/2, hight/2, 0)
    p4 = gmsh.model.occ.addPoint(width_finite/2, 0, 0)
    p5 = gmsh.model.occ.addPoint(width/2, -hight/2, 0)

    # Create segments of the contour
    l1 = gmsh.model.occ.addLine(p0, p1)
    l2 = gmsh.model.occ.addLine(p1, p2)
    l3 = gmsh.model.occ.addLine(p2, p3)
    l4 = gmsh.model.occ.addLine(p3, p4)
    l5 = gmsh.model.occ.addLine(p4, p5)
    l6 = gmsh.model.occ.addLine(p5, p0)

    # Create a wire (closed contour)
    wire = gmsh.model.occ.addWire([l1, l2, l3, l4, l5, l6])

    # Create the surface
    surface = gmsh.model.occ.addPlaneSurface([wire])
    gmsh.model.occ.synchronize()

    Automatic = 2
    gmsh.option.setNumber("Mesh.Algorithm", Automatic)   # To set the "Automatic" algorithm / Change if necessary

    if mesh_size == 0:
        raise ValueError("mesh_size should not be null")
    elif mesh_size > 1:
        raise ValueError("mesh_size should not be greater than 1")
    elif width >= hight:
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size * width)
    else:
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size * hight)

    # Generate the mesh
    gmsh.model.mesh.generate(2)

    # Check if folder exists, if not create it
    if not os.path.exists(save_mesh_folder):
        os.makedirs(save_mesh_folder)
    
    # Define the path to save the file
    output_path = os.path.join(save_mesh_folder, mesh_name)

    gmsh.write(output_path)
    print(f"{mesh_name} saved in {output_path} successfully")

    # Close Gmsh
    gmsh.finalize()

    return output_path

def ifa_gmsh(Lenght, Hight, Small_Lenght, Small_Hight, gap_F, mesh_name, save_mesh_folder, mesh_size):
    # Initialize Gmsh
    gmsh.initialize()
    gmsh.model.add("IFA_antenna")

    # Create the base rectangle
    base = gmsh.model.occ.addRectangle(0, 0, 0, Lenght, Hight)

    # Create the inverted "F"
    # Vertical bar 1 of the "F"
    f_vertical_1 = gmsh.model.occ.addRectangle(0, Hight, 0, Small_Lenght, Small_Hight)
    # Vertical bar 2 of the "F"
    f_vertical_2 = gmsh.model.occ.addRectangle(gap_F, Hight, 0, Small_Lenght, Small_Hight)

    # Top horizontal bar of the "F"
    f_top = gmsh.model.occ.addRectangle(0, Hight + Small_Hight - Small_Lenght, 0, Lenght, Small_Lenght)

    # Fuse the shapes
    antenna, _ = gmsh.model.occ.fuse([(2, base)], [(2, f_vertical_1), (2, f_vertical_2), (2, f_top)])
    gmsh.model.occ.synchronize()

    Automatic = 2
    gmsh.option.setNumber("Mesh.Algorithm", Automatic)   # To set the "Automatic" algorithm / Change if necessary

    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)

    # Generate the mesh
    gmsh.model.mesh.generate(2)

    # Check if folder exists, if not create it
    if not os.path.exists(save_mesh_folder):
        os.makedirs(save_mesh_folder)
    
    # Define the path to save the file
    output_path = os.path.join(save_mesh_folder, mesh_name)

    gmsh.write(output_path)
    print(f"{mesh_name} saved in {output_path} successfully")

    # Close Gmsh
    gmsh.finalize()

    return output_path

def strip_gmsh(Longueur, largeur, mesh_name, save_mesh_folder, mesh_size):
    # Initialize Gmsh
    gmsh.initialize()
    gmsh.model.add("Strip_Antenna")

    # Create points
    p0 = gmsh.model.occ.addPoint(0, 0, 0)
    p1 = gmsh.model.occ.addPoint(Longueur, 0, 0)
    p2 = gmsh.model.occ.addPoint(Longueur, largeur, 0)
    p3 = gmsh.model.occ.addPoint(0, largeur, 0)

    # Create segments
    l1 = gmsh.model.occ.addLine(p0, p1)
    l2 = gmsh.model.occ.addLine(p1, p2)
    l3 = gmsh.model.occ.addLine(p2, p3)
    l4 = gmsh.model.occ.addLine(p3, p0)

    # Create surface
    wire = gmsh.model.occ.addWire([l1, l2, l3, l4])  # Closed contour
    strip_antenna = gmsh.model.occ.addPlaneSurface([wire])  # Surface
    gmsh.model.occ.synchronize()

    Automatic = 2
    gmsh.option.setNumber("Mesh.Algorithm", Automatic)   # To set the "Automatic" algorithm / Change if necessary

    if mesh_size == 0:
        raise ValueError("mesh_size should not be null")
    elif mesh_size > 1:
        raise ValueError("mesh_size should not be greater than 1")
    elif Longueur >= largeur:
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size * Longueur)
    else:
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size * largeur)

    # Generate the mesh
    gmsh.model.mesh.generate(2)
    
    # Check if folder exists, if not create it
    if not os.path.exists(save_mesh_folder):
        os.makedirs(save_mesh_folder)
    
    # Define the path to save the file
    output_path = os.path.join(save_mesh_folder, mesh_name)

    gmsh.write(output_path)
    print(f"{mesh_name} saved in {output_path} successfully")

    # Close Gmsh
    gmsh.finalize()

    return output_path