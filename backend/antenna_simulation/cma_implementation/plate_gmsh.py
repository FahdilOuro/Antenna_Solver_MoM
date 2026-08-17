import gmsh

def construct_plate_geometry(Long, larg):
    gmsh.model.add("plate")

    # Define the points of the plate
    p1 = gmsh.model.geo.addPoint(0, 0, 0)
    p2 = gmsh.model.geo.addPoint(Long, 0, 0)
    p3 = gmsh.model.geo.addPoint(Long, larg, 0)
    p4 = gmsh.model.geo.addPoint(0, larg, 0)

    # Create lines between the points
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    # Create a curve loop and a surface
    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    s = gmsh.model.geo.addPlaneSurface([cl])

    # Synchronize the model
    gmsh.model.geo.synchronize()

    return s
