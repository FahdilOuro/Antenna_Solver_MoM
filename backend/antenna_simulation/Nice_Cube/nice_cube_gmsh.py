import gmsh


def create_nice_cube_geometry(C, space_btw_cubesat_and_solar_panel, l_c_s):
    # list of points for the cubeSat
    A_point_1 = gmsh.model.occ.addPoint(-C/2, -C/2, -3*C)       # point 1
    A_point_2 = gmsh.model.occ.addPoint(C/2, -C/2, -3*C)        # point 2
    A_point_3 = gmsh.model.occ.addPoint(C/2, C/2, -3*C)         # point 3
    A_point_4 = gmsh.model.occ.addPoint(-C/2, C/2, -3*C)        # point 4

    B_point_1 = gmsh.model.occ.addPoint(-C/2, -C/2, 0)          # point 5
    B_point_2 = gmsh.model.occ.addPoint(C/2, -C/2, 0)           # point 6
    B_point_3 = gmsh.model.occ.addPoint(C/2, C/2, 0)            # point 7
    B_point_4 = gmsh.model.occ.addPoint(-C/2, C/2, 0)           # point 8

    # creating lines for the cubeSat
    line_1 = gmsh.model.occ.addLine(A_point_1, A_point_2)
    line_2 = gmsh.model.occ.addLine(A_point_2, A_point_3)
    line_3 = gmsh.model.occ.addLine(A_point_3, A_point_4)
    line_4 = gmsh.model.occ.addLine(A_point_4, A_point_1)

    line_5 = gmsh.model.occ.addLine(B_point_1, B_point_2)
    line_6 = gmsh.model.occ.addLine(B_point_2, B_point_3)
    line_7 = gmsh.model.occ.addLine(B_point_3, B_point_4)
    line_8 = gmsh.model.occ.addLine(B_point_4, B_point_1)

    line_9 = gmsh.model.occ.addLine(A_point_1, B_point_1)
    line_10 = gmsh.model.occ.addLine(A_point_2, B_point_2)
    line_11 = gmsh.model.occ.addLine(A_point_3, B_point_3)
    line_12 = gmsh.model.occ.addLine(A_point_4, B_point_4)

    # creating curve loops for the cubeSat
    curve_loop_1 = gmsh.model.occ.addCurveLoop([line_1, line_2, line_3, line_4])
    curve_loop_2 = gmsh.model.occ.addCurveLoop([line_5, line_6, line_7, line_8])
    curve_loop_3 = gmsh.model.occ.addCurveLoop([line_1, line_10, -line_5, -line_9])
    curve_loop_4 = gmsh.model.occ.addCurveLoop([line_2, line_11, -line_6, -line_10])
    curve_loop_5 = gmsh.model.occ.addCurveLoop([line_3, line_12, -line_7, -line_11])
    curve_loop_6 = gmsh.model.occ.addCurveLoop([line_4, line_9, -line_8, -line_12])

    # creating surfaces for the cubeSat
    surface_1 = gmsh.model.occ.addPlaneSurface([curve_loop_1])
    surface_2 = gmsh.model.occ.addPlaneSurface([curve_loop_2])
    surface_3 = gmsh.model.occ.addPlaneSurface([curve_loop_3])
    surface_4 = gmsh.model.occ.addPlaneSurface([curve_loop_4])
    surface_5 = gmsh.model.occ.addPlaneSurface([curve_loop_5])
    surface_6 = gmsh.model.occ.addPlaneSurface([curve_loop_6])

    # point list for the solar panel
    solar_panel_A_point_1 = gmsh.model.occ.addPoint(C/2 + space_btw_cubesat_and_solar_panel, C/2, 0)            # point 9
    solar_panel_A_point_2 = gmsh.model.occ.addPoint(C/2 + space_btw_cubesat_and_solar_panel, -C/2, 0)           # point 10
    solar_panel_A_point_3 = gmsh.model.occ.addPoint(C/2 + space_btw_cubesat_and_solar_panel + 3*C, -C/2, 0)     # point 11
    solar_panel_A_point_4 = gmsh.model.occ.addPoint(C/2 + space_btw_cubesat_and_solar_panel + 3*C, C/2, 0)      # point 12

    solar_panel_B_point_1 = gmsh.model.occ.addPoint(C/2, C/2 + space_btw_cubesat_and_solar_panel, 0)            # point 13
    solar_panel_B_point_2 = gmsh.model.occ.addPoint(C/2, C/2 + space_btw_cubesat_and_solar_panel + 3*C, 0)      # point 14
    solar_panel_B_point_3 = gmsh.model.occ.addPoint(-C/2, C/2 + space_btw_cubesat_and_solar_panel + 3*C, 0)     # point 15
    solar_panel_B_point_4 = gmsh.model.occ.addPoint(-C/2, C/2 + space_btw_cubesat_and_solar_panel, 0)           # point 16

    solar_panel_C_point_1 = gmsh.model.occ.addPoint(-C/2 - space_btw_cubesat_and_solar_panel, C/2, 0)            # point 17
    solar_panel_C_point_2 = gmsh.model.occ.addPoint(-C/2 - space_btw_cubesat_and_solar_panel, -C/2, 0)           # point 18
    solar_panel_C_point_3 = gmsh.model.occ.addPoint(-C/2 - space_btw_cubesat_and_solar_panel - 3*C, -C/2, 0)     # point 19
    solar_panel_C_point_4 = gmsh.model.occ.addPoint(-C/2 - space_btw_cubesat_and_solar_panel - 3*C, C/2, 0)      # point 20

    solar_panel_D_point_1 = gmsh.model.occ.addPoint(-C/2, -C/2 - space_btw_cubesat_and_solar_panel, 0)           # point 21
    solar_panel_D_point_2 = gmsh.model.occ.addPoint(-C/2, -C/2 - space_btw_cubesat_and_solar_panel - 3*C, 0)     # point 22
    solar_panel_D_point_3 = gmsh.model.occ.addPoint(C/2, -C/2 - space_btw_cubesat_and_solar_panel - 3*C, 0)      # point 23
    solar_panel_D_point_4 = gmsh.model.occ.addPoint(C/2, -C/2 - space_btw_cubesat_and_solar_panel, 0)            # point 24

    # creating lines for the solar panel
    solar_panel_A_line_1 = gmsh.model.occ.addLine(solar_panel_A_point_1, solar_panel_A_point_2)
    solar_panel_A_line_2 = gmsh.model.occ.addLine(solar_panel_A_point_2, solar_panel_A_point_3)
    solar_panel_A_line_3 = gmsh.model.occ.addLine(solar_panel_A_point_3, solar_panel_A_point_4)
    solar_panel_A_line_4 = gmsh.model.occ.addLine(solar_panel_A_point_4, solar_panel_A_point_1)

    solar_panel_B_line_1 = gmsh.model.occ.addLine(solar_panel_B_point_1, solar_panel_B_point_2)
    solar_panel_B_line_2 = gmsh.model.occ.addLine(solar_panel_B_point_2, solar_panel_B_point_3)
    solar_panel_B_line_3 = gmsh.model.occ.addLine(solar_panel_B_point_3, solar_panel_B_point_4)
    solar_panel_B_line_4 = gmsh.model.occ.addLine(solar_panel_B_point_4, solar_panel_B_point_1)

    solar_panel_C_line_1 = gmsh.model.occ.addLine(solar_panel_C_point_1, solar_panel_C_point_2)
    solar_panel_C_line_2 = gmsh.model.occ.addLine(solar_panel_C_point_2, solar_panel_C_point_3)
    solar_panel_C_line_3 = gmsh.model.occ.addLine(solar_panel_C_point_3, solar_panel_C_point_4)
    solar_panel_C_line_4 = gmsh.model.occ.addLine(solar_panel_C_point_4, solar_panel_C_point_1)

    solar_panel_D_line_1 = gmsh.model.occ.addLine(solar_panel_D_point_1, solar_panel_D_point_2)
    solar_panel_D_line_2 = gmsh.model.occ.addLine(solar_panel_D_point_2, solar_panel_D_point_3)
    solar_panel_D_line_3 = gmsh.model.occ.addLine(solar_panel_D_point_3, solar_panel_D_point_4)
    solar_panel_D_line_4 = gmsh.model.occ.addLine(solar_panel_D_point_4, solar_panel_D_point_1)

    # creating curve loop for the solar panel
    solar_panel_A_curve_loop = gmsh.model.occ.addCurveLoop([solar_panel_A_line_1, solar_panel_A_line_2, solar_panel_A_line_3, solar_panel_A_line_4])
    solar_panel_B_curve_loop = gmsh.model.occ.addCurveLoop([solar_panel_B_line_1, solar_panel_B_line_2, solar_panel_B_line_3, solar_panel_B_line_4])
    solar_panel_C_curve_loop = gmsh.model.occ.addCurveLoop([solar_panel_C_line_1, solar_panel_C_line_2, solar_panel_C_line_3, solar_panel_C_line_4])
    solar_panel_D_curve_loop = gmsh.model.occ.addCurveLoop([solar_panel_D_line_1, solar_panel_D_line_2, solar_panel_D_line_3, solar_panel_D_line_4])

    # creating surface for the solar panel
    solar_panel_A_surface = gmsh.model.occ.addPlaneSurface([solar_panel_A_curve_loop])
    solar_panel_B_surface = gmsh.model.occ.addPlaneSurface([solar_panel_B_curve_loop])
    solar_panel_C_surface = gmsh.model.occ.addPlaneSurface([solar_panel_C_curve_loop])
    solar_panel_D_surface = gmsh.model.occ.addPlaneSurface([solar_panel_D_curve_loop])

    # Creating connection between the cubeSat and the solar panel with 2 small surfaces per face of the cubeSat
    connection_A_1_point_1 = gmsh.model.occ.addPoint(C/2, C/2 - l_c_s, 0)                                           # point 25
    connection_A_1_point_2 = gmsh.model.occ.addPoint(C/2 + space_btw_cubesat_and_solar_panel, C/2 - l_c_s, 0)       # point 26

    connection_A_2_point_1 = gmsh.model.occ.addPoint(C/2, -C/2 + l_c_s, 0)                                          # point 27
    connection_A_2_point_2 = gmsh.model.occ.addPoint(C/2 + space_btw_cubesat_and_solar_panel, -C/2 + l_c_s, 0)      # point 28

    connection_B_1_point_1 = gmsh.model.occ.addPoint(-C/2 + l_c_s, C/2, 0)                                          # point 29
    connection_B_1_point_2 = gmsh.model.occ.addPoint(-C/2 + l_c_s, C/2 + space_btw_cubesat_and_solar_panel, 0)      # point 30

    connection_B_2_point_1 = gmsh.model.occ.addPoint(C/2 - l_c_s, C/2, 0)                                           # point 31
    connection_B_2_point_2 = gmsh.model.occ.addPoint(C/2 - l_c_s, C/2 + space_btw_cubesat_and_solar_panel, 0)       # point 32
    
    connection_C_1_point_1 = gmsh.model.occ.addPoint(-C/2, C/2 - l_c_s, 0)                                          # point 33
    connection_C_1_point_2 = gmsh.model.occ.addPoint(-C/2 - space_btw_cubesat_and_solar_panel, C/2 - l_c_s, 0)      # point 34

    connection_C_2_point_1 = gmsh.model.occ.addPoint(-C/2, -C/2 + l_c_s, 0)                                         # point 35
    connection_C_2_point_2 = gmsh.model.occ.addPoint(-C/2 - space_btw_cubesat_and_solar_panel, -C/2 + l_c_s, 0)     # point 36

    connection_D_1_point_1 = gmsh.model.occ.addPoint(-C/2 + l_c_s, -C/2, 0)                                         # point 37
    connection_D_1_point_2 = gmsh.model.occ.addPoint(-C/2 + l_c_s, -C/2 - space_btw_cubesat_and_solar_panel, 0)     # point 38

    connection_D_2_point_1 = gmsh.model.occ.addPoint(C/2 - l_c_s, -C/2, 0)                                          # point 39
    connection_D_2_point_2 = gmsh.model.occ.addPoint(C/2 - l_c_s, -C/2 - space_btw_cubesat_and_solar_panel, 0)      # point 40

    connection_A_1_line_1 = gmsh.model.occ.addLine(B_point_3, connection_A_1_point_1)
    connection_A_1_line_2 = gmsh.model.occ.addLine(connection_A_1_point_1, connection_A_1_point_2)
    connection_A_1_line_3 = gmsh.model.occ.addLine(connection_A_1_point_2, solar_panel_A_point_1)
    connection_A_1_line_4 = gmsh.model.occ.addLine(solar_panel_A_point_1, B_point_3)

    connection_A_2_line_1 = gmsh.model.occ.addLine(B_point_2, connection_A_2_point_1)
    connection_A_2_line_2 = gmsh.model.occ.addLine(connection_A_2_point_1, connection_A_2_point_2)
    connection_A_2_line_3 = gmsh.model.occ.addLine(connection_A_2_point_2, solar_panel_A_point_2)
    connection_A_2_line_4 = gmsh.model.occ.addLine(solar_panel_A_point_2, B_point_2)

    connection_B_1_line_1 = gmsh.model.occ.addLine(B_point_4, connection_B_1_point_1)
    connection_B_1_line_2 = gmsh.model.occ.addLine(connection_B_1_point_1, connection_B_1_point_2)
    connection_B_1_line_3 = gmsh.model.occ.addLine(connection_B_1_point_2, solar_panel_B_point_4)
    connection_B_1_line_4 = gmsh.model.occ.addLine(solar_panel_B_point_4, B_point_4)

    connection_B_2_line_1 = gmsh.model.occ.addLine(B_point_3, solar_panel_B_point_1)
    connection_B_2_line_2 = gmsh.model.occ.addLine(solar_panel_B_point_1, connection_B_2_point_2)
    connection_B_2_line_3 = gmsh.model.occ.addLine(connection_B_2_point_2, connection_B_2_point_1)
    connection_B_2_line_4 = gmsh.model.occ.addLine(connection_B_2_point_1, B_point_3)

    connection_C_1_line_1 = gmsh.model.occ.addLine(B_point_4, connection_C_1_point_1)
    connection_C_1_line_2 = gmsh.model.occ.addLine(connection_C_1_point_1, connection_C_1_point_2)
    connection_C_1_line_3 = gmsh.model.occ.addLine(connection_C_1_point_2, solar_panel_C_point_1)
    connection_C_1_line_4 = gmsh.model.occ.addLine(solar_panel_C_point_1, B_point_4)

    connection_C_2_line_1 = gmsh.model.occ.addLine(B_point_1, connection_C_2_point_1)
    connection_C_2_line_2 = gmsh.model.occ.addLine(connection_C_2_point_1, connection_C_2_point_2)
    connection_C_2_line_3 = gmsh.model.occ.addLine(connection_C_2_point_2, solar_panel_C_point_2)
    connection_C_2_line_4 = gmsh.model.occ.addLine(solar_panel_C_point_2, B_point_1)

    connection_D_1_line_1 = gmsh.model.occ.addLine(B_point_1, connection_D_1_point_1)
    connection_D_1_line_2 = gmsh.model.occ.addLine(connection_D_1_point_1, connection_D_1_point_2)
    connection_D_1_line_3 = gmsh.model.occ.addLine(connection_D_1_point_2, solar_panel_D_point_1)
    connection_D_1_line_4 = gmsh.model.occ.addLine(solar_panel_D_point_1, B_point_1)

    connection_D_2_line_1 = gmsh.model.occ.addLine(B_point_2, connection_D_2_point_1)
    connection_D_2_line_2 = gmsh.model.occ.addLine(connection_D_2_point_1, connection_D_2_point_2)
    connection_D_2_line_3 = gmsh.model.occ.addLine(connection_D_2_point_2, solar_panel_D_point_4)
    connection_D_2_line_4 = gmsh.model.occ.addLine(solar_panel_D_point_4, B_point_2)

    connection_A_1_curve_loop = gmsh.model.occ.addCurveLoop([connection_A_1_line_1, connection_A_1_line_2, connection_A_1_line_3, connection_A_1_line_4])
    connection_A_1_surface = gmsh.model.occ.addPlaneSurface([connection_A_1_curve_loop])

    connection_A_2_curve_loop = gmsh.model.occ.addCurveLoop([connection_A_2_line_1, connection_A_2_line_2, connection_A_2_line_3, connection_A_2_line_4])
    connection_A_2_surface = gmsh.model.occ.addPlaneSurface([connection_A_2_curve_loop])

    connection_B_1_curve_loop = gmsh.model.occ.addCurveLoop([connection_B_1_line_1, connection_B_1_line_2, connection_B_1_line_3, connection_B_1_line_4])
    connection_B_1_surface = gmsh.model.occ.addPlaneSurface([connection_B_1_curve_loop])

    connection_B_2_curve_loop = gmsh.model.occ.addCurveLoop([connection_B_2_line_1, connection_B_2_line_2, connection_B_2_line_3, connection_B_2_line_4])
    connection_B_2_surface = gmsh.model.occ.addPlaneSurface([connection_B_2_curve_loop])

    connection_C_1_curve_loop = gmsh.model.occ.addCurveLoop([connection_C_1_line_1, connection_C_1_line_2, connection_C_1_line_3, connection_C_1_line_4])
    connection_C_1_surface = gmsh.model.occ.addPlaneSurface([connection_C_1_curve_loop])

    connection_C_2_curve_loop = gmsh.model.occ.addCurveLoop([connection_C_2_line_1, connection_C_2_line_2, connection_C_2_line_3, connection_C_2_line_4])
    connection_C_2_surface = gmsh.model.occ.addPlaneSurface([connection_C_2_curve_loop])

    connection_D_1_curve_loop = gmsh.model.occ.addCurveLoop([connection_D_1_line_1, connection_D_1_line_2, connection_D_1_line_3, connection_D_1_line_4])
    connection_D_1_surface = gmsh.model.occ.addPlaneSurface([connection_D_1_curve_loop])

    connection_D_2_curve_loop = gmsh.model.occ.addCurveLoop([connection_D_2_line_1, connection_D_2_line_2, connection_D_2_line_3, connection_D_2_line_4])
    connection_D_2_surface = gmsh.model.occ.addPlaneSurface([connection_D_2_curve_loop])

    # Collect all surfaces into a list of tuples (dimension, tag)
    surfaces_to_fuse = [
        surface_1, surface_2, surface_3, surface_4, surface_5, surface_6, 
        solar_panel_A_surface, solar_panel_B_surface, solar_panel_C_surface, solar_panel_D_surface, 
        connection_A_1_surface, connection_A_2_surface, 
        connection_B_1_surface, connection_B_2_surface, 
        connection_C_1_surface, connection_C_2_surface, 
        connection_D_1_surface, connection_D_2_surface
    ]

    all_input_entities = [(2, s) for s in surfaces_to_fuse]

    # Synchronize to ensure OCC kernel knows all entities
    gmsh.model.occ.synchronize()

    # Use Fragment to merge overlapping boundaries
    # This is much more reliable than iterative 'fuse' for complex surface junctions
    new_entities, _ = gmsh.model.occ.fragment(all_input_entities, [])

    # Remove any duplicate points or lines that might remain
    gmsh.model.occ.removeAllDuplicates()

    # Synchronize again to apply changes
    gmsh.model.occ.synchronize()

    # Create a Physical Surface group
    # This is vital for solvers to see the CubeSat as a single assembly
    fused_surface_tags = [tag for dim, tag in new_entities if dim == 2]
    gmsh.model.addPhysicalGroup(2, fused_surface_tags, name="CubeSat_Shell")

def create_nice_cube_geometry_v2(C, gap, bracket_w):
    # This function can be implemented similarly to the first one, but with a different approach to creating the geometry.
    # For example, we could create the cubeSat and solar panel as separate volumes and then use boolean operations to combine them.
    height = 3 * C
    panel_width = C
    panel_length = 3 * C

    # --- 1. Main CubeSat Body (3U) ---
    # We create a box and then keep only its boundary surfaces
    body_vol = gmsh.model.occ.addBox(-C/2, -C/2, -height, C, C, height)
    gmsh.model.occ.synchronize()
    
    # Get the surfaces of the body and remove the volume to keep it as a shell
    body_surfaces = gmsh.model.occ.getEntities(2)
    
    # --- 2. Solar Panels and Connections Loop ---
    all_panels = []
    all_connections = []

    # Basic shape for one panel (on the X+ side)
    # We define it once and then rotate it 4 times
    for i in range(4):
        # Create a Solar Panel surface
        panel = gmsh.model.occ.addRectangle(C/2 + gap, -C/2, 0, panel_length, panel_width)
        
        # Create two small connection brackets
        # Bracket 1 (Top-ish)
        conn1 = gmsh.model.occ.addRectangle(C/2, C/2 - bracket_w, 0, gap, bracket_w)
        # Bracket 2 (Bottom-ish)
        conn2 = gmsh.model.occ.addRectangle(C/2, -C/2, 0, gap, bracket_w)
        
        # Rotate them around the Z-axis by 90 degrees * i
        angle = i * (1.570796) # 90 degrees in radians
        gmsh.model.occ.rotate([(2, panel), (2, conn1), (2, conn2)], 0, 0, 0, 0, 0, 1, angle)
        
        all_panels.append(panel)
        all_connections.extend([conn1, conn2])

    # --- 3. Boolean Fragment (Conformal Mesh) ---
    # This replaces 'fuse' and 'removeDuplicates'. 
    # It ensures the mesh is shared at the junctions.
    all_surface_tags = [s[1] for s in body_surfaces] + all_panels + all_connections
    input_entities = [(2, tag) for tag in all_surface_tags]
    
    # Fragment creates a clean topology where surfaces meet
    dim_tags, _ = gmsh.model.occ.fragment(input_entities, [])
    gmsh.model.occ.synchronize()

    # --- 4. Physical Groups ---
    # Grouping surfaces for the solver
    fused_tags = [tag for dim, tag in dim_tags if dim == 2]
    gmsh.model.addPhysicalGroup(2, fused_tags, name="CubeSat_Shell")