"""
loop_antenna_mesh.py

Recreates, with Gmsh, the planar structure used in the MATLAB/AToM script
characteristic_modes_AToM.m:

    pixelSize = 0.04;                  % [m] edge length of one pixel
    Mask = 2*ones(6, 12);              % 6 rows (y) x 12 columns (x) pixel canvas
    Mask(2:end-1, 2:end-1) = 0;        % hollow center -> loop shape

Mask(2:end-1, 2:end-1) = 0 zeroes out every pixel except the outer ring,
so the metal is a 1-pixel-wide rectangular loop (picture-frame shape):

    outer size  : 12 * pixelSize x 6 * pixelSize = 0.48 m x 0.24 m
    strip width : 1  * pixelSize               = 0.04 m
    inner (hole): 10 * pixelSize x 4 * pixelSize = 0.40 m x 0.16 m

Both rectangles are centered on the origin.

Requirements:
    pip install gmsh
"""

import gmsh

def create_loop_antenna_mesh(pixel_size, n_rows, n_cols, border_pixels):
    """
    Creates a 2D triangular mesh of a rectangular loop antenna using Gmsh.
    The mesh is saved to 'loop_antenna.msh' and the geometry to 'loop_antenna.step'.
    """
    occ = gmsh.model.occ

    outer_x = n_cols * pixel_size                            # 0.48 m
    outer_y = n_rows * pixel_size                            # 0.24 m
    inner_x = (n_cols - 2 * border_pixels) * pixel_size      # 0.40 m
    inner_y = (n_rows - 2 * border_pixels) * pixel_size      # 0.16 m

    # Outer and inner rectangles, both centered at the origin, in the XY plane (z = 0)
    outer = occ.addRectangle(-outer_x / 2, -outer_y / 2, 0, outer_x, outer_y)
    inner = occ.addRectangle(-inner_x / 2, -inner_y / 2, 0, inner_x, inner_y)

    # Boolean cut: outer - inner -> rectangular loop / picture-frame shape
    loop_dimtags, _ = occ.cut([(2, outer)], [(2, inner)])
    occ.synchronize()

    # Physical group, useful when the mesh is later imported into an EM solver
    gmsh.model.addPhysicalGroup(2, [tag for (_dim, tag) in loop_dimtags], name="LoopAntenna")
