import os
from types import SimpleNamespace

def setup_gmsh_file_paths(name, gmsh_folder_path='data/gmsh_files/', mat_folder_path='data/antennas_mesh/'):
    """
    Ensures the target directory exists and constructs the full paths 
    for the mesh (.msh) and geometry (.brep) files.
    
    Args:
        name (str): The base name for the files (e.g., "plate").
        gmsh_folder_path (str): The directory where files will be stored.
        
    Returns:
        tuple: (plate_msh_gmsh, plate_geo_gmsh)
    """
    
    # 1. Define filenames with specific extensions
    mesh_name = f"{name}.msh"
    geo_name = f"{name}.brep"
    mat_name = f"{name}.mat"

    # 2. Check if the destination folder exists, if not, create it
    if not os.path.exists(gmsh_folder_path):
        os.makedirs(gmsh_folder_path)
        print(f"Directory created: {gmsh_folder_path}")

    # 3. Construct the absolute or relative paths for the files
    file_msh_path = os.path.join(gmsh_folder_path, mesh_name)
    file_geo_path = os.path.join(gmsh_folder_path, geo_name)
    file_mat_path = os.path.join(gmsh_folder_path, mat_name)

    # Store paths in a namespace for dot notation access
    paths = SimpleNamespace(
        msh = file_msh_path,
        geo = file_geo_path,
        mat = file_mat_path,
    )

    return paths