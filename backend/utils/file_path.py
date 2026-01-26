import os
from types import SimpleNamespace

def setup_save_file_paths(name, gmsh_folder_path='data/gmsh_files/', mat_folder_path='data/antennas_mesh/'):
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
    remesh_name = f"{name + '_remesh'}.msh"
    geo_name = f"{name}.brep"
    mat_name = f"{name}.mat"

    # 2. Check if the destination folders exist, if not, create them
    if not os.path.exists(gmsh_folder_path):
        os.makedirs(gmsh_folder_path)
        print(f"Directory created: {gmsh_folder_path}")
    if not os.path.exists(mat_folder_path):
        os.makedirs(mat_folder_path)
        print(f"Directory created: {mat_folder_path}")
        

    # 3. Construct the absolute or relative paths for the files
    file_msh_path = os.path.join(gmsh_folder_path, mesh_name)
    file_remsh_path = os.path.join(gmsh_folder_path, remesh_name)
    file_geo_path = os.path.join(gmsh_folder_path, geo_name)
    file_mat_path = os.path.join(mat_folder_path, mat_name)

    # Store paths in a namespace for dot notation access
    paths = SimpleNamespace(
        msh = file_msh_path,
        remsh = file_remsh_path,
        geo = file_geo_path,
        mat = file_mat_path
    )

    return paths