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
    mesh_name                      = f"{name}.msh"
    remesh_name                    = f"{name + '_remesh'}.msh"
    geo_name                       = f"{name}.brep"
    mat_name                       = f"{name}.mat"
    mat_mesh1_name                 = f"{name}_mesh1.mat"
    mat_mesh2_name                 = f"{name}_mesh2.mat"
    mat_impedance_name             = f"{name}_impedance.mat"
    mat_current_name               = f"{name}_current.mat"
    mat_gain_power_name            = f"{name}_gain_power.mat"
    mat_freq_sweep_name            = f"{name}_freq_sweep.mat"
    mat_polar_rhcp_gain_power_name = f"{name}_polar_rhcp_gain_power.mat"
    mat_polar_lhcp_gain_power_name = f"{name}_polar_lhcp_gain_power.mat"

    # 2. Define additional folders for different types of data
    mat_mesh1_folder_path      = 'data/antennas_mesh1/'
    mat_mesh2_folder_path      = 'data/antennas_mesh2/'
    mat_current_folder_path    = 'data/antennas_current/'
    mat_impedance_folder_path  = 'data/antennas_impedance/'
    mat_gain_power_folder_path = 'data/antennas_gain_power/'
    mat_freq_sweep_folder_path = 'data/antennas_sweep/'

    # 2. Check if the destination folders exist, if not, create them
    if not os.path.exists(gmsh_folder_path):
        os.makedirs(gmsh_folder_path)
        print(f"Directory created: {gmsh_folder_path}")
    if not os.path.exists(mat_folder_path):
        os.makedirs(mat_folder_path)
        print(f"Directory created: {mat_folder_path}")
    if not os.path.exists(mat_mesh1_folder_path):
        os.makedirs(mat_mesh1_folder_path)
        print(f"Directory created: {mat_mesh1_folder_path}")
    if not os.path.exists(mat_mesh2_folder_path):
        os.makedirs(mat_mesh2_folder_path)
        print(f"Directory created: {mat_mesh2_folder_path}")
    if not os.path.exists(mat_impedance_folder_path):
        os.makedirs(mat_impedance_folder_path)
        print(f"Directory created: {mat_impedance_folder_path}")
    if not os.path.exists(mat_current_folder_path):
        os.makedirs(mat_current_folder_path)
        print(f"Directory created: {mat_current_folder_path}")
    if not os.path.exists(mat_gain_power_folder_path):
        os.makedirs(mat_gain_power_folder_path)
        print(f"Directory created: {mat_gain_power_folder_path}")
    if not os.path.exists(mat_freq_sweep_folder_path):
        os.makedirs(mat_freq_sweep_folder_path)
        print(f"Directory created: {mat_freq_sweep_folder_path}")

    # 3. Construct the absolute or relative paths for the files
    file_msh_path                       = os.path.join(gmsh_folder_path, mesh_name)
    file_remsh_path                     = os.path.join(gmsh_folder_path, remesh_name)
    file_geo_path                       = os.path.join(gmsh_folder_path, geo_name)
    file_mat_path                       = os.path.join(mat_folder_path, mat_name)
    file_mat_mesh1_path                 = os.path.join(mat_mesh1_folder_path, mat_mesh1_name)
    file_mat_mesh2_path                 = os.path.join(mat_mesh2_folder_path, mat_mesh2_name)
    file_mat_impedance_path             = os.path.join(mat_impedance_folder_path, mat_impedance_name)
    file_mat_current_path               = os.path.join(mat_current_folder_path, mat_current_name)
    file_mat_gain_power_path            = os.path.join(mat_gain_power_folder_path, mat_gain_power_name)
    file_mat_freq_sweep_path            = os.path.join(mat_freq_sweep_folder_path, mat_freq_sweep_name)
    file_mat_polar_rhcp_gain_power_path = os.path.join(mat_gain_power_folder_path, mat_polar_rhcp_gain_power_name)
    file_mat_polar_lhcp_gain_power_path = os.path.join(mat_gain_power_folder_path, mat_polar_lhcp_gain_power_name)

    # Store paths in a namespace for dot notation access
    paths = SimpleNamespace(
        name                      = name,
        msh                       = file_msh_path,
        remsh                     = file_remsh_path,
        geo                       = file_geo_path,
        mat                       = file_mat_path,
        mat_mesh1                 = file_mat_mesh1_path,
        mat_mesh2                 = file_mat_mesh2_path,
        mat_impedance             = file_mat_impedance_path,
        mat_current               = file_mat_current_path,
        mat_gain_power            = file_mat_gain_power_path,
        mat_freq_sweep            = file_mat_freq_sweep_path,
        mat_polar_rhcp_gain_power = file_mat_polar_rhcp_gain_power_path,
        mat_polar_lhcp_gain_power = file_mat_polar_lhcp_gain_power_path
    )

    return paths