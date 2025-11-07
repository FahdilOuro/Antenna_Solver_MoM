from backend.src.scattering_algorithm.scattering_algorithm import *
from backend.utils.gmsh_function import *

def create_plate_mesh(x_rect, y_rect, mesh_size, model_name, save_mesh_folder, mesh_name):
    """
    Create a rectangular plate mesh using Gmsh and save it to a .msh file.
    """
    gmsh.initialize()
    gmsh.model.add(model_name)

    surface_tag = rectangle_surface(x_rect, y_rect)
    print(f"Created rectangular surface with tag {surface_tag}")

    gmsh.model.geo.synchronize()

    # --- Step 3: Mesh generation ---
    apply_mesh_size(mesh_size)
    generate_surface_mesh()

    write(save_mesh_folder, mesh_name)

    gmsh.finalize()

def main():
    """Main entry point of the plate simulation."""
    print("=== Starting plate simulation ===")

    # --- Step 1: Define plate parameters ---
    plate_length = 5.0     # meters
    plate_width = 5.0      # meters
    plate_mesh_size = 0.13  # meters
    model_name = "plate_scattering"
    mesh_name = "plate_mesh.msh"
    save_mesh_folder = 'data/gmsh_files/'
    file_msh = save_mesh_folder + mesh_name
    path_file_mat = 'data/antennas_mesh/plate_scattering.mat'
    save_mesh_folder = 'data/gmsh_files/'

    # --- Step 2: Create rectangular plate using utility function ---
    x_rect = [0.0, plate_length, plate_length, 0.0]
    y_rect = [0.0, 0.0, plate_width, plate_width]

    create_plate_mesh(x_rect, y_rect, plate_mesh_size, model_name, save_mesh_folder, mesh_name)

    extract_msh_to_mat(file_msh, path_file_mat)

    # --- Step 4: Load and run scattering computation ---
    print("=== Loading mesh and computing scattering ===")
    frequency = 75e6 
    
    # Definition of wave direction and polarization vectors
    wave_incident_direction = np.array([0, 0, -1])
    polarization=np.array([1, 0, 0])

    # --- Step 5: Display results ---
    scattering_algorithm(path_file_mat, frequency, wave_incident_direction, polarization, show=False)


if __name__ == "__main__":
    main()
