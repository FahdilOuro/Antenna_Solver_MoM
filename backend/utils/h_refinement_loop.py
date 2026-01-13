from backend.src.scattering_algorithm.scattering_algorithm import scattering_algorithm
from backend.utils.error_estimator   import *
from backend.utils.gmsh_function import extract_ModelMsh_to_mat
from backend.utils.h_refinement_func import *


def run_refinement_cycle(frequency, wave_incident_direction, polarization, config, sizes, grid_points, r_vicinity, paths):
    """
    Executes the iterative mesh refinement loop.
    
    Args:
        config: SimpleNamespace containing max_iterations, threshold_percentage, etc.
        sizes: The current mesh size array (modified in-place).
        grid_points: The static background grid coordinates.
        r_vicinity: The radius used for KDTree spatial lookups.
        paths: SimpleNamespace containing file paths (msh, geo, mat).
        solver_params: Dictionary containing frequency, direction, and polarization.
    """
    show_image = True
    _, _, _, surface_current_density = scattering_algorithm(paths.mat, frequency, wave_incident_direction, polarization, show=show_image)

    # Pack solver parameters to pass into the loop
    solver_params = {
        'frequency': frequency,
        'direction': wave_incident_direction,
        'polarization': polarization,
        'surface_current_density': surface_current_density
    }
    
    for i in range(config.max_iterations):
        print(f"\n>>> Starting Iteration {i + 2}/{config.max_iterations}")

        # 1. Mesh Analysis & Element Selection
        # Extract centroids from the current mesh to evaluate error location
        centroids = get_mesh_centroids(paths.msh)
        
        # In a real scenario, surface_current_density is retrieved from the previous solver run
        # For the first iteration, it should be initialized or computed once before the loop
        errors = simple_estimation(solver_params['surface_current_density'])
        
        # Identify elements exceeding the error threshold
        high_error_indices = np.where(errors > config.threshold_percentage)[0]
        candidates = centroids[high_error_indices]

        # 2. Map mesh errors to the background grid
        # Use KDTree to find grid points within the influence radius of high-error centroids
        grid_tree = KDTree(grid_points)
        affected_nested = grid_tree.query_ball_point(candidates, r_vicinity)
        
        # Flatten the list and get unique indices of grid points to refine
        g_selected_idx = np.unique([idx for sublist in affected_nested for idx in sublist])

        # Ensure the indices are integers (addressing the IndexError)
        g_selected_idx = g_selected_idx.astype(int)

        # Check if any points were actually found in the vicinity
        # If the array is empty, we raise an error to stop the execution
        if g_selected_idx.size == 0:
            raise ValueError(
                f"Refinement aborted: No grid points were found within the specified "
                f"radius_vicinity. Check your threshold or input coordinates."
            )

        # 3. Update Sizes
        old_sizes = sizes.copy()
        for idx in g_selected_idx:
            sizes[idx] *= config.refinement_factor_gamma

        # Prepare data for Gmsh refinement function
        grid_points_to_refine = grid_points[g_selected_idx]
        target_size_value = sizes[g_selected_idx][0] if g_selected_idx.size > 0 else None

        # 4. Generate Refined Mesh
        gmsh.initialize()
        gmsh.model.add(f"Iteration_{i+1}")
        setup_performance_config()

        # Apply the refinement logic using the specialized grid function
        define_mesh_by_grid_refined(paths.geo, grid_points_to_refine, target_size_value, config.initial_mesh_size)

        # Save and export the model
        gmsh.write(paths.msh)
        extract_ModelMsh_to_mat(paths.mat)
        gmsh.finalize()

        # 5. Turn the show_image to True if it's the last iteration
        if i == config.max_iterations - 1: show_image = True

        # 6. Run Physics Solver on the new mesh
        # This updates the surface_current_density for the next iteration's error estimation
        _, _, _, scd = scattering_algorithm(paths.mat, solver_params['frequency'], solver_params['direction'], solver_params['polarization'], show=show_image)
        solver_params['surface_current_density'] = scd

        # 6. Logging Statistics
        sizes_changed = np.sum(sizes != old_sizes)
        print(f"Points refined in this step: {len(g_selected_idx)}")
        print(f"Total modified points in grid: {sizes_changed}")
        print(f"Size range: [{sizes.min():.6f}, {sizes.max():.6f}]")

    print("\n>>> Refinement process completed.")