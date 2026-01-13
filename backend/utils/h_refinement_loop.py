from backend.src.scattering_algorithm.scattering_algorithm import scattering_algorithm
from backend.src.radiation_algorithm.radiation_algorithm import radiation_algorithm
from backend.utils.error_estimator   import *
from backend.utils.gmsh_function import extract_ModelMsh_to_mat
from backend.utils.h_refinement_func import *


def run_refinement_cycle(solver_function, config, sizes, grid_points, r_vicinity, paths, **solver_kwargs):
    """
    Executes the iterative mesh refinement loop with a flexible solver function.
    
    Args:
        solver_function: The physics solver to use (scattering_algorithm or radiation_algorithm).
        config: SimpleNamespace containing max_iterations, threshold_percentage, etc.
        sizes: The current mesh size array (modified in-place).
        grid_points: The static background grid coordinates.
        r_vicinity: The radius used for KDTree spatial lookups.
        paths: SimpleNamespace containing file paths (msh, geo, mat).
        **solver_kwargs: All keyword arguments needed by the solver function.
                        For scattering: frequency, wave_incident_direction, polarization
                        For radiation: frequency, feed_point, voltage_amplitude, monopole, etc.
    """
    show_image = True
    
    # Initial solver run to get the first surface current density
    _, _, _, surface_current_density = solver_function(paths.mat, show=show_image, **solver_kwargs)
    
    for i in range(config.max_iterations):
        print(f"\n>>> Starting Iteration {i + 2}/{config.max_iterations}")

        # 1. Mesh Analysis & Element Selection
        centroids = get_mesh_centroids(paths.msh)
        
        # Estimate errors based on surface current density
        errors = simple_estimation(surface_current_density)
        
        # Identify elements exceeding the error threshold
        high_error_indices = np.where(errors > config.threshold_percentage)[0]
        candidates = centroids[high_error_indices]

        # 2. Map mesh errors to the background grid
        grid_tree = KDTree(grid_points)
        affected_nested = grid_tree.query_ball_point(candidates, r_vicinity)
        
        # Flatten the list and get unique indices of grid points to refine
        g_selected_idx = np.unique([idx for sublist in affected_nested for idx in sublist])
        g_selected_idx = g_selected_idx.astype(int)

        # Check if any points were found
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

        # Apply the refinement logic
        define_mesh_by_grid_refined(paths.geo, grid_points_to_refine, target_size_value, config.initial_mesh_size)

        # Save and export the model
        gmsh.write(paths.msh)
        extract_ModelMsh_to_mat(paths.mat)
        gmsh.finalize()

        # 5. Show image on last iteration
        if i == config.max_iterations - 1:
            show_image = True

        # 6. Run Physics Solver on the new mesh
        _, _, _, surface_current_density = solver_function(paths.mat, show=show_image, **solver_kwargs)

        # 7. Logging Statistics
        sizes_changed = np.sum(sizes != old_sizes)
        print(f"Points refined in this step: {len(g_selected_idx)}")
        print(f"Total modified points in grid: {sizes_changed}")
        print(f"Size range: [{sizes.min():.6f}, {sizes.max():.6f}]")

    print("\n>>> Refinement process completed.")

def run_scattering_refinement(config, sizes, grid_points, r_vicinity, paths, 
                               frequency, wave_incident_direction, polarization):
    """
    Wrapper for scattering-based refinement.
    """
    return run_refinement_cycle(
        solver_function=scattering_algorithm,
        config=config,
        sizes=sizes,
        grid_points=grid_points,
        r_vicinity=r_vicinity,
        paths=paths,
        frequency=frequency,
        wave_incident_direction=wave_incident_direction,
        polarization=polarization
    )


# Example 2: Using radiation algorithm
def run_radiation_refinement(config, sizes, grid_points, r_vicinity, paths,
                              frequency, feed_point, voltage_amplitude=1,
                              monopole=False, simulate_array_antenna=False,
                              load_lumped_elements=False, LoadPoint=None,
                              LoadValue=None, LoadDir=None):
    """
    Wrapper for radiation-based refinement.
    """
    return run_refinement_cycle(
        solver_function=radiation_algorithm,
        config=config,
        sizes=sizes,
        grid_points=grid_points,
        r_vicinity=r_vicinity,
        paths=paths,
        frequency=frequency,
        feed_point=feed_point,
        voltage_amplitude=voltage_amplitude,
        monopole=monopole,
        simulate_array_antenna=simulate_array_antenna,
        load_lumped_elements=load_lumped_elements,
        LoadPoint=LoadPoint,
        LoadValue=LoadValue,
        LoadDir=LoadDir
    )