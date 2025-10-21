import numpy as np

from efield.efield1 import calculate_electric_magnetic_field_at_point
from efield.efield2 import radiation_intensity_distribution_over_sphere_surface
from efield.efield3 import antenna_directivity_pattern

if __name__ == "__main__":
    """
        Main script for antenna analysis.

        Main steps:
            1. Load files containing antenna data (mesh, current, gain/power).
            2. Compute and display electromagnetic fields at a given point.
            3. Compute the radiation intensity distribution over a sphere.
            4. Generate directivity patterns.
    """

    # Input files for different types of antennas
    filename_mesh2_plate = 'data/antennas_mesh2/plate_mesh2.mat'
    filename_current_plate = 'data/antennas_current/plate_current.mat'
    filename_gain_power_plate = 'data/antennas_gain_power/plate_gain_power.mat'

    filename_mesh2_platecoarse = 'data/antennas_mesh2/platecoarse_mesh2.mat'
    filename_current_platecoarse = 'data/antennas_current/platecoarse_current.mat'
    filename_gain_power_platecoarse = 'data/antennas_gain_power/platecoarse_gain_power.mat'

    filename_mesh2_platefine = 'data/antennas_mesh2/platefine_mesh2.mat'
    filename_current_platefine = 'data/antennas_current/platefine_current.mat'
    filename_gain_power_platefine = 'data/antennas_gain_power/platefine_gain_power.mat'

    filename_mesh2_dipole = 'data/antennas_mesh2/dipole_mesh2.mat'
    filename_current_dipole = 'data/antennas_current/dipole_current.mat'
    filename_gain_power_dipole = 'data/antennas_gain_power/dipole_gain_power.mat'

    filename_mesh2_bowtie = 'data/antennas_mesh2/bowtie_mesh2.mat'
    filename_current_bowtie = 'data/antennas_current/bowtie_current.mat'
    filename_gain_power_bowtie = 'data/antennas_gain_power/bowtie_gain_power.mat'

    filename_mesh2_slot = 'data/antennas_mesh2/slot_mesh2.mat'
    filename_current_slot = 'data/antennas_current/slot_current.mat'
    filename_gain_power_slot = 'data/antennas_gain_power/slot_gain_power.mat'

    # Dense mesh for a sphere used in radiation calculations
    filename_sphere_dense = 'data/sphere_mesh/sphere_dense.mat'


    # Observation point for field calculations
    observation_point = np.array([5, 0, 0])

    # Lists of input files grouped by antenna type
    filename_mesh2 = [filename_mesh2_plate, filename_mesh2_platecoarse, filename_mesh2_platefine, filename_mesh2_dipole, filename_mesh2_bowtie, filename_mesh2_slot]
    filename_current = [filename_current_plate, filename_current_platecoarse, filename_current_platefine, filename_current_dipole, filename_current_bowtie, filename_current_slot]
    filename_gain_power = [filename_gain_power_plate, filename_gain_power_platecoarse, filename_gain_power_platefine, filename_gain_power_dipole, filename_gain_power_bowtie, filename_gain_power_slot]

    # Loop over each antenna type to perform calculations and visualizations
    for mesh2, current, gain_power in zip(filename_mesh2, filename_current, filename_gain_power):
        print(f"Processing antenna with mesh: {mesh2}")

        # Step 1: Calculate electric and magnetic fields at a given observation point
        print("Calculating electric and magnetic fields at observation point...")
        calculate_electric_magnetic_field_at_point(mesh2, current, observation_point)

        # Step 2: Radiation intensity distribution over sphere surface
        print("Calculating radiation intensity distribution over sphere surface...")
        radiation_intensity_distribution_over_sphere_surface(mesh2, current, filename_sphere_dense)

        # Step 3: Generate antenna directivity pattern
        print("Generating antenna directivity pattern...")
        antenna_directivity_pattern(mesh2, current, gain_power)

        # Empty line to separate outputs for different antennas
        print('')