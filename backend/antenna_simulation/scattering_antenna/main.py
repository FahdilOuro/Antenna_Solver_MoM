from backend.src.scattering_algorithm.scattering_algorithm import *

import time

def process_antenna(filename):
    base_name = os.path.splitext(os.path.basename(filename))[0]
    print(f"Processing antenna {base_name}")

if __name__ == "__main__":

    filename_mesh1_plate = 'data/antennas_mesh/plate.mat'
    filename_mesh1_platecoarse = 'data/antennas_mesh/platecoarse.mat'
    filename_mesh1_platefine = 'data/antennas_mesh/platefine.mat'
    filename_mesh1_dipole = 'data/antennas_mesh/dipole.mat'
    filename_mesh1_bowtie = 'data/antennas_mesh/bowtie.mat'
    filename_mesh1_slot = 'data/antennas_mesh/slot.mat'


    filename_mesh1 = [filename_mesh1_plate, filename_mesh1_platecoarse, filename_mesh1_platefine, filename_mesh1_dipole, filename_mesh1_bowtie, filename_mesh1_slot]

    frequency = 75e6  # The frequency of the electromagnetic field can be defined for each antenna
    # Definition of wave direction and polarization vectors
    wave_incident_direction = np.array([0, 0, -1])

    # To simulate only one antenna
    # Start time measurement
    start_time = time.time()

    process_antenna(filename_mesh1_plate)
    scattering_algorithm(filename_mesh1_plate, 3e8, wave_incident_direction, polarization=np.array([1, 0, 0]))

    elapsed_time = time.time() - start_time
    print(f"Elapsed time for processing the studied antenna: {elapsed_time:.6f} seconds")
    print('\n')

    start_time = time.time()

    process_antenna(filename_mesh1_platecoarse)
    scattering_algorithm(filename_mesh1_platecoarse, 3e8, wave_incident_direction, polarization=np.array([1, 0, 0]))

    elapsed_time = time.time() - start_time
    print(f"Elapsed time for processing the studied antenna: {elapsed_time:.6f} seconds")
    print('\n')

    start_time = time.time()

    process_antenna(filename_mesh1_platefine)
    scattering_algorithm(filename_mesh1_platefine, 3e8, wave_incident_direction, polarization=np.array([1, 0, 0]))

    elapsed_time = time.time() - start_time
    print(f"Elapsed time for processing the studied antenna: {elapsed_time:.6f} seconds")
    print('\n')

    start_time = time.time()

    process_antenna(filename_mesh1_dipole)
    scattering_algorithm(filename_mesh1_dipole, 75e6, wave_incident_direction, polarization=np.array([0, 1, 0]))

    elapsed_time = time.time() - start_time
    print(f"Elapsed time for processing the studied antenna: {elapsed_time:.6f} seconds")
    print('\n')

    start_time = time.time()

    process_antenna(filename_mesh1_bowtie)
    scattering_algorithm(filename_mesh1_bowtie, 75e7, wave_incident_direction, polarization=np.array([0, 1, 0]))

    elapsed_time = time.time() - start_time
    print(f"Elapsed time for processing the studied antenna: {elapsed_time:.6f} seconds")
    print('\n')

    start_time = time.time()

    process_antenna(filename_mesh1_slot)
    scattering_algorithm(filename_mesh1_slot, 75e6, wave_incident_direction, polarization=np.array([1, 0, 0]))

    elapsed_time = time.time() - start_time
    print(f"Elapsed time for processing the studied antenna: {elapsed_time:.6f} seconds")
    print('\n')

    '''# To simulate all the antennas
    for filename in filename_mesh1:
        # Start time measurement
        start_time = time.time()

        scattering_algorithm(filename, frequency)

        # Measure elapsed time
        elapsed_time = time.time() - start_time
        print(f"Elapsed time for processing the studied antenna: {elapsed_time:.6f} seconds")
        print('\n')'''
