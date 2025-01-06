from test_setup import setup_project_root
setup_project_root()

from rwg.rwg4 import *

filename_plate_to_load = 'data/antennas_mesh2/plate_mesh2.mat'
filename_impedance_plate = 'data/antennas_impedance/plate_impedance.mat'

wave_incident_direction = np.array([0, 0, -1])
polarization = np.array([1, 0, 0])

frequency, omega, mu, epsilon, light_speed_c, eta, voltage, current_plate = (
    calculate_current_scattering(filename_plate_to_load, filename_impedance_plate, wave_incident_direction,
                                 polarization))

print(f"Shape of current = {current_plate.shape}")

save_folder_name = 'data/antennas_current/'
DataManager_rwg4.save_data_fro_scattering(filename_plate_to_load, save_folder_name, frequency, omega, mu, epsilon, light_speed_c,
                           eta, wave_incident_direction, polarization, voltage, current_plate)