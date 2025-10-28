import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import savemat
from scipy.io import loadmat

from backend.rwg.rwg2 import DataManager_rwg2
from backend.utils.dipole_parameters import compute_dipole_center_moment, compute_e_h_field


def compute_electric_magnetic_field_at_point_for_single_frequency(frequency, triangles, edges, current, observation_point, eta, light_speed_c):
    omega = 2 * np.pi * frequency              # Angular frequency (rad/s)
    k = omega / light_speed_c                  # Wave number (rad/m)
    complex_k = 1j * k                         # Complex component of wave number

    dipole_center, dipole_moment = compute_dipole_center_moment(triangles, edges, current)
    e_field_total, h_field_total, *_ = compute_e_h_field(observation_point, eta, complex_k, dipole_moment, dipole_center)

    return e_field_total, h_field_total

def efield_1_sweep(filename_mesh2_to_load, frequencies, currents, observation_point):
    base_name = os.path.splitext(os.path.basename(filename_mesh2_to_load))[0].replace('_mesh2', '')

    _, triangles, edges, *_ = DataManager_rwg2.load_data(filename_mesh2_to_load)

    epsilon = 8.854e-12  # Vacuum permittivity (F/m)
    mu = 1.257e-6        # Vacuum permeability (H/m)
    light_speed_c = 1 / np.sqrt(epsilon * mu)
    eta = np.sqrt(mu / epsilon)

    num_freq = len(frequencies)
    e_fields = np.zeros((3, num_freq), dtype=np.complex128)
    h_fields = np.zeros((3, num_freq), dtype=np.complex128)

    for i, (freq, current) in enumerate(zip(frequencies, currents)):
        e_field, h_field = compute_electric_magnetic_field_at_point_for_single_frequency(
            freq, triangles, edges, current, observation_point, eta, light_speed_c
        )
        e_fields[:, i] = e_field
        h_fields[:, i] = h_field

    save_efield_1_data(base_name, frequencies, e_fields, observation_point)

    return e_fields, h_fields

def save_efield_1_data(filename, frequencies, e_fields, observation_point):
    """
    Saves the data into a .mat file similar to the MATLAB script.
    
    Parameters:
        filename : str — .mat file name
        frequencies : ndarray (N_freq,)
        e_fields : ndarray (3, N_freq)
        observation_point : ndarray (3,)
    """

    output_dir = "data/antennas_sweep/"
    os.makedirs(output_dir, exist_ok=True)
    output_matfile = os.path.join(output_dir, f"{filename}_radiatedfield_1_sweep.mat")

    mat_data = {
        'f': frequencies,
        'E': e_fields,
        'ObservationPoint': observation_point
    }
    savemat(output_matfile, mat_data)

def load_efield_1_data(filepath):
    """
    Loads data saved by save_efield_1_data from a .mat file.

    Parameters:
        filepath : str — .mat file name

    Returns:
        frequencies : ndarray (N_freq,)
        e_fields : ndarray (3, N_freq)
        observation_point : ndarray (3,)
    """
    mat_data = loadmat(filepath)
    frequencies = mat_data['f'].squeeze()
    e_fields = mat_data['E'].squeeze()
    observation_point = mat_data['ObservationPoint'].squeeze()
    return frequencies, e_fields, observation_point

def plot_efield_components(frequencies, e_fields):
    """
        Plots the amplitudes of Ex, Ey, Ez components as a function of frequency (in MHz).
    """
    fig_size = 12
    Fibonacci = (1 + np.sqrt(5)) / 2
    plt.figure(figsize=(fig_size, fig_size / Fibonacci))
    freqs_mhz = frequencies / 1e6

    plt.plot(freqs_mhz, np.abs(e_fields[0, :]), label='|Ex|')
    plt.plot(freqs_mhz, np.abs(e_fields[1, :]), label='|Ey|')
    plt.plot(freqs_mhz, np.abs(e_fields[2, :]), label='|Ez|')
    plt.grid(True)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Electric Field Amplitude (V/m)")
    plt.title("Electric Field Components")
    plt.legend()
    plt.show()

def plot_ex_phase(frequencies, e_fields):
    """
        Plots the unambiguous phase of the Ex component (frequency in MHz).
    """
    fig_size = 12
    Fibonacci = (1 + np.sqrt(5)) / 2
    plt.figure(figsize=(fig_size, fig_size / Fibonacci))
    freqs_mhz = frequencies / 1e6
    phase_ex = np.unwrap(np.angle(e_fields[0, :]))

    plt.plot(freqs_mhz, phase_ex, label='Phase Ex')
    plt.grid(True)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Ex Phase (rad)")
    plt.title("Phase of the Ex Component")
    plt.legend()
    plt.show()

def plot_ey_phase(frequencies, e_fields):
    """
        Plots the unambiguous phase of the Ey component (frequency in MHz).
    """
    fig_size = 12
    Fibonacci = (1 + np.sqrt(5)) / 2
    plt.figure(figsize=(fig_size, fig_size / Fibonacci))
    freqs_mhz = frequencies / 1e6
    phase_ey = np.unwrap(np.angle(e_fields[1, :]))

    plt.plot(freqs_mhz, phase_ey, label='Phase Ey')
    plt.grid(True)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Ey Phase (rad)")
    plt.title("Phase of the Ey Component")
    plt.legend()
    plt.show()

def plot_ez_phase(frequencies, e_fields):
    """
        Plots the unambiguous phase of the Ez component (frequency in MHz).
    """
    fig_size = 12
    Fibonacci = (1 + np.sqrt(5)) / 2
    plt.figure(figsize=(fig_size, fig_size / Fibonacci))
    freqs_mhz = frequencies / 1e6
    phase_ez = np.unwrap(np.angle(e_fields[2, :]))

    plt.plot(freqs_mhz, phase_ez, label='Phase Ez')
    plt.grid(True)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Ez Phase (rad)")
    plt.title("Phase of the Ez Component")
    plt.legend()
    plt.show()
