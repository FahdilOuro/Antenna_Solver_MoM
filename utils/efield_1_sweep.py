import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import savemat
from scipy.io import loadmat

from rwg.rwg2 import DataManager_rwg2
from utils.dipole_parameters import compute_dipole_center_moment, compute_e_h_field


def compute_electric_magnetic_field_at_point_for_single_frequency(frequency, triangles, edges, current, observation_point, eta, light_speed_c):
    omega = 2 * np.pi * frequency              # Pulsation angulaire (rad/s)
    k = omega / light_speed_c                  # Nombre d'onde (en rad/m)
    complex_k = 1j * k                         # Composante complexe du nombre d'onde

    dipole_center, dipole_moment = compute_dipole_center_moment(triangles, edges, current)
    e_field_total, h_field_total, *_ = compute_e_h_field(observation_point, eta, complex_k, dipole_moment, dipole_center)

    return e_field_total, h_field_total

def efield_1_sweep(filename_mesh2_to_load, frequencies, currents, observation_point):
    base_name = os.path.splitext(os.path.basename(filename_mesh2_to_load))[0].replace('_mesh2', '')

    _, triangles, edges, *_ = DataManager_rwg2.load_data(filename_mesh2_to_load)

    epsilon = 8.854e-12  # Permittivité du vide (F/m)
    mu = 1.257e-6        # Perméabilité magnétique du vide (H/m)
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
    Enregistre les données dans un fichier .mat à la manière du script MATLAB.
    
    Paramètres :
        filename : str — nom de fichier .mat
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
    Charge les données sauvegardées par save_efield_1_data depuis un fichier .mat.

    Paramètres :
        filepath : str — nom de fichier .mat

    Retourne :
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
    Affiche les amplitudes des composantes Ex, Ey, Ez en fonction de la fréquence (en MHz).
    """
    fig_size = 12
    Fibonacci = (1 + np.sqrt(5)) / 2
    plt.figure(figsize=(fig_size, fig_size / Fibonacci))
    freqs_mhz = frequencies / 1e6

    plt.plot(freqs_mhz, np.abs(e_fields[0, :]), label='|Ex|')
    plt.plot(freqs_mhz, np.abs(e_fields[1, :]), label='|Ey|')
    plt.plot(freqs_mhz, np.abs(e_fields[2, :]), label='|Ez|')
    plt.grid(True)
    plt.xlabel("Fréquence (MHz)")
    plt.ylabel("Amplitude du champ électrique (V/m)")
    plt.title("Composantes du champ électrique")
    plt.legend()
    plt.show()

def plot_ex_phase(frequencies, e_fields):
    """
    Affiche la phase non ambigüe de la composante Ex (fréquence en MHz).
    """
    fig_size = 12
    Fibonacci = (1 + np.sqrt(5)) / 2
    plt.figure(figsize=(fig_size, fig_size / Fibonacci))
    freqs_mhz = frequencies / 1e6
    phase_ex = np.unwrap(np.angle(e_fields[0, :]))

    plt.plot(freqs_mhz, phase_ex, label='Phase Ex')
    plt.grid(True)
    plt.xlabel("Fréquence (MHz)")
    plt.ylabel("Phase de Ex (rad)")
    plt.title("Phase de la composante Ex")
    plt.legend()
    plt.show()

def plot_ey_phase(frequencies, e_fields):
    """
    Affiche la phase non ambigüe de la composante Ey (fréquence en MHz).
    """
    fig_size = 12
    Fibonacci = (1 + np.sqrt(5)) / 2
    plt.figure(figsize=(fig_size, fig_size / Fibonacci))
    freqs_mhz = frequencies / 1e6
    phase_ey = np.unwrap(np.angle(e_fields[1, :]))

    plt.plot(freqs_mhz, phase_ey, label='Phase Ey')
    plt.grid(True)
    plt.xlabel("Fréquence (MHz)")
    plt.ylabel("Phase de Ey (rad)")
    plt.title("Phase de la composante Ey")
    plt.legend()
    plt.show()

def plot_ez_phase(frequencies, e_fields):
    """
    Affiche la phase non ambigüe de la composante Ez (fréquence en MHz).
    """
    fig_size = 12
    Fibonacci = (1 + np.sqrt(5)) / 2
    plt.figure(figsize=(fig_size, fig_size / Fibonacci))
    freqs_mhz = frequencies / 1e6
    phase_ez = np.unwrap(np.angle(e_fields[2, :]))

    plt.plot(freqs_mhz, phase_ez, label='Phase Ez')
    plt.grid(True)
    plt.xlabel("Fréquence (MHz)")
    plt.ylabel("Phase de Ez (rad)")
    plt.title("Phase de la composante Ez")
    plt.legend()
    plt.show()
