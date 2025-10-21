import os
import numpy as np
from scipy.io import savemat
from scipy.io import loadmat
from src.radiation_algorithm.radiation_algorithm import radiation_algorithm


def generate_freq_step(fLow, fHigh, step=2e6):
    nPoints = int((fHigh - fLow) // step) + 1
    frequencies = [fLow + i * step for i in range(nPoints)]

    return frequencies

def frequency_sweep(mat_file, frequencies, feed_point, voltage_amplitude=1, load_lumped_elements=False, LoadPoint=None, LoadValue=None, LoadDir=None):
    Z0 = 50
    s11_db = []
    impedances = []
    currents = []
    gap_currents = []
    gap_voltages = []
    feed_powers = []
    index_feeding_edges_list = []
    nPoints = len(frequencies)

    for idx, frequency in enumerate(frequencies):
        impedance, current, gap_current, gap_voltage, feed_power, index_feeding_edges, _ = radiation_algorithm(mat_file, frequency, 
                                                                                                               feed_point, voltage_amplitude=voltage_amplitude, 
                                                                                                               show=False,
                                                                                                               load_lumped_elements=load_lumped_elements, 
                                                                                                               LoadPoint=LoadPoint, LoadValue=LoadValue, 
                                                                                                               LoadDir=LoadDir)
        impedances.append(impedance)
        currents.append(current)
        gap_currents.append(gap_current)
        gap_voltages.append(gap_voltage)
        feed_powers.append(feed_power)
        index_feeding_edges_list.append(index_feeding_edges)
        s11 = (impedance - Z0) / (impedance + Z0)
        s11_db.append(20 * np.log10(abs(s11)))
        print(f"Simulation {idx+1}/{nPoints} | f = {frequency/1e6:.2f} MHz | S11 = {s11_db[-1]:.2f} dB")

    antenna_name = os.path.splitext(os.path.basename(mat_file))[0]
    output_dir = "data/antennas_sweep"
    os.makedirs(output_dir, exist_ok=True)
    output_matfile = os.path.join(output_dir, f"{antenna_name}_freq_sweep.mat")

    # Convert lists to arrays where appropriate
    data_to_save = {
        'frequencies': np.array(frequencies),
        'impedances': np.array(impedances),
        's11_db': np.array(s11_db),
        'currents': np.array(currents),  # shape: (nPoints, N)
        'gap_currents': np.array(gap_currents),
        'gap_voltages': np.array(gap_voltages),
        'feed_powers': np.array(feed_powers),
        'index_feeding_edges': np.array(index_feeding_edges_list),
    }

    savemat(output_matfile, data_to_save)

    return output_matfile

def load_freq_sweep_data(filepath):
    data = loadmat(filepath)

    frequencies = data.get('frequencies', np.array([])).squeeze()
    impedances = data.get('impedances', np.array([])).squeeze()
    s11_db = data.get('s11_db', np.array([])).squeeze()
    currents = data.get('currents', np.array([])).squeeze()
    gap_currents = data.get('gap_currents', np.array([])).squeeze()
    gap_voltages = data.get('gap_voltages', np.array([])).squeeze()
    feed_powers = data.get('feed_powers', np.array([])).squeeze()
    index_feeding_edges = data.get('index_feeding_edges', np.array([])).squeeze()

    return frequencies, impedances, s11_db, currents, gap_currents, gap_voltages, feed_powers, index_feeding_edges
