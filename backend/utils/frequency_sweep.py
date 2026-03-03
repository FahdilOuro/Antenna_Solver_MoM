from matplotlib import pyplot as plt
import numpy as np
from scipy.io import savemat
from scipy.io import loadmat

from backend.src.radiation_algorithm.radiation_algorithm import radiation_algorithm


def generate_freq_step(fLow, fHigh, step=2e6):
    nPoints = int((fHigh - fLow) // step) + 1
    frequencies = [fLow + i * step for i in range(nPoints)]

    return frequencies

def frequency_sweep(path, freq_start, freq_stop,feed_point, step=2e6, voltage_amplitude=1,
                    excitation_unit_vector=None, gap_width=0.05, voltage_phase=None, load_lumped_elements=False,
                    LoadPoint=None, LoadValue=None, LoadDir=None, Z0=50):
    s11_db = []
    impedances = []
    currents = []
    gap_currents = []
    feed_powers = []

    frequencies = generate_freq_step(freq_start, freq_stop, step)
    nPoints = len(frequencies)
    print(f"Starting frequency sweep from {freq_start/1e6:.2f} MHz to {freq_stop/1e6:.2f} MHz with {nPoints} points.\n")

    for idx, frequency in enumerate(frequencies):
        _, _, current, _, gap_current, _, impedance, feed_power = radiation_algorithm(path, frequency,
                                                                                    feed_point,
                                                                                    voltage_amplitude=voltage_amplitude,
                                                                                    excitation_unit_vector=excitation_unit_vector,
                                                                                    gap_width=gap_width,
                                                                                    voltage_phase=voltage_phase,
                                                                                    show=True,
                                                                                    load_lumped_elements=load_lumped_elements,
                                                                                    LoadPoint=LoadPoint, LoadValue=LoadValue,
                                                                                    LoadDir=LoadDir)
        impedances.append(impedance)
        currents.append(current)
        gap_currents.append(gap_current)
        feed_powers.append(feed_power)
        s11 = (impedance - Z0) / (impedance + Z0)
        s11_db.append(20 * np.log10(abs(s11)))
        print(f"Simulation {idx+1}/{nPoints} | f = {frequency/1e6:.2f} MHz | S11 = {s11_db[-1]:.2f} dB")

    # Convert lists to arrays where appropriate
    data_to_save = {
        'frequencies': np.array(frequencies),
        'impedances': np.array(impedances),
        's11_db': np.array(s11_db),
        'currents': np.array(currents),
        'gap_currents': np.array(gap_currents),
        'feed_powers': np.array(feed_powers)
    }

    savemat(path.mat_freq_sweep, data_to_save)

def load_freq_sweep_data(filepath):
    data = loadmat(filepath)
    frequencies = data.get('frequencies', np.array([])).squeeze()
    impedances = data.get('impedances', np.array([])).squeeze()
    s11_db = data.get('s11_db', np.array([])).squeeze()
    currents = data.get('currents', np.array([])).squeeze()
    gap_currents = data.get('gap_currents', np.array([])).squeeze()
    feed_powers = data.get('feed_powers', np.array([])).squeeze()

    return frequencies, impedances, s11_db, currents, gap_currents, feed_powers

def plot_s11(path, freq_unit='MHz'):
    """
    Plot S11 parameter vs frequency.
    
    Args:
        path: Path object containing mat_freq_sweep file
        freq_unit: Frequency unit ('Hz', 'kHz', 'MHz', 'GHz'). Default is 'MHz'
    """
    frequencies, _, s11_db, _, _, _ = load_freq_sweep_data(path.mat_freq_sweep)

    print(frequencies)
    print(s11_db)
    
    # Convert frequencies based on unit
    unit_factors = {'Hz': 1, 'kHz': 1e3, 'MHz': 1e6, 'GHz': 1e9}
    freq_divisor = unit_factors.get(freq_unit, 1e6)
    frequencies_converted = frequencies / freq_divisor
    
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies_converted, s11_db, marker='o')
    plt.title('S11 Parameter vs Frequency')
    plt.xlabel(f'Frequency ({freq_unit})')
    plt.ylabel('S11 (dB)')
    plt.grid()
    plt.axhline(-10, color='red', linestyle='--', label='-10 dB Threshold')
    plt.legend()
    plt.show()
