import os
from matplotlib import pyplot as plt
import numpy as np
import plotly.graph_objects as go

def plot_smith_chart(impedances, frequencies, fC=None, save_image=False):
    fig_size = 15
    Fibonacci = (1 + np.sqrt(5)) / 2

    fig = go.Figure()

    Normalized_Z0 = 50  # Normalized impedance reference (Z0)

    smith_points = []
    for idx, Z in enumerate(impedances):
        r = Z.real / Normalized_Z0
        x = Z.imag / Normalized_Z0
        smith_points.append(dict(
            real=r,
            imag=x,
            freq=frequencies[idx],
            label=f'Z = {Z:.3f} at {frequencies[idx]*1e-6:.2f} MHz'
        ))

    if fC is not None and fC in frequencies:
        idx_fC = frequencies.index(fC)
        # Plot all points except the highlighted one
        fig.add_trace(go.Scattersmith(
            real=[pt['real'] for i, pt in enumerate(smith_points) if i != idx_fC],
            imag=[pt['imag'] for i, pt in enumerate(smith_points) if i != idx_fC],
            mode='markers+lines',
            marker=dict(size=4, color='blue'),
            name='Impedance Points',
            text=[pt['label'] for i, pt in enumerate(smith_points) if i != idx_fC],
            hoverinfo='text'
        ))
        # Highlight the specific frequency in red and show impedance value in legend
        pt = smith_points[idx_fC]
        fig.add_trace(go.Scattersmith(
            real=[pt['real']],
            imag=[pt['imag']],
            mode='markers',
            marker=dict(size=10, color='red', symbol='circle'),
            name=f"fC={fC*1e-6:.2f} MHz, Z={impedances[idx_fC]:.2f}",
            text=[f"{pt['label']}"],
            hoverinfo='text'
        ))
    else:
        fig.add_trace(go.Scattersmith(
            real=[pt['real'] for pt in smith_points],
            imag=[pt['imag'] for pt in smith_points],
            mode='markers+lines',
            marker=dict(size=4, color='blue'),
            name='Impedance Points',
            text=[pt['label'] for pt in smith_points],
            hoverinfo='text'
        ))

    fig.update_layout(
        title='',
        showlegend=True,
        width=1000,
        height=int(1000 / Fibonacci)
    )

    fig.show()

    if save_image:
        output_dir_fig_image = "data/fig_image/"
        if not os.path.exists(output_dir_fig_image):
            os.makedirs(output_dir_fig_image)
            print(f"File created : {output_dir_fig_image}")
        pdf_path = os.path.join(output_dir_fig_image, "ifa_M_opti3_Smith_chart.pdf")
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=10, b=10)
        )
        fig.write_image(pdf_path, format="pdf")
        print(f"\nImage saved in PDF format (transparent background, minimal margins) : {pdf_path}\n")

def plot_single_impedance(Z):
    Normalized_Z0 = 50  # Normalized impedance reference (Z0)

    r = Z.real / Normalized_Z0
    x = Z.imag / Normalized_Z0

    fig = go.Figure()
    fig.add_trace(go.Scattersmith(
        real=[r],
        imag=[x],
        mode='markers',
        marker=dict(size=14, color='green', symbol='circle'),
        name=f'Z = {Z}',
        text=[f'Z = {Z}'],
        hoverinfo='text'
    ))
    fig.update_layout(
        title=f'{Z} on Smith Chart',
        showlegend=True
    )
    fig.show()

def calculate_Q(frequencies, s11_db, f_resonance):
    bandwidth_criterion = -3  # -3 dB bandwidth criterion
    s11_db = np.array(s11_db)
    print(f"\nS11 in dB : {s11_db}")
    threshold = np.min(s11_db) - bandwidth_criterion
    print(f"Threshold for -3 dB bandwidth : {threshold:.2f} dB")

    # Find indices where S11 is below the threshold
    in_band = np.where(s11_db <= threshold)[0]
    print(f"indices where S11 <= {threshold:.2f} dB : {in_band}")

    if len(in_band) < 2:
        print("Unable to determine the -3 dB bandwidth.")
        return None

    f_low = frequencies[in_band[0]]
    f_high = frequencies[in_band[-1]]
    bandwidth = f_high - f_low

    Q = f_resonance / bandwidth

    print(f"\nCalculation of Q:")
    print(f"→ Bandwidth (-3 dB) : {(bandwidth / 1e6):.2f} MHz")
    print(f"→ Quality factor Q  : {Q:.2f}")

    return Q

def plot_impedance_curve(impedances, fLow, fHigh, f_resonance=None):
    plt.style.use('seaborn-v0_8-talk')
    plt.rcParams['font.family'] = 'Lucida Console'
    plt.rcParams['font.size'] = 11
    frequencies = np.linspace(fLow, fHigh, len(impedances))
    frequencies_mhz = np.array(frequencies) / 1e6
    real_parts = [z.real for z in impedances]
    imag_parts = [z.imag for z in impedances]

    fig_size = 12
    Fibonacci = (1 + np.sqrt(5)) / 2
    plt.figure(figsize=(fig_size, fig_size / Fibonacci))
    plt.plot(frequencies_mhz, real_parts, label="Resistance (Re(Z))", color='red', linewidth=2.5)
    plt.plot(frequencies_mhz, imag_parts, label="Reactance (Im(Z))", color='blue', linewidth=2.5)

    # If f_resonance is given, draw the vertical line
    if f_resonance is not None:
        idx_res = np.argmin(np.abs(frequencies - f_resonance))
        f_res_mhz = frequencies_mhz[idx_res]
        R_res = real_parts[idx_res]
        X_res = imag_parts[idx_res]
        plt.axvline(f_res_mhz, color='green', linestyle='--', 
                    label=f"Resonance: {f_res_mhz:.2f} MHz\nRe(Z)={R_res:.2f} Ω, Im(Z)={X_res:.2f} Ω")

    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Impedance (Ω)")
    plt.title("Impedance vs Frequency")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_s11_curve(s11_db, fLow, fHigh, fC=None, show_min=False):
    plt.style.use('seaborn-v0_8-talk')
    plt.rcParams['font.family'] = 'Lucida Console'
    plt.rcParams['font.size'] = 11
    frequencies = np.linspace(fLow, fHigh, len(s11_db))
    frequencies_mhz = np.array(frequencies) / 1e6
    s11_db = np.array(s11_db)

    # Find the minimum of S11
    min_index = np.argmin(s11_db)
    f_resonance = frequencies[min_index] / 1e6
    s11_min = s11_db[min_index]

    # Plotting
    fig_size = 12
    Fibonacci = (1 + np.sqrt(5)) / 2
    plt.figure(figsize=(fig_size, fig_size / Fibonacci))
    plt.plot(frequencies_mhz, s11_db, label="S11 (dB)", color='blue', linewidth=2.5)
    
    if show_min:
        plt.plot(f_resonance, s11_min, 'ro', label=f"Resonance: {f_resonance:.2f} MHz (S11={s11_min:.2f} dB)", linewidth=2.5)
    
    if fC is not None:
        fC_mhz = fC / 1e6
        idx_fc = np.argmin(np.abs(frequencies - fC))
        s11_fc = s11_db[idx_fc]
        plt.axvline(fC_mhz, color='green', linestyle='--', 
                    label=f"fC = {fC_mhz:.2f} MHz (S11={s11_fc:.2f} dB)")

    plt.xlabel("Frequency (MHz)")
    plt.ylabel("S11 (dB)")
    plt.title("S11 vs Frequency")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_s11_curve_CST_MoM(s11_db, fLow, fHigh, fC=None, cst_freq_mhz=None, cst_s11_db=None):
    plt.style.use('seaborn-v0_8-talk')
    plt.rcParams['font.family'] = 'Lucida Console'
    plt.rcParams['font.size'] = 11
    frequencies = np.linspace(fLow, fHigh, len(s11_db))
    frequencies_mhz = frequencies / 1e6
    s11_db = np.array(s11_db)

    # Find the minimum of S11 (Python)
    min_index = np.argmin(s11_db)
    f_resonance = frequencies[min_index] / 1e6
    s11_min = s11_db[min_index]

    fig_size = 12
    Fibonacci = (1 + np.sqrt(5)) / 2
    plt.figure(figsize=(fig_size, fig_size / Fibonacci))
    
    # Python-computed curve
    plt.plot(frequencies_mhz, s11_db, label="S11 (MoM_solver)", color='blue', linewidth=2.5)
    plt.plot(f_resonance, s11_min, 'ro', label=f"MoM_solver: {f_resonance:.2f} MHz (S11={s11_min:.2f} dB)", linewidth=2.5)
    
    # CST curve if provided
    if cst_freq_mhz is not None and cst_s11_db is not None:
        plt.plot(cst_freq_mhz, cst_s11_db, label="S11 (CST)", color='red', linestyle='--', linewidth=2.5)
        # Find the minimum of S11 (CST)
        cst_s11_db = np.array(cst_s11_db)
        min_cst_index = np.argmin(cst_s11_db)
        cst_f_resonance = cst_freq_mhz[min_cst_index]
        cst_s11_min = cst_s11_db[min_cst_index]
        plt.plot(cst_f_resonance, cst_s11_min, 'ms', label=f"CST: {cst_f_resonance:.2f} MHz (S11={cst_s11_min:.2f} dB)", markersize=10)

    # Central frequency
    if fC is not None:
        fC_mhz = fC / 1e6
        idx_fc = np.argmin(np.abs(frequencies - fC))
        s11_fc = s11_db[idx_fc]
        plt.axvline(fC_mhz, color='green', linestyle='--', 
                    label=f"fC = {fC_mhz:.2f} MHz (S11={s11_fc:.2f} dB)", linewidth=2.5)

    plt.xlabel("Frequency (MHz)")
    plt.ylabel("S11 (dB)")
    plt.title("S11 vs Frequency")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_s11_curve_MoM_vs_Experiment(s11_db_mom, fLow, fHigh, s11_db_exp=None, exp_freq_mhz=None, fC=None):
    """
    Plot S11 comparison between MoM_solver and experimental measurements.

    Parameters:
        s11_db_mom: S11 values (dB) from MoM_solver (array-like)
        fLow: Start frequency (Hz)
        fHigh: End frequency (Hz)
        s11_db_exp: S11 values (dB) from experiment (array-like, optional)
        exp_freq_mhz: Frequencies (MHz) for experimental data (array-like, optional)
        fC: Central frequency (Hz, optional)
    """
    plt.style.use('seaborn-v0_8-talk')
    plt.rcParams['font.family'] = 'Lucida Console'
    plt.rcParams['font.size'] = 11

    frequencies = np.linspace(fLow, fHigh, len(s11_db_mom))
    frequencies_mhz = frequencies / 1e6
    s11_db_mom = np.array(s11_db_mom)

    # Find minimum S11 for MoM_solver
    min_index_mom = np.argmin(s11_db_mom)
    f_resonance_mom = frequencies_mhz[min_index_mom]
    s11_min_mom = s11_db_mom[min_index_mom]

    fig_size = 12
    Fibonacci = (1 + np.sqrt(5)) / 2
    fig = plt.figure(figsize=(fig_size, fig_size / Fibonacci))

    # Plot MoM_solver curve
    plt.plot(frequencies_mhz, s11_db_mom, label="S11 (MoM_solver)", color='blue', linewidth=2.5)
    plt.plot(f_resonance_mom, s11_min_mom, 'ro', label=f"MoM_solver: {f_resonance_mom:.2f} MHz (S11={s11_min_mom:.2f} dB)", markersize=8)

    # Plot experimental curve if provided
    if s11_db_exp is not None and exp_freq_mhz is not None:
        s11_db_exp = np.array(s11_db_exp)
        exp_freq_mhz = np.array(exp_freq_mhz)
        plt.plot(exp_freq_mhz, s11_db_exp, label="S11 (Experiment)", color='orange', linestyle='--', linewidth=2.5)
        # Find minimum S11 for experiment
        min_exp_index = np.argmin(s11_db_exp)
        exp_f_resonance = exp_freq_mhz[min_exp_index]
        exp_s11_min = s11_db_exp[min_exp_index]
        plt.plot(exp_f_resonance, exp_s11_min, 'ms', label=f"Experiment: {exp_f_resonance:.2f} MHz (S11={exp_s11_min:.2f} dB)", markersize=10)

    # Central frequency marker
    if fC is not None:
        fC_mhz = fC / 1e6
        idx_fc = np.argmin(np.abs(frequencies_mhz - fC_mhz))
        s11_fc = s11_db_mom[idx_fc]
        plt.axvline(fC_mhz, color='green', linestyle='--',
                    label=f"fC = {fC_mhz:.2f} MHz (S11={s11_fc:.2f} dB)", linewidth=2.5)

    plt.xlabel("Frequency (MHz)")
    plt.ylabel("S11 (dB)")
    plt.title("S11 Comparison: MoM_solver vs Experimental Measurement")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    output_dir_fig_image = "data/fig_image/"
    if not os.path.exists(output_dir_fig_image):
        os.makedirs(output_dir_fig_image)
        print(f"Directory created: {output_dir_fig_image}")
    pdf_path = os.path.join(output_dir_fig_image, "MoM_solver_vs_experimental.pdf")
    fig.patch.set_alpha(0.0)  # Transparent background
    plt.savefig(pdf_path, format="pdf", transparent=True)
    print(f"\nImage saved as PDF (transparent background, minimal margins): {pdf_path}\n")
    plt.show()

def plot_impedance_curve_CST_MoM(impedances, fLow, fHigh, f_resonance,
                                 cst_freq_mhz=None, cst_re_z=None, cst_im_z=None):
    # Clean style
    plt.style.use('fivethirtyeight')
    plt.rcParams['font.family'] = 'JetBrains Mono'

    # Frequencies associated with the Python curve
    frequencies = np.linspace(fLow, fHigh, len(impedances))
    frequencies_mhz = frequencies / 1e6
    real_parts = np.real(impedances)
    imag_parts = np.imag(impedances)

    # Find the closest resonance frequency
    idx_res = np.argmin(np.abs(frequencies - f_resonance))
    f_res_mhz = frequencies_mhz[idx_res]
    R_res = real_parts[idx_res]
    X_res = imag_parts[idx_res]

    # Plot
    fig_size = 12
    Fibonacci = (1 + np.sqrt(5)) / 2
    plt.figure(figsize=(fig_size, fig_size / Fibonacci))

    # Python curve
    plt.plot(frequencies_mhz, real_parts, label="Re(Z) (Python)", color='red', linewidth=2.5)
    plt.plot(frequencies_mhz, imag_parts, label="Im(Z) (Python)", color='blue', linewidth=2.5)

    # CST curve if provided
    if cst_freq_mhz is not None and cst_re_z is not None and cst_im_z is not None:
        plt.plot(cst_freq_mhz, cst_re_z, label="Re(Z) (CST)", color='darkred', linestyle='--', linewidth=2)
        plt.plot(cst_freq_mhz, cst_im_z, label="Im(Z) (CST)", color='darkblue', linestyle='--', linewidth=2)

    # Resonance frequency
    plt.axvline(f_res_mhz, color='green', linestyle='--',
                label=f"Resonance: {f_res_mhz:.2f} MHz\nRe(Z)={R_res:.2f} Ω, Im(Z)={X_res:.2f} Ω")

    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Impedance (Ω)")
    plt.title("Impedance vs Frequency")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def load_cst_data(filepath):
    """
    Load CST-exported data in text/tab-delimited format.
    Returns: frequencies [MHz], S11 magnitudes, S11 phases [°]
    """
    frequencies = []
    mag_s11 = []
    phase_s11 = []

    with open(filepath, 'r') as f:
        for line in f:
            # Ignore comment lines
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue  # invalid line
            freq = float(parts[0])
            mag = float(parts[1])
            phase = float(parts[2])
            frequencies.append(freq)
            mag_s11.append(mag)
            phase_s11.append(phase)
    
    return frequencies, mag_s11, phase_s11

def s11_to_impedance(mag, phase_deg, z0=50):
    """Convert S11 (magnitude, phase in degrees) to complex impedance."""
    phase_rad = np.deg2rad(phase_deg)
    gamma = mag * np.exp(1j * phase_rad)
    z = z0 * (1 + gamma) / (1 - gamma)
    return z

def plot_smith_chart_CST_MoM(frequencies_own, Z_own, frequencies_cst, mag_cst, phase_cst, fC=None, z0=50):
    fig_size = 15
    Fibonacci = (1 + np.sqrt(5)) / 2
    plt.figure(figsize=(fig_size, fig_size / Fibonacci))

    fig = go.Figure()

    # === Own data ===
    norm_r_own = [z.real / z0 for z in Z_own]
    norm_x_own = [z.imag / z0 for z in Z_own]
    labels_own = [f'Z = {z:.3f} @ {f*1e-6:.2f} MHz' for z, f in zip(Z_own, frequencies_own)]

    fig.add_trace(go.Scattersmith(
        real=norm_r_own,
        imag=norm_x_own,
        mode='lines+markers',
        marker=dict(size=4, color='blue'),
        name='MoM solver',
        text=labels_own,
        hoverinfo='text'
    ))

    # === CST data ===
    Z_cst = [s11_to_impedance(m, p, z0) for m, p in zip(mag_cst, phase_cst)]
    norm_r_cst = [z.real / z0 for z in Z_cst]
    norm_x_cst = [z.imag / z0 for z in Z_cst]
    labels_cst = [f'Z = {z:.3f} @ {f:.2f} MHz' for z, f in zip(Z_cst, frequencies_cst)]

    fig.add_trace(go.Scattersmith(
        real=norm_r_cst,
        imag=norm_x_cst,
        mode='lines+markers',
        marker=dict(size=4, color='green'),
        name='CST',
        text=labels_cst,
        hoverinfo='text'
    ))

    # === Highlight central frequency fC on both curves ===
    if fC is not None:
        # For MoM solver
        idx_own = np.argmin(np.abs(np.array(frequencies_own) - fC))
        z_own_fc = Z_own[idx_own]
        fig.add_trace(go.Scattersmith(
            real=[z_own_fc.real / z0],
            imag=[z_own_fc.imag / z0],
            mode='markers',
            marker=dict(size=12, color='red', symbol='diamond'),
            name=f'fC MoM ({fC/1e6:.2f} MHz)',
            text=[f'fC MoM: Z={z_own_fc:.2f} @ {fC/1e6:.2f} MHz'],
            hoverinfo='text'
        ))

        # For CST
        frequencies_cst_arr = np.array(frequencies_cst)
        idx_cst = np.argmin(np.abs(frequencies_cst_arr - fC/1e6))
        z_cst_fc = Z_cst[idx_cst]
        fig.add_trace(go.Scattersmith(
            real=[z_cst_fc.real / z0],
            imag=[z_cst_fc.imag / z0],
            mode='markers',
            marker=dict(size=12, color='orange', symbol='diamond'),
            name=f'fC CST ({frequencies_cst[idx_cst]:.2f} MHz)',
            text=[f'fC CST: Z={z_cst_fc:.2f} @ {frequencies_cst[idx_cst]:.2f} MHz'],
            hoverinfo='text'
        ))

    fig.update_layout(
        title='Superimposed Smith Chart - MoM Solver vs CST',
        showlegend=True,
        width=fig_size * 80,
        height=int((fig_size / Fibonacci) * 80)
    )

    fig.show()