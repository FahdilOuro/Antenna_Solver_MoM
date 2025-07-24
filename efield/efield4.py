import os
from matplotlib import pyplot as plt
import numpy as np
import plotly.graph_objects as go

def plot_smith_chart(impedances, frequencies, fC=None, save_image=False):
    # fig_size and plt.figure are not used by plotly, only by matplotlib
    # To set figure size in plotly, use width and height in update_layout
    fig_size = 15
    Fibonacci = (1 + np.sqrt(5)) / 2

    fig = go.Figure()

    Normalized_Z0 = 50  # Normalized impedance reference (Z0)

    # Prepare data for Scattersmith
    smith_points = []
    for idx, Z in enumerate(impedances):
        r = Z.real / Normalized_Z0  # Normalized resistance
        x = Z.imag / Normalized_Z0  # Normalized reactance
        smith_points.append(dict(
            real=r,
            imag=x,
            freq=frequencies[idx],
            label=f'Z = {Z:.3f} at {frequencies[idx]*1e-6} MHz'
        ))

    # Plot all points except fC in default color
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
        # Highlight the specific frequency in red
        pt = smith_points[idx_fC]
        fig.add_trace(go.Scattersmith(
            real=[pt['real']],
            imag=[pt['imag']],
            mode='markers',
            marker=dict(size=10, color='red', symbol='circle'),
            name=f'Impedance at fC={fC*1e-6} MHz',
            text=[pt['label']],
            hoverinfo='text'
        ))
    else:
        # Plot all points normally if fC is not specified or not found
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
        width=1000,  # Set your desired width here
        height=int(1000 / Fibonacci)  # Set your desired height here
    )

    fig.show()

    if save_image:
        # Chemin du dossier de sortie
        output_dir_fig_image = "data/fig_image/"
        
        # Crée le dossier s'il n'existe pas
        if not os.path.exists(output_dir_fig_image):
            os.makedirs(output_dir_fig_image)
            print(f"Dossier créé : {output_dir_fig_image}")
        
        # Nom du fichier PDF à enregistrer
        pdf_path = os.path.join(output_dir_fig_image, "ifa_M_opti3_Smith_chart.pdf")
        
        # Mettre le fond transparent et supprimer les marges blanches
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=10, b=10)
        )
        
        # Sauvegarde de la figure
        fig.write_image(pdf_path, format="pdf")
        print(f"\nImage sauvegardée au format PDF (fond transparent, marges minimales) : {pdf_path}\n")

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
    print(f"\nS11 en dB : {s11_db}")
    threshold = np.min(s11_db) - bandwidth_criterion
    print(f"Seuil pour la bande passante à -3 dB : {threshold:.2f} dB")

    # Trouver les indices où S11 est en dessous du seuil
    in_band = np.where(s11_db <= threshold)[0]
    print(f"Indices où S11 <= {threshold:.2f} dB : {in_band}")

    if len(in_band) < 2:
        print("Impossible de déterminer la bande passante à -3 dB.")
        return None

    f_low = frequencies[in_band[0]]
    f_high = frequencies[in_band[-1]]
    bandwidth = f_high - f_low

    Q = f_resonance / bandwidth

    print(f"\nCalcul de Q :")
    print(f"→ Bande passante (-3 dB) : {(bandwidth / 1e6):.2f} MHz")
    print(f"→ Facteur de qualité Q  : {Q:.2f}")

    return Q

def plot_impedance_curve(impedances, fLow, fHigh, f_resonance=None):
    plt.style.use('fivethirtyeight')
    plt.rcParams['font.family'] = 'JetBrains Mono'
    frequencies = np.linspace(fLow, fHigh, len(impedances))
    frequencies_mhz = np.array(frequencies) / 1e6
    real_parts = [z.real for z in impedances]
    imag_parts = [z.imag for z in impedances]

    fig_size = 12
    Fibonacci = (1 + np.sqrt(5)) / 2
    plt.figure(figsize=(fig_size, fig_size / Fibonacci))
    plt.plot(frequencies_mhz, real_parts, label="Résistance (Re(Z))", color='red', linewidth=2.5)
    plt.plot(frequencies_mhz, imag_parts, label="Réactance (Im(Z))", color='blue', linewidth=2.5)

    # Si f_resonance est donné, tracer la ligne verticale
    if f_resonance is not None:
        idx_res = np.argmin(np.abs(frequencies - f_resonance))
        f_res_mhz = frequencies_mhz[idx_res]
        R_res = real_parts[idx_res]
        X_res = imag_parts[idx_res]
        plt.axvline(f_res_mhz, color='green', linestyle='--', 
                    label=f"Résonance: {f_res_mhz:.2f} MHz\nRe(Z)={R_res:.2f} Ω, Im(Z)={X_res:.2f} Ω")

    plt.xlabel("Fréquence (MHz)")
    plt.ylabel("Impédance (Ω)")
    plt.title("Évolution de l'impédance vs Fréquence")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_s11_curve(s11_db, fLow, fHigh, fC=None, show_min=False):
    plt.style.use('fivethirtyeight')
    plt.rcParams['font.family'] = 'JetBrains Mono'
    frequencies = np.linspace(fLow, fHigh, len(s11_db))
    frequencies_mhz = np.array(frequencies) / 1e6
    s11_db = np.array(s11_db)

    # Trouver le minimum de S11
    min_index = np.argmin(s11_db)
    f_resonance = frequencies[min_index] / 1e6
    s11_min = s11_db[min_index]

    # Tracé
    fig_size = 12
    Fibonacci = (1 + np.sqrt(5)) / 2
    plt.figure(figsize=(fig_size, fig_size / Fibonacci))
    plt.plot(frequencies_mhz, s11_db, label="S11 (dB)", color='blue', linewidth=2.5)
    
    if show_min:
        plt.plot(f_resonance, s11_min, 'ro', label=f"Résonance: {f_resonance:.2f} MHz (S11={s11_min:.2f} dB)", linewidth=2.5)
    
    if fC is not None:
        fC_mhz = fC / 1e6
        idx_fc = np.argmin(np.abs(frequencies - fC))
        s11_fc = s11_db[idx_fc]
        plt.axvline(fC_mhz, color='green', linestyle='--', 
                    label=f"fC = {fC_mhz:.2f} MHz (S11={s11_fc:.2f} dB)")

    plt.xlabel("Fréquence (MHz)")
    plt.ylabel("S11 (dB)")
    plt.title("Courbe de S11 vs Fréquence")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_s11_curve_CST_MoM(s11_db, fLow, fHigh, fC=None, cst_freq_mhz=None, cst_s11_db=None):
    plt.style.use('fivethirtyeight')
    plt.rcParams['font.family'] = 'JetBrains Mono'
    frequencies = np.linspace(fLow, fHigh, len(s11_db))
    frequencies_mhz = frequencies / 1e6
    s11_db = np.array(s11_db)

    # Trouver le minimum de S11 (Python)
    min_index = np.argmin(s11_db)
    f_resonance = frequencies[min_index] / 1e6
    s11_min = s11_db[min_index]

    fig_size = 12
    Fibonacci = (1 + np.sqrt(5)) / 2
    plt.figure(figsize=(fig_size, fig_size / Fibonacci))
    
    # Courbe calculée en Python
    plt.plot(frequencies_mhz, s11_db, label="S11 (MoM_solver)", color='blue', linewidth=2.5)
    plt.plot(f_resonance, s11_min, 'ro', label=f"MoM_solver: {f_resonance:.2f} MHz (S11={s11_min:.2f} dB)", linewidth=2.5)
    
    # Courbe CST si fournie
    if cst_freq_mhz is not None and cst_s11_db is not None:
        plt.plot(cst_freq_mhz, cst_s11_db, label="S11 (CST)", color='red', linestyle='--', linewidth=2.5)
        # Trouver le minimum de S11 (CST)
        cst_s11_db = np.array(cst_s11_db)
        min_cst_index = np.argmin(cst_s11_db)
        cst_f_resonance = cst_freq_mhz[min_cst_index]
        cst_s11_min = cst_s11_db[min_cst_index]
        plt.plot(cst_f_resonance, cst_s11_min, 'ms', label=f"CST: {cst_f_resonance:.2f} MHz (S11={cst_s11_min:.2f} dB)", markersize=10)

    # Fréquence centrale
    if fC is not None:
        fC_mhz = fC / 1e6
        idx_fc = np.argmin(np.abs(frequencies - fC))
        s11_fc = s11_db[idx_fc]
        plt.axvline(fC_mhz, color='green', linestyle='--', 
                    label=f"fC = {fC_mhz:.2f} MHz (S11={s11_fc:.2f} dB)", linewidth=2.5)

    plt.xlabel("Fréquence (MHz)")
    plt.ylabel("S11 (dB)")
    plt.title("Courbe de S11 vs Fréquence")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_impedance_curve_CST_MoM(impedances, fLow, fHigh, f_resonance,
                         cst_freq_mhz=None, cst_re_z=None, cst_im_z=None):
    # Style propre
    plt.style.use('fivethirtyeight')
    plt.rcParams['font.family'] = 'JetBrains Mono'

    # Fréquences associées à la courbe Python
    frequencies = np.linspace(fLow, fHigh, len(impedances))
    frequencies_mhz = frequencies / 1e6
    real_parts = np.real(impedances)
    imag_parts = np.imag(impedances)

    # Trouver la fréquence de résonance la plus proche
    idx_res = np.argmin(np.abs(frequencies - f_resonance))
    f_res_mhz = frequencies_mhz[idx_res]
    R_res = real_parts[idx_res]
    X_res = imag_parts[idx_res]

    # Tracé
    fig_size = 12
    Fibonacci = (1 + np.sqrt(5)) / 2
    plt.figure(figsize=(fig_size, fig_size / Fibonacci))

    # Courbe Python
    plt.plot(frequencies_mhz, real_parts, label="Re(Z) (Python)", color='red', linewidth=2.5)
    plt.plot(frequencies_mhz, imag_parts, label="Im(Z) (Python)", color='blue', linewidth=2.5)

    # Courbe CST si fournie
    if cst_freq_mhz is not None and cst_re_z is not None and cst_im_z is not None:
        plt.plot(cst_freq_mhz, cst_re_z, label="Re(Z) (CST)", color='darkred', linestyle='--', linewidth=2)
        plt.plot(cst_freq_mhz, cst_im_z, label="Im(Z) (CST)", color='darkblue', linestyle='--', linewidth=2)

    # Fréquence de résonance
    plt.axvline(f_res_mhz, color='green', linestyle='--',
                label=f"Résonance: {f_res_mhz:.2f} MHz\nRe(Z)={R_res:.2f} Ω, Im(Z)={X_res:.2f} Ω")

    plt.xlabel("Fréquence (MHz)")
    plt.ylabel("Impédance (Ω)")
    plt.title("Évolution de l'impédance vs Fréquence")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def load_cst_data(filepath):
    """
    Charge les données CST exportées en format texte/tabulé.
    Retourne : fréquences [MHz], magnitudes S11, phases S11 [°]
    """
    frequencies = []
    mag_s11 = []
    phase_s11 = []

    with open(filepath, 'r') as f:
        for line in f:
            # Ignorer les lignes de commentaire
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue  # ligne invalide
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

    # === Données "propres" ===
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

    # === Données CST ===
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

    # === Mettre en évidence la fréquence fC sur les deux courbes ===
    if fC is not None:
        # Pour MoM solver
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

        # Pour CST
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
        title='Superposition Smith Chart - MoM Solveur vs CST',
        showlegend=True,
        width=fig_size * 80,  # 12*80=960px
        height=int((fig_size / Fibonacci) * 80)  # ~742px
    )

    fig.show()