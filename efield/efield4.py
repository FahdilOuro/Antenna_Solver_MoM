from matplotlib import pyplot as plt
import numpy as np
import plotly.graph_objects as go

def plot_smith_chart(impedances, frequencies, fC=None):
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
        title='Abaque de Smith',
        showlegend=True,
        width=1000,  # Set your desired width here
        height=int(1000 / Fibonacci)  # Set your desired height here
    )

    fig.show()

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

