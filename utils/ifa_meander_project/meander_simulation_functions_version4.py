from src.radiation_algorithm.radiation_algorithm import radiation_algorithm
import numpy as np

def simulate_frequency_sweep(frequencies, fC, ifa_meander_mat, feed_point, voltage_amplitude=0.5, Z0=50):
    s11_db = []
    impedances = []
    nPoints = len(frequencies)
    fc_index = np.where(frequencies == fC)[0][0]
    for idx, frequency in enumerate(frequencies):
        visualiser = (frequency == fC)
        impedance, _ = radiation_algorithm(ifa_meander_mat, frequency, feed_point, voltage_amplitude, show=visualiser)
        impedances.append(impedance)
        s11 = (impedance - Z0) / (impedance + Z0)
        s11_db.append(20 * np.log10(abs(s11)))
        print(f"Simulation {idx+1}/{nPoints} | f = {frequency/1e6:.2f} MHz | S11 = {s11_db[-1]:.2f} dB")
    # Résultats
    min_index = np.argmin(s11_db)
    f_resonance = frequencies[min_index]
    Z_at_res = impedances[min_index]
    Z_at_fc = impedances[fc_index]
    R_res = Z_at_res.real
    X_res = Z_at_res.imag

    print(f"\n📡 Résultats de simulation :")
    print(f"→ Fréquence de résonance = {f_resonance / 1e6:.2f} MHz")
    print(f"→ Impédance à f_res      = {Z_at_res:.2f} Ω")
    print(f"→ Impédance à fC         = {Z_at_fc:.2f} Ω")
    print(f"→ S11 à f_res            = {s11_db[min_index]:.2f} dB")

    return s11_db, impedances, R_res, X_res, f_resonance, min_index

def simulate_freq_loop_version4(
    fLow, fHigh, nPoints, fC, accuracy,
    ifa_meander_mat, feed_point, distance_short,
    wid, L, hauteur, largeur, L_short, Nombre_meandre
):
    Z0 = 50  # Impédance caractéristique
    frequencies = np.linspace(fLow, fHigh, nPoints)
    s11_db = []
    impedances = []
    voltage_amplitude = 0.5
    has_converged = False

    # Variables ajustables
    new_distance_short = distance_short
    new_wid = wid
    new_Nombre_meandre = Nombre_meandre

    s11_db, impedances, R_res, X_res, f_resonance, min_index = simulate_frequency_sweep(
        frequencies, fC, ifa_meander_mat, feed_point, voltage_amplitude, Z0
        )

    # Erreurs
    freq_error = abs((fC - f_resonance) / fC)
    R_error = abs(R_res - Z0) / Z0
    X_error = abs(X_res) / Z0
    s11_min = s11_db[min_index]

    # --- Critères de convergence ---
    if freq_error < accuracy and X_error < 0.1 and s11_min < -10:
        has_converged = True
        print(f"\n✅ Convergence atteinte !")
    else:
        print("\n❌ Pas de convergence —> Réajustement couplé intelligent...\n")

        # --- Ajustement simultané des méandres et de la largeur ---
        if f_resonance >= fHigh:
            print("📉 f trop haute —> + méandres")
            new_Nombre_meandre = min(new_Nombre_meandre + 1, int((L / hauteur) * 2))

        elif f_resonance <= fLow:
            print("📈 f trop basse —> - méandres")
            new_Nombre_meandre = max(new_Nombre_meandre - 1, 1)

        # --- Ajustement distance_short pour correction fine de fréquence
        freq_corr_factor = (fC / f_resonance) ** 0.5
        new_distance_short *= freq_corr_factor

        R_diff = Z0 - R_res
        if abs(R_diff) > 1:
            wid_corr_factor = 1 + 0.3 * (R_diff / Z0)
            wid_corr_factor = np.clip(wid_corr_factor, 0.8, 1.2)
            new_wid *= wid_corr_factor

        # --- Ajustement de la réactance (X)
        if abs(X_res) > 1:
            X_corr = 1 - 0.2 * np.sign(X_res) * min(X_error, 0.5)
            new_distance_short *= X_corr

        # --- Sécurité : limites physiques ---
        new_wid = np.clip(new_wid, 0.5e-3, largeur / 2)
        new_distance_short = np.clip(new_distance_short, 0.5e-3, hauteur - new_wid)

        print(f"\n📐 Paramètres ajustés intelligemment :")
        print(f"• Distance short-feed : {new_distance_short * 1e3:.2f} mm")
        print(f"• Largeur de trace    : {new_wid * 1e3:.2f} mm")
        print(f"• Nombre de méandres  : {new_Nombre_meandre}\n")

    # --- Retour ---
    return s11_db, f_resonance, new_distance_short, new_wid, new_Nombre_meandre, has_converged, impedances