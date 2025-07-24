from src.radiation_algorithm.radiation_algorithm import radiation_algorithm
import numpy as np

def simulate_frequency_sweep(frequencies, fC, ifa_meander_mat, feed_point, voltage_amplitude, Z0=50):
    s11_db = []
    impedances = []
    nPoints = len(frequencies)
    for idx, frequency in enumerate(frequencies):
        visualiser = (frequency == fC)
        impedance, *_ = radiation_algorithm(ifa_meander_mat, frequency, feed_point, voltage_amplitude, show=visualiser)
        impedances.append(impedance)
        s11 = (impedance - Z0) / (impedance + Z0)
        s11_db.append(20 * np.log10(abs(s11)))
        print(f"Simulation {idx+1}/{nPoints} | f = {frequency/1e6:.2f} MHz | S11 = {s11_db[-1]:.2f} dB")
    # Résultats
    min_index = np.argmin(s11_db)
    f_resonance = frequencies[min_index]
    Z_at_res = impedances[min_index]
    R_res = Z_at_res.real
    X_res = Z_at_res.imag

    print(f"\n📡 Résultats de simulation :")
    print(f"→ Fréquence de résonance = {f_resonance / 1e6:.2f} MHz")
    print(f"→ Impédance à f_res      = {Z_at_res:.2f} Ω")

    return s11_db, impedances, R_res, X_res, f_resonance, min_index

def simulate_freq_loop_test(
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

    alpha = 0.1  # Empirique — à ajuster selon résultats

    # --- Critères de convergence ---
    if freq_error < accuracy and X_error < 0.1 and s11_min < -10:
        has_converged = True
        print(f"\n✅ Convergence atteinte !")
    else:
        print("\n❌ Pas de convergence —> Réajustement des paramètres...\n")

        # [1] Nombre de méandres (grossier)
        if f_resonance >= fHigh:
            print("📉 Fréquence trop haute —> + méandres")
            new_Nombre_meandre = min(new_Nombre_meandre + 1, int((L / hauteur) * 2))
        elif f_resonance <= fLow:
            print("📈 Fréquence trop basse —> - méandres")
            new_Nombre_meandre = max(new_Nombre_meandre - 1, 1)

        # [2] Réglage fréquence par longueur électrique (distance short)
        freq_corr_factor = (fC / f_resonance) ** 0.5
        new_distance_short *= freq_corr_factor

        # [3] Réglage de R (partie réelle) via largeur de trace
        R_diff = Z0 - R_res
        if abs(R_diff) > 1:
            wid_corr_factor = 1 + 0.3 * (R_diff / Z0)
            wid_corr_factor = np.clip(wid_corr_factor, 0.8, 1.2)
            new_wid *= wid_corr_factor

            # [3.5] Compensation de l'effet indirect sur f_res
            delta_wid = new_wid - wid
            wid_effect = 1 - alpha * (delta_wid / wid)
            new_distance_short *= wid_effect


        # [4] Réglage de X (partie imaginaire) via décalage fin
        if abs(X_res) > 1:
            X_corr = 1 - 0.2 * np.sign(X_res) * min(X_error, 0.5)
            new_distance_short *= X_corr

        # --- Sécurité : limites physiques ---
        new_wid = np.clip(new_wid, 0.5e-3, largeur / 2)
        new_distance_short = np.clip(new_distance_short, 0.5e-3, hauteur - new_wid)

        print(f"🔧 Nouveaux paramètres :")
        print(f"• Distance short-feed : {new_distance_short * 1e3:.2f} mm")
        print(f"• Largeur de piste    : {new_wid * 1e3:.2f} mm")
        print(f"• Nombre de méandres  : {new_Nombre_meandre}\n")

    # --- Retour ---
    return s11_db, f_resonance, new_distance_short, new_wid, new_Nombre_meandre, has_converged, impedances

# deuxieme version de la fonction simulate_freq_loop_test


def simulate_freq_loop_test_version_2(
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


def loop_in_interval(fLow, fHigh, nPoints, fC, accuracy, ifa_meander_mat, feed_point, distance_short, wid, hauteur):
    print("\n############### loop_in_interval ###############################\n")
    Z0 = 50  # Impédance caractéristique en ohms
    frequencies = np.linspace(fLow, fHigh, nPoints)
    s11_db = []
    voltage_amplitude = 0.5
    has_converged = False
    count = 0
    show = False
    impedances = []
    new_distance_short = distance_short
    new_wid = wid

    index_fC = 0

    for frequency in frequencies:
        print(f"Simulation Numéro {count + 1}\n")
        if frequency == fC:
            show = True
            index_fC = count
        else:
            show = False
        impedance, *_ = radiation_algorithm(ifa_meander_mat, frequency, feed_point, voltage_amplitude, show)
        impedances.append(impedance)
        s11 = (impedance - Z0) / (impedance + Z0)
        s11_db.append(20 * np.log10(abs(s11)))
        print(f"paramètre S11 = {s11_db[count]} db\n")
        count += 1

    # Trouver la fréquence de résonance (minimum S11)
    min_index = np.argmin(s11_db)

    # Trouver l'index de la valeur de fC ou la plus proche de fC
    # index_fC = np.argmin(np.abs(frequencies - fC))

    f_resonance = frequencies[min_index]
    R_I_min_index = impedances[min_index].real
    print(f"R_I_min_index = {R_I_min_index}")
    print(f"\nFréquence de résonance : {f_resonance / 1e6:.2f} MHz\n")

    # Comparaison à la fréquence de coupure
    error = abs((fC - f_resonance) / fC)
    s11_db_min_index = s11_db[min_index]

    if error < accuracy:
        if s11_db_min_index < -10:
            has_converged = True
            print(f" Convergence atteinte : |f_res - fC| = {error:.2f} Hz ≤ {accuracy}")
            return s11_db, f_resonance, new_distance_short, new_wid, has_converged, impedances
        else:
            print(" ")
            print("Opti Freq found but no matching yet ! \n")
            print(f"R_I_min_index = {R_I_min_index} Ohm\n")
            new_distance_short = distance_short * pow((Z0 / R_I_min_index), 2)
    else:

        new_distance_short = distance_short * (Z0 / R_I_min_index)
        print("f_resonance > fLow and f_resonance < fHigh\n")
        new_wid = wid * pow((fC / f_resonance), 2)
        
        if new_wid < 0.5 / 1000:
            new_wid = 0.5 / 1000

        DSF_max = hauteur - new_wid
        
        if new_distance_short > DSF_max:
            print("new_distance_short > DSF_max\n")
            new_distance_short = DSF_max
            # new_distance_short = DSF_max - distance_short * np.sqrt(Z0 / R_I_min_index)
            print(f"new_distance_short = {new_distance_short * 1000}\n")

        if new_distance_short < 0.5 / 1000 or new_distance_short < new_wid:
            print("new_distance_short < 0.5 / 1000\n")
            new_distance_short = 0.5 / 1000 + new_wid
            print(f"new_distance_short = {new_distance_short * 1000}\n")

        print(f" Pas de convergence : |f_res - fC| = {error:.2f} Hz > {accuracy}")
        print(f"\n1...........short feed ...... dans la fonction = {new_distance_short * 1000}\n")
    
    return s11_db, f_resonance, new_distance_short, new_wid, has_converged, impedances