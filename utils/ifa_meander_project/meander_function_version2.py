import gmsh
import numpy as np
import math
import matplotlib.pyplot as plt
from utils.gmsh_function import *
from rwg.rwg1 import *
from rwg.rwg2 import *
from rwg.rwg3 import *
from rwg.rwg4 import *
from rwg.rwg5 import *

def calculate_nPoints(fLow, fHigh, fC, min_points=4):
    step = (fHigh - fLow) / (min_points - 1)
    if (fC - fLow) % step == 0:
        return min_points
    else:
        # Trouve le plus petit nPoints qui inclut fC
        nPoints = min_points
        while True:
            frequencies = np.linspace(fLow, fHigh, nPoints)
            if fC in frequencies:
                return nPoints
            nPoints += 1

def ifa_creation_new(L, largeur, hauteur, width, L_short = 1 / 1000):
    # — Initialisation
    x0 = 0
    y0 = hauteur / 2 - width / 2
    hauteur = hauteur - width

    # — Calcul du nombre de brins verticaux
    # N = int(np.floor((largeur / min_slot - 1)))
    N = int(np.floor((L - largeur) / hauteur))
    print(f"Number of meanders {N}")
    distance_meandre = (largeur - L_short) / N
    print(f"distance meandres {distance_meandre}")

    # longueur_meandre = (N + 1) * min_slot + N * hauteur

    x = np.zeros(2 * N + 3)
    y = np.zeros(2 * N + 3)

    x[0] = x0
    y[0] = y0

    direction = -1
    idx = 0
    calcul_actuel_longueur = 0
    horizontal = False
    vertical = False

    idx += 1
    x[idx] = x[idx - 1] + L_short
    y[idx] = y[idx - 1]

    for k in range(1, N + 1):
        # Vertical
        idx += 1
        x[idx] = x[idx - 1]
        y[idx] = y[idx - 1] + direction * hauteur
        calcul_actuel_longueur += hauteur
        direction = -direction

        # Horizontal
        idx += 1
        x[idx] = x[idx - 1] + distance_meandre - width / 4
        y[idx] = y[idx - 1]
        # calcul_actuel_longueur += distance_meandre - width / 2
    
    # print(f"last index = {idx}")

    # Horizontal
    idx += 1
    x[idx] = x[idx - 1] + distance_meandre - width / 4
    y[idx] = y[idx - 1]
    # calcul_actuel_longueur += distance_meandre - width / 2

    # Ajouter le dernier petit segment correctif
    # idx += 1
    x[idx] = x[idx - 1]
    y[idx] = y[idx - 1] + direction * hauteur
    calcul_actuel_longueur += distance_meandre
    
    """ print("\nlongueur_obtenue =", calcul_actuel_longueur)
    print("longueur_desiree =", L, "\n") """

    return x[:idx+1], y[:idx+1], N

def ifa_creation_optimisation(L, largeur, hauteur, width, Nombre_meandre, L_short = 2):
    # — Initialisation
    x0 = 0
    y0 = hauteur / 2 - width / 2
    hauteur = hauteur - width

    # — Calcul du nombre de brins verticaux
    # N = int(np.floor((largeur / min_slot - 1)))
    N = Nombre_meandre
    print(f"Number of meanders {N}")
    distance_meandre = (largeur - L_short) / N
    print(f"distance meandres {distance_meandre}")

    # longueur_meandre = (N + 1) * min_slot + N * hauteur

    x = np.zeros(2 * N + 3)
    y = np.zeros(2 * N + 3)

    x[0] = x0
    y[0] = y0

    direction = -1
    idx = 0
    calcul_actuel_longueur = 0
    horizontal = False
    vertical = False

    idx += 1
    x[idx] = x[idx - 1] + L_short
    y[idx] = y[idx - 1]

    for k in range(1, N + 1):
        # Vertical
        idx += 1
        x[idx] = x[idx - 1]
        y[idx] = y[idx - 1] + direction * hauteur
        calcul_actuel_longueur += hauteur
        direction = -direction

        # Horizontal
        idx += 1
        x[idx] = x[idx - 1] + distance_meandre - width / 4
        y[idx] = y[idx - 1]
        # calcul_actuel_longueur += distance_meandre - width / 2
    
    # print(f"last index = {idx}")

    # Horizontal
    idx += 1
    x[idx] = x[idx - 1] + distance_meandre - width / 4
    y[idx] = y[idx - 1]
    # calcul_actuel_longueur += distance_meandre - width / 2

    # Ajouter le dernier petit segment correctif
    # idx += 1
    x[idx] = x[idx - 1]
    y[idx] = y[idx - 1] + direction * hauteur
    calcul_actuel_longueur += distance_meandre
    
    """ print("\nlongueur_obtenue =", calcul_actuel_longueur)
    print("longueur_desiree =", L, "\n") """

    return x[:idx+1], y[:idx+1]

def trace_meander_new(x, y, Width):
    """
    Génère le contour épais (meander) autour min_slot'une ligne polygonale donnée.
    
    Paramètres :
        x : array-like, abscisses de la ligne centrale
        y : array-like, ordonnées de la ligne centrale
        Width : hauteur totale du contour (centré sur la ligne)
        
    Retourne :
        x_meander, y_meander : coordonnées du contour
    """
    x = np.array(x)
    y = np.array(y)
    n = len(x)

    x_meander = np.zeros(2 * n)
    y_meander = np.zeros(2 * n)

    # Premier point
    if x[0] == x[1] and y[0] > y[1]:
        x_meander[0]     = x[0] + Width / 2
        x_meander[2*n-1] = x[0] - Width / 2
        y_meander[0]     = y[0]
        y_meander[2*n-1] = y[0]
    elif x[0] == x[1] and y[0] < y[1]:
        x_meander[0]     = x[0] - Width / 2
        x_meander[2*n-1] = x[0] + Width / 2
        y_meander[0]     = y[0]
        y_meander[2*n-1] = y[0]
    elif y[0] == y[1]:
        x_meander[0]     = x[0]
        x_meander[2*n-1] = x[0]
        y_meander[0]     = y[0] + Width / 2
        y_meander[2*n-1] = y[0] - Width / 2

    # Dernier point
    if y[n-2] == y[n-1]:
        x_meander[n-1] = x[n-1]
        x_meander[n]   = x[n-1]
        y_meander[n-1] = y[n-1] + Width / 2
        y_meander[n]   = y[n-1] - Width / 2
    elif x[n-2] == x[n-1] and y[n-2] > y[n-1]:
        x_meander[n-1] = x[n-1] + Width / 2
        x_meander[n]   = x[n-1] - Width / 2
        y_meander[n-1] = y[n-1] - Width / 2  # modif
        y_meander[n]   = y[n-1] - Width / 2  # modif
    elif x[n-2] == x[n-1] and y[n-2] < y[n-1]:
        x_meander[n-1] = x[n-1] - Width / 2
        x_meander[n]   = x[n-1] + Width / 2
        y_meander[n-1] = y[n-1] + Width / 2  # modif
        y_meander[n]   = y[n-1] + Width / 2  # modif

    # Points intermédiaires
    j = 2 * n - 2
    for i in range(1, n - 1):
        if y[i-1] == y[i] and x[i] == x[i+1] and y[i] > y[i+1]:
            x_meander[i] = x[i] + Width / 2
            y_meander[i] = y[i] + Width / 2
            x_meander[j] = x[i] - Width / 2
            y_meander[j] = y[i] - Width / 2

        elif x[i-1] == x[i] and y[i] == y[i+1] and y[i-1] > y[i+1]:
            x_meander[i] = x[i] + Width / 2
            y_meander[i] = y[i] + Width / 2
            x_meander[j] = x[i] - Width / 2
            y_meander[j] = y[i] - Width / 2

        elif y[i-1] == y[i] and x[i] == x[i+1] and y[i] < y[i+1]:
            x_meander[i] = x[i] - Width / 2
            y_meander[i] = y[i] + Width / 2
            x_meander[j] = x[i] + Width / 2
            y_meander[j] = y[i] - Width / 2

        elif x[i-1] == x[i] and y[i] == y[i+1] and y[i-1] < y[i+1]:
            x_meander[i] = x[i] - Width / 2
            y_meander[i] = y[i] + Width / 2
            x_meander[j] = x[i] + Width / 2
            y_meander[j] = y[i] - Width / 2

        j -= 1

    return x_meander, y_meander

def antenna_ifa_meander(meander_x, meander_y, terminal_x, terminal_y, feed_x, feed_y, save_mesh_folder, mesh_name, mesh_size):
    gmsh.initialize()
    model_name  = "IFA_meander"
    gmsh.model.add("model_name")

    ifa_meander = rectangle_surface(meander_x, meander_y)
    # print("tag du ifa_meander =", ifa_meander)

    ifa_feed = rectangle_surface(feed_x, feed_y)
    # print("tag du ifa_meander =", ifa_meander)

    # Creation du terminal
    terminal = rectangle_surface(terminal_x, terminal_y)

    # Fusion du terminal et du meander
    antenna_ifa_meander, _ = gmsh.model.occ.fuse([(2, ifa_meander)], [(2, terminal), (2, ifa_feed)])
    # print("tag du fused =", antenna_ifa_meander)

    # Synchronisation et sauvegarde
    gmsh.model.occ.synchronize()
    
    apply_mesh_size(mesh_size)

    # Afficher le modèle dans l’interface Gmsh
    gmsh.model.mesh.generate(2)

    # run()

    write(save_mesh_folder, mesh_name)

    gmsh.finalize()

def radiation_ifa(mesh1, frequency, feed_point, voltage_amplitude, show):
    # Chargement du fichier de maillage
    p, t = load_mesh_file(mesh1)

    # Définition des points et triangles à partir du maillage
    points = Points(p)
    triangles = Triangles(t)

    # Filtrage des triangles invalides et calcul des propriétés géométriques (aires, centres)
    triangles.filter_triangles()
    triangles.calculate_triangles_area_and_center(points)

    # Affiche les dimensions principales de l'antenne
    base_name = os.path.splitext(os.path.basename(mesh1))[0]

    # Définition des arêtes et calcul de leurs longueurs
    edges = triangles.get_edges()
    filter_complexes_jonctions(points, triangles, edges)          # Filtrage des jonctions complexes pour simplifier la structure du maillage

    edges.compute_edges_length(points)

    # Sauvegarde des données du maillage traité
    save_folder_name_mesh1 = 'data/antennas_mesh1/'
    save_file_name_mesh1 = DataManager_rwg1.save_data(mesh1, save_folder_name_mesh1, points, triangles, edges)

    # Chargement des données sauvegardées
    filename_mesh1_to_load = save_folder_name_mesh1 + save_file_name_mesh1

    # Définition et calcul des triangles barycentriques
    barycentric_triangles = Barycentric_triangle()
    barycentric_triangles.calculate_barycentric_center(points, triangles)

    # Calcul des vecteurs RHO pour les arêtes
    vecteurs_rho = Vecteurs_Rho()
    vecteurs_rho.calculate_vecteurs_rho(points, triangles, edges, barycentric_triangles)

    # Sauvegarde des données des triangles barycentriques et vecteurs RHO
    save_folder_name_mesh2 = 'data/antennas_mesh2/'
    save_file_name_mesh2 = DataManager_rwg2.save_data(filename_mesh1_to_load, save_folder_name_mesh2, barycentric_triangles, vecteurs_rho)

    # Chargement des données pour le maillage traité
    filename_mesh2_to_load = save_folder_name_mesh2 + save_file_name_mesh2

    # Calcul des constantes électromagnétiques et de la matrice d'impédance Z
    omega, mu, epsilon, light_speed_c, eta, matrice_z = calculate_z_matrice(triangles,
                                                                            edges,
                                                                            barycentric_triangles,
                                                                            vecteurs_rho,
                                                                            frequency)

    # Sauvegarde des données d'impédance
    save_folder_name_impedance = 'data/antennas_impedance/'
    save_file_name_impedance = DataManager_rwg3.save_data(filename_mesh2_to_load, save_folder_name_impedance, frequency, omega, mu, epsilon, light_speed_c, eta, matrice_z)

    # Chargement des données d'impédance
    filename_impedance = save_folder_name_impedance + save_file_name_impedance

    # Calcul du courant induit sur l'antenne par l'onde incidente
    frequency, omega, mu, epsilon, light_speed_c, eta, voltage, current, gap_current, gap_voltage, impedance, feed_power = calculate_current_radiation(filename_mesh2_to_load, filename_impedance, feed_point, voltage_amplitude)

    

    # Sauvegarde des données de courant
    save_folder_name_current = 'data/antennas_current/'
    save_file_name_current = DataManager_rwg4.save_data_for_radiation(filename_mesh2_to_load, save_folder_name_current, frequency, omega, mu, epsilon, light_speed_c, eta, voltage, current, gap_current, gap_voltage, impedance, feed_power)

    # Calcul des courants de surface à partir du courant total
    surface_current_density = calculate_current_density(current, triangles, edges, vecteurs_rho)

    # Visualisation des courants de surface
    if show:
        antennas_name = os.path.splitext(os.path.basename(filename_mesh2_to_load))[0].replace('_mesh2', ' antenna surface current in radiation mode')
        fig = visualize_surface_current(points, triangles, surface_current_density, feed_point, antennas_name)
        fig.show()

    return impedance, surface_current_density

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

    for idx, frequency in enumerate(frequencies):
        show = (frequency == fC)
        impedance, _ = radiation_ifa(ifa_meander_mat, frequency, feed_point, voltage_amplitude, show)
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

    for idx, frequency in enumerate(frequencies):
        show = (frequency == fC)
        impedance, _ = radiation_ifa(ifa_meander_mat, frequency, feed_point, voltage_amplitude, show)
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

            if R_res > Z0:
                print("↪ R trop haute —> augmenter wid")
                new_wid *= 1.1  # +10%
        elif f_resonance <= fLow:
            print("📈 f trop basse —> - méandres")
            new_Nombre_meandre = max(new_Nombre_meandre - 1, 1)

            if R_res < Z0:
                print("↪ R trop basse —> diminuer wid")
                new_wid *= 0.9  # -10%

        # --- Ajustement distance_short pour correction fine de fréquence
        freq_corr_factor = (fC / f_resonance) ** 0.5
        new_distance_short *= freq_corr_factor

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

# Troisieme version de la fonction simulate_freq_loop_test

def simulate_freq_loop_test_version_3(
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

    for idx, frequency in enumerate(frequencies):
        show = (frequency == fC)
        impedance, _ = radiation_ifa(ifa_meander_mat, frequency, feed_point, voltage_amplitude, show)
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

    # Erreurs
    freq_error = abs((fC - f_resonance) / fC)
    R_error = abs(R_res - Z0) / Z0
    X_error = abs(X_res) / Z0
    s11_min = s11_db[min_index]

    # --- Critères de convergence ---
    if freq_error < accuracy and R_error < 0.1 and X_error < 0.1 and s11_min < -10:
        has_converged = True
        print(f"\n✅ Convergence atteinte !")
    else:
        print("\n❌ Pas de convergence —> Ajustement basé sur les erreurs...\n")

        # Ajustement du nombre de méandres selon la fréquence trouvée
        if f_resonance > fC * (1 + accuracy):
            print("📉 Fréquence trop haute —> + méandres")
            new_Nombre_meandre = min(new_Nombre_meandre + 1, int((L / hauteur) * 2))
        elif f_resonance < fC * (1 - accuracy):
            print("📈 Fréquence trop basse —> - méandres")
            new_Nombre_meandre = max(new_Nombre_meandre - 1, 1)

        # Ajustement fin de la distance short-feed selon l'erreur de fréquence
        freq_corr_factor = (fC / f_resonance)
        new_distance_short *= freq_corr_factor

        # Ajustement de la largeur de piste selon l'erreur sur la résistance
        if abs(R_res - Z0) > 1:
            if R_res < Z0:
                print("↪ R trop basse —> augmenter wid")
                new_wid *= 1 + 0.1 * R_error
            else:
                print("↪ R trop haute —> diminuer wid")
                new_wid *= 1 - 0.1 * R_error

        # Ajustement de la distance short-feed selon l'erreur sur la réactance
        if abs(X_res) > 1:
            print("↪ X non nulle —> ajustement fin de distance_short")
            new_distance_short *= 1 - 0.1 * np.sign(X_res) * min(X_error, 0.5)

        # --- Sécurité : limites physiques ---
        new_wid = np.clip(new_wid, 0.5e-3, largeur / 2)
        # La distance short-feed doit être supérieure à la largeur + marge de sécurité
        min_distance_short = new_wid + 2e-3
        max_distance_short = hauteur - new_wid
        new_distance_short = np.clip(new_distance_short, min_distance_short, max_distance_short)

        print(f"🔧 Nouveaux paramètres :")
        print(f"• Distance short-feed : {new_distance_short * 1e3:.2f} mm")
        print(f"• Largeur de piste    : {new_wid * 1e3:.2f} mm")
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
        impedance, _ = radiation_ifa(ifa_meander_mat, frequency, feed_point, voltage_amplitude, show)
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



def plot_s11_curve(fLow, fHigh, nPoints, s11_db, fC=None):

    frequencies = np.linspace(fLow, fHigh, nPoints)
    frequencies_mhz = np.array(frequencies) / 1e6
    s11_db = np.array(s11_db)

    # Trouver le minimum de S11
    min_index = np.argmin(s11_db)
    f_resonance = frequencies[min_index] / 1e6
    s11_min = s11_db[min_index]

    # Tracé
    fig_size = 8
    Fibonacci = (1 + np.sqrt(5)) / 2
    plt.figure(figsize=(fig_size, fig_size / Fibonacci))
    plt.plot(frequencies_mhz, s11_db, label="S11 (dB)", color='blue')
    plt.plot(f_resonance, s11_min, 'ro', label=f"Résonance: {f_resonance:.2f} MHz")
    
    if fC:
        plt.axvline(fC / 1e6, color='green', linestyle='--', label=f"fC = {fC/1e6:.2f} MHz")

    plt.xlabel("Fréquence (MHz)")
    plt.ylabel("S11 (dB)")
    plt.title("Courbe de S11 vs Fréquence")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_impedance(fLow, fHigh, nPoints, impedances, s11_db, fC=None):
    # Vecteur de fréquences
    frequencies = np.linspace(fLow, fHigh, nPoints)
    frequencies_mhz = frequencies / 1e6

    # Partie réelle de l’impédance
    impedances_real = np.real(impedances)

    # Trouver la fréquence de résonance (minimum de S11)
    min_index = np.argmin(s11_db)
    f_resonance = frequencies[min_index] / 1e6
    impedance_resonance = impedances_real[min_index]

    # Préparer le tracé
    fig_size = 8
    golden_ratio = (1 + np.sqrt(5)) / 2
    plt.figure(figsize=(fig_size, fig_size / golden_ratio))

    # Courbe d'impédance réelle
    plt.plot(frequencies_mhz, impedances_real, label="Re(Z)", color='red')

    # Marqueur de résonance
    plt.plot(f_resonance, impedance_resonance, 'bo', label=f"Résonance: {f_resonance:.2f} MHz")

    # Si une fréquence cible est donnée
    if fC is not None:
        fC_mhz = fC / 1e6
        index_fC = np.argmin(np.abs(frequencies - fC))
        impedance_fC = impedances_real[index_fC]
        plt.plot(fC_mhz, impedance_fC, 'go', label=f"fC: {fC_mhz:.2f} MHz")
        plt.axvline(fC_mhz, color='green', linestyle='--')

    # Mise en forme
    plt.xlabel("Fréquence (MHz)")
    plt.ylabel("Impédance réelle (Ohm)")
    plt.title("Impédance réelle vs Fréquence")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()