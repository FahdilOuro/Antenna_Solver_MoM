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

def calculate_nPoints(fLow, fHigh, fC, min_points=5):
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
    
    print(f"last index = {idx}")

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
    
    print("\nlongueur_obtenue =", calcul_actuel_longueur)
    print("longueur_desiree =", L, "\n")

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
    
    print(f"last index = {idx}")

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
    
    print("\nlongueur_obtenue =", calcul_actuel_longueur)
    print("longueur_desiree =", L, "\n")

    return x[:idx+1], y[:idx+1]

def ifa_creation(a, b, first_min_slot, other_min_slot, m_max=100):
    gap = 0
    # Essayer différentes valeurs de m pour trouver un wid acceptable
    for m in reversed(range(1, m_max + 1)):
        numerator = a - first_min_slot - (m - 1) * other_min_slot
        if numerator > 0:
            wid = numerator / m
            break
    else:
        raise ValueError("Impossible de placer au moins un brin avec ces paramètres.")

    x_dip = []
    y_dip = []

    # Point de départ (en haut à gauche)
    x_dip.append(gap)
    y_dip.append(b / 2)

    # Premier slot horizontal (first_min_slot)
    x_dip.append(gap + first_min_slot)
    y_dip.append(b / 2)

    # Zigzag horizontal (de gauche à droite)
    for i in range(m):
        base_x = gap + first_min_slot + i * (wid + other_min_slot)
        if i % 2 == 0:
            # Descente
            x_dip.append(base_x + wid)
            y_dip.append(b / 2)
            x_dip.append(base_x + wid)
            y_dip.append(-b / 2 + wid)
        else:
            # Montée
            x_dip.append(base_x)
            y_dip.append(-b / 2 + wid)
            x_dip.append(base_x)
            y_dip.append(b / 2)

    # Dernier point à droite
    if m % 2 == 0:
        x_dip.append(a)
        y_dip.append(b / 2)
    else:
        x_dip.append(a)
        y_dip.append(-b / 2)

    # Deuxième moitié (retour vers la gauche)
    for i in range(m):
        base_x = a - i * (wid + other_min_slot)
        if (m - i) % 2 == 1:
            # Montée
            x_dip.append(base_x - wid)
            y_dip.append(-b / 2)
            x_dip.append(base_x - wid)
            y_dip.append(b / 2 - wid)
        else:
            # Descente
            x_dip.append(base_x)
            y_dip.append(b / 2 - wid)
            x_dip.append(base_x)
            y_dip.append(-b / 2)

    # Retour final à gauche
    x_dip.append(gap)
    y_dip.append(b / 2 - wid)

    x = np.array(x_dip)
    y = np.array(y_dip)

    return x, y, wid

def ifa_creation_center(L, largeur, hauteur, width, min_slot):
    """
    Trace un dipôle méandré en 2D avec des points de contour bien délimités.
    Le tracé commence au point (x0, y0) et se développe vers le bas.
    """
    # — Validation des entrées
    for val, name in zip([L, largeur, hauteur, width, min_slot], ['L', 'largeur', 'hauteur', 'width', 'min_slot']):
        if not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
            raise ValueError(f"{name} doit etre un scalaire numérique réel.")
    if L <= 0:
        raise ValueError("L doit etre strictement positif.")
    if width <= 0:
        raise ValueError("width doit etre strictement positif.")
    if hauteur < 0:
        raise ValueError("hauteur doit etre positif ou nul.")
    if min_slot <= 0:
        raise ValueError("min_slot doit etre strictement positif.")
    if largeur <= 0:
        raise ValueError("largeur doit etre strictement positif.")
    if min_slot <= width:
        raise ValueError("min_slot doit etre strictement superieur à width.")

    # — Initialisation
    x0 = 0
    y0 = hauteur / 2 - width / 2
    hauteur = hauteur - width

    # — Calcul du nombre de brins verticaux
    N = int(np.floor((largeur / min_slot - 1)))
    longueur_meandre = (N + 1) * min_slot + N * hauteur

    print("longueur_meandre =", longueur_meandre)
    print("longueur_desiree =", L, "\n")

    x = np.zeros(2 * N + 2)
    y = np.zeros(2 * N + 2)

    x[0] = x0
    y[0] = y0

    direction = -1
    idx = 0
    calcul_actuel_longueur = 0
    horizontal = False
    vertical = False

    for k in range(1, N + 1):
        # Horizontal
        idx += 1
        x[idx] = x[idx - 1] + min_slot
        y[idx] = y[idx - 1]
        calcul_actuel_longueur += min_slot
        if calcul_actuel_longueur + hauteur >= L:
            rem = L - calcul_actuel_longueur
            vertical = True
            break

        # Vertical
        idx += 1
        x[idx] = x[idx - 1]
        y[idx] = y[idx - 1] + direction * hauteur
        calcul_actuel_longueur += hauteur
        direction = -direction
        if calcul_actuel_longueur + min_slot >= L:
            rem = L - calcul_actuel_longueur
            horizontal = True
            break

    # Ajouter le dernier petit segment correctif
    if not horizontal and not vertical:
        idx += 1
        x[idx] = x[idx - 1] + min_slot
        y[idx] = y[idx - 1]
        calcul_actuel_longueur += min_slot
    elif horizontal:
        idx += 1
        x[idx] = x[idx - 1] + rem
        y[idx] = y[idx - 1]
        calcul_actuel_longueur += rem
    elif vertical:
        idx += 1
        x[idx] = x[idx - 1]
        y[idx] = y[idx - 1] + direction * rem
        calcul_actuel_longueur += rem

    return x[:idx+1], y[:idx+1], calcul_actuel_longueur

def trace_meander(x, y, Width):
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
        y_meander[n-1] = y[n-1]
        y_meander[n]   = y[n-1]
    elif x[n-2] == x[n-1] and y[n-2] < y[n-1]:
        x_meander[n-1] = x[n-1] - Width / 2
        x_meander[n]   = x[n-1] + Width / 2
        y_meander[n-1] = y[n-1]
        y_meander[n]   = y[n-1]

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

def simulate_freq_loop(fLow, fHigh, nPoints, fC, accuracy, ifa_meander_mat, feed_point, distance_short, wid, L, hauteur, largeur, L_short, Nombre_meandre):
    Z0 = 50  # Impédance caractéristique en ohms
    frequencies = np.linspace(fLow, fHigh, nPoints)
    s11_db = []
    voltage_amplitude = 0.5
    has_converged = False
    count = 0
    show = False
    impedances = []
    sf_list = []
    new_distance_short = distance_short
    new_wid = wid
    new_Nombre_meandre = Nombre_meandre

    for frequency in frequencies:
        print(f"Simulation Numéro {count + 1}")
        '''if count == nPoints - 1: 
            show = True'''
        if frequency == fC:
            show = True
        else:
            show = False
        impedance, _ = radiation_ifa(ifa_meander_mat, frequency, feed_point, voltage_amplitude, show)
        impedances.append(impedance)
        s11 = (impedance - Z0) / (impedance + Z0)
        s11_db.append(20 * np.log10(abs(s11)))
        print(f"paramètre S11 = {s11_db[count]} db")
        count += 1

    # Trouver la fréquence de résonance (minimum S11)
    min_index = np.argmin(s11_db)
    f_resonance = frequencies[min_index]
    R_I_min_index = impedances[min_index].real
    print(f"R_I_min_index = {R_I_min_index}")
    print(f"\nFréquence de résonance : {f_resonance / 1e6:.2f} MHz")

    sf_list.append(new_distance_short)

    # Comparaison à la fréquence de coupure
    error = abs((fC - f_resonance) / fC)
    s11_db_min_index = s11_db[min_index]
    if error < accuracy:
        if s11_db_min_index < -10:
            has_converged = True
            print(f" Convergence atteinte : |f_res - fC| = {error:.2f} Hz ≤ {accuracy}")
        else:
            print(f"R_I_min_index = {R_I_min_index} $\Omega$")
            # new_distance_short = distance_short * pow((Z0 / R_I_min_index), 2) 
            new_distance_short = distance_short * pow((Z0 / R_I_min_index), 2)
            # new_distance_short = 7.77 / 1000
            if new_distance_short < 0.5 / 1000:
                new_distance_short = 0.5 / 1000
            if new_distance_short > hauteur:
                new_distance_short = hauteur - wid
    else:
        distance_meandre = (largeur - L_short) / Nombre_meandre
        if distance_meandre < 0.5 / 1000:
            distance_meandre = 0.5 / 1000
        # new_Nombre_meandre = np.floor((largeur - L_short) / (wid + distance_meandre))
        # new_Nombre_meandre = int(np.floor(Nombre_meandre * pow((fC / f_resonance), 2)))
        new_Nombre_meandre = int(math.ceil((L / hauteur) * f_resonance / fC)) + 1 
        new_wid = wid * pow((fC / f_resonance), 2)
        # new_distance_short = distance_short * (Z0 / R_I_min_index)
        new_distance_short = distance_short * (Z0 / R_I_min_index)
        print(f"\n0...........short feed ...... dans la fonction = {new_distance_short * 1000}\n")
        if new_distance_short < 0.5 / 1000:
            new_distance_short = 0.5 / 1000
        if new_distance_short > hauteur:
            new_distance_short = hauteur
        print(f" Pas de convergence : |f_res - fC| = {error:.2f} Hz > {accuracy}")
        print(f"\n1...........short feed ...... dans la fonction = {new_distance_short * 1000}\n")

    return s11_db, f_resonance, new_distance_short, new_wid, new_Nombre_meandre, has_converged


def plot_s11_curve(fLow, fHigh, nPoints, s11_db, fC=None):

    frequencies = np.linspace(fLow, fHigh, nPoints)
    frequencies_mhz = np.array(frequencies) / 1e6
    s11_db = np.array(s11_db)

    # Trouver le minimum de S11
    min_index = np.argmin(s11_db)
    f_resonance = frequencies[min_index] / 1e6
    s11_min = s11_db[min_index]

    # Tracé
    plt.figure(figsize=(8, 4))
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