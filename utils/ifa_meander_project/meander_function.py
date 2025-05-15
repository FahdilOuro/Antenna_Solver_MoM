import gmsh
import numpy as np
import matplotlib.pyplot as plt
from utils.gmsh_function import *
from rwg.rwg1 import *
from rwg.rwg2 import *
from rwg.rwg3 import *
from rwg.rwg4 import *
from rwg.rwg5 import *


'''def ifa_creation(a, b, wid, min_slot):
    gap = 0
    n = (a - gap) / (wid + min_slot)
    m = int(np.floor(n))

    # Recalcul du slot minimal pour s'ajuster exactement à la hauteur `a`
    min_slot = ((a - gap) - m * wid) / m

    x_dip = []
    y_dip = []

    # Point de départ (en haut à gauche)
    x_dip.append(gap)
    y_dip.append(b / 2)

    x_dip.append(min_slot)
    y_dip.append(b / 2)

    j = 2  # compteur de points

    # Première moitié du zigzag (de gauche à droite)
    for i in range(m):
        if i % 2 == 0:
            # Descente
            x_dip.append(i * a / m + wid + min_slot)
            y_dip.append(b / 2)

            x_dip.append(i * a / m + wid + min_slot)
            y_dip.append(-b / 2 + wid)
        else:
            # Montée
            x_dip.append(i * a / m + min_slot)
            y_dip.append(-b / 2 + wid)

            x_dip.append(i * a / m + min_slot)
            y_dip.append(b / 2)

    # Dernier point à droite
    if m % 2 == 0:
        x_dip.append(a)
        y_dip.append(b / 2)
    else:
        x_dip.append(a)
        y_dip.append(-b / 2)

    # Deuxième moitié du zigzag (de droite à gauche)
    for i in range(m):
        if (m - i) % 2 == 1:
            # Montée
            x_dip.append(a - (i * a / m) - wid)
            y_dip.append(-b / 2)

            x_dip.append(a - (i * a / m) - wid)
            y_dip.append(b / 2 - wid)
        else:
            # Descente
            x_dip.append(a - (i * a / m))
            y_dip.append(b / 2 - wid)

            x_dip.append(a - (i * a / m))
            y_dip.append(-b / 2)

    # Retour final à gauche
    x_dip.append(gap)
    y_dip.append(b / 2 - wid)

    x = np.array(x_dip)
    y = np.array(y_dip)

    return x, y'''

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

    run()

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
        fig = visualize_surface_current(points, triangles, surface_current_density, antennas_name)
        fig.show()

    return impedance, surface_current_density

def simulate_freq_loop(fLow, fHigh, nPoints, fC, accuracy, ifa_meander_mat, feed_point):
    Z0 = 50  # Impédance caractéristique en ohms
    frequencies = np.linspace(fLow, fHigh, nPoints)
    s11_db = []
    voltage_amplitude = 0.5
    has_converged = False
    count = 0
    show = False

    for frequency in frequencies:
        print(f"Simulation Numéro {count + 1}")
        if count == nPoints - 1: 
            show = True
        impedance, _ = radiation_ifa(ifa_meander_mat, frequency, feed_point, voltage_amplitude, show)
        s11 = (impedance - Z0) / (impedance + Z0)
        s11_db.append(20 * np.log10(abs(s11)))
        print(f"paramètre S11 = {s11_db[count]} db")
        count += 1

    # Trouver la fréquence de résonance (minimum S11)
    min_index = np.argmin(s11_db)
    f_resonance = frequencies[min_index]
    print(f"\nFréquence de résonance : {f_resonance / 1e6:.2f} MHz")

    # Comparaison à la fréquence de coupure
    error = abs(f_resonance - fC)
    if error <= accuracy:
        has_converged = True
        print(f" Convergence atteinte : |f_res - fC| = {error:.2f} Hz ≤ {accuracy}")
    else:
        print(f" Pas de convergence : |f_res - fC| = {error:.2f} Hz > {accuracy}")

    return s11_db, f_resonance, has_converged


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