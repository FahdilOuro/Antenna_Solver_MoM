from matplotlib import pyplot as plt
from rwg.rwg1 import *
from rwg.rwg2 import *
from rwg.rwg3 import *
from rwg.rwg4 import *
from utils.efield_1_sweep import load_efield_1_data


def scattering_algorithm_E_field(mesh, frequency, wave_incident_direction, polarization, load_from_matlab=True, show=False):
    # Chargement du fichier de maillage
    p, t = load_mesh_file(mesh,load_from_matlab)

    # Définition des points et triangles à partir du maillage
    points = Points(p)
    triangles = Triangles(t)

    # Filtrage des triangles invalides et calcul des propriétés géométriques (aires, centres)
    triangles.filter_triangles()
    triangles.calculate_triangles_area_and_center(points)

    # Définition des arêtes et calcul de leurs longueurs
    edges = triangles.get_edges()
    edges.compute_edges_length(points)

    # Filtrage des jonctions complexes pour simplifier la structure du maillage
    filter_complexes_jonctions(points, triangles, edges)

    # Sauvegarde des données du maillage traité
    save_folder_name_mesh1 = 'data/antennas_mesh1/'
    save_file_name_mesh1 = DataManager_rwg1.save_data(mesh, save_folder_name_mesh1, points, triangles, edges)

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
    save_file_name_impedance = DataManager_rwg3.save_data(filename_mesh2_to_load, save_folder_name_impedance, frequency,
                                                          omega, mu, epsilon, light_speed_c, eta, matrice_z)

    # Chargement des données d'impédance
    filename_impedance = save_folder_name_impedance + save_file_name_impedance

    # Calcul du courant induit sur l'antenne par l'onde incidente
    frequency, omega, mu, epsilon, light_speed_c, eta, voltage, current = calculate_current_scattering(filename_mesh2_to_load, filename_impedance,
                                                                                                       wave_incident_direction, polarization)

    # Sauvegarde des données de courant
    save_folder_name_current = 'data/antennas_current/'
    save_file_name_current = DataManager_rwg4.save_data_for_scattering(filename_mesh2_to_load, save_folder_name_current, frequency,
                                                        omega, mu, epsilon, light_speed_c, eta, wave_incident_direction,
                                                        polarization, voltage, current)

    return points, edges, matrice_z, current

def save_scattering_result_from_antenna(filename_antenna_receiving, filename_antenna_emitting, frequencies, impedances, OutputVoltage, PowerConjMatch):
    output_dir = "data/antennas_sweep/"
    os.makedirs(output_dir, exist_ok=True)
    output_matfile = os.path.join(output_dir, f"{filename_antenna_receiving}_scattering_from_{filename_antenna_emitting}_sweep.mat")

    data_to_save = {
        'frequencies': frequencies,
        'impedances': impedances,
        'OutputVoltage': OutputVoltage,
        'PowerConjMatch': PowerConjMatch
        }
    
    savemat(output_matfile, data_to_save)

def load_scattering_result_from_antenna(filename):
    data_to_load = loadmat(filename)
    frequencies = data_to_load.get('frequencies').squeeze()
    impedances = data_to_load.get('impedances').squeeze()
    OutputVoltage = data_to_load.get('OutputVoltage').squeeze()
    PowerConjMatch = data_to_load.get('PowerConjMatch').squeeze()
    return frequencies, impedances, OutputVoltage, PowerConjMatch

def plot_PowerConjMatch(filename):
    frequencies, _, _, PowerConjMatch = load_scattering_result_from_antenna(filename)
    plt.style.use('fivethirtyeight')
    plt.rcParams['font.family'] = 'JetBrains Mono'
    fig_size = 12
    Fibonacci = (1 + np.sqrt(5)) / 2
    plt.figure(figsize=(fig_size, fig_size / Fibonacci))
    plt.plot(frequencies, PowerConjMatch)
    plt.title('Power Conjugate Match vs Frequency')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Conjugate Match (W)')
    plt.grid(True)
    plt.show()

def Scattering_E_field_from_antenna(receiving_antenna_filename_mesh1, radiatedfield_1_sweep_filename, impedances, feed_point):

    wave_incident_direction = np.array([0, 0, -1])

    frequencies, e_fields, _ = load_efield_1_data(radiatedfield_1_sweep_filename)

    n_freq = e_fields.shape[1]

    OutputVoltage = np.zeros(n_freq, dtype=complex)
    PowerConjMatch = np.zeros(n_freq)

    for i in range(n_freq):
        print(f"simulation number {i+1} / {n_freq}")
        polarization_i = e_fields[:, i]
        frequency = frequencies[i]

        # Appel à l'algorithme de diffusion
        points, edges, matrice_z, current = scattering_algorithm_E_field(receiving_antenna_filename_mesh1, frequency, wave_incident_direction, polarization_i)

        index_feeding_edges = find_feed_edges(points, edges, feed_point)

        Imp = impedances[i]  # scalaire complexe

        # Produit scalaire vectorisé
        feed_currents = current[index_feeding_edges] * edges.edges_length[index_feeding_edges]
        FeedCurReceived = np.sum(feed_currents)  # somme des contributions

        FeedVolReceived = FeedCurReceived * Imp

        OutputVoltage[i] = FeedVolReceived
        PowerConjMatch[i] = (1/8) * np.abs(FeedVolReceived)**2 / np.real(Imp)
        
        antenna_receiving = os.path.splitext(os.path.basename(receiving_antenna_filename_mesh1))[0]
        antenna_emitting = os.path.splitext(os.path.basename(radiatedfield_1_sweep_filename))[0].replace('_radiatedfield_1_sweep', '')

        # Sauvegarde des résultats dans un fichier .mat
        save_scattering_result_from_antenna(antenna_receiving, antenna_emitting, frequencies, impedances, OutputVoltage, PowerConjMatch)
