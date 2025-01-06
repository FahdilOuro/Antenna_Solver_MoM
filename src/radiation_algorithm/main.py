from src.radiation_algorithm.radiation_algorithm import *

def traitement_de_(filename):
    base_name = os.path.splitext(os.path.basename(filename))[0]
    print(f"Traitement de l'antenne {base_name}")

if __name__ == "__main__":

    filename_mesh1_strip2 = 'data/antennas_mesh/strip2.mat'

    if not os.path.exists(filename_mesh1_strip2):
        print(f"Le fichier '{filename_mesh1_strip2}' est introuvable.")
    else:
        # Mesure du temps de début
        start_time = time.time()

        traitement_de_(filename_mesh1_strip2)
        radiation_algorithm(filename_mesh1_strip2, 75e6)

        elapsed_time = time.time() - start_time
        print(f"Temps écoulé pour le traitement de l'antenne étudiée : {elapsed_time:.6f} secondes")
        print('\n')
