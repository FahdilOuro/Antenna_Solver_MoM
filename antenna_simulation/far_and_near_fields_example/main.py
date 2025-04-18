import numpy as np

from efield.efield1 import calculate_electric_magnetic_field_at_point
from efield.efield2 import radiation_intensity_distribution_over_sphere_surface
from efield.efield3 import antenna_directivity_pattern

if __name__ == "__main__":
    """
        Script principal pour l'analyse des antennes.

        Étapes principales :
            1. Chargement des fichiers contenant les données des antennes (maillage, courant, gain/power).
            2. Calcul et affichage des champs électromagnétiques au niveau d'un point donné.
            3. Calcul de la distribution d'intensité du rayonnement sur une sphère.
            4. Génération des diagrammes de directivité.
    """

    # Fichiers d'entrée pour différents types d'antennes
    filename_mesh2_plate = 'data/antennas_mesh2/plate_mesh2.mat'
    filename_current_plate = 'data/antennas_current/plate_current.mat'
    filename_gain_power_plate = 'data/antennas_gain_power/plate_gain_power.mat'

    filename_mesh2_platecoarse = 'data/antennas_mesh2/platecoarse_mesh2.mat'
    filename_current_platecoarse = 'data/antennas_current/platecoarse_current.mat'
    filename_gain_power_platecoarse = 'data/antennas_gain_power/platecoarse_gain_power.mat'

    filename_mesh2_platefine = 'data/antennas_mesh2/platefine_mesh2.mat'
    filename_current_platefine = 'data/antennas_current/platefine_current.mat'
    filename_gain_power_platefine = 'data/antennas_gain_power/platefine_gain_power.mat'

    filename_mesh2_dipole = 'data/antennas_mesh2/dipole_mesh2.mat'
    filename_current_dipole = 'data/antennas_current/dipole_current.mat'
    filename_gain_power_dipole = 'data/antennas_gain_power/dipole_gain_power.mat'

    filename_mesh2_bowtie = 'data/antennas_mesh2/bowtie_mesh2.mat'
    filename_current_bowtie = 'data/antennas_current/bowtie_current.mat'
    filename_gain_power_bowtie = 'data/antennas_gain_power/bowtie_gain_power.mat'

    filename_mesh2_slot = 'data/antennas_mesh2/slot_mesh2.mat'
    filename_current_slot = 'data/antennas_current/slot_current.mat'
    filename_gain_power_slot = 'data/antennas_gain_power/slot_gain_power.mat'

    # Maillage dense pour une sphère utilisée dans les calculs de rayonnement
    filename_sphere_dense = 'data/sphere_mesh/sphere_dense.mat'


    # Point d'observation pour le calcul des champs
    observation_point = np.array([5, 0, 0])

    # Listes des fichiers d'entrée regroupés par type d'antenne
    filename_mesh2 = [filename_mesh2_plate, filename_mesh2_platecoarse, filename_mesh2_platefine, filename_mesh2_dipole, filename_mesh2_bowtie, filename_mesh2_slot]
    filename_current = [filename_current_plate, filename_current_platecoarse, filename_current_platefine, filename_current_dipole, filename_current_bowtie, filename_current_slot]
    filename_gain_power = [filename_gain_power_plate, filename_gain_power_platecoarse, filename_gain_power_platefine, filename_gain_power_dipole, filename_gain_power_bowtie, filename_gain_power_slot]

    # Boucle sur chaque type d'antenne pour effectuer les calculs et visualisations
    for mesh2, current, gain_power in zip(filename_mesh2, filename_current, filename_gain_power):
        print(f"Processing antenna with mesh: {mesh2}")

        # Étape 1 : Calcul du champ électrique et magnétique à un point donné
        print("Calculating electric and magnetic fields at observation point...")
        calculate_electric_magnetic_field_at_point(mesh2, current, observation_point)

        # Étape 2 : Distribution de l'intensité du rayonnement sur une sphère
        print("Calculating radiation intensity distribution over sphere surface...")
        radiation_intensity_distribution_over_sphere_surface(mesh2, current, filename_sphere_dense)

        # Étape 3 : Génération du diagramme de directivité
        print("Generating antenna directivity pattern...")
        antenna_directivity_pattern(mesh2, current, gain_power)

        # Ligne vide pour séparer les sorties des différentes antennes
        print('')