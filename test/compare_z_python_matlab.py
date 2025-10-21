from scipy.io import loadmat
import numpy as np

# Charger les fichiers .mat
fichier3 = loadmat('data/antennas_impedance/plate_impedance_matlab.mat')
fichier1 = loadmat('data/antennas_impedance/plate_impedance_with_matlab_file.mat')
fichier2 = loadmat('data/antennas_impedance/plate_impedance.mat')

# Extraire les données pertinentes
variables_fichier1 = {'f', 'omega', 'mu_', 'epsilon_', 'c_', 'eta_', 'Z'}
variables_fichier2 = {'frequency', 'omega', 'mu', 'epsilon', 'light_speed_c', 'eta', 'matrice_z'}

# Créer un mapping des variables pour vérifier l'équivalence
mapping_variables = {
    'f': 'frequency',
    'omega': 'omega',
    'mu_': 'mu',
    'epsilon_': 'epsilon',
    'c_': 'light_speed_c',
    'eta_': 'eta',
    'Z': 'matrice_z'
}

# Comparer les données
for var1, var2 in mapping_variables.items():
    print(var1 + ' et ' + var2)
    data1 = fichier1.get(var1)
    data2 = fichier2.get(var2)

    if data1 is not None and data2 is not None:
        are_equal = np.allclose(data1, data2, atol=1e-8)
        print(f"Comparaison entre '{var1}' et '{var2}': {'ÉGAL' if are_equal else 'DIFFÉRENT'}")
    else:
        print(f"Variable manquante dans l'un des fichiers : '{var1}' ou '{var2}'")


python_with_matlab_matrice_z = fichier1['Z']
python_matrice_z = fichier2['matrice_z']
matlab_matrice_z = fichier3['Z']

print("Shape of matlab_matrice_z =", matlab_matrice_z.shape)
print("Shape of python_matrice_z =", python_matrice_z.shape)
print("Affichage matlab")
print(matlab_matrice_z[0, 0])
print(matlab_matrice_z[100, 100])
print(matlab_matrice_z[8, 66])
print(matlab_matrice_z[0, 7])
print(matlab_matrice_z[175, 175])
print("Affichage python")
print(python_matrice_z[0, 0])
print(python_matrice_z[100, 100])
print(python_matrice_z[8, 66])
print(python_matrice_z[0, 7])
print(python_matrice_z[175, 175])
print("Affichage python avec donnee matlab")
print(python_with_matlab_matrice_z[0, 0])
print(python_with_matlab_matrice_z[100, 100])
print(python_with_matlab_matrice_z[8, 66])
print(python_with_matlab_matrice_z[0, 7])
print(python_with_matlab_matrice_z[175, 175])
'''
frequency = fichier2['frequency']
print("frequency type = ", type(frequency))
print("frequency shape = ", frequency.shape)
print("frequency = ", frequency)
print("frequency squeeze = ", frequency.squeeze())

omega = fichier2['omega']
print("omega type = ", type(omega))
print("omega shape = ", omega.shape)
print("omega = ", omega)
print("omega squeeze = ", omega.squeeze())

mu = fichier2['mu']
print("mu type = ", type(mu))
print("mu shape = ", mu.shape)
print("mu = ", mu)
print("mu squeeze = ", mu.squeeze())

epsilon = fichier2['epsilon']
print("epsilon type = ", type(epsilon))
print("epsilon shape = ", epsilon.shape)
print("epsilon = ", epsilon)
print("epsilon squeeze = ", epsilon.squeeze())

light_speed_c = fichier2['light_speed_c']
print("light_speed_c type = ", type(light_speed_c))
print("light_speed_c shape = ", light_speed_c.shape)
print("light_speed_c = ", light_speed_c)
print("light_speed_c squeeze = ", light_speed_c.squeeze())

eta = fichier2['eta']
print("eta type = ", type(eta))
print("eta shape = ", eta.shape)
print("eta = ", eta)
print("eta squeeze = ", eta.squeeze())

matrice_z = fichier2['matrice_z']
print("matrice_z type = ", type(matrice_z))
print("matrice_z shape = ", matrice_z.shape)
print("matrice_z = \n", matrice_z)
print("matrice_z squeeze = ", matrice_z.squeeze().shape)
'''

