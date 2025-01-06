from scipy.io import loadmat

# Charger les fichiers .mat
fichier_curent_matlab = loadmat('data/antennas_current/plate_current_matlab.mat')
fichier_curent_python = loadmat('data/antennas_current/plate_current.mat')

curent_matlab = fichier_curent_matlab['I'].squeeze()
curent_python = fichier_curent_python['current'].squeeze()

print("curent_matlab")
print(curent_matlab.shape)
print(f"curent_matlab[0] = {curent_matlab[0]}")
print(f"curent_matlab[100] = {curent_matlab[100]}")
print(f"curent_matlab[150] = {curent_matlab[150]}")
print(f"curent_matlab[35] = {curent_matlab[35]}")
print("curent_python")
print(curent_python.shape)
print(f"curent_python[0] = {curent_python[0]}")
print(f"curent_python[100] = {curent_python[100]}")
print(f"curent_python[150] = {curent_python[150]}")
print(f"curent_python[35] = {curent_python[35]}")