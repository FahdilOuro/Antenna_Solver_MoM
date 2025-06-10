import math
import numpy as np


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

def ifa_creation(largeur, hauteur, width, distance_meandre):
    # Initialisation de x0 et y0 --> pour se positionner sur la ligne central de la piste
    x0 = 0
    y0 = hauteur - width / 2
    hauteur_ligne_central = hauteur - width

    # Calcul du nombre de brins verticaux
    N = math.floor(largeur / (width + distance_meandre))
    # print(f"Number of meanders {N}")
    
    # Recalculer le nombre de min_slot
    distance_meandre = (largeur / N) - width
    # print(f"New distance meandres {distance_meandre * 1000} mm")

    x = np.zeros(2 * N + 2)
    y = np.zeros(2 * N + 2)

    x[0] = x0
    y[0] = y0

    direction = -1
    idx = 0

    # Horizontal
    idx += 1
    x[idx] = x[idx - 1] + distance_meandre + width / 2
    y[idx] = y[idx - 1]

    # dist_center = distance_meandre + width / 2

    k = 2
    while k < N+1:
        # Vertical
        idx += 1
        x[idx] = x[idx - 1]
        y[idx] = y[idx - 1] + direction * hauteur_ligne_central
        direction = -direction
        
        # Horizontal
        idx += 1
        x[idx] = x[idx - 1] + distance_meandre + width
        y[idx] = y[idx - 1]

        k += 1

    # Vertical
    idx += 1
    x[idx] = x[idx - 1]
    y[idx] = y[idx - 1] + direction * hauteur_ligne_central

    return x[:idx+1], y[:idx+1], N, distance_meandre