import numpy as np


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

def ifa_creation_new_v4(L, largeur, hauteur, width, L_short = 1 / 1000):
    # — Initialisation
    x0 = 0
    y0 = hauteur - width / 2
    hauteur = hauteur - width

    # — Calcul du nombre de brins verticaux
    # N = int(np.floor((largeur / min_slot - 1)))
    N = int(np.floor((L - largeur) / hauteur))
    print(f"Number of meanders {N}")
    distance_meandre = (largeur - L_short) / N
    print(f"distance meandres {distance_meandre * 1000} mm")

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

    return x[:idx+1], y[:idx+1], N, distance_meandre

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

def ifa_creation_optimisation_v4(L, largeur, hauteur, width, Nombre_meandre, L_short = 2):
    # — Initialisation
    x0 = 0
    y0 = hauteur - width / 2
    hauteur = hauteur - width

    # — Calcul du nombre de brins verticaux
    # N = int(np.floor((largeur / min_slot - 1)))
    N = Nombre_meandre
    print(f"Number of meanders {N}")
    distance_meandre = (largeur - L_short) / N
    print(f"distance meandres {distance_meandre * 1000} mm")


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

    return x[:idx+1], y[:idx+1], distance_meandre

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