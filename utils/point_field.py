import numpy as np


def radiated_scattered_field_at_a_point(point, eta, parameter_k, dipole_moment, dipole_center):
    """
        Calcule des champs électriques E et magnétique H rayonnés/dispersés
        par un ensemble de dipôles au point d'observation donné.

        Paramètres :
            * point (np.n-d-array) : Coordonnées du point d'observation, de taille (3,).
            * eta (float) : Impédance caractéristique du milieu.
            * parameter_k (complex float) : Nombre d'onde complexe, où omega est la pulsation angulaire et c la vitesse de la lumière.
            * dipole_moment (np.n-d-array) : Moments dipolaires des dipôles, de taille (3, N), où N est le nombre de dipôles.
            * dipole_center (np.n-d-array) : Positions des dipôles dans l'espace, de taille (3, N).

        Retourne :
            * e_field (np.n-d-array) : Champ électrique au point d'observation, de taille (3, N).
            * h_field (np.n-d-array) : Champ magnétique au point d'observation, de taille (3, N).

        Description :
            * Cette fonction modélise les champs électromagnétiques rayonnés et dispersés par un tableau de dipôles dans un milieu donné.
            * Elle utilise des formules classiques dérivées des équations de Maxwell pour un dipôle oscillant.

        Calculs principaux :
            1. Distance et vecteur entre le point d'observation et chaque dipôle.
            2. Facteurs géométriques et d'atténuation en fonction de la distance.
            3. Contribution des moments dipolaires au champ électrique et magnétique.
    """
    # Constante normalisée pour les champs
    c = 4 * np.pi
    constant_h = parameter_k / c
    constant_e = eta / c

    # Calcul du vecteur entre le point d'observation et les centres des dipôles
    r = point.reshape(3, 1) - dipole_center       # (3, N) : vecteurs distance
    r_norm = np.sqrt(np.sum(r ** 2, axis=0))      # Norme de r : (1, N)
    r_norm2 = r_norm ** 2                         # Distance au carré : (1, N)

    # Calcul de l'atténuation exponentielle due à la distance
    exp = np.exp(-parameter_k * r_norm)    # Facteur exponentiel : (N)

    # Calcul des facteurs géométriques
    parameter_c = (1 / r_norm2) * (1 + (1 / (parameter_k * r_norm)))  # Facteur c (1, N)
    parameter_d = np.sum(r * dipole_moment, axis=0) / r_norm2  # Facteur d (1, N)

    # Calcul du terme Paramètre M
    parameter_m = parameter_d * r  # (3, N)

    # Calcul du champ magnétique H
    h_field = constant_h * np.cross(dipole_moment, r, axis=0) * parameter_c * exp  # (3, N)

    # Calcul du champ électrique E
    e_field = constant_e * ((parameter_m - dipole_moment) * (parameter_k / r_norm + parameter_c) + 2 * parameter_m * parameter_c) * exp  # (3, N)

    return e_field, h_field