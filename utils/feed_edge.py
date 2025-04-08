import numpy as np

def find_feed_indices(p, p_feed):
    """
    Trouve les indices des points de p_feed dans p.

    Paramètres :
    - p : np.array de taille (3, N), contenant les coordonnées de tous les points.
    - p_feed : np.array de taille (3, M), sous-ensemble de p.

    Retour :
    - indices_feed : np.array de taille (1, M) avec les indices des colonnes de p correspondant à p_feed.
    """
    indices_feed = []

    for i in range(p_feed.shape[1]):
        # Trouver l'index de la colonne de p_feed dans p
        match = np.where((p.T == p_feed[:, i]).all(axis=1))[0]

        if match.size == 0:
            raise ValueError(f"Le point {p_feed[:, i]} n'a pas été trouvé dans p.")
        else:
            indices_feed.append(match[0])  # Stocker l' indice trouvé

    return np.array(indices_feed)

import numpy as np

def create_edge_feed(indices_feed):
    """
    Génère le tableau edge_feed en connectant chaque point successif de indices_feed.

    Paramètres :
    - indices_feed : np.array de taille (M,) contenant les indices des points de p_feed dans p.

    Retour :
    - edge_feed : np.array de taille (2, M-1) connectant les points successifs.
    """
    num_points = len(indices_feed)

    if num_points < 2:
        raise ValueError("indices_feed doit contenir au moins deux indices pour former une arête.")

    # Générer les arêtes en connectant chaque point au suivant
    edge_feed = np.vstack((indices_feed[:-1], indices_feed[1:]))

    return edge_feed

def find_matching_edges(edge_obj, edge_feed):
    """
    Vérifie si les arêtes de `edge_feed` existent déjà dans `edge_obj` (dans les deux sens).

    Cette fonction compare les arêtes de `edge_feed` avec celles contenues dans `edge_obj`, 
    en considérant que les arêtes sont non orientées (c'est-à-dire que (a, b) est équivalent à (b, a)). 
    Elle retourne les indices des arêtes correspondantes dans `edge_obj`.

    Paramètres :
    ----------
    - edge_obj : Edges
        Objet de la classe `Edges` contenant les arêtes du maillage.
    - edge_feed : np.array de taille (2, M)
        Tableau contenant les arêtes à vérifier (chaque colonne est une arête).

    Retour :
    -------
    - matching_indices : np.array de forme (K,)
        Tableau contenant les indices des colonnes de `edge_obj` qui correspondent à celles de `edge_feed`.
        K représente le nombre d'arêtes trouvées.
    """
    matching_indices = []

    # Extraire les arêtes sous forme de tableau 2xN
    edge = np.vstack((edge_obj.first_points, edge_obj.second_points))

    # Trier les arêtes pour considérer (a, b) et (b, a) comme équivalents
    edge_sorted = np.sort(edge, axis=0)  
    edge_feed_sorted = np.sort(edge_feed, axis=0)  

    # Vérifier quelles arêtes de edge_feed existent dans edge
    for i in range(edge_feed_sorted.shape[1]):
        match = np.where((edge_sorted.T == edge_feed_sorted[:, i]).all(axis=1))[0]

        if match.size > 0:
            matching_indices.append(match[0])  # Ajouter l'indice trouvé

    return np.array(matching_indices)  # Retourne un tableau NumPy des indices trouvés
