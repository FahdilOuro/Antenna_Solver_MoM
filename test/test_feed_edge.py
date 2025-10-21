import scipy.io
from utils.feed_edge import *

# Charger les données .mat
data_path = "data/antennas_mesh/radiate_plate_gmsh"
# data_path = "data/antennas_mesh/radiate_bowtie_gmsh"


data = scipy.io.loadmat(data_path)
p = data['p']
p_feed = data['p_feed']

# Affichage taille : 
print(f"taille p : {p.shape[1]}")
print(f"taille p_feed : {p_feed.shape[1]}")

# Trouver les indices
indices_feed = find_feed_indices(p, p_feed)

# Afficher les résultats
print("Indices des points de p_feed dans p :", indices_feed)

# Générer le tableau edge_feed
edge_feed = create_edge_feed(indices_feed)

# Afficher edge_feed
print("edge_feed :\n", edge_feed)

# Trouver les indices des arêtes de edge_feed qui existent dans edge
"""matching_indices = find_matching_edges(edge, edge_feed)

# Afficher les indices correspondants
print("Indices des arêtes correspondantes dans edge :", matching_indices)"""