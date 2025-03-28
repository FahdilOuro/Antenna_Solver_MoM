import os
import gmsh
import numpy as np

from utils.gmsh_function import feed_edge, is_point_on_boundary, save_gmsh_log
from utils.refinement_function import load_high_current_points_from_file, save_high_current_points_to_file


def plate_gmsh(longueur, hauteur, mesh_name, feed_point, length_feed_edge, angle, save_mesh_folder, high_current_points_list=np.array([3, 0]), iteration=1, mesh_size=1):
    # Initialisation de Gmsh
    gmsh.initialize()
    gmsh.model.add(os.path.splitext(os.path.basename(mesh_name))[0])

    size_min = 0.5  # Ajuste selon le niveau de détail souhaité
    size_max = 5.0  # Taille maximale des éléments
    threshold = 2.5  # Valeur de transition entre `size_min` et `size_max`


    pos_folder = 'data/pos/'

    # Vérifier si le dossier existe, sinon le créer
    if not os.path.exists(pos_folder):
        os.makedirs(pos_folder)

    # Nom du fichier de points du maillage adaptatif
    filename = f'data/pos/{os.path.splitext(os.path.basename(mesh_name))[0]}.pos'

    # Vérifier si le dossier existe, sinon le créer
    if not os.path.exists(save_mesh_folder):
        os.makedirs(save_mesh_folder)

    # Utilisation d'Open Cascade pour créer un rectangle (x, y, z, largeur, hauteur)
    plate = gmsh.model.occ.addRectangle(0, 0, 0, longueur, hauteur)
    gmsh.model.occ.synchronize()

    feed_edge(plate, feed_point, length_feed_edge, mesh_name, angle)

    Automatic = 2
    gmsh.option.setNumber("Mesh.Algorithm", Automatic)   # To set The "Automatic" algorithm / Change if necessary

    # gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)

    if iteration <= 0:
        raise ValueError("iteration value should start with '1'")
    elif iteration == 1:
        # Vérifier si le fichier existe et le réinitialiser
        if os.path.exists(filename):
            os.remove(filename)  # Supprimer le fichier existant s'il existe

        # Créer un nouveau fichier vierge avec l'en-tête
        with open(filename, 'w') as file:
            file.write("x y z\n")
    else:
        # Sauver les nouveaux points de rafinage dans le fichier
        save_high_current_points_to_file(high_current_points_list, filename)
        # Liste pour stocker tous les champs de distance
        distance_fields = []

        # Charger les points actuels à chaque itération
        all_points = load_high_current_points_from_file(filename)

        # Ajouter les nouveaux points à la géométrie (mais ne pas les garder)
        point_tags = []
        for x, y, z in all_points:
            if not is_point_on_boundary(plate, (x, y, z)):
                tag = gmsh.model.occ.addPoint(x, y, z)
                point_tags.append(tag)  # Garder les tags pour les supprimer après

        # Définir un champ de distance pour cette itération
        f = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(f, "PointsList", point_tags)

        distance_fields.append(f)

        # Créer un champ de seuil lié à ce champ de distance
        g = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(g, "InField", f)
        gmsh.model.mesh.field.setNumber(g, "SizeMin", size_min)
        gmsh.model.mesh.field.setNumber(g, "SizeMax", size_max)
        gmsh.model.mesh.field.setNumber(g, "DistMin", threshold / 2)
        gmsh.model.mesh.field.setNumber(g, "DistMax", threshold)

        # Fusionner tous les champs de distance pour l'itération
        min_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", distance_fields)

        # Appliquer le champ de maillage fusionné
        gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

        gmsh.model.occ.synchronize()

        # Supprimer les points ajoutés à la géométrie (les tags sont utilisés pour les supprimer)
        for tag in point_tags:
            gmsh.model.occ.remove([(0, tag)])  # Supprimer chaque point individuellement en passant une liste

    # Génération du maillage 2D sur la surface
    gmsh.model.mesh.generate(2)

    # Définir le chemin où enregistrer le fichier
    output_path = os.path.join(save_mesh_folder, mesh_name)

    gmsh.write(output_path)
    print(f"{mesh_name} saved in {output_path} successfully")

    # Sauvegarde des logs 
    save_gmsh_log(mesh_name, output_path)

    gmsh.finalize()

    return output_path