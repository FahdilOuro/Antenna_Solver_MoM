import numpy as np
import triangle as tr
import plotly.figure_factory as ff
import plotly.graph_objects as go
import os
from scipy.io import savemat

class AdaptiveMeshRefiner:
    def __init__(self):
        self.points = np.empty((3, 0))                # Tableau vide de forme (3, 0)
        self.triangles = np.empty((4, 0), dtype=int)  # Tableau vide de forme (4, 0)

    def initial_meshing(self, antenna_geometrie, filename, save_folder_name, refinement_level="high"):
        """
        Effectue la triangulation initiale sur l'antenne avec un niveau de raffinage sélectionnable.
        
        - refinement_level : "low", "medium", "high" (défaut) pour changer les options de `triangle`.
        """
        # Choix des options selon le niveau de raffinement
        options_map = {
            "very_low": "p",
            "low": "pq20",
            "moderate": "pq20a0.1",
            "medium": "pq20a0.01",
            "high": "pq20a0.00001"
        }

        # Vérifier si le niveau existe, sinon utiliser le mode "high" par défaut
        options = options_map.get(refinement_level, options_map["very_low"])

        print(f"Utilisation des options de triangulation : {options}")

        # Application de la triangulation avec les options sélectionnées
        antenna_mesh = tr.triangulate(antenna_geometrie, options)

        self.transform_data(antenna_mesh)
        self.data_save(filename, save_folder_name)

    def generate_new_points(self, selected_triangles):
        """
        Génère de nouveaux points aux barycentres des triangles sélectionnés.
        Mise à jour directe de `self.points`.
        """
        new_points = []

        for tri_idx in selected_triangles:
            v1, v2, v3 = self.triangles[:3, tri_idx]
            p1, p2, p3 = self.points[:2, v1], self.points[:2, v2], self.points[:2, v3]

            x_center = (p1[0] + p2[0] + p3[0]) / 3
            y_center = (p1[1] + p2[1] + p3[1]) / 3

            new_points.append([x_center, y_center, 0])

        if new_points:
            new_points = np.array(new_points).T
            self.points = np.concatenate((self.points, new_points), axis=1)

    def adaptative_meshing(self, antenna_geometrie, selected_triangles, filename, save_folder_name):
        """
        Applique le raffinage adaptatif sur le maillage en mettant à jour les données.
        """
        if selected_triangles is None or selected_triangles.size == 0:
            raise ValueError("Erreur : Aucun triangle sélectionné pour le raffinage. Veuillez fournir des indices valides.")
        
        self.generate_new_points(selected_triangles)

        # Mise à jour directe des points dans la structure de triangulation
        antenna_geometrie['vertices'] = self.points[:2, :].T

        # Nouvelle triangulation avec les mêmes options
        antenna_mesh = tr.triangulate(antenna_geometrie, "pYY")

        print("Après raffinage")
        self.transform_data(antenna_mesh)
        self.data_save(filename, save_folder_name)

    def transform_data(self, antenna_mesh):
        """
        Met à jour les structures points et triangles avec le nouveau maillage.
        """
        nbr_of_points = antenna_mesh['vertices'].shape[0]
        nbr_of_triangles = antenna_mesh['triangles'].shape[0]
        print(f"Nombre de points = {nbr_of_points}")
        print(f"Nombre de triangles = {nbr_of_triangles}")

        self.triangles.resize((4, nbr_of_triangles), refcheck=False)
        self.triangles[:3, :] = antenna_mesh['triangles'].T

        self.points.resize((3, nbr_of_points), refcheck=False)
        self.points[:2, :] = antenna_mesh['vertices'].T

        print(f"Matrice points shape = {self.points.shape}")
        print(f"Matrice triangles shape = {self.triangles.shape}")

    def show_mesh(self, feed_point = None):
        # Normalisation pour l'aspect ratio
        x_, y_, z_ = self.points
        fig_scale = max(max(x_) - min(x_), max(y_) - min(y_))
        x_scale = (max(x_) - min(x_)) / fig_scale
        y_scale = (max(y_) - min(y_)) / fig_scale
        z_scale = 0.3  # Z scale arbitraire pour une meilleure visualisation
        # Création de la figure 3D avec Plotly
        fig = ff.create_trisurf(
            x=x_,
            y=y_,
            z=z_,
            simplices=self.triangles[:3, :].T,
            color_func=list(range(len(self.triangles[:3, :].T))),  # Couleurs basées sur les indices des triangles
            show_colorbar=False,
            title="Maillage triangulaire",
            aspectratio=dict(x=x_scale, y=y_scale, z=z_scale)
        )
        if feed_point is not None:
            color="red"     # Put it in black if you want
            # Ajouter le point spécifique
            scatter = go.Scatter3d(
                x=[feed_point[0]],
                y=[feed_point[1]],
                z=[feed_point[2]],
                mode="markers",
                marker=dict(size=6, color=color, opacity=1.0),
                name="Point d'alimentation"
            )

            # Ajouter le point à la figure
            fig.add_trace(scatter)

        # Afficher la figure
        fig.show()

    def data_save(self, filename, save_folder_name):
        data = {'p': self.points, 't': self.triangles}
        save_file_name = filename + '.mat'
        full_save_path = os.path.join(save_folder_name, save_file_name)
        if not os.path.exists(save_folder_name):
            os.makedirs(save_folder_name)
            print(f"Directory '{save_folder_name}' created.")
        savemat(full_save_path, data)
        print(f"Data saved successfully to {full_save_path}")
        return save_file_name
