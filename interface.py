import tkinter as tk
from PIL import Image as PILImage, ImageTk
import numpy as np
from PIL import Image
import pandas as pd
import function
from tkinter import messagebox
import json
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import ttk


class Application(tk.Tk):
    def __init__(self, debug=False):
        super().__init__()
        self.debug = debug

        self.title("Interface Tkinter avec navigation")

        # Dimension et centrage de la fenêtre
        self.centrer_fenetre(300, 80)

        # Mise au premier plan
        self.lift()
        self.attributes("-topmost", True)
        self.after_idle(self.attributes, "-topmost", False)

        # Données nécessaires
        self.data_users = function.import_data_users("data_users.json")
        self.users = function.import_users(self.data_users)

        self.list_frame = [
            (Frame_ifmod(self), (300, 80)),
            (Frame_datamod(self, "data.json"), (200, 420)),
            (Frame_ifimport(self), (300, 80)),
            (Frame_datachoice(self), (0, 0)),
            (Frame_importimages(self), (160, 140)),
            (Frame_importexif(self), (160, 140)),
            (Frame_createmetadata(self), (160, 140)),
            (Frame_choix1(self), (200, 150)),
            (
                Frame_graph(self),
                (self.winfo_screenwidth(), self.winfo_screenheight() - 80),
            ),
            (Frame_choixutilisateur(self), (400, 300)),
            (Frame_top10(self), (160, 140)),
            (Frame_choiximage(self), (0, 0)),
            (Frame_prepa(self), (160, 140)),
            (Frame_choix2(self), (220, 300)),
            (Frame_pred1(self), (0, 0)),
            (Frame_pred2(self), (0, 0)),
            (Frame_usergraph(self), (400, 200)),
        ]

        # Créer la fenêtre de navigation si en mode debug
        if debug:
            self.navbar = Navbar(self)

        # Initialiser la première frame
        frame1 = self.list_frame[0][0]
        frame1.grid(row=1, column=0, sticky="nsew")

        # Configuration de la grille pour l'extension de l'interface
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Variables de fonctionnement initiales
        self.data_mod = 0
        self.if_import = 0
        self.info_a_traiter = ""
        self.data = json.load(open("data.json", encoding="utf-8"))
        self.dataframes = {}
        self.animation_angle = 0
        self.df_all_exif = pd.DataFrame()
        self.df_image_data_user = pd.DataFrame()
        self.top10 = pd.DataFrame()
        self.selected = []
        self.image_data_global = []
        self.image_data_user = []
        self.filtered_row = pd.DataFrame()
        self.recommendations = pd.DataFrame()
        self.prediction = ""
        self.user = ""

    def centrer_fenetre(self, largeur: int, hauteur: int):
        """Centrer la fenêtre sur l'écran."""
        largeur_ecran = self.winfo_screenwidth()
        hauteur_ecran = self.winfo_screenheight()
        x = (largeur_ecran - largeur) // 2
        y = (hauteur_ecran - hauteur) // 2
        self.geometry(f"{largeur}x{hauteur}+{x-10}+{y-40}")

    def change_frame(self, index_frame: int):
        """Changer de frame."""
        if 0 < index_frame <= len(self.list_frame):
            if index_frame in {15}:
                self.list_frame[index_frame - 1][0].affiche_images(self.recommendations, self.df_image_data_user)  # type: ignore
            elif index_frame in {16}:
                self.list_frame[index_frame - 1][0].affiche_image(self.filtered_row["image"].values[0], self.prediction)  # type: ignore
            for i in range(len(self.list_frame)):
                frame, size = self.list_frame[i]
                if index_frame == 4 and i + 1 == 4:
                    self.update_data_json("data.json")
                    size = frame.get_size(self.data)  # type: ignore
                elif index_frame == 12 and i + 1 == 12:
                    size = frame.get_size(self.top10)  # type: ignore
                elif index_frame == 15 and i + 1 == 15:
                    size = frame.get_size()  # type: ignore
                elif index_frame == 16 and i + 1 == 16:
                    size = frame.get_size()  # type: ignore
                if i + 1 == index_frame:
                    self.centrer_fenetre(*size)
                    frame.grid(row=1, column=0, sticky="nsew")
                else:
                    frame.grid_forget()
            if index_frame in {5, 6, 7, 11, 13}:
                self.list_frame[index_frame - 1][0].start_animation()  # type: ignore
            elif index_frame in {9}:
                self.list_frame[index_frame - 1][0].create_graph(self.df_image_data_user)  # type: ignore
            
    def update_data_json(self, path: str):
        """Mettre à jour les données JSON."""
        self.data = json.load(open(path, encoding="utf-8"))


class Navbar(tk.Toplevel):
    def __init__(self, parent: Application):
        super().__init__(parent)

        # Définir la taille de la fenêtre de la barre de navigation
        size = len(parent.list_frame)
        self.title("Barre de Navigation")
        self.geometry(f"{size * (12 + 55)}x50+10+10")
        self.configure(bg="#333")

        # Créer les boutons de navigation pour chaque frame
        for i in range(size):
            btn_frame = tk.Button(
                self,
                text=f"Frame {i + 1}",
                command=lambda i=i: parent.change_frame(i + 1),
            )
            btn_frame.pack(side="left", padx=5)


class Frame_ifimport(tk.Frame):
    def __init__(self, parent: Application):
        super().__init__(parent)
        self.parent = parent

        # Créer une étiquette
        label = tk.Label(self, text="Souhaitez-vous faire l'importation des images")
        label.pack(pady=10)

        # Créer un conteneur Frame pour les boutons
        button_frame = tk.Frame(self)
        button_frame.pack()

        # Créer un bouton "Oui"
        yes_button = tk.Button(button_frame, text="Oui", command=self.on_yes)
        yes_button.pack(side=tk.LEFT, padx=10)

        # Créer un bouton "Non"
        no_button = tk.Button(button_frame, text="Non", command=self.on_no)
        no_button.pack(side=tk.RIGHT, padx=10)

    def on_yes(self):
        """Action à effectuer lorsque l'utilisateur clique sur 'Oui'."""
        self.parent.if_import = 1
        self.parent.change_frame(4)

    def on_no(self):
        """Action à effectuer lorsque l'utilisateur clique sur 'Non'."""
        self.parent.if_import = 0
        self.parent.change_frame(4)


class Frame_ifmod(tk.Frame):
    def __init__(self, parent: Application):
        super().__init__(parent)
        self.parent = parent

        # Créer une étiquette
        label = tk.Label(self, text="Souhaitez-vous modifier le fichier Json")
        label.pack(pady=10)

        # Créer un conteneur Frame pour les boutons
        button_frame = tk.Frame(self)
        button_frame.pack()

        # Créer un bouton "Oui"
        yes_button = tk.Button(button_frame, text="Oui", command=self.on_yes)
        yes_button.pack(side=tk.LEFT, padx=10)

        # Créer un bouton "Non"
        no_button = tk.Button(button_frame, text="Non", command=self.on_no)
        no_button.pack(side=tk.RIGHT, padx=10)

    def on_yes(self):
        """Action à effectuer lorsque l'utilisateur clique sur 'Oui'."""
        self.parent.data_mod = 1
        self.parent.change_frame(2)

    def on_no(self):
        """Action à effectuer lorsque l'utilisateur clique sur 'Non'."""
        self.parent.data_mod = 0
        self.parent.change_frame(3)


class Frame_datamod(tk.Frame):
    def __init__(self, parent: Application, json_file: str):
        super().__init__(parent)
        self.parent = parent
        self.json_file = json_file
        self.load_json()
        self.create_widgets()

    def load_json(self):
        """Charger le fichier JSON"""
        try:
            with open(self.json_file, "r") as f:
                self.data = json.load(f)
        except Exception as e:
            messagebox.showerror(
                "Erreur", f"Impossible de charger le fichier JSON: {e}"
            )
            self.data = {}

    def save_json(self):
        """Sauvegarder les modifications dans le fichier JSON"""
        try:
            with open(self.json_file, "w") as f:
                json.dump(self.data, f, indent=4)
            messagebox.showinfo("Succès", "Fichier JSON sauvegardé avec succès!")
        except Exception as e:
            messagebox.showerror(
                "Erreur", f"Impossible de sauvegarder le fichier JSON: {e}"
            )

    def create_widgets(self):
        """Créer l'interface pour éditer les éléments du fichier JSON"""
        # Formulaire pour ajouter une nouvelle clé
        self.new_key_label = tk.Label(self, text="Ajouter une nouvelle clé:")
        self.new_key_label.pack(pady=10)

        self.new_key_entry = tk.Entry(self)
        self.new_key_entry.pack(pady=5)

        self.add_key_button = tk.Button(
            self, text="Ajouter Clé", command=self.add_new_key
        )
        self.add_key_button.pack(pady=5)

        # Liste déroulante pour sélectionner la clé principale
        self.key_selector = tk.StringVar(self)
        self.key_selector.set("exoplanet")  # Valeur par défaut

        self.keys_ = list(self.data.keys())
        self.dropdown = tk.OptionMenu(
            self, self.key_selector, *self.keys_, command=self.update_form
        )
        self.dropdown.pack(pady=10)

        # Champs pour chaque élément
        self.folder_label = tk.Label(self, text="Folder:")
        self.folder_label.pack()

        self.folder_entry = tk.Entry(self)
        self.folder_entry.pack(pady=5)

        self.sparql_label = tk.Label(self, text="SPARQL:")
        self.sparql_label.pack()

        self.sparql_entry = tk.Entry(self)
        self.sparql_entry.pack(pady=5)

        self.prefix_label = tk.Label(self, text="Prefix:")
        self.prefix_label.pack()

        self.prefix_entry = tk.Entry(self)
        self.prefix_entry.pack(pady=5)

        # Boutons pour sauvegarder et supprimer
        self.save_button = tk.Button(
            self, text="Sauvegarder", command=self.save_changes
        )
        self.save_button.pack(pady=5)

        self.delete_key_button = tk.Button(
            self, text="Supprimer Clé", command=self.delete_key
        )
        self.delete_key_button.pack(pady=1)

        self.quit_button = tk.Button(self, text="Quitter", command=self.change_frame)
        self.quit_button.pack(pady=10)

        self.update_form()

    def update_form(self, *args):
        """Mettre à jour les champs avec les données du fichier JSON"""
        selected_key = self.key_selector.get()
        if selected_key in self.data:
            entry_data = self.data[selected_key]
            self.folder_entry.delete(0, tk.END)
            self.folder_entry.insert(0, entry_data.get("folder", ""))
            self.sparql_entry.delete(0, tk.END)
            self.sparql_entry.insert(0, entry_data.get("sparql", ""))
            self.prefix_entry.delete(0, tk.END)
            self.prefix_entry.insert(0, entry_data.get("prefix", ""))

    def save_changes(self):
        """Sauvegarder les modifications dans le JSON"""
        selected_key = self.key_selector.get()
        folder = self.folder_entry.get()
        sparql = self.sparql_entry.get()
        prefix = self.prefix_entry.get()
        if selected_key in self.data:
            self.data[selected_key] = {
                "folder": folder,
                "sparql": sparql,
                "prefix": prefix,
            }
            self.save_json()

    def add_new_key(self):
        """Ajouter une nouvelle clé au fichier JSON"""
        new_key = self.new_key_entry.get().strip()
        if not new_key:
            messagebox.showerror("Erreur", "Le nom de la clé ne peut pas être vide.")
            return
        if new_key in self.data:
            messagebox.showerror("Erreur", "La clé existe déjà dans le JSON.")
            return
        folder = self.folder_entry.get()
        sparql = self.sparql_entry.get()
        prefix = self.prefix_entry.get()
        self.data[new_key] = {"folder": folder, "sparql": sparql, "prefix": prefix}
        self.keys_ = list(self.data.keys())
        self.key_selector.set(new_key)
        self.dropdown["menu"].delete(0, "end")
        for key in self.keys_:
            self.dropdown["menu"].add_command(
                label=key, command=tk._setit(self.key_selector, key)
            )
        self.new_key_entry.delete(0, tk.END)
        self.save_json()

    def delete_key(self):
        """Supprimer la clé sélectionnée du fichier JSON"""
        selected_key = self.key_selector.get()
        if selected_key not in self.data:
            messagebox.showerror("Erreur", "La clé sélectionnée n'existe pas.")
            return
        confirmation = messagebox.askyesno(
            "Confirmation",
            f"Êtes-vous sûr de vouloir supprimer la clé '{selected_key}' et toutes ses données ?",
        )
        if confirmation:
            del self.data[selected_key]
            self.keys_ = list(self.data.keys())
            self.key_selector.set(self.keys_[0] if self.keys_ else "")
            self.dropdown["menu"].delete(0, "end")
            for key in self.keys_:
                self.dropdown["menu"].add_command(
                    label=key, command=tk._setit(self.key_selector, key)
                )
            self.save_json()

    def change_frame(self):
        """Changer de frame"""
        self.parent.change_frame(3)


class Frame_datachoice(tk.Frame):
    def __init__(self, parent: Application):
        super().__init__(parent)
        self.parent = parent
        self.var = tk.StringVar(value="rien")

        # Label pour demander la base de données à utiliser
        self.label = tk.Label(
            self, text="Quelle base de données\nsouhaitez-vous utiliser ?"
        )
        self.label.pack(pady=3)

        # Frame pour les boutons radio
        self.frame_rbt = tk.Frame(self)
        self.frame_rbt.pack(expand=True, fill="both")

        # Bouton pour soumettre la réponse
        tk.Button(self, text="Soumettre", command=self.traiter_reponse).pack(
            anchor=tk.CENTER, pady=3
        )

    def traiter_reponse(self):
        """Traiter la réponse de l'utilisateur et changer de frame en conséquence."""
        var = self.var.get()
        self.parent.info_a_traiter = var
        if var != "rien":
            self.parent.change_frame(5)
        else:
            self.parent.change_frame(1)

    def get_size(self, data: dict[str, dict[str, str]]):
        """Créer les boutons radio pour chaque clé de données et retourner la taille de la fenêtre."""
        for key in data.keys():
            tk.Radiobutton(
                self.frame_rbt, text=key.title(), variable=self.var, value=key
            ).pack(anchor=tk.CENTER, pady=1)
        tk.Radiobutton(
            self.frame_rbt, text="Tout", variable=self.var, value="tout"
        ).pack(anchor=tk.CENTER, pady=1)
        tk.Radiobutton(
            self.frame_rbt, text="Rien", variable=self.var, value="rien"
        ).pack(anchor=tk.CENTER, pady=1)
        return 300, 30 * (len(data) + 2) + 55


class Frame_travail(tk.Frame):
    def __init__(self, parent: Application):
        super().__init__(parent)
        self.parent = parent
        self.create_widgets()

        self.running = False  # Pour savoir si l'animation doit tourner ou non
        self.angle = 0  # Angle de rotation initial
        self.animation_id = None  # ID de l'animation pour pouvoir l'arrêter

    def create_widgets(self):
        # Label pour afficher "Importation des images"
        self.label = tk.Label(self, text="temporaire")
        self.label.pack()

        # Canvas pour l'animation de chargement
        self.canvas = tk.Canvas(self, width=100, height=100)
        self.canvas.pack()

    def start_animation(self):
        # Démarrer l'animation seulement si elle n'est pas déjà en cours
        if not self.running:
            self.angle = self.parent.animation_angle
            self.running = True
            self.animate()
            threading.Thread(target=self.travail).start()

    def travail(self):
        # Simuler un travail en cours (cera remplacé par le vrai travail)
        if self.parent.debug:
            time.sleep(2)
        self.stop_animation()

    def stop_animation(self):
        # Arrêter l'animation en annulant le `after` si en cours
        if self.animation_id:
            self.parent.after_cancel(self.animation_id)
            self.running = False
            self.animation_id = None
            self.parent.animation_angle = self.angle

    def animate(self):
        if self.running:
            # Effacer le canvas
            self.canvas.delete("all")

            # Dessiner un cercle de chargement
            self.draw_loading_circle(self.angle)

            # Mettre à jour l'angle pour la rotation
            self.angle += 5  # Augmenter l'angle pour la rotation

            # Si l'angle dépasse 360°, le remettre à zéro
            if self.angle >= 360:
                self.angle = 0

            # Continuer l'animation après 50 ms
            self.animation_id = self.parent.after(50, self.animate)

    def draw_loading_circle(self, angle):
        # Calculer les coordonnées pour dessiner un arc de cercle
        x_center = self.canvas.winfo_height() // 2
        y_center = self.canvas.winfo_width() // 2
        radius = x_center // 2

        # Dessiner un arc de cercle représentant l'animation de chargement
        self.canvas.create_arc(
            x_center - radius,
            y_center - radius,
            x_center + radius,
            y_center + radius,
            start=angle,
            extent=45,
            outline="gray",
            width=5,
        )


class Frame_importimages(Frame_travail):
    def __init__(self, parent: Application):
        super().__init__(parent)
        self.label.configure(text="Importation des images")

    def travail(self):
        # Importer les données en fonction des paramètres
        self.parent.dataframes = function.get_data(
            self.parent.info_a_traiter, self.parent.if_import, 150, self.parent.data
        )
        super().travail()
        # Changer de frame après le travail
        self.parent.change_frame(6)


class Frame_importexif(Frame_travail):
    def __init__(self, parent: Application):
        super().__init__(parent)
        self.label.configure(text="Importation des exif")

    def travail(self):
        # Extraire les données EXIF des images
        self.parent.df_all_exif = function.extract_exif_date(self.parent.dataframes)
        super().travail()
        # Changer de frame après le travail
        self.parent.change_frame(7)


class Frame_createmetadata(Frame_travail):
    def __init__(self, parent: Application):
        super().__init__(parent)
        self.label.configure(text="Importation des metadata")

    def travail(self):
        # Étiqueter les images avec les métadonnées
        self.parent.df_image_data_user = function.etiquetage_image(
            self.parent.df_all_exif
        )
        super().travail()
        # Changer de frame après le travail
        self.parent.change_frame(8)


class Frame_choix1(tk.Frame):
    def __init__(self, parent: Application):
        super().__init__(parent)

        # Label pour la visualisation des données
        self.label = tk.Label(self, text="Visualisation des données")
        self.label.pack(pady=2)

        # Bouton pour naviguer vers la visualisation des données
        self.btn_to_frame2 = tk.Button(
            self, text="Visualisation", command=lambda: parent.change_frame(9)
        )
        self.btn_to_frame2.pack(pady=10)

        # Label pour la suite de la recommandation / utilisateur
        self.label = tk.Label(self, text="Suite recommandation / utilisateur")
        self.label.pack(pady=2)

        # Bouton pour naviguer vers la suite de la recommandation / utilisateur
        self.btn_to_frame3 = tk.Button(
            self, text="Suite", command=lambda: parent.change_frame(10)
        )
        self.btn_to_frame3.pack(pady=10)


class Frame_graph(tk.Frame):
    def __init__(self, parent: Application):
        super().__init__(parent)
        self.label = tk.Label(self, text="Visualisation des données")
        self.label.pack(pady=2)
        self.canvas = None

        # Bouton pour revenir
        self.btn_to_frame1 = tk.Button(
            self, text="Revenir", command=lambda: parent.change_frame(8)
        )
        self.btn_to_frame1.pack(pady=1)

    def create_graph(self, df_image_data_user: pd.DataFrame):
        # Créer la figure avec des sous-graphique
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))
        axs = axs.flatten()

        # Sous-graphique 1 : Courbe de la taille des pixels en ordre croissant
        df_sorted = df_image_data_user.sort_values(by="taille_pixels")
        axs[0].plot(
            range(1, len(df_sorted) + 1),
            df_sorted["taille_pixels"],
            marker="o",
            linestyle="-",
            color="purple",
        )
        axs[0].set_xlabel("Images triées de la plus petite à la plus grande")
        axs[0].set_ylabel("Taille en pixels")
        axs[0].set_title("Évolution de la taille des images")
        axs[0].grid(True)
        axs[0].ticklabel_format(style="plain", axis="y")

        # Sous-graphique 2 : Nombre d'images par orientation
        df_orientation = df_image_data_user["orientation"].value_counts()
        axs[1].bar(df_orientation.index, df_orientation, color="green")
        axs[1].set_xlabel("Orientation")
        axs[1].set_ylabel("Nombre d'images")
        axs[1].set_title("Nombre d'images par orientation")

        # Sous-graphique 3 : Nombre d'images par couleur dominante
        df_color = df_image_data_user["dominant_color_rgb"].value_counts()
        axs[2].bar(df_color.index, df_color)
        axs[2].set_xlabel("Couleur dominante")
        axs[2].set_ylabel("Nombre d'images")
        axs[2].set_title("Nombre d'images par couleur dominante")

        # Sous-graphique 4 : Histogramme de la répartition des valeurs de luminosité
        axs[3].hist(
            df_image_data_user["luminosity"],
            bins=20,
            color="skyblue",
            edgecolor="black",
        )
        axs[3].set_xlabel("Luminosité")
        axs[3].set_ylabel("Fréquence")
        axs[3].set_title("Répartition des valeurs de luminosité")
        axs[3].grid(axis="y", linestyle="--", alpha=0.7)

        # Sous-graphique 5 : Nombre de photos par année
        photos_par_annee = df_image_data_user.groupby("date_prise_vue").size()
        axs[4].bar(photos_par_annee.index, photos_par_annee, color="skyblue")
        axs[4].set_title("Nombre de photos par année")
        axs[4].set_xlabel("Année")
        axs[4].set_ylabel("Nombre de photos")
        axs[4].tick_params(axis="x", rotation=45)

        # Supprimer le sixième subplot vide
        fig.delaxes(axs[5])

        # Ajuster l'espacement entre les sous-graphiques
        plt.subplots_adjust(hspace=0.6, wspace=0.4)

        if self.canvas:
            self.canvas.destroy()

        # Affichage de la figure dans le canvas Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        self.canvas = canvas.get_tk_widget()
        self.canvas.pack(fill=tk.BOTH, expand=True)


class Frame_choixutilisateur(tk.Frame):
    def __init__(self, parent: Application):
        super().__init__(parent)
        self.parent = parent

        # Titre de la Frame
        self.label_title = tk.Label(
            self, text="Gestion des Utilisateurs", font=("Arial", 16)
        )
        self.label_title.grid(row=0, column=0, columnspan=2, pady=10)

        # Entrée pour le nom de l'utilisateur
        self.label_name = tk.Label(self, text="Nom de l'utilisateur :")
        self.label_name.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.entry_name = tk.Entry(self)
        self.entry_name.grid(row=1, column=1, padx=10, pady=5)

        # Bouton pour ajouter un utilisateur
        self.button_add_user = tk.Button(
            self, text="Ajouter Utilisateur", command=self.add_user
        )
        self.button_add_user.grid(row=2, column=0, columnspan=2, pady=10)

        # Combobox pour sélectionner un utilisateur
        self.label_select_user = tk.Label(self, text="Sélectionner un utilisateur :")
        self.label_select_user.grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.combobox_users = ttk.Combobox(self)
        self.combobox_users.grid(row=3, column=1, padx=10, pady=5)

        # Afficher l'utilisateur sélectionné
        self.label_selected_user = tk.Label(
            self, text="Utilisateur sélectionné : Aucun"
        )
        self.label_selected_user.grid(row=5, column=0, columnspan=2, pady=10)

        # Bouton pour valider l'utilisateur sélectionné
        self.button_validate_user = tk.Button(
            self, text="Valider Utilisateur", command=self.validate_user
        )
        self.button_validate_user.grid(row=4, column=0, columnspan=2, pady=10)

        # Mise à jour de la Combobox avec les utilisateurs ajoutés
        self.update_combobox()

    def add_user(self):
        """Ajouter un utilisateur à la liste s'il n'existe pas déjà."""
        name = self.entry_name.get()
        if name:
            # Vérifier si l'utilisateur existe déjà
            if name in self.parent.users:
                messagebox.showwarning(
                    "Utilisateur déjà existant", f"L'utilisateur '{name}' existe déjà."
                )
            else:
                self.parent.users.append(name)
                self.entry_name.delete(
                    0, tk.END
                )  # Effacer le champ de texte après ajout
                self.update_combobox()  # Mettre à jour la combobox
                messagebox.showinfo(
                    "Utilisateur ajouté",
                    f"L'utilisateur '{name}' a été ajouté avec succès.",
                )
        else:
            messagebox.showwarning(
                "Nom manquant", "Veuillez entrer un nom d'utilisateur."
            )

    def update_combobox(self):
        """Mettre à jour la combobox avec la liste des utilisateurs."""
        self.combobox_users["values"] = self.parent.users
        if self.parent.users:
            # Activer la sélection et le premier élément de la combobox
            self.combobox_users.current(0)
            self.combobox_users.bind("<<ComboboxSelected>>", self.on_user_select)

    def on_user_select(self, event):
        """Mettre à jour l'affichage de l'utilisateur sélectionné."""
        selected_user = self.combobox_users.get()
        self.label_selected_user.config(
            text=f"Utilisateur sélectionné : {selected_user}"
        )

    def validate_user(self):
        """Valider l'utilisateur sélectionné et afficher un message de confirmation."""
        selected_user = self.combobox_users.get()
        if selected_user:
            self.parent.user = selected_user
            self.parent.change_frame(11)
        else:
            messagebox.showwarning(
                "Aucun utilisateur sélectionné", "Veuillez sélectionner un utilisateur."
            )


class Frame_top10(Frame_travail):
    def __init__(self, parent: Application):
        super().__init__(parent)
        self.label.configure(text="Compilation des metadata")

    def travail(self):
        # Filtrer les 10 meilleures images
        self.parent.top10 = function.filter_10_image(self.parent.df_image_data_user)
        super().travail()
        # Changer de frame après le travail
        self.parent.change_frame(12)


class Frame_choiximage(tk.Frame):
    def __init__(self, parent: Application, nb_images: int = 10):
        super().__init__(parent)
        self.parent = parent
        self.nb_images = nb_images
        self.image_frame = None
        self.select_button = None

        # Calculer le nombre d'images par ligne et le nombre de lignes nécessaires
        self.nb_images_sur_2 = nb_images // 2
        self.nb_ligne = -(-nb_images // self.nb_images_sur_2)

    def show_selected_images(self):
        """Récupérer les images sélectionnées et changer de frame."""
        for i, checkbox in enumerate(self.checkboxes_var):
            img_path = self.df["image"][self.id_used[i]]
            if checkbox.get() == 1:  # Si la case est cochée
                self.parent.selected.append(("Favorite", img_path))
            else:
                self.parent.selected.append(("NotFavorite", img_path))
        self.parent.change_frame(13)

    def get_size(self, df: pd.DataFrame):
        """Afficher les images avec des cases à cocher et retourner la taille de la fenêtre."""
        self.df = df

        if self.image_frame:
            self.image_frame.destroy()

        # Créer une frame pour les images
        self.image_frame = tk.Frame(self)
        self.image_frame.pack()

        self.checkboxes_var = []
        self.image_widgets = []
        self.id_used = []

        # Taille des images
        image_largeur = 200
        image_hauteur = 200

        # Afficher les images et les cases à cocher
        for i in range(min(self.nb_images, len(df))):
            while True:
                random_index = np.random.randint(0, len(df))
                if random_index not in self.id_used:
                    self.id_used.append(random_index)
                    break
            img_path = df["image"][random_index]
            image = PILImage.open(img_path)
            image.thumbnail((image_largeur, image_hauteur))  # Redimensionner l'image
            image_tk = ImageTk.PhotoImage(image)
            image_widget = tk.Label(self.image_frame, image=image_tk)
            image_widget.image = image_tk  # type: ignore # Garder une référence à l'image

            # Disposer les images sur deux lignes avec trois images par ligne
            row = (
                i // self.nb_images_sur_2
            )  # Pour avoir nb_images_sur_2 images par ligne
            col = i % self.nb_images_sur_2  # Colonnes 0, 1, 2
            image_widget.grid(row=row * 2, column=col)

            self.image_widgets.append(image_widget)

            var = tk.IntVar(value=0)
            checkbox = tk.Checkbutton(self.image_frame, text="Favorite", variable=var)
            checkbox.grid(
                row=row * 2 + 1, column=col
            )  # Positionner le checkbox sous l'image

            self.checkboxes_var.append(var)

        if self.select_button:
            self.select_button.destroy()

        # Ajouter un bouton pour récupérer les images sélectionnées
        self.select_button = tk.Button(
            self, text="Sélectionner", command=self.show_selected_images
        )
        self.select_button.pack()

        # Définir la taille de la fenêtre
        largeur_fenetre = (image_largeur + 20) * self.nb_images_sur_2 + 10
        hauteur_fenetre = (image_hauteur + 20) * self.nb_ligne + 50
        return largeur_fenetre, hauteur_fenetre


class Frame_prepa(Frame_travail):
    def __init__(self, parent: Application):
        super().__init__(parent)
        self.label.configure(text="Compilation des metadata")

    def travail(self):
        # Compilation des metadata globales et utilisateur
        self.parent.image_data_global = function.list_metadata_all(
            self.parent.df_image_data_user
        )
        self.parent.image_data_user = function.list_metadata_utilisateur(
            self.parent.df_image_data_user, self.parent.selected
        )

        # Exécution des algorithmes de clustering et de classification
        self.parent.recommendations = function.clustering_algo(
            self.parent.image_data_global, self.parent.image_data_user
        )
        self.parent.filtered_row, self.parent.prediction = function.classification_algo(
            self.parent.image_data_user,
            self.parent.image_data_global,
            self.parent.selected,
            self.parent.df_image_data_user,
        )

        # Mise à jour des données utilisateur avec les résultats des prédictions
        self.parent.data_users[self.parent.user] = {
            "pred1": self.parent.recommendations.to_dict(),
            "pred2": self.parent.filtered_row.to_dict(),
        }

        # Appel de la méthode travail de la classe parente et changement de frame
        super().travail()
        self.parent.change_frame(14)


class Frame_choix2(tk.Frame):
    def __init__(self, parent: Application):
        super().__init__(parent)
        self.parent = parent

        # Boutons pour naviguer
        self.label = tk.Label(self, text="Prediction algo cluster")
        self.label.pack(pady=2)

        self.btn_to_frame2 = tk.Button(
            self, text="Prediction 1", command=lambda: parent.change_frame(15)
        )
        self.btn_to_frame2.pack(pady=10)

        self.label = tk.Label(self, text="Prediction algo classification")
        self.label.pack(pady=2)

        self.btn_to_frame3 = tk.Button(
            self, text="Prediction 2", command=lambda: parent.change_frame(16)
        )
        self.btn_to_frame3.pack(pady=10)

        self.label = tk.Label(self, text="Donné utilisateur")
        # self.label.pack(pady=2)

        self.btn_to_frame3 = tk.Button(
            self, text="graphique", command=lambda: parent.change_frame(17)
        )
        # self.btn_to_frame3.pack(pady=10)

        self.label = tk.Label(self, text="Retour au choix utilisateur")
        self.label.pack(pady=2)

        self.btn_to_frame3 = tk.Button(
            self, text="Retour", command=lambda: parent.change_frame(10)
        )
        self.btn_to_frame3.pack(pady=10)

        self.label = tk.Label(self, text="Merci d'avoir utiliser notre application")
        self.label.pack(pady=2)

        self.btn_to_frame3 = tk.Button(self, text="Exit", command=self.exit)
        self.btn_to_frame3.pack(pady=10)

    def exit(self):
        # Sauvegarder les données des utilisateurs avant de quitter
        function.save_data_users(self.parent.data_users)
        self.parent.quit()
        self.parent.destroy()


class Frame_pred1(tk.Frame):
    def __init__(self, parent: Application):
        super().__init__(parent)
        self.parent = parent
        self.frame = None
        self.button = None
        label = tk.Label(self, text="Recomendation de 3 images")
        label.pack(pady=5)

    def get_size(self):
        return 230 * 3, self.hauteur + 100

    def affiche_images(self, recommendations, df_all_exif_clean):
        # Obtenir les chemins des images à partir des recommandations
        indices_lignes = recommendations.index.tolist()
        image_paths = [
            df_all_exif_clean.iloc[indices_lignes[0]]["image"],
            df_all_exif_clean.iloc[indices_lignes[1]]["image"],
            df_all_exif_clean.iloc[indices_lignes[2]]["image"],
        ]

        frame = tk.Frame(self)
        frame.pack()
        if self.frame:
            self.frame.destroy()
        self.frame = frame
        hauteur = 0

        # Afficher les images dans la fenêtre tkinter
        for i, path in enumerate(image_paths):
            img = Image.open(path)
            img.thumbnail(
                (200, 200)
            )  # Redimensionner l'image pour s'adapter à la fenêtre
            img_tk = ImageTk.PhotoImage(img)

            label = tk.Label(frame, image=img_tk)
            label.image = img_tk  # type: ignore # Garder une référence à l'image
            label.grid(row=0, column=i, padx=10, pady=10)
            hauteur = max(hauteur,img_tk.height())

        self.hauteur = hauteur

        button = tk.Button(
            self, text="Retour", command=lambda: self.parent.change_frame(14)
        )
        button.pack(pady=5)
        if self.button:
            self.button.destroy()
        self.button = button


class Frame_pred2(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.label = None
        self.label2 = None
        self.button = None
        label = tk.Label(self, text="Image testée")
        label.pack(pady=5)

    def get_size(self):
        return 420, self.hauteur + 130

    def affiche_image(self, image_path, prediction):
        # Charger et redimensionner l'image
        img = Image.open(image_path)
        img.thumbnail((300, 300))  # Redimensionner l'image pour s'adapter à la fenêtre
        img_tk = ImageTk.PhotoImage(img)

        # Afficher l'image
        label = tk.Label(self, image=img_tk)
        label.image = img_tk  # type: ignore # Garder une référence à l'image
        label.pack()

        if self.label:
            self.label.destroy()
        self.label = label

        self.hauteur = img_tk.height()

        # Afficher la prédiction
        prediction_text = (
            f"Prediction: {'Favorite' if prediction == 'Favorite' else 'Non favorite'}"
        )
        prediction_label = tk.Label(self, text=prediction_text, font=("Helvetica", 14))
        prediction_label.pack(pady=5)

        if self.label2:
            self.label2.destroy()
        self.label2 = prediction_label

        # Bouton pour revenir à la frame précédente
        button = tk.Button(
            self, text="Retour", command=lambda: self.parent.change_frame(14)
        )
        button.pack(pady=5)
        if self.button:
            self.button.destroy()
        self.button = button


class Frame_usergraph(tk.Frame):
    def __init__(self, parent: Application):
        super().__init__(parent)
        self.parent = parent

        # Label de bienvenue
        label = tk.Label(
            self, text="Bienvenue dans la Frame non fini", font=("Arial", 20)
        )
        label.pack(pady=50)

        # Bouton pour revenir à la frame précédente
        button = tk.Button(
            self, text="Retour", command=lambda: self.parent.change_frame(14)
        )
        button.pack()


if __name__ == "__main__":
    app = Application(True)
    app.mainloop()
