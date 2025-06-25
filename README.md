# Système de recommandation d’images basé sur les préférences utilisateur

## Présentation

Ce projet a pour objectif de construire un système de recommandation d’images en Python, en exploitant des techniques de data mining et d’apprentissage automatique. Il automatise la collecte, l’annotation, l’analyse et la visualisation des images, puis propose des recommandations personnalisées selon les préférences de l’utilisateur.

## Fonctionnalités principales

- **Collecte automatisée d’images** via requêtes SPARQL sur Wikidata et téléchargement dans le dossier `images/`.
- **Extraction et stockage des métadonnées** (taille, orientation, couleur dominante, luminosité, date, etc.) à partir des fichiers EXIF et analyse d’image.
- **Annotation et étiquetage** automatique et manuel des images (tags, couleurs, etc.).
- **Profil utilisateur** construit à partir de la sélection d’images favorites.
- **Analyses de données** sur les préférences utilisateurs et les caractéristiques des images.
- **Visualisation** interactive des données (matplotlib, tkinter).
- **Système de recommandation** basé sur le clustering (KMeans) et la classification (SVM).
- **Interface graphique** pour la sélection, l’annotation et la recommandation d’images.

## Structure du projet

- `images/` : dossiers contenant les images téléchargées et leurs métadonnées CSV.
- `data.json` : configuration des types d’images à collecter.
- `data_users.json` : stockage des profils utilisateurs.
- `function.py` : fonctions utilitaires pour l’analyse et la manipulation des images.
- `interface.py` : interfaces graphiques (Tkinter).
- `projet.ipynb` : notebook principal, intégrant l’ensemble du pipeline.
- `requirements.txt` : dépendances Python du projet.
- `README.md` : ce fichier.

## Installation

1. Clonez ce dépôt.
2. Installez les dépendances :
   ```sh
   pip install -r requirements.txt
   ```
3. Lancez le notebook `projet.ipynb` ou exécutez les scripts selon vos besoins.

## Utilisation

- Lancez le notebook.
- Suivez les instructions de l’interface pour importer les images, sélectionner vos préférées, visualiser les analyses et obtenir des recommandations personnalisées.

## Sources des données

Les images sont collectées automatiquement depuis Wikidata, uniquement parmi les images sous licence libre.



