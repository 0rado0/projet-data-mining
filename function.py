import sys
from SPARQLWrapper import SPARQLWrapper, JSON
import requests
import shutil
import os
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm
import random
import json


# Sauvegarde le dataframe dans un fichier CSV
def save_data(dataframe: pd.DataFrame, dossier: str):
    dataframe.to_csv(dossier + "/data.csv", index=False)


# Télécharge une image depuis une URL et la sauvegarde dans un dossier spécifié
def download_image(url: str, dossier: str, id: int, prefix: str):
    headers = {"User-Agent": "Mozilla/5.0"}

    # Configuration de la stratégie de réessai
    retry_strategy = Retry(
        total=5,  # Nombre total de réessais
        status_forcelist=[500, 502, 503, 504],  # Codes de statut à réessayer
        allowed_methods=["HEAD", "GET", "OPTIONS"],  # Méthodes à réessayer
        backoff_factor=1,  # Facteur de backoff exponentiel
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    request = session.get(
        url, allow_redirects=True, headers=headers, stream=True, timeout=60
    )
    if request.status_code == 200:
        with open(
            dossier + "/" + prefix + str(id) + os.path.splitext(url)[1], "wb"
        ) as image:
            request.raw.decode_content = True
            shutil.copyfileobj(request.raw, image)
        return dossier + "/" + prefix + str(id) + os.path.splitext(url)[1]
    return "None"


# Exécute une requête SPARQL et retourne les résultats
def get_results(endpoint_url: str, query: str):
    user_agent = "WDQS-example Python/%s.%s" % (
        sys.version_info[0],
        sys.version_info[1],
    )
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()


# Importe des données depuis une requête SPARQL et sauvegarde les images
def import_data(dossier: str, query: str, prefix: str):
    if not os.path.exists(dossier):
        os.mkdir(dossier)

    endpoint_url = "https://query.wikidata.org/sparql"

    array = []
    results = get_results(endpoint_url, query)
    print(results)

    if isinstance(results, dict):
        for result in results["results"]["bindings"]:
            array.append(
                (
                    results["results"]["bindings"].index(result) + 1,
                    result["image"]["value"],
                )
            )

    dataframe = pd.DataFrame(array, columns=["id", "image"])
    dataframe = dataframe.astype(dtype={"id": "int", "image": "<U200"})

    dataframe["image"] = dataframe.apply(
        lambda row: download_image(row["image"], dossier, row["id"], prefix), axis=1
    )

    save_data(dataframe, dossier)
    return dataframe


# Obtient les données en fonction de l'entrée utilisateur et du drapeau d'importation
def get_data(
    info_a_traiter: str, importation: int, Limit: int, Data: dict[str, dict[str, str]]
):
    if not os.path.exists("images"):
        os.mkdir("images")
    dataframes = {}
    if info_a_traiter == "tout":
        for key, value in Data.items():
            query = f"""
            SELECT ?item ?itemLabel ?image
            WHERE
            {{
            ?item wdt:P31 wd:{value["sparql"]}.
            ?item wdt:P18 ?image.
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
            }}
            limit {Limit}
            """
            if importation == 1:
                dataframes[value["folder"]] = import_data(
                    "images/" + value["folder"], query, value["prefix"]
                )
            elif importation == 0:
                dataframes[value["folder"]] = pd.read_csv(
                    "images/" + value["folder"] + "/data.csv"
                )
    elif info_a_traiter and info_a_traiter in Data.keys():
        query = f"""
        SELECT ?item ?itemLabel ?image
        WHERE
        {{
        ?item wdt:P31 wd:{Data[info_a_traiter]["sparql"]}.
        ?item wdt:P18 ?image.
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
        }}
        limit {Limit}
        """
        if importation == 1:
            dataframes[Data[info_a_traiter]["folder"]] = import_data(
                "images/" + Data[info_a_traiter]["folder"],
                query,
                Data[info_a_traiter]["prefix"],
            )
        elif importation == 0:
            dataframes[Data[info_a_traiter]["folder"]] = pd.read_csv(
                "images/" + Data[info_a_traiter]["folder"] + "/data.csv"
            )
    return dataframes


# Extrait la date EXIF d'une image
def get_exif_date(img_path: str):
    try:
        img = Image.open(img_path)
        exif_data = img.getexif()
        if exif_data is not None:
            for tag, value in exif_data.items():
                if TAGS.get(tag, tag) == "DateTime":
                    return value
    except:
        return None


# Extrait les dates EXIF d'un dictionnaire de dataframes
def extract_exif_date(dict_df: dict[str, pd.DataFrame]):
    all_df = []
    for df in dict_df.values():
        df["date_prise_vue"] = df["image"].apply(get_exif_date)
        all_df.append(df.dropna(subset=["date_prise_vue"]))
    return pd.concat(all_df, ignore_index=True)


# Obtient les métadonnées d'une image
def get_image_metadata(image_path: str, colors_dict: dict[str, list[int]]):
    try:
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)

        # Taille et dimensions
        width, height = img.size
        file_size = width * height  # Nombre total de pixels

        # Orientation
        orientation = (
            "Paysage" if width > height else "Portrait" if height > width else "Carre"
        )

        # Comparer la couleur moyenne à chaque couleur dans le dictionnaire
        avg_color = img_array.mean(axis=(0, 1))  # Couleurs moyennes R, G, B
        closest_color = min(
            colors_dict,
            key=lambda color: np.sqrt(
                np.sum((np.array(avg_color) - np.array(colors_dict[color])) ** 2)
            ),
        )

        # Luminosité
        luminosity = (
            0.2126 * img_array[:, :, 0].mean()
            + 0.7152 * img_array[:, :, 1].mean()
            + 0.0722 * img_array[:, :, 2].mean()
        )

        return [file_size, orientation, closest_color, round(luminosity, 1)]

    except Exception as e:
        print(f"Erreur avec {image_path} : {e}")
        return [None, None, None, None]


# Étiquette les images avec des métadonnées
def etiquetage_image(df_all_exif: pd.DataFrame):
    colors_dict = {
        "Rouge": [255, 0, 0],
        "Vert": [0, 255, 0],
        "Bleu": [0, 0, 255],
        "Jaune": [255, 255, 0],
        "Cyan": [0, 255, 255],
        "Magenta": [255, 0, 255],
        "Blanc": [255, 255, 255],
        "Noir": [0, 0, 0],
    }

    df_all_exif[
        ["taille_pixels", "orientation", "dominant_color_rgb", "luminosity"]
    ] = df_all_exif["image"].apply(
        lambda x: pd.Series(get_image_metadata(x, colors_dict))
    )

    df_all_exif_clean = df_all_exif.dropna(
        subset=["taille_pixels", "orientation", "dominant_color_rgb", "luminosity"]
    )

    df_all_exif_clean["date_prise_vue"] = df_all_exif_clean["date_prise_vue"].astype(
        str
    )
    df_all_exif_clean["date_prise_vue"] = df_all_exif_clean["date_prise_vue"].str[:4]

    df_all_exif_clean["tag"] = df_all_exif_clean["image"].str.split("/").str[1]
    return df_all_exif_clean


# Visualise les données en utilisant matplotlib
def visualisation(df: pd.DataFrame):
    # Courbe : Taille des pixels triée par ordre croissant
    plt.figure(figsize=(10, 5))
    df_sorted = df.sort_values(by="taille_pixels")
    plt.plot(
        range(1, len(df_sorted) + 1),
        df_sorted["taille_pixels"],
        marker="o",
        linestyle="-",
        color="purple",
    )
    plt.xlabel("Images triées de la plus petite à la plus grande")
    plt.ylabel("Taille en pixels")
    plt.title("Évolution de la taille des images")
    plt.grid()
    plt.ticklabel_format(
        style="plain", axis="y"
    )  # Affiche les nombres en tant qu'entiers sur l'axe y
    plt.show()

    # Diagramme à barres : Nombre d'images par orientation
    plt.figure(figsize=(10, 5))
    df_orientation = df["orientation"].value_counts()
    plt.bar(df_orientation.index, df_orientation, color="green")
    plt.xlabel("Orientation")
    plt.ylabel("Nombre d'images")
    plt.title("Nombre d'images par orientation")
    plt.show()

    # Diagramme à barres : Nombre d'images par couleur dominante
    plt.figure(figsize=(10, 5))
    df_color = df["dominant_color_rgb"].value_counts()
    plt.bar(df_color.index, df_color)
    plt.xlabel("Couleur dominante")
    plt.ylabel("Nombre d'images")
    plt.title("Nombre d'images par couleur dominante")
    plt.show()

    # Histogramme des valeurs de luminosité
    plt.figure(figsize=(10, 5))
    plt.hist(df["luminosity"], bins=20, color="skyblue", edgecolor="black")
    plt.xlabel("Luminosité")
    plt.ylabel("Fréquence")
    plt.title("Répartition des valeurs de luminosité")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

    # Diagramme à barres : Nombre de photos par année
    photos_par_annee = df.groupby("date_prise_vue").size()
    plt.figure(figsize=(10, 6))
    photos_par_annee.plot(kind="bar", color="skyblue")
    plt.title("Nombre de photos par année")
    plt.xlabel("Année")
    plt.ylabel("Nombre de photos")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Calcule la diversité pour filtrer les images
def calculate_diversity(row, covered_col1, covered_col2, covered_col5):
    diversity = 0
    if row["orientation"] not in covered_col1:
        diversity += 1
    if row["dominant_color_rgb"] not in covered_col2:
        diversity += 1
    if row["tag"] not in covered_col5:
        diversity += 1
    return diversity


# Filtre les 10 meilleures images basées sur la diversité
def filter_10_image(df: pd.DataFrame):
    col1_values = df["orientation"].unique()
    col2_values = df["dominant_color_rgb"].unique()
    col5_values = df["tag"].unique()

    covered_col1 = set()
    covered_col2 = set()
    covered_col5 = set()

    selected_rows = []

    while (
        len(covered_col1) < len(col1_values)
        or len(covered_col2) < len(col2_values)
        or len(covered_col5) < len(col5_values)
    ):
        best_diversity = -1
        best_row = None

        for _, row in df.iterrows():
            if (
                row["orientation"] not in covered_col1
                or row["dominant_color_rgb"] not in covered_col2
                or row["tag"] not in covered_col5
            ):
                diversity = calculate_diversity(
                    row, covered_col1, covered_col2, covered_col5
                )
                if diversity > best_diversity:
                    best_diversity = diversity
                    best_row = row

        if best_row is not None:
            selected_rows.append(best_row)
            covered_col1.add(best_row["orientation"])
            covered_col2.add(best_row["dominant_color_rgb"])
            covered_col5.add(best_row["tag"])

    remaining_rows_size = len(df[~df.index.isin([row.name for row in selected_rows])])

    while len(selected_rows) < min(10, remaining_rows_size):
        remaining_rows = df[~df.index.isin([row.name for row in selected_rows])]
        random_row = remaining_rows.sample(n=1).iloc[0]
        selected_rows.append(random_row)

    top_10 = pd.DataFrame(selected_rows)
    return top_10.reset_index(drop=True)


# Liste les métadonnées pour toutes les images
def list_metadata_all(df_all_exif_clean: pd.DataFrame):
    image_data_global = []

    for image_path in df_all_exif_clean["image"]:
        result = df_all_exif_clean[df_all_exif_clean["image"] == image_path]
        if not result.empty:
            for _, row in result.iterrows():
                image_data_global.append(
                    [
                        row["dominant_color_rgb"],
                        row["tag"],
                        row["luminosity"],
                        row["taille_pixels"],
                        row["orientation"],
                    ]
                )
        else:
            print(f"Image non trouvée dans le DataFrame : {image_path}")
    return image_data_global


# Liste les métadonnées pour les images sélectionnées
def list_metadata_utilisateur(
    df_all_exif_clean: pd.DataFrame, selected: list[tuple[int, str]]
):
    image_data_user = []

    for image_path in selected:
        result = df_all_exif_clean[df_all_exif_clean["image"] == image_path[1]]
        if not result.empty:
            for _, row in result.iterrows():
                image_data_user.append(
                    [
                        row["dominant_color_rgb"],
                        row["tag"],
                        row["luminosity"],
                        row["taille_pixels"],
                        row["orientation"],
                    ]
                )
        else:
            print(f"Image non trouvée dans le DataFrame : {image_path}")
    return image_data_user


# Algorithme de clustering pour la recommandation d'images
def clustering_algo(
    data: list[list[str | int]], image_data_user: list[list[str | int]]
):
    label_encoders: dict[int, LabelEncoder] = {}
    for i, val in enumerate(data[0]):
        if isinstance(val, str):
            label_encoders[i] = LabelEncoder()

    encoded_data = []
    for i, column in enumerate(zip(*data)):
        if i in label_encoders:
            encoded_data.append(label_encoders[i].fit_transform(column))
        else:
            encoded_data.append(column)

    X = list(zip(*encoded_data))

    k = 5  # Nombre de clusters
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(X)
    clusters = kmeans.labels_

    data_with_clusters = pd.DataFrame(
        data,
        columns=[
            "dominant_color_rgb",
            "tag",
            "luminosity",
            "taille_pixels",
            "orientation",
        ],
    )
    data_with_clusters["Cluster"] = clusters

    def recommend_items(cluster, data_with_clusters):
        items_in_cluster = data_with_clusters[data_with_clusters["Cluster"] == cluster]
        recommended_items = items_in_cluster.sample(n=3)
        return recommended_items

    user_interaction = image_data_user[0]
    encoded_interaction = [label_encoders[i].transform([val])[0] if i in label_encoders else val for i, val in enumerate(user_interaction)]  # type: ignore
    cluster = kmeans.predict([encoded_interaction])[0]  # type: ignore
    recommendations = recommend_items(cluster, data_with_clusters)
    return recommendations


# Algorithme de classification pour la prédiction d'images
def classification_algo(
    data: list[list[str | int]],
    image_data_global: list[list[str | int]],
    selected: list[tuple[int, str]],
    df_all_exif_clean: pd.DataFrame,
):
    result = [status for status, _ in selected]

    label_encoders: dict[int, LabelEncoder] = {}
    for i, val in enumerate(data[0]):
        if isinstance(val, str):
            label_encoders[i] = LabelEncoder()

    encoded_data = []
    for i, column in enumerate(zip(*data)):
        if i in label_encoders:
            encoded_data.append(label_encoders[i].fit_transform(column))
        else:
            encoded_data.append(column)

    X = list(zip(*encoded_data))
    y = result

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    classifier = svm.SVC()
    classifier.fit(X_train, y_train)

    sample_item = random.choice(image_data_global)

    encoded_item = [label_encoders[i].transform([val])[0] if i in label_encoders else val for i, val in enumerate(sample_item)]  # type: ignore

    try:
        prediction = classifier.predict([encoded_item])[0]  # type: ignore
    except:
        prediction = "NotFavorite"

    df = df_all_exif_clean
    filtered_row = df[
        (df["dominant_color_rgb"] == sample_item[0])
        & (df["tag"] == sample_item[1])
        & (df["luminosity"] == sample_item[2])
        & (df["taille_pixels"] == sample_item[3])
        & (df["orientation"] == sample_item[4])
    ]

    return filtered_row, prediction


# Importe les données des utilisateurs
def import_users(data_users: dict):
    return list(data_users.keys())


# Importe les données depuis un fichier JSON
def import_data_users(url: str):
    try:
        return json.load(open(url, encoding="utf-8"))
    except:
        return {}


# Sauvegarde les données des utilisateurs dans un fichier JSON
def save_data_users(dict: dict):
    with open("data_users.json", "w") as file:
        json.dump(dict, file, ensure_ascii=False)
