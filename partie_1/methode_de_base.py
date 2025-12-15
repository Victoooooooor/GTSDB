import json
import os
import random as r
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def plot_image(image, title=None):
    """
    Affiche une image
    """
    plt.figure()
    plt.imshow(image)
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.show()

def load_image(image_path):
    """
    charge une image
    """
    if not os.path.exists(image_path):
        print(f"erreur: Image introuvable({image_path})") 
    return Image.open(image_path)

def save_image(image, save_path):
    """
    Sauvegarde une image
    """
    image.save(save_path)
    print(f"Image sauvegardée : {save_path}")

def load_annotation(annotation_path):
    """
    charge les anotations
    Retourne un dict: { "00000.ppm": [ {"coord":(x_min,y_min,x_max,y_max),"class_id":id}, ... ], ... }
    """
    dico = {}
    if not os.path.exists(annotation_path):
        print("erreur: fichier d'annotations introuvable")
        return dico

    with open(annotation_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            #filename; x_min; y_min; x_max; y_max; class_id
            parts = line.split(";")
            if len(parts) < 6:
                continue # on sait jamais

            filename = parts[0].strip()
            x_min = int(parts[1])
            y_min = int(parts[2])
            x_max = int(parts[3])
            y_max = int(parts[4])
            class_id = int(parts[5])

            #plusieur annotation par image parfois
            if filename not in dico.keys():
                dico[filename] = []

            dico[filename].append({
                "coord": (x_min, y_min, x_max, y_max),
                "class_id": class_id
            })

    return dico

def save_json(data, json_path):
    """
    Sauvegarde un JSON.
    """
    dir_name = os.path.dirname(json_path)
    if dir_name != "":
        os.makedirs(dir_name, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Json {json_path} sauvegardé")

def load_json(json_path):
    """
    Charge un fichier JSON
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def create_json_imagetes_class(imagettes_root_dir, json_path):
    """
    Parcourt le dossier des imagettes (rangées par classe) et crée un JSON
    associant chaque fichier à sa classe.
    """
    samples = []

    for class_name in os.listdir(imagettes_root_dir):
        class_dir = os.path.join(imagettes_root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        try:
            class_id = int(class_name)
        except ValueError:
            print(f"Classe ignorée (nom non numérique) : {class_name}")
            continue

        for fname in os.listdir(class_dir):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            # chemin relatif depuis imagettes_root_dir
            rel_path = os.path.join(class_name, fname)
            samples.append({
                "path": rel_path,
                "class_id": class_id
            })

    save_json(samples, json_path)
    print(f"{len(samples)} imagettes indexées dans {json_path}")
