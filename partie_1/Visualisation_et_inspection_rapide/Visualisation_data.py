import os
import random as r
from PIL import ImageDraw
import methode_de_base as mb

#a) Afficher quelques images GTSDB avec leurs boîtes englobantes (dessiner le rectangle autour du panneau).
def draw_boite_sur_image(img, coord, color="red", largeur=2):
    """
    Dessine un rectangle sur une image.
    """
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    draw.rectangle(coord,outline=color,width=largeur)

        
    return img_copy

def chose_nbr_random_image(image_dir,dico_annotation, nbr_image,min_annotations=1,extension=".ppm"):
    """
    Choisis de facon random nbr d'image
    Retourne liste(path_image, image_name)
    """
    if not os.path.exists(image_dir):
        print("erreur: image dir non trouvée, chemin invalide")
        return []

    fichiers = [f for f in os.listdir(image_dir) if f.lower().endswith(extension)]
    if len(fichiers) == 0:
        print(f"erreur: aucune image {extension} trouvée dans le dossier")
        return []

    candidats = [
        f for f in fichiers
        if f in dico_annotation and len(dico_annotation[f]) >= min_annotations
    ]

    if len(candidats) == 0:
        print(f"erreur: aucune image avec au moins {min_annotations} annotation(s)")
        return []

    k = min(nbr_image, len(candidats))
    images_choisies = r.sample(candidats, k)

    liste_couple_path_name = [(os.path.join(image_dir, image), image) for image in images_choisies]
    return liste_couple_path_name

#a) Afficher quelques images GTSDB avec leurs boîtes englobantes (dessiner le rectangle autour du panneau).
def visualise_with_box(image_dir, annotation_file_path, nbr_de_visualistion=5,min=1):
    """
    image_dir : dossier contenant les images
    annotation_file_path : chemin vers gt.txt
    nbr_de_visualistion: le nbr que l'on veux visualiser
    min d'annotation
    """
    dico_annotation = mb.load_annotation(annotation_file_path)
    liste_couple_path_name = chose_nbr_random_image(image_dir,dico_annotation, nbr_de_visualistion,min)

    for image_path, image_name in liste_couple_path_name:
        image =  mb.load_image(image_path)
        image_with_box=image.copy()
        annotations = dico_annotation.get(image_name,[]) 
        for annotation in annotations:
            image_with_box =  draw_boite_sur_image(image_with_box, annotation["coord"])
        
        print(f"l'image :{image_path} nbr annottions:{len(annotations)}.")
        mb.plot_image(image_with_box, title=image_name)

#b)Commenter : taille des panneaux, variations de point de vue, luminosité,occlusions, etc.

"""
Les images de la base GTSDB montrent des panneaux de tailles très variées : 
certains apparaissent gros et bien visibles, tandis que d’autres sont beaucoup 
plus petits car éloignés. Les conditions de prise de vue changent d’une image 
à l’autre, avec des variations de luminosité (plein soleil, zones d’ombre) ainsi 
que des angles de vue différents. Après inspection de plus d’une centaine d’images,
on constate qu’il n’y a aucune occlusion : les panneaux sont toujours entièrement 
visibles. En revanche, certaines images présentent un léger flou dû au mouvement 
ou à la distance. La base reste très diversifiée, avec de nombreux types de panneaux 
(stop, cédez-le-passage, limitations de vitesse, direction…).
"""