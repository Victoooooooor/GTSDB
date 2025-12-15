import os
from PIL import Image
import methode_de_base as mb

def extraire_from_image(image, coord):
    """
    Extrait une sous-image à partir des coord.
    coord = (x_min, y_min, x_max, y_max)
    """
    return image.crop(coord)

def resize(image, size=(64, 64)):
    """
    Redimensionne une image
    """
    return image.resize(size, Image.BILINEAR)

def resize_all_image(image_dir, annotation_file_path,output_dir="",format=(64,64),extension="png"):
    """
    On va parcourir chaque image selon les anotations disponible et extraire chaque panneaux pour les placer dans un format fixe
    """
    dico_annotation = mb.load_annotation(annotation_file_path)
    dossier_sortie = os.path.join(output_dir, f"{format[0]}x{format[1]}/")
    nbr=0
    for image_name,annotations in dico_annotation.items():
        image = mb.load_image(os.path.join(image_dir, image_name))
        for idx, annotation in enumerate(annotations):
            tmp = extraire_from_image(image,annotation["coord"])
            tmp = resize(tmp,format)    
            base_name, _ = os.path.splitext(image_name)
            tmp_image_name= f"{base_name}_{idx}.{extension}"
            
            tmp_dossier_sortie=os.path.join(dossier_sortie, f"{annotation['class_id']}/")
            os.makedirs(tmp_dossier_sortie, exist_ok=True)
            mb.save_image(tmp,os.path.join(tmp_dossier_sortie, tmp_image_name))
            nbr+=1

    print(f"Extraction de {nbr} panneaux.")


