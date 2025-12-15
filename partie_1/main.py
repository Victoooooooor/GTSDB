import methode_de_base as mb
from Visualisation_et_inspection_rapide.Visualisation_data import visualise_with_box
from Extraction_de_imagettes_de_panneaux.resizeur import resize_all_image
from Filtrage_des_classes.Pretraitement import print_repartition_classes
from Filtrage_des_classes.Pretraitement import calculs_repartition_classes
from Filtrage_des_classes.Pretraitement import filtre_imagettes_selon_liste
from Filtrage_des_classes.Pretraitement import split_train_val_test
from Filtrage_des_classes.Pretraitement import print_repartition_pourcentage_par_classe

image_dir = "./data/Image/"
annotation_file = "./data/gt.txt"
imagettes_output ="./data/"
imagettes_dir_64x64="./data/64x64/"
# la variable si dessous est la pour stocker la liste des couple path_image et class qui a ete retenue apres le filtre selon une valuer min de 20   
json_couple_path_class_train = imagettes_dir_64x64+"train.json"
json_couple_path_class_val = imagettes_dir_64x64+"val.json"
json_couple_path_class_test = imagettes_dir_64x64+"test.json"


#Visualisation et inspection rapide :
def visualise():
    visualise_with_box(image_dir,annotation_file,6,3)

visualise()
#rep questions dans le fichier.py

#Extraction de imagettes de panneaux :
def resize_all():
    resize_all_image(image_dir,annotation_file,imagettes_output)

#Filtrage des classes:
def split_data():
    #pour visualié:
    print_repartition_classes(calculs_repartition_classes(annotation_file))

    liste_couple_path_class = filtre_imagettes_selon_liste(imagettes_dir_64x64,annotation_file)
    train, val, test = split_train_val_test(liste_couple_path_class)

    #on verif qu'on garde l'homogeneité
    print_repartition_pourcentage_par_classe(train, title="TRAIN")
    print_repartition_pourcentage_par_classe(val,   title="VAL")
    print_repartition_pourcentage_par_classe(test,  title="TEST")

    mb.save_json(train, json_couple_path_class_train)
    mb.save_json(val, json_couple_path_class_val)
    mb.save_json(test, json_couple_path_class_test)




"""
Consigne : 
Pour initialiser, il faut mettre les images de 0 à 599.ppm dans un dossier data/Image et dans data le fichier gt.txt, ou remplacer les chemins au-dessus.
Ensuite, resize_all va faire tout ce qui est resize et sauvegarder le tout dans un fichier imagettes_output/ (format de l’image), donc normalement imagettes_output/64x64/.

Ensuite, split_data() va charger l’annotation, faire des calculs pour la répartition, et supprimer les classes < 20.
Puis ça va rouvrir les images resize pour créer une liste globale de couples (path, class), pour ensuite, avec split_train_val_test,
les diviser dans nos 3 blocs train, val et test. J’ai ajouté des JSON pour chacun d’entre eux pour que ce soit plus simple à réutiliser.

"""

resize_all()
split_data()