from partie_1.methode_de_base import load_json
from partie_1.Filtrage_des_classes.Pretraitement import filtre_imagettes_selon_liste
from partie_2.Conception_du_MLP.build_model import *
from partie_2.Entrainement.train import *
from partie_2.Entrainement.stats import *
from partie_2.Preparation_des_donnees.prepare_data import *
from partie_2.Evaluation.evaluation import *
import time


image_dir = "./data/Image/"
annotation_file = "./data/gt.txt"
imagettes_dir_64x64="./data/64x64/"


json_couple_path_class_train = imagettes_dir_64x64+"train.json"
json_couple_path_class_val = imagettes_dir_64x64+"val.json"
json_couple_path_class_test = imagettes_dir_64x64+"test.json"

def entrainement_complet(epochs=50,batch_size=64,size_image=64*64*3):
    """
    c juste la bloucle d'entrainement complet
    """
    train = load_json(json_couple_path_class_train)
    val = load_json(json_couple_path_class_val)

    class_selectioner = filtre_imagettes_selon_liste(imagettes_dir_64x64,annotation_file)

    train_dataset = contruire_dataset(train,batch_size=batch_size, shuffle=True, augment=True)
    val_dataset = contruire_dataset(val,batch_size=batch_size, shuffle=True, augment=True)
    
    start = time.time()
    model,history = entrainer_mlp(build_model(len(class_selectioner),size_image),train_dataset,val_dataset,epochs)
    end = time.time()
    print(f"Temps d'entraînement du MLP : {end - start:.2f} secondes")

    tracer_courbes(history)

    save_model(model)
    save_history(history)

def test_complet(batch_size=64,model_path="./partie_2/models/MLP_64x64.keras"):
    """
    test complet et reachargement du model
    """
    train = load_json(json_couple_path_class_train)
    test  = load_json(json_couple_path_class_test)

    train_ds = contruire_dataset(train, batch_size=batch_size, shuffle=False, augment=False)
    test_ds  = contruire_dataset(test,  batch_size=batch_size, shuffle=False, augment=False)

    model = load_model(model_path)
    evaluer_performances(model, train_ds, test_ds)

    y_true, y_pred = get_predictions(model, test_ds)
    cm = afficher_matrice_confusion(y_true,y_pred,title="Matrice de confusion - MLP (Test)")

    analyser_conflits_classes(cm,)

entrainement_complet()
#test_complet()