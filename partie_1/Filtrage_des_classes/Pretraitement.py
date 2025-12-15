import os
from sklearn.model_selection import train_test_split
from .. import methode_de_base as mb

def calculs_repartition_classes(annotation_file_path):
    """
    Calculs la repartition des des classs sans rien de plus
    """
    repartition_class={}
    dico_annotation = mb.load_annotation(annotation_file_path)

    for annotations in dico_annotation.values():
        for ann in annotations:  
            class_id = ann["class_id"]     

            if class_id not in repartition_class:
                repartition_class[class_id] = 1
            else:
                repartition_class[class_id] += 1
    return repartition_class

def print_repartition_classes(repartition_class):
    nbr_total=0 #verif debug
    for id_class, nbr in sorted(repartition_class.items(), key=lambda x: x[1], reverse=True):
        print(f"id classe : {id_class:<3}  -  {nbr:>3}")
        nbr_total+=nbr
    print(f"nbr total de val {nbr_total}.")

def find_reprtition_sup_val(repartition_class,min=20):
    """
    filtrage simple sup a un min
    """
    selection_class=[]
    for id_class, nbr in sorted(repartition_class.items()):
        if(nbr>min):
            selection_class.append(id_class)
    print(f"la liste selectioné : {selection_class} \n taille : {len(selection_class)}")
    return selection_class

def load_imagettes_par_classe(imagettes_dir,extension="png"):
    """
    Charge toutes les imagettes triées par classe.
    Retourne un dictionnaire : {class_id : [liste chemins imagettes]}
    """
    dico = {}
    for class_name in os.listdir(imagettes_dir):
        class_path = os.path.join(imagettes_dir, class_name)
        if os.path.isdir(class_path):
            dico[int(class_name)] = [os.path.join(class_path+"/", f) for f in os.listdir(class_path) if f.lower().endswith(extension)]
    return dico

def filtre_imagettes_selon_liste(imagettes_dir,annotation_file_path):
    """
    filtre selon la seltection faite avec find_reprtition_sup_val et renvoie la liste de couple (path_image,class)
    """
    selection_class = find_reprtition_sup_val(calculs_repartition_classes(annotation_file_path))
    liste_couple_path_class=[]
    dico = load_imagettes_par_classe(imagettes_dir)
    for class_id,paths in dico.items():
        if(class_id in selection_class):
            for path in paths:
                liste_couple_path_class.append((path,class_id))
    return liste_couple_path_class

def split_train_val_test(data, p_train=0.7, p_test=0.15):
    "on split"
    paths = [x[0] for x in data]
    labels = [x[1] for x in data]
    X_train, X_tmp, y_train, y_tmp = train_test_split(paths,labels,train_size=p_train,stratify=labels)
    p_test_rel = p_test / (1 - p_train)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp,y_tmp,test_size=p_test_rel,stratify=y_tmp)
    train = list(zip(X_train, y_train))
    val   = list(zip(X_val, y_val))
    test  = list(zip(X_test, y_test))

    taille_totale=len(data)
    print("repartion du split:")
    print(f"    -train: {round(len(train)/taille_totale*100)}")
    print(f"    -val: {round(len(val)/taille_totale*100)}")
    print(f"    -test: {round(len(test)/taille_totale*100)}")
    return train, val, test

def repartition_pourcentage_par_classe(data):
    repartition = {}
    total = len(data)
    if total == 0:
        return repartition
    for _, cid in data:
        if cid not in repartition:
            repartition[cid] = 0
        repartition[cid] += 1

    for cid in repartition:
        repartition[cid] = (repartition[cid] / total) * 100

    return repartition

def print_repartition_pourcentage_par_classe(data, title="SPLIT"):
    rep = repartition_pourcentage_par_classe(data)
    print(f"\n{title} (total = {len(data)})")
    for cid in sorted(rep):
        print(f"Classe {cid:<3} : {rep[cid]:5.2f}%")
