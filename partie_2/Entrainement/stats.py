import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

from partie_2.Preparation_des_donnees.prepare_data import decode_class_y

def tracer_courbes(history, title="MLP", file_path="./partie_2/models/Courbes_MLP_64x64"):
    """
    b) Courbes loss / accuracy train + val
    """
    h = history.history

    plt.figure()
    plt.plot(h["loss"])
    plt.plot(h["val_loss"])
    plt.title(f"{title} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "val"])
    plt.savefig(f"{file_path}_loss.png")
    plt.show()

    plt.figure()
    plt.plot(h["accuracy"])
    plt.plot(h["val_accuracy"])
    plt.title(f"{title} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train", "val"])
    plt.savefig(f"{file_path}_accuracy.png")
    plt.show()

def afficher_matrice_confusion(y_true,y_pred,title="Matrice de confusion",file_path="./partie_2/models/Matrice_de_confusion_MLP_64x64.png"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))

    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    class_names = [decode_class_y(cid) for cid in labels]

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_title(title)

    fig.savefig(file_path)
    plt.show()

    return cm


def analyser_conflits_classes(cm, top_k=3):
    cm_off_diag = cm.copy()
    np.fill_diagonal(cm_off_diag, 0)

    confusions = []
    for i in range(cm_off_diag.shape[0]):
        for j in range(cm_off_diag.shape[1]):
            n = cm_off_diag[i, j]
            if n > 0:
                confusions.append((i, j, int(n)))

    confusions.sort(key=lambda x: x[2], reverse=True)
    confusions = confusions[:top_k]

    print("\nClasses les plus confondues :")
    for i, j, n in confusions:
        print(f"{i} ({decode_class_y(i)})  ==>  {j} ({decode_class_y(j)}) : {n} erreurs")

    return confusions