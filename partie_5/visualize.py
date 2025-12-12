import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import ShapesDataset
from model import EncoderDecoder
from metrics import iou_score, dice_score  


# =======================================
#           Fonction d’affichage
# ======================================= 

def show_predictions(model, loader, device="cpu", num_samples=5, threshold=0.5):
    model.eval()

    imgs, masks = next(iter(loader))
    imgs = imgs.to(device)
    masks = masks.to(device)

    with torch.no_grad():
        preds = model(imgs)
        preds_bin = (preds > threshold).float()

    imgs = imgs.cpu().numpy()
    masks = masks.cpu().numpy()
    preds = preds.cpu().numpy()
    preds_bin = preds_bin.cpu().numpy()

    for i in range(num_samples):
        # IoU et Dice pour l'image i
        iou = iou_score(torch.tensor(preds[i]), torch.tensor(masks[i])).item()
        dice = dice_score(torch.tensor(preds[i]), torch.tensor(masks[i])).item()

        fig, ax = plt.subplots(1, 4, figsize=(15, 4))

        ax[0].imshow(imgs[i][0], cmap="gray")
        ax[0].set_title("Entrée")
        ax[0].axis("off")

        ax[1].imshow(masks[i][0], cmap="gray")
        ax[1].set_title("Contour réel")
        ax[1].axis("off")

        ax[2].imshow(preds[i][0], cmap="viridis")
        ax[2].set_title(f"Prédiction (proba)\nIoU={iou:.3f}  Dice={dice:.3f}")
        ax[2].axis("off")

        ax[3].imshow(preds_bin[i][0], cmap="gray")
        ax[3].set_title("Prédiction binaire")
        ax[3].axis("off")

        plt.tight_layout()
        plt.show()



# =======================================
#       Charger dataset test (ou val)
# =======================================

test_set = ShapesDataset(
    img_dir="shapes_dataset/test/images",
    mask_dir="shapes_dataset/test/masks"
)

test_loader = DataLoader(test_set, batch_size=8, shuffle=False)


# =======================================
#    Charger modèle entraîné
# =======================================

device = "cuda" if torch.cuda.is_available() else "cpu"
model = EncoderDecoder().to(device)
model.load_state_dict(torch.load("best_shape_model.pt", map_location=device))


# =======================================
#        Affichage des prédictions
# =======================================

show_predictions(model, test_loader, device=device, num_samples=5)
