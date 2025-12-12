import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import ShapesDataset
from model import EncoderDecoder

# ==========================
#     MÉTRIQUES
# ==========================

def iou_score(pred, target, threshold=0.5, eps=1e-6):
    pred_bin = (pred > threshold).float()

    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection

    return (intersection + eps) / (union + eps)


def dice_score(pred, target, threshold=0.5, eps=1e-6):
    pred_bin = (pred > threshold).float()

    intersection = (pred_bin * target).sum()
    total = pred_bin.sum() + target.sum()

    return (2 * intersection + eps) / (total + eps)


# ==========================
#     VISUALISATION
# ==========================

def show_predictions(model, loader, device="cpu", num_samples=5, threshold=0.5):

    model.eval()

    # Un seul batch pour afficher quelques images
    imgs, masks = next(iter(loader))

    imgs = imgs.to(device)
    masks = masks.to(device)

    with torch.no_grad():
        preds = model(imgs)
        preds_bin = (preds > threshold).float()

    # CPU arrays
    imgs_np = imgs.cpu().numpy()
    masks_np = masks.cpu().numpy()
    preds_np = preds.cpu().numpy()
    preds_bin_np = preds_bin.cpu().numpy()

    for i in range(num_samples):

        # ====== MÉTRIQUES ======
        iou = iou_score(preds[i], masks[i]).item()
        dice = dice_score(preds[i], masks[i]).item()

        # ====== AFFICHAGE ======
        fig, ax = plt.subplots(1, 4, figsize=(16, 4))

        ax[0].imshow(imgs_np[i][0], cmap="gray")
        ax[0].set_title("Entrée (multi-formes)")
        ax[0].axis("off")

        ax[1].imshow(masks_np[i][0], cmap="gray")
        ax[1].set_title("Contour réel")
        ax[1].axis("off")

        ax[2].imshow(preds_np[i][0], cmap="viridis")
        ax[2].set_title(f"Prédiction (proba)\nIoU={iou:.3f}   Dice={dice:.3f}")
        ax[2].axis("off")

        ax[3].imshow(preds_bin_np[i][0], cmap="gray")
        ax[3].set_title("Prédiction binaire")
        ax[3].axis("off")

        plt.tight_layout()
        plt.show()


# ==========================
#     CHARGEMENT DATASET TEST
# ==========================

test_set = ShapesDataset(
    img_dir="shapes_multi/test/images",
    mask_dir="shapes_multi/test/masks"
)

test_loader = DataLoader(test_set, batch_size=8, shuffle=False)


# ==========================
#     CHARGEMENT MODÈLE
# ==========================

device = "cuda" if torch.cuda.is_available() else "cpu"

model = EncoderDecoder().to(device)
model.load_state_dict(torch.load("best_shape_multi_model.pt", map_location=device))


# ==========================
#     LANCER VISUALISATION
# ==========================

show_predictions(model, test_loader, device=device, num_samples=5)
