from torch.utils.data import DataLoader

from dataset import ShapesDataset
from dataset import ShapesDataset
from model import EncoderDecoder
import torch
import torch.nn as nn
from tqdm import tqdm

train_set = ShapesDataset(
    img_dir="shapes_multi/train/images",
    mask_dir="shapes_multi/train/masks"
)

val_set = ShapesDataset(
    img_dir="shapes_multi/val/images",
    mask_dir="shapes_multi/val/masks"
)

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=16, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = EncoderDecoder().to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=15):
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs = imgs.to(device)
            masks = masks.to(device)

            preds = model(imgs)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.detach().item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)

                preds = model(imgs)
                loss = criterion(preds, masks)
                val_loss += loss.detach().item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{epochs} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_shape_multi_model.pt")
            print(">>> Nouveau meilleur modèle sauvegardé !")

    print("Entraînement terminé.")

train_model(model, train_loader, val_loader, criterion, optimizer, epochs=15)
