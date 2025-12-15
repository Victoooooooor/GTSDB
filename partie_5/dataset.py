import torch
from torch.utils.data import Dataset
import cv2
import os

class ShapesDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.files = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.files[idx])
        mask_path = os.path.join(self.mask_dir, self.files[idx])

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(float) / 255.0
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(float) / 255.0

        # (H, W) → (1, H, W)
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        # S'assure que mask contient 0 ou 1
        mask = (mask > 0.5).float()

        return img, mask
