import shutil
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# ==========================================================
#              PARAMÈTRES DU DATASET
# ==========================================================

IMG_SIZE = 128
N_SAMPLES = 3000   # Total images à générer
OUTPUT_DIR = "shapes_dataset"  # Nom du dossier de sortie

os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "masks"),  exist_ok=True)


# ==========================================================
#              FONCTIONS UTILITAIRES
# ==========================================================

def random_square(img):
    """Carré avec rotation robuste et coord. clampées."""
    
    size = np.random.randint(20, 60)

    # Centre sûr (marge suffisante)
    cx = np.random.randint(30, IMG_SIZE - 30)
    cy = np.random.randint(30, IMG_SIZE - 30)

    half = size // 2

    # Carré centré
    square = np.array([
        [cx - half, cy - half],
        [cx + half, cy - half],
        [cx + half, cy + half],
        [cx - half, cy + half],
    ], dtype=np.float32)

    # Rotation
    angle = np.random.randint(0, 360)
    M = cv2.getRotationMatrix2D((int(cx), int(cy)), angle, 1.0)
    square = cv2.transform(np.array([square]), M)[0]

    # Clamp des coordonnées dans l'image
    square[:, 0] = np.clip(square[:, 0], 0, IMG_SIZE - 1)
    square[:, 1] = np.clip(square[:, 1], 0, IMG_SIZE - 1)

    cv2.fillPoly(img, [square.astype(np.int32)], 1)


def random_circle(img):
    """Dessine un disque avec position + taille aléatoires."""
    radius = np.random.randint(10, 40)
    x = np.random.randint(radius, IMG_SIZE - radius)
    y = np.random.randint(radius, IMG_SIZE - radius)
    cv2.circle(img, (x, y), radius, 1, -1)


def random_triangle(img):
    """Dessine un triangle aléatoire (position, rotation, taille), robuste aux dépassements."""
    
    size = np.random.randint(20, 60)

    # Position de base : on force un espace de sécurité
    cx = np.random.randint(30, IMG_SIZE - 30)
    cy = np.random.randint(30, IMG_SIZE - 30)

    # Triangle initial centré autour de (cx, cy)
    p1 = (cx, cy - size//2)
    p2 = (cx - size//2, cy + size//2)
    p3 = (cx + size//2, cy + size//2)

    triangle = np.array([p1, p2, p3], dtype=np.float32)

    # Rotation
    angle = np.random.randint(0, 360)
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    # Transformation correcte
    triangle = cv2.transform(np.array([triangle]), M)[0]

    # Sécurité : clamp dans l'image
    triangle[:, 0] = np.clip(triangle[:, 0], 0, IMG_SIZE-1)
    triangle[:, 1] = np.clip(triangle[:, 1], 0, IMG_SIZE-1)

    # Dessin
    cv2.fillPoly(img, [triangle.astype(np.int32)], 1)



def generate_contour(mask):
    """Contour = mask - erode(mask)"""
    kernel = np.ones((3,3), np.uint8)
    eroded = cv2.erode(mask, kernel)
    contour = mask - eroded
    contour[contour < 0] = 0
    return contour


# ==========================================================
#              GÉNÉRATION DU DATASET
# ==========================================================

shapes = [random_square, random_circle, random_triangle]

print("Génération des images...")
for i in tqdm(range(N_SAMPLES)):

    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    # Choisir une forme aléatoire
    shape_fn = np.random.choice(shapes)
    shape_fn(img) 


    # Générer le contour correspondant
    contour = generate_contour(img)

    # Sauvegarde
    cv2.imwrite(f"{OUTPUT_DIR}/images/{i:05d}.png", img*255)
    cv2.imwrite(f"{OUTPUT_DIR}/masks/{i:05d}.png", contour*255)

print("Dataset généré dans :", OUTPUT_DIR)

# ==========================================================
#              SPLIT TRAIN / VAL / TEST (70 / 15 / 15)
# ==========================================================

print("Création du split train/val/test...")

all_files = sorted(os.listdir(f"{OUTPUT_DIR}/images"))

train_files, temp_files = train_test_split(all_files, test_size=0.30, random_state=42)
val_files, test_files   = train_test_split(temp_files, test_size=0.50, random_state=42)

for split in ["train", "val", "test"]:
    os.makedirs(f"{OUTPUT_DIR}/{split}/images", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/{split}/masks", exist_ok=True)

def copy_split(files, split_name):
    for f in files:
        shutil.copy(f"{OUTPUT_DIR}/images/{f}", f"{OUTPUT_DIR}/{split_name}/images/{f}")
        shutil.copy(f"{OUTPUT_DIR}/masks/{f}", f"{OUTPUT_DIR}/{split_name}/masks/{f}")

copy_split(train_files, "train")
copy_split(val_files, "val")
copy_split(test_files, "test")

print("Split terminé !")
print(f"Train : {len(train_files)} images")
print(f"Val   : {len(val_files)} images")
print(f"Test  : {len(test_files)} images")
