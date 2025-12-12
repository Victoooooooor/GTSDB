import numpy as np
import cv2
import os
from tqdm import tqdm
import shutil
from sklearn.model_selection import train_test_split

# ==========================================================
#              PARAMÈTRES DU DATASET
# ==========================================================

IMG_SIZE = 128
N_SAMPLES = 3000
OUTPUT_DIR = "shapes_multi"

os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "masks"),  exist_ok=True)


# ==========================================================
#              FONCTIONS DE FORMES
# ==========================================================

def random_square(img):
    size = np.random.randint(20, 60)
    cx = np.random.randint(30, IMG_SIZE - 30)
    cy = np.random.randint(30, IMG_SIZE - 30)
    half = size // 2

    square = np.array([
        [cx - half, cy - half],
        [cx + half, cy - half],
        [cx + half, cy + half],
        [cx - half, cy + half],
    ], dtype=np.float32)

    angle = np.random.randint(0, 360)
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    square = cv2.transform(np.array([square]), M)[0]

    square[:, 0] = np.clip(square[:, 0], 0, IMG_SIZE - 1)
    square[:, 1] = np.clip(square[:, 1], 0, IMG_SIZE - 1)

    cv2.fillPoly(img, [square.astype(np.int32)], 1)


def random_circle(img):
    radius = np.random.randint(10, 40)
    cx = np.random.randint(radius, IMG_SIZE - radius)
    cy = np.random.randint(radius, IMG_SIZE - radius)
    cv2.circle(img, (cx, cy), radius, 1, -1)


def random_triangle(img):
    size = np.random.randint(20, 60)
    cx = np.random.randint(30, IMG_SIZE - 30)
    cy = np.random.randint(30, IMG_SIZE - 30)

    p1 = (cx, cy - size//2)
    p2 = (cx - size//2, cy + size//2)
    p3 = (cx + size//2, cy + size//2)

    triangle = np.array([p1, p2, p3], dtype=np.float32)

    angle = np.random.randint(0, 360)
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    triangle = cv2.transform(np.array([triangle]), M)[0]

    triangle[:, 0] = np.clip(triangle[:, 0], 0, IMG_SIZE-1)
    triangle[:, 1] = np.clip(triangle[:, 1], 0, IMG_SIZE-1)

    cv2.fillPoly(img, [triangle.astype(np.int32)], 1)


def generate_contour(mask):
    kernel = np.ones((3,3), np.uint8)
    eroded = cv2.erode(mask, kernel)
    contour = mask - eroded
    contour[contour < 0] = 0
    return contour


def generate_multi_shape_image(min_shapes=2, max_shapes=5):
    """ Génère une image contenant entre 2 et 5 formes """
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    shapes = [random_square, random_circle, random_triangle]

    nb = np.random.randint(min_shapes, max_shapes + 1)

    for _ in range(nb):
        shape_fn = np.random.choice(shapes)
        shape_fn(img)

    contour = generate_contour(img)
    return img, contour


# ==========================================================
#              GÉNÉRATION DU DATASET
# ==========================================================

print("Génération des images multi-formes...")

for i in tqdm(range(N_SAMPLES)):
    img, contour = generate_multi_shape_image()

    cv2.imwrite(f"{OUTPUT_DIR}/images/{i:05d}.png", img * 255)
    cv2.imwrite(f"{OUTPUT_DIR}/masks/{i:05d}.png", contour * 255)

print("Images générées.")


# ==========================================================
#              SPLIT 70 / 15 / 15
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
