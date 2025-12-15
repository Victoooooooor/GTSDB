import tensorflow as tf
import os

def save_model(model, model_path="./partie_2/models/MLP_64x64.keras"):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"Modèle sauvegardé dans : {model_path}")


def load_model(model_path="./models/MLP_64x64.keras"):
    if not os.path.exists(model_path):
        print("erreur: pas de modele")
        return 
    model = tf.keras.models.load_model(model_path)
    print(f"Modèle chargé depuis : {model_path}")
    return model

def build_model(num_classes,size_image=64*64*3):
    """
    size_image = taille du vecteur d'entrée après mais plat ex: 64*64*3
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(size_image,)),
        tf.keras.layers.Dense(size_image//2, activation="relu"),
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    return model
