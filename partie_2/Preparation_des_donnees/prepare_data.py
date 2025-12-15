import tensorflow as tf

def load_image_tenserflow_et_preparation(path, label, augment=False):
        #lecture
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)

        #normalisation
        img = tf.cast(img, tf.float32) / 255.0

        #augmentation 
        # b) Data augmentation modérée : petites rotations, translations,légères variations de luminosité
        if augment:
            img = tf.image.random_brightness(img, max_delta=0.15)
            img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
            # Flip horizontalement (plus sûr dans un tf.function)
            img = tf.image.random_flip_left_right(img)

        # On aplati vue que c'est un MLP
        x = tf.reshape(img, [-1])
        y = tf.cast(label, tf.int32)
        return x, y

def contruire_dataset(couples,batch_size=64, shuffle=True, augment=False):
    paths  = [p for p, _ in couples]
    labels = [c for _, c in couples]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)

    ds = ds.map(lambda p, l: load_image_tenserflow_et_preparation(p, l,augment=augment),num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def decode_class_y(y):
    """
    Convertit un id de classe en nom lisible (string)
    """
    class_names = {
        0: "speed limit 20",
        1: "speed limit 30",
        2: "speed limit 50",
        3: "speed limit 60",
        4: "speed limit 70",
        5: "speed limit 80",
        6: "restriction ends 80",
        7: "speed limit 100",
        8: "speed limit 120",
        9: "no overtaking",
        10: "no overtaking (trucks)",
        11: "priority at next intersection",
        12: "priority road",
        13: "give way",
        14: "stop",
        15: "no traffic both ways",
        16: "no trucks",
        17: "no entry",
        18: "danger",
        19: "bend left",
        20: "bend right",
        21: "bend",
        22: "uneven road",
        23: "slippery road",
        24: "road narrows",
        25: "construction",
        26: "traffic signal",
        27: "pedestrian crossing",
        28: "school crossing",
        29: "cycles crossing",
        30: "snow",
        31: "animals",
        32: "restriction ends",
        33: "go right",
        34: "go left",
        35: "go straight",
        36: "go right or straight",
        37: "go left or straight",
        38: "keep right",
        39: "keep left",
        40: "roundabout",
        41: "restriction ends (overtaking)",
        42: "restriction ends (overtaking trucks)",
    }

    return class_names.get(int(y), f"unknown class {y}")