import tensorflow as tf
import os,json

def save_history(history, path="./partie_2/models/history_mlp_64x64.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(history.history, f, indent=4)
    print(f"Historique sauvegardé dans : {path}")

def build_callbacks(path_best_model="./partie_2/models/best_model.keras", patience=5):
    cb_best = tf.keras.callbacks.ModelCheckpoint(filepath=path_best_model,monitor="val_loss",save_best_only=True,save_weights_only=False,verbose=1)

    cb_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=patience,restore_best_weights=True,verbose=1)

    return [cb_best, cb_early]

def entrainer_mlp(model, train_ds, val_ds, epochs=50,best_model_path="./partie_2/models/best_mlp.keras"):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    callbacks=build_callbacks(
        path_best_model=best_model_path,
        patience=6
    )

    history = model.fit(train_ds,validation_data=val_ds,epochs=epochs, callbacks=callbacks)
    return model,history
