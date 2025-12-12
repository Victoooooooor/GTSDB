import numpy as np

def evaluer_performances(model, train_ds, test_ds):
    print("\n--- Performances TRAIN ---")
    train_loss, train_acc = model.evaluate(train_ds, verbose=0)
    print(f"Loss : {train_loss:.4f} | Accuracy : {train_acc:.4f}")

    print("\n--- Performances TEST ---")
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Loss : {test_loss:.4f} | Accuracy : {test_acc:.4f}")

    return (train_loss, train_acc), (test_loss, test_acc)

def get_predictions(model, dataset):
    """
    Récupère les labels réels et prédits
    """
    y_true = []
    y_pred = []

    for x, y in dataset:
        preds = model.predict(x, verbose=0)
        preds = np.argmax(preds, axis=1)

        y_true.extend(y.numpy())
        y_pred.extend(preds)

    return np.array(y_true), np.array(y_pred)