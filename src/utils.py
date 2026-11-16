"""Plotting, metrics and helper functions."""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def save_class_indices(class_indices, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(class_indices, f, indent=2)


def load_class_indices(path):
    with open(path) as f:
        return json.load(f)


def plot_history(history, output_dir, show=False):
    os.makedirs(output_dir, exist_ok=True)
    h = history.history
    epochs = range(len(h["accuracy"]))

    plt.figure()
    plt.plot(epochs, h["accuracy"], "r", label="Training accuracy")
    plt.plot(epochs, h["val_accuracy"], "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "accuracy.png"), dpi=150, bbox_inches="tight")

    plt.figure()
    plt.plot(epochs, h["loss"], "r", label="Training loss")
    plt.plot(epochs, h["val_loss"], "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss.png"), dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close("all")


def evaluate_model(model, val_gen, output_dir, show=False):
    """Compute confusion matrix + classification report on the validation set."""
    os.makedirs(output_dir, exist_ok=True)
    val_gen.reset()
    steps = int(np.ceil(val_gen.samples / val_gen.batch_size))
    probs = model.predict(val_gen, steps=steps, verbose=1).flatten()
    y_pred = (probs > 0.5).astype(int)
    y_true = val_gen.classes

    labels = [k for k, _ in sorted(val_gen.class_indices.items(), key=lambda kv: kv[1])]
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close("all")

    report = classification_report(y_true, y_pred, target_names=labels)
    print(report)
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    wrong = np.where(y_true != y_pred)[0]
    print(f"Misclassified: {len(wrong)} / {len(y_true)}")
    with open(os.path.join(output_dir, "misclassified.txt"), "w") as f:
        for i in wrong:
            f.write(f"{val_gen.filepaths[i]}\ttrue={labels[y_true[i]]}\tpred={labels[y_pred[i]]}\tprob={probs[i]:.3f}\n")
    return cm, report
