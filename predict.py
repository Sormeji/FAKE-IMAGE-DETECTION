"""Predict whether image(s) are real or fake.

Usage:
    python predict.py path/to/image.jpg
    python predict.py path/to/folder --show
"""
import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

import config
from src.utils import load_class_indices

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def predict_image(model, idx_to_class, path):
    img = load_img(path, target_size=config.IMG_SIZE)
    x = np.expand_dims(img_to_array(img) / 255.0, axis=0)
    prob = float(model.predict(x, verbose=0)[0][0])  # probability of class index 1
    idx = int(prob > 0.5)
    label = idx_to_class[idx]
    confidence = prob if idx == 1 else 1 - prob
    return label, confidence


def show_image(path, title):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path", help="Image file or folder of images")
    p.add_argument("--model", default=config.MODEL_PATH)
    p.add_argument("--show", action="store_true", help="Display each image with its prediction")
    args = p.parse_args()

    model = load_model(args.model)
    class_indices = load_class_indices(config.CLASS_INDICES_PATH)
    idx_to_class = {v: k for k, v in class_indices.items()}

    if os.path.isdir(args.path):
        files = sorted(os.path.join(args.path, f) for f in os.listdir(args.path) if f.lower().endswith(EXTS))
    else:
        files = [args.path]

    for f in files:
        label, conf = predict_image(model, idx_to_class, f)
        print(f"{os.path.basename(f)}: {label} ({conf:.1%})")
        if args.show:
            show_image(f, f"Predicted: {label} ({conf:.1%})")


if __name__ == "__main__":
    main()
