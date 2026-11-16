"""Train the fake-image detector.

Usage:
    python train.py --data_dir "/path/to/DATASET"
"""
import argparse
import os

import config
from src.data import get_generators
from src.model import build_model
from src.utils import evaluate_model, plot_history, save_class_indices

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def parse_args():
    p = argparse.ArgumentParser(description="Train fake image detector")
    p.add_argument("--data_dir", default=config.DATA_DIR, help="Folder containing TRAIN/ and VALIDATION/")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--show", action="store_true", help="Display plots interactively")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    train_gen, val_gen = get_generators(args.data_dir, config.IMG_SIZE, args.batch_size)
    save_class_indices(train_gen.class_indices, config.CLASS_INDICES_PATH)
    print("Class mapping:", train_gen.class_indices)

    model = build_model(config.IMG_SIZE, args.lr, config.DROPOUT)
    model.summary()

    callbacks = [
        ModelCheckpoint(config.MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    ]

    # steps_per_epoch/validation_steps are left unset so each epoch sees the full dataset
    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=callbacks,
    )

    plot_history(history, config.OUTPUT_DIR, show=args.show)
    evaluate_model(model, val_gen, config.OUTPUT_DIR, show=args.show)
    print(f"\nBest model saved to: {config.MODEL_PATH}")
    print(f"Plots and reports saved to: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
