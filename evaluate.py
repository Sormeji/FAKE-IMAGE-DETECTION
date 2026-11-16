"""Evaluate a trained model on the validation set.

Usage:
    python evaluate.py --data_dir "/path/to/DATASET"
"""
import argparse

from tensorflow.keras.models import load_model

import config
from src.data import get_generators
from src.utils import evaluate_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=config.DATA_DIR)
    p.add_argument("--model", default=config.MODEL_PATH)
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    _, val_gen = get_generators(args.data_dir, config.IMG_SIZE, config.BATCH_SIZE)
    model = load_model(args.model)
    evaluate_model(model, val_gen, config.OUTPUT_DIR, show=args.show)


if __name__ == "__main__":
    main()
