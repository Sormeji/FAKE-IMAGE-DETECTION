"""Central configuration for the Fake Image Detection project."""
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Dataset layout expected:
#   DATA_DIR/TRAIN/<FAKE|REAL>/*.jpg
#   DATA_DIR/VALIDATION/<FAKE|REAL>/*.jpg
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

IMG_SIZE = (150, 150)
BATCH_SIZE = 20
EPOCHS = 50
LEARNING_RATE = 1e-4
DROPOUT = 0.5

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.keras")
CLASS_INDICES_PATH = os.path.join(MODEL_DIR, "class_indices.json")
