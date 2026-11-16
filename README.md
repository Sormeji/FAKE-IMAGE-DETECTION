# Fake Image Detection (CNN)

A convolutional neural network (TensorFlow/Keras) that classifies images as **REAL** or **FAKE**.

## Project structure

```
fake-image-detection/
├── config.py          # image size, batch size, learning rate, paths
├── train.py           # train + evaluate + save best model
├── evaluate.py        # confusion matrix / classification report on validation set
├── prepare_dataset.py # download/arrange CIFAKE into TRAIN/VALIDATION
├── predict.py         # predict a single image or a whole folder
├── requirements.txt
├── src/
│   ├── data.py        # data generators & augmentation
│   ├── model.py       # CNN architecture
│   └── utils.py       # plots, metrics, helpers
├── models/            # saved model + class_indices.json (generated)
└── outputs/           # plots and reports (generated)
```

## Getting a dataset (CIFAKE)

This project works with [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
(60k real + 60k AI-generated images, 32x32). `prepare_dataset.py` arranges it into the layout below:

```bash
# auto-download (may ask for Kaggle login) ...
python prepare_dataset.py --download --output data
# ... or from a manually downloaded + unzipped copy
python prepare_dataset.py --source "/path/to/cifake" --output data
# small subset for a quick test
python prepare_dataset.py --download --output data --max_train 2000 --max_val 400

python train.py --data_dir data
```

Cite: Krizhevsky & Hinton (2009); Bird & Lotfi (2023), arXiv:2303.14126.

## Dataset layout

```
DATASET/
├── TRAIN/
│   ├── FAKE/
│   └── REAL/
└── VALIDATION/
    ├── FAKE/
    └── REAL/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
# Train (saves models/best_model.keras and plots in outputs/)
python train.py --data_dir "/path/to/DATASET" --epochs 50

# Re-evaluate a saved model
python evaluate.py --data_dir "/path/to/DATASET"

# Predict
python predict.py "/path/to/DATASET/TEST/20.jpg" --show
python predict.py "/path/to/DATASET/TEST"
```

## Improvements over the original notebook code

- Split into reusable modules instead of copy-pasted scripts
- Dataset path is a command-line argument (no hard-coded paths)
- Fixed the syntax error (missing `)` on `classification_report`)
- Validation generator uses `shuffle=False`, so the confusion matrix is correctly aligned
- Full dataset used each epoch (removed fixed `steps_per_epoch=100` / `validation_steps=50`, which can exceed dataset size)
- Added Dropout, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
- Class label mapping saved to `class_indices.json`, so `predict.py` never mixes up FAKE/REAL
- Prediction shows a confidence score
- Plots, reports and misclassified file list saved to `outputs/`
