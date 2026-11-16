"""Data loading and augmentation."""
import os
from collections import Counter

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def count_images(directory):
    """Return number of images per class sub-folder."""
    counts = Counter()
    for sub in sorted(os.listdir(directory)):
        path = os.path.join(directory, sub)
        if os.path.isdir(path):
            counts[sub] = len(os.listdir(path))
    return counts


def get_generators(data_dir, img_size, batch_size):
    """Create train and validation generators.

    The validation generator uses shuffle=False so predictions line up
    with `generator.classes` when building the confusion matrix.
    """
    train_dir = os.path.join(data_dir, "TRAIN")
    val_dir = os.path.join(data_dir, "VALIDATION")
    for d in (train_dir, val_dir):
        if not os.path.isdir(d):
            raise FileNotFoundError(
                f"Folder not found: {d}\n"
                "Expected layout: <data_dir>/TRAIN/{FAKE,REAL} and <data_dir>/VALIDATION/{FAKE,REAL}"
            )

    print("Training distribution:  ", dict(count_images(train_dir)))
    print("Validation distribution:", dict(count_images(val_dir)))

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=img_size, batch_size=batch_size, class_mode="binary"
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir, target_size=img_size, batch_size=batch_size,
        class_mode="binary", shuffle=False,
    )
    return train_gen, val_gen
