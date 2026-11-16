"""Download CIFAKE (optional) and arrange it into the TRAIN / VALIDATION layout
that train.py expects.

Usage:
    # Option A: auto-download via kagglehub (pip install kagglehub)
    python prepare_dataset.py --download --output data

    # Option B: you downloaded + unzipped CIFAKE manually from Kaggle
    python prepare_dataset.py --source "/path/to/cifake" --output data

    # Quick test run with a small subset (e.g. 2000 train / 400 val images per class)
    python prepare_dataset.py --download --output data --max_train 2000 --max_val 400

Result:
    <output>/TRAIN/{FAKE,REAL}
    <output>/VALIDATION/{FAKE,REAL}
"""
import argparse
import os
import random
import shutil

KAGGLE_HANDLE = "birdy654/cifake-real-and-ai-generated-synthetic-images"
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def find_dir(root, name):
    """Find a sub-folder called `name` (case-insensitive) anywhere under root."""
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            if d.lower() == name.lower():
                return os.path.join(dirpath, d)
    return None


def find_class_dir(split_dir, cls):
    for d in os.listdir(split_dir):
        if d.lower() == cls.lower() and os.path.isdir(os.path.join(split_dir, d)):
            return os.path.join(split_dir, d)
    raise FileNotFoundError(f"No '{cls}' folder inside {split_dir}")


def copy_split(src_split, dst_split, max_per_class, seed):
    rng = random.Random(seed)
    for cls in ("FAKE", "REAL"):
        src = find_class_dir(src_split, cls)
        files = [f for f in os.listdir(src) if f.lower().endswith(EXTS)]
        rng.shuffle(files)
        if max_per_class:
            files = files[:max_per_class]
        dst = os.path.join(dst_split, cls)
        os.makedirs(dst, exist_ok=True)
        for f in files:
            shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
        print(f"  {os.path.basename(dst_split)}/{cls}: {len(files)} images")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", help="Folder containing the unzipped CIFAKE dataset (train/ and test/)")
    p.add_argument("--download", action="store_true", help="Download CIFAKE using kagglehub")
    p.add_argument("--output", default="data", help="Where to create TRAIN/ and VALIDATION/")
    p.add_argument("--max_train", type=int, default=None, help="Max train images per class")
    p.add_argument("--max_val", type=int, default=None, help="Max validation images per class")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    source = args.source
    if args.download:
        import kagglehub  # pip install kagglehub
        source = kagglehub.dataset_download(KAGGLE_HANDLE)
        print("Downloaded to:", source)
    if not source:
        p.error("Provide --source <folder> or use --download")

    train_src = find_dir(source, "train")
    test_src = find_dir(source, "test")
    if not train_src or not test_src:
        raise SystemExit(f"Could not find train/ and test/ folders under {source}")

    print("Creating dataset...")
    copy_split(train_src, os.path.join(args.output, "TRAIN"), args.max_train, args.seed)
    copy_split(test_src, os.path.join(args.output, "VALIDATION"), args.max_val, args.seed)
    print(f"\nDone. Now run:\n  python train.py --data_dir \"{args.output}\"")


if __name__ == "__main__":
    main()
