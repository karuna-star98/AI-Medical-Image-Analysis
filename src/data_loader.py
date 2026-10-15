"""Dataset loading. TensorFlow is imported lazily so non-TF commands still work."""
from pathlib import Path

from src.config import BATCH_SIZE, IMAGE_EXTS, IMG_SIZE, SEED, VAL_SPLIT


def _count_images(folder: Path) -> int:
    return sum(1 for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def dataset_summary(data_dir) -> dict:
    """{'train': {'NORMAL': n, 'PNEUMONIA': n}, 'val': {...}, 'test': {...}}"""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {data_dir}\nSee data/README.md "
            "(or run `python main.py synthetic` to create test data).")
    summary = {}
    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        if split_dir.exists():
            summary[split] = {c.name: _count_images(c)
                              for c in sorted(split_dir.iterdir()) if c.is_dir()}
    return summary


def class_weights(data_dir) -> dict:
    """Balance NORMAL vs PNEUMONIA (the Kaggle set has ~3x more pneumonia)."""
    counts = dataset_summary(data_dir)["train"]
    names = sorted(counts)
    total = sum(counts.values())
    return {i: total / (len(names) * counts[n]) for i, n in enumerate(names)}


def load_datasets(data_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE,
                  val_split=VAL_SPLIT, seed=SEED):
    """Return (train_ds, val_ds, test_ds, class_names).

    Images load as 1-channel grayscale, 0-255 floats. Labels are 0/1 floats.
    The test set is NOT shuffled so predictions line up with labels.
    """
    import tensorflow as tf
    data_dir = Path(data_dir)
    summary = dataset_summary(data_dir)
    if "train" not in summary or "test" not in summary:
        raise FileNotFoundError(f"{data_dir} must contain train/ and test/ folders.")

    common = dict(labels="inferred", label_mode="binary", color_mode="grayscale",
                  image_size=img_size, batch_size=batch_size)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "train", validation_split=val_split, subset="training",
        seed=seed, shuffle=True, **common)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "train", validation_split=val_split, subset="validation",
        seed=seed, shuffle=False, **common)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "test", shuffle=False, **common)

    class_names = list(train_ds.class_names)
    auto = tf.data.AUTOTUNE
    return (train_ds.prefetch(auto), val_ds.prefetch(auto),
            test_ds.prefetch(auto), class_names)
