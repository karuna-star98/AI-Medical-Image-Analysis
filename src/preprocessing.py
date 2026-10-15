"""Image preprocessing helpers (OpenCV + NumPy only - no TensorFlow needed).

IMPORTANT: models in this project expect RAW pixel values 0-255.
Rescaling happens *inside* the model, so training and inference stay consistent.
"""
from pathlib import Path

import cv2
import numpy as np

from src.config import IMG_SIZE, IMAGE_EXTS


def load_grayscale(path, size=IMG_SIZE):
    """Read an image as grayscale uint8 and resize to `size` (height, width)."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return resize_gray(img, size)


def resize_gray(img, size=IMG_SIZE):
    """Resize a grayscale array. OpenCV wants (width, height)."""
    return cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_AREA)


def to_model_input(img_uint8):
    """(H, W) uint8 -> (1, H, W, 1) float32, still on the 0-255 scale."""
    return img_uint8.astype("float32")[None, ..., None]


def apply_clahe(img_uint8, clip_limit=2.0, tile=8):
    """Contrast Limited Adaptive Histogram Equalisation - boosts local contrast.
    Used for the visual demo; not part of the training pipeline."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return clahe.apply(img_uint8)


def list_images(folder):
    """Recursively list image files in a folder (sorted)."""
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def preprocessing_demo(image_path, out_path):
    """Save a figure: original -> resized -> CLAHE -> normalised histogram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    original = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise FileNotFoundError(image_path)
    resized = resize_gray(original)
    enhanced = apply_clahe(resized)
    normalised = resized.astype("float32") / 255.0

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].imshow(original, cmap="gray"); ax[0].set_title(f"Original {original.shape[::-1]}")
    ax[1].imshow(resized, cmap="gray"); ax[1].set_title(f"Resized {resized.shape[::-1]}")
    ax[2].imshow(enhanced, cmap="gray"); ax[2].set_title("CLAHE (contrast demo)")
    ax[3].hist(normalised.ravel(), bins=50, color="steelblue")
    ax[3].set_title("Normalised pixels (0-1)")
    for a in ax[:3]:
        a.axis("off")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
