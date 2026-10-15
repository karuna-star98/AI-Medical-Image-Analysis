"""Generate SYNTHETIC chest-X-ray-like images so the whole pipeline can be tested
without downloading anything. These are cartoons, NOT real medical images.

NORMAL     : two dark lung fields, bright spine, heart shadow, faint ribs.
PNEUMONIA  : same, plus bright cloudy 'opacities' inside the lungs.
"""
from pathlib import Path

import cv2
import numpy as np

from src.config import SIM_DIR, SEED

SIZE = 256


def _blob(h, w, cx, cy, sx, sy):
    y, x = np.mgrid[0:h, 0:w]
    return np.exp(-(((x - cx) ** 2) / (2 * sx ** 2) + ((y - cy) ** 2) / (2 * sy ** 2)))


def make_image(pneumonia: bool, rng: np.random.Generator) -> np.ndarray:
    h = w = SIZE
    img = np.full((h, w), rng.uniform(25, 45), np.float32)

    # Torso
    torso = np.zeros((h, w), np.float32)
    cv2.ellipse(torso, (w // 2, h // 2 + 10), (int(w * 0.40), int(h * 0.46)), 0, 0, 360, 1.0, -1)
    img += torso * rng.uniform(70, 95)

    # Lungs (dark) - slight random shape / position jitter
    lungs = np.zeros((h, w), np.float32)
    jitter = rng.integers(-6, 7, size=4)
    cv2.ellipse(lungs, (w // 2 - 52 + jitter[0], h // 2 + jitter[1]),
                (38 + jitter[2], 78), 0, 0, 360, 1.0, -1)
    cv2.ellipse(lungs, (w // 2 + 52 - jitter[0], h // 2 + jitter[1]),
                (38 - jitter[2], 78), 0, 0, 360, 1.0, -1)
    img -= lungs * rng.uniform(45, 65)

    # Spine + heart
    img += _blob(h, w, w // 2, h // 2, 8, 110) * rng.uniform(25, 40)
    img += _blob(h, w, w // 2 - 20, h // 2 + 35, 28, 30) * rng.uniform(35, 55)

    # Ribs
    for k in range(7):
        y0 = 55 + k * 20 + rng.integers(-2, 3)
        rib = np.zeros((h, w), np.float32)
        cv2.ellipse(rib, (w // 2, y0 + 30), (95, 22), 0, 200, 340, 1.0, 2)
        img += cv2.GaussianBlur(rib, (0, 0), 2) * 14

    # Disease: cloudy opacities inside the lung fields
    if pneumonia:
        for _ in range(rng.integers(1, 4)):
            side = rng.choice([-1, 1])
            cx = w // 2 + side * rng.integers(35, 65)
            cy = h // 2 + rng.integers(-55, 55)
            img += _blob(h, w, cx, cy, rng.uniform(12, 26), rng.uniform(14, 30)) * rng.uniform(30, 75)

    img = cv2.GaussianBlur(img, (0, 0), 1.5)
    img += rng.normal(0, rng.uniform(3, 7), (h, w))
    return np.clip(img, 0, 255).astype(np.uint8)


def generate_dataset(out_dir=SIM_DIR, n_train=400, n_val=80, n_test=120,
                     pneumonia_fraction=0.6, seed=SEED):
    """Write PNGs to out_dir/{train,val,test}/{NORMAL,PNEUMONIA}/. Returns counts."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    counts = {}
    for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        n_pn = int(n * pneumonia_fraction)
        for label, count in (("NORMAL", n - n_pn), ("PNEUMONIA", n_pn)):
            folder = out_dir / split / label
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                img = make_image(label == "PNEUMONIA", rng)
                cv2.imwrite(str(folder / f"sim_{label.lower()}_{i:04d}.png"), img)
            counts[f"{split}/{label}"] = count
    return counts
