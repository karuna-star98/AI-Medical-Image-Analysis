"""Central configuration. Change values here instead of hunting through code."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REAL_DIR = DATA_DIR / "chest_xray"
SIM_DIR = DATA_DIR / "simulated"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

IMG_SIZE = (224, 224)      # (height, width) - 224 suits MobileNetV2
BATCH_SIZE = 32
SEED = 42
VAL_SPLIT = 0.2            # validation carved out of train (official val set has only 16 images)

# Folder names sorted alphabetically => label 0 = NORMAL, label 1 = PNEUMONIA
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
THRESHOLD = 0.5            # probability above this => PNEUMONIA

# Triage simulation priority bands (probability of pneumonia)
URGENT_AT = 0.85
REVIEW_AT = 0.50

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

DISCLAIMER = ("Educational project only - NOT a medical device. "
              "Not for clinical use or diagnosis.")


def ensure_dirs():
    for d in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
