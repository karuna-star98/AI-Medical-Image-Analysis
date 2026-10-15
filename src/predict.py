"""Inference helpers shared by the CLI, triage simulation and Flask API."""
import numpy as np

from src.config import CLASS_NAMES, THRESHOLD
from src.preprocessing import load_grayscale, resize_gray, to_model_input

_MODEL_CACHE = {}


def load_trained_model(model_path):
    import tensorflow as tf
    key = str(model_path)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = tf.keras.models.load_model(model_path)
    return _MODEL_CACHE[key]


def _describe(prob, threshold=THRESHOLD):
    pred = int(prob >= threshold)
    return {
        "label": CLASS_NAMES[pred],
        "probability_pneumonia": round(float(prob), 4),
        "confidence": round(float(prob if pred else 1 - prob), 4),
    }


def predict_array(model, gray_uint8, threshold=THRESHOLD):
    """Predict from an in-memory grayscale image (any size)."""
    x = to_model_input(resize_gray(gray_uint8))
    prob = float(model.predict(x, verbose=0)[0][0])
    return _describe(prob, threshold)


def predict_image(model, path, threshold=THRESHOLD):
    x = to_model_input(load_grayscale(path))
    prob = float(model.predict(x, verbose=0)[0][0])
    return _describe(prob, threshold)


def predict_paths(model, paths, batch_size=32):
    """Batch prediction -> array of P(PNEUMONIA), one per path."""
    probs = []
    for i in range(0, len(paths), batch_size):
        batch = np.concatenate([to_model_input(load_grayscale(p)) for p in paths[i:i + batch_size]])
        probs.extend(model.predict(batch, verbose=0).ravel().tolist())
    return np.array(probs)
