"""Evaluate a saved model on the held-out TEST set and save all result artifacts."""
import json

import numpy as np

from src.config import OUTPUTS_DIR, THRESHOLD, ensure_dirs
from src.data_loader import load_datasets
from src.metrics import compute_metrics, text_report
from src.predict import load_trained_model
from src.visualize import plot_confusion_matrix, plot_prediction_grid, plot_roc


def evaluate(model_path, data_dir, tag=None, threshold=THRESHOLD):
    ensure_dirs()
    tag = tag or model_path.stem.replace("_best", "")
    model = load_trained_model(model_path)
    _, _, test_ds, class_names = load_datasets(data_dir)

    images, y_true, y_prob = [], [], []
    for x, y in test_ds:                               # test_ds is not shuffled
        y_prob.extend(model.predict_on_batch(x).ravel().tolist())
        y_true.extend(y.numpy().ravel().tolist())
        images.append(x.numpy().astype("uint8"))
    images = np.concatenate(images)
    y_true, y_prob = np.array(y_true), np.array(y_prob)

    metrics = compute_metrics(y_true, y_prob, threshold)
    report = text_report(y_true, y_prob, class_names, threshold)
    print("\n" + report)
    for k, v in metrics.items():
        print(f"{k:>20}: {v:.4f}" if isinstance(v, float) else f"{k:>20}: {v}")

    (OUTPUTS_DIR / f"{tag}_metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUTPUTS_DIR / f"{tag}_classification_report.txt").write_text(report)
    plot_confusion_matrix(y_true, y_prob, class_names, OUTPUTS_DIR / f"{tag}_confusion_matrix.png", threshold)
    plot_roc(y_true, y_prob, OUTPUTS_DIR / f"{tag}_roc_curve.png")
    plot_prediction_grid(images, y_true, y_prob, class_names,
                         OUTPUTS_DIR / f"{tag}_predictions.png", threshold=threshold)
    plot_prediction_grid(images, y_true, y_prob, class_names,
                         OUTPUTS_DIR / f"{tag}_errors.png", only_errors=True, threshold=threshold)
    print(f"\nSaved results to {OUTPUTS_DIR}")
    return metrics
