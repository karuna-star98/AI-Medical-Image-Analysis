"""All plots (matplotlib only - no TensorFlow). Every function saves a PNG."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

from src.preprocessing import list_images, load_grayscale


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_class_distribution(summary: dict, path):
    """Bar chart of image counts per split/class. `summary` from dataset_summary()."""
    splits = list(summary)
    classes = sorted({c for s in summary.values() for c in s})
    x = np.arange(len(splits)); width = 0.8 / len(classes)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, c in enumerate(classes):
        vals = [summary[s].get(c, 0) for s in splits]
        bars = ax.bar(x + i * width, vals, width, label=c)
        ax.bar_label(bars, fontsize=8)
    ax.set_xticks(x + width * (len(classes) - 1) / 2); ax.set_xticklabels(splits)
    ax.set_ylabel("Images"); ax.set_title("Dataset class distribution"); ax.legend()
    return _save(fig, path)


def show_samples(data_dir, class_names, path, n_per_class=4, split="train"):
    """Grid of example images per class."""
    fig, axes = plt.subplots(len(class_names), n_per_class,
                             figsize=(3 * n_per_class, 3.2 * len(class_names)), squeeze=False)
    for r, cls in enumerate(class_names):
        files = list_images(Path(data_dir) / split / cls)[:n_per_class]
        for c in range(n_per_class):
            ax = axes[r][c]; ax.axis("off")
            if c < len(files):
                ax.imshow(load_grayscale(files[c]), cmap="gray")
                ax.set_title(cls, fontsize=10)
    fig.suptitle("Sample images", fontsize=13)
    return _save(fig, path)


def plot_history(history: dict, path):
    """Accuracy + loss curves from a Keras history dict."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for key, label in (("accuracy", "train"), ("val_accuracy", "validation")):
        if key in history:
            ax[0].plot(history[key], label=label)
    ax[0].set_title("Accuracy"); ax[0].set_xlabel("Epoch"); ax[0].legend()
    for key, label in (("loss", "train"), ("val_loss", "validation")):
        if key in history:
            ax[1].plot(history[key], label=label)
    ax[1].set_title("Loss"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    return _save(fig, path)


def plot_confusion_matrix(y_true, y_prob, class_names, path, threshold=0.5):
    y_pred = (np.asarray(y_prob).ravel() >= threshold).astype(int)
    cm = confusion_matrix(np.asarray(y_true).astype(int).ravel(), y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(class_names)
    ax.set_yticks([0, 1]); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, path)


def plot_roc(y_true, y_prob, path):
    y_true = np.asarray(y_true).astype(int).ravel()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("ROC curve"); ax.legend(loc="lower right")
    return _save(fig, path)


def plot_prediction_grid(images, y_true, y_prob, class_names, path,
                         n=12, only_errors=False, threshold=0.5, seed=0):
    """Predicted-vs-actual gallery. Green title = correct, red = wrong."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob >= threshold).astype(int)
    idx = np.where(y_pred != y_true)[0] if only_errors else np.arange(len(y_true))
    if len(idx) == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.choice(idx, size=min(n, len(idx)), replace=False)

    cols = 4; rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(np.squeeze(images[i]), cmap="gray", vmin=0, vmax=255)
        ok = y_pred[i] == y_true[i]
        ax.set_title(f"Actual: {class_names[y_true[i]]}\nPred: {class_names[y_pred[i]]} "
                     f"({y_prob[i]:.2f})", fontsize=9, color="green" if ok else "red")
    fig.suptitle("Misclassified examples" if only_errors else "Predicted vs actual", fontsize=13)
    return _save(fig, path)
