"""Grad-CAM: highlights which image regions pushed the model toward 'PNEUMONIA'.
Explainability is what makes medical AI trustworthy to doctors."""
from pathlib import Path

import cv2
import numpy as np

from src.preprocessing import load_grayscale, to_model_input


def make_gradcam(model, img_batch, layer_name="last_conv"):
    """Return a heatmap in [0, 1] with the spatial size of the conv layer."""
    import tensorflow as tf
    try:
        conv_layer = model.get_layer(layer_name)
    except ValueError:
        raise ValueError(f"Layer '{layer_name}' not found. Grad-CAM is wired for the "
                         "custom CNN (model_type='cnn').")
    grad_model = tf.keras.models.Model(model.inputs, [conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_batch, training=False)
        score = preds[:, 0]
    grads = tape.gradient(score, conv_out)
    pooled = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(conv_out * pooled[:, None, None, :], axis=-1)[0]
    cam = tf.nn.relu(cam).numpy()
    return cam / cam.max() if cam.max() > 0 else cam


def overlay_heatmap(gray_uint8, cam, alpha=0.4):
    """Blend the heatmap over the grayscale image -> RGB uint8."""
    heat = cv2.resize(cam, (gray_uint8.shape[1], gray_uint8.shape[0]))
    heat = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)
    base = cv2.cvtColor(gray_uint8, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(base, 1 - alpha, heat, alpha, 0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


def save_gradcam_figure(model, image_path, out_path, prediction=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gray = load_grayscale(image_path)
    cam = make_gradcam(model, to_model_input(gray))
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
    ax[0].imshow(gray, cmap="gray"); ax[0].set_title("Input"); ax[0].axis("off")
    ax[1].imshow(overlay_heatmap(gray, cam))
    title = "Grad-CAM"
    if prediction:
        title += f" - {prediction['label']} ({prediction['probability_pneumonia']:.2f})"
    ax[1].set_title(title); ax[1].axis("off")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return out_path
