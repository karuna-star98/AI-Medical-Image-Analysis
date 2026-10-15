"""Model definitions. Both models take RAW grayscale pixels (0-255), shape (H, W, 1),
and output P(PNEUMONIA) in [0, 1]. Rescaling + augmentation live INSIDE the model.

  build_cnn()        - small custom CNN, trains from scratch (Option: easiest)
  build_mobilenet()  - MobileNetV2 transfer learning (Option: best accuracy, still light)
"""
import tensorflow as tf
from tensorflow.keras import layers, models

from src.config import IMG_SIZE


def build_augmentation():
    """Random changes applied ONLY while training (Keras disables them at inference).
    Horizontal flips are left out on purpose: in a chest X-ray the heart sits on one
    side, so a mirrored image is anatomically unrealistic."""
    return models.Sequential([
        layers.RandomRotation(0.03),            # about +/- 10 degrees
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomContrast(0.1),
    ], name="augmentation")


def build_cnn(input_shape=(*IMG_SIZE, 1)):
    inputs = layers.Input(shape=input_shape)
    x = build_augmentation()(inputs)
    x = layers.Rescaling(1.0 / 255)(x)

    for filters in (32, 64, 128):                       # feature extraction blocks
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu", name="last_conv")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inputs, outputs, name="medical_cnn")


def build_mobilenet(input_shape=(*IMG_SIZE, 1), freeze_base=True):
    inputs = layers.Input(shape=input_shape)
    x = build_augmentation()(inputs)
    x = layers.Concatenate()([x, x, x])                 # grayscale -> 3 channels
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)     # MobileNetV2 expects [-1, 1]

    base = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet",
        input_shape=(input_shape[0], input_shape[1], 3))
    base.trainable = not freeze_base

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inputs, outputs, name="medical_mobilenetv2")


def build_model(model_type="cnn"):
    if model_type == "cnn":
        return build_cnn()
    if model_type == "mobilenet":
        return build_mobilenet()
    raise ValueError("model_type must be 'cnn' or 'mobilenet'")


def compile_model(model, learning_rate=1e-3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")])
    return model


def unfreeze_top_layers(model, n_layers=30, learning_rate=1e-5):
    """Fine-tuning step for the MobileNet model: unfreeze the last n layers of the
    base network and recompile with a much smaller learning rate."""
    base = next(l for l in model.layers if isinstance(l, tf.keras.Model))
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    # keep BatchNorm layers frozen - safer on small datasets
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    return compile_model(model, learning_rate)
