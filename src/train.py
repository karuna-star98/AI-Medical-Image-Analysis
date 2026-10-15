"""Training pipeline: load data -> build model -> fit -> save best model + curves."""
import json

import tensorflow as tf

from src.config import MODELS_DIR, OUTPUTS_DIR, SEED, ensure_dirs
from src.data_loader import class_weights, load_datasets
from src.model import build_model, compile_model, unfreeze_top_layers
from src.visualize import plot_history


def _callbacks(model_path, log_path):
    return [
        tf.keras.callbacks.ModelCheckpoint(str(model_path), monitor="val_auc", mode="max",
                                           save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=5,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2,
                                             verbose=1),
        tf.keras.callbacks.CSVLogger(str(log_path)),
    ]


def train(data_dir, model_type="cnn", epochs=15, fine_tune_epochs=0,
          use_class_weights=True, tag=None):
    """Returns the path of the best saved model."""
    ensure_dirs()
    tf.keras.utils.set_random_seed(SEED)
    tag = tag or model_type
    model_path = MODELS_DIR / f"{tag}_best.keras"
    log_path = OUTPUTS_DIR / f"{tag}_training_log.csv"

    train_ds, val_ds, _, class_names = load_datasets(data_dir)
    print("Classes:", class_names)
    weights = class_weights(data_dir) if use_class_weights else None
    print("Class weights:", weights)

    model = compile_model(build_model(model_type))
    model.summary()

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        class_weight=weights, callbacks=_callbacks(model_path, log_path))
    merged = {k: list(v) for k, v in history.history.items()}

    if model_type == "mobilenet" and fine_tune_epochs > 0:
        print("\n--- Fine-tuning top MobileNet layers ---")
        unfreeze_top_layers(model)
        ft = model.fit(train_ds, validation_data=val_ds, epochs=fine_tune_epochs,
                       class_weight=weights, callbacks=_callbacks(model_path, log_path))
        for k, v in ft.history.items():
            merged.setdefault(k, []).extend(v)

    (OUTPUTS_DIR / f"{tag}_history.json").write_text(json.dumps(merged, indent=2))
    plot_history(merged, OUTPUTS_DIR / f"{tag}_training_curves.png")
    print(f"\nBest model saved to: {model_path}")
    return model_path
