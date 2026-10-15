"""Smoke tests that run WITHOUT TensorFlow (a tiny fake model stands in).
Run:  pytest -q
"""
import io
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import predict as predict_mod
from src.data_loader import class_weights, dataset_summary
from src.metrics import compute_metrics
from src.preprocessing import load_grayscale, to_model_input
from src.synthetic import generate_dataset


class FakeModel:
    """Scores an image by its mean brightness - enough to test the plumbing."""
    def predict(self, x, verbose=0):
        return np.clip(x.mean(axis=(1, 2, 3)) / 255.0 * 2, 0, 1)[:, None]


def test_synthetic_dataset_layout(tmp_path):
    generate_dataset(tmp_path, n_train=10, n_val=4, n_test=6)
    s = dataset_summary(tmp_path)
    assert set(s) == {"train", "val", "test"}
    assert s["train"]["PNEUMONIA"] == 6 and s["train"]["NORMAL"] == 4
    w = class_weights(tmp_path)
    assert w[0] > w[1]            # minority class gets the larger weight


def test_preprocessing_shape_and_range(tmp_path):
    generate_dataset(tmp_path, 4, 2, 2)
    f = next((tmp_path / "train" / "NORMAL").iterdir())
    img = load_grayscale(f)
    assert img.shape == (224, 224) and img.dtype == np.uint8
    x = to_model_input(img)
    assert x.shape == (1, 224, 224, 1) and x.max() <= 255


def test_metrics_perfect_and_worst():
    y = np.array([0, 0, 1, 1])
    assert compute_metrics(y, [0.1, 0.2, 0.8, 0.9])["accuracy"] == 1.0
    m = compute_metrics(y, [0.9, 0.8, 0.1, 0.2])
    assert m["accuracy"] == 0.0 and m["fn"] == 2 and m["fp"] == 2


def test_predict_output_format(tmp_path):
    generate_dataset(tmp_path, 4, 2, 2)
    f = next((tmp_path / "train" / "PNEUMONIA").iterdir())
    r = predict_mod.predict_image(FakeModel(), f)
    assert r["label"] in ("NORMAL", "PNEUMONIA") and 0 <= r["probability_pneumonia"] <= 1


def test_flask_api(tmp_path, monkeypatch):
    from src import app as app_mod
    monkeypatch.setattr(app_mod, "get_model", lambda: FakeModel())
    client = app_mod.app.test_client()
    assert client.get("/health").json == {"status": "ok"}
    assert client.post("/predict").status_code == 400            # no file
    img = np.full((300, 300), 120, np.uint8)
    ok, buf = cv2.imencode(".png", img)
    r = client.post("/predict", data={"image": (io.BytesIO(buf.tobytes()), "x.png")},
                    content_type="multipart/form-data")
    assert r.status_code == 200 and "label" in r.json and "disclaimer" in r.json
    bad = client.post("/predict", data={"image": (io.BytesIO(b"abc"), "x.txt")},
                      content_type="multipart/form-data")
    assert bad.status_code == 400


def test_triage_worklist(tmp_path, monkeypatch):
    import src.triage as triage
    generate_dataset(tmp_path / "d", 6, 2, 6)
    monkeypatch.setattr(triage, "load_trained_model", lambda p: FakeModel())
    monkeypatch.setattr(triage, "OUTPUTS_DIR", tmp_path / "out")
    csv_path, counts = triage.run_triage("fake.keras", tmp_path / "d" / "test")
    assert csv_path.exists() and sum(counts.values()) == 6
