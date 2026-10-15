"""Virtual hospital simulation: a folder of 'incoming studies' is scored by the model and
turned into a PRIORITISED RADIOLOGIST WORKLIST - the way real AI triage products
(e.g. Aidoc, Qure.ai) assist radiologists. The AI never diagnoses; it re-orders the queue."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import CLASS_NAMES, OUTPUTS_DIR, REVIEW_AT, URGENT_AT, ensure_dirs
from src.predict import load_trained_model, predict_paths
from src.preprocessing import list_images


def priority_for(prob):
    if prob >= URGENT_AT:
        return "URGENT"
    if prob >= REVIEW_AT:
        return "REVIEW"
    return "ROUTINE"


def run_triage(model_path, folder, tag="triage"):
    ensure_dirs()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    paths = list_images(folder)
    if not paths:
        raise FileNotFoundError(f"No images found in {folder}")
    model = load_trained_model(model_path)
    probs = predict_paths(model, paths)

    rows = []
    for p, prob in zip(paths, probs):
        truth = p.parent.name if p.parent.name in CLASS_NAMES else ""
        rows.append({"file": str(p), "ground_truth": truth,
                     "probability_pneumonia": round(float(prob), 4),
                     "priority": priority_for(prob)})
    rows.sort(key=lambda r: r["probability_pneumonia"], reverse=True)
    for rank, r in enumerate(rows, 1):
        r["queue_position"] = rank

    csv_path = OUTPUTS_DIR / f"{tag}_worklist.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["queue_position", "file", "ground_truth",
                                          "probability_pneumonia", "priority"])
        w.writeheader(); w.writerows(rows)

    counts = {k: sum(r["priority"] == k for r in rows) for k in ("URGENT", "REVIEW", "ROUTINE")}
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(counts.keys(), counts.values(), color=["#c0392b", "#e6a117", "#2e8b57"])
    ax.bar_label(bars); ax.set_title(f"Simulated AI triage of {len(rows)} studies")
    ax.set_ylabel("Studies")
    fig.tight_layout(); fig.savefig(OUTPUTS_DIR / f"{tag}_priority_summary.png", dpi=130)
    plt.close(fig)

    print(f"Worklist saved: {csv_path}\nPriority counts: {counts}")
    return csv_path, counts
