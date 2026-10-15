# Virtual simulation of a hospital workflow

Real hospital systems (PACS/RIS) are not available to students, so we simulate the *decision-support* part.

| Real hospital | Simulation here |
|---|---|
| Patient scan stored in PACS | image file in `data/chest_xray/test/` |
| Radiologist's final report | folder label (NORMAL / PNEUMONIA) = ground truth |
| AI server scoring new studies | `predict_paths()` batch inference |
| Worklist re-ordering | `triage.py` → `outputs/triage_worklist.csv` |
| Radiologist verifies AI | Grad-CAM heatmap + probability shown |
| Clinical API integration | Flask `/predict` endpoint |

## Steps
1. `python main.py explore` - show dataset size, class balance, sample images.
2. `python main.py train ...` - train the model on the "historical archive" (train split).
3. `python main.py evaluate ...` - measure performance on unseen "new patients" (test split).
4. `python main.py triage --folder data/chest_xray/test` - treat the test images as today's incoming studies.
   Output: ranked CSV with URGENT (p ≥ 0.85), REVIEW (≥ 0.50), ROUTINE.
5. Open the CSV: compare `priority` to `ground_truth`. Count how many true pneumonia cases landed in URGENT/REVIEW (the benefit) and how many NORMAL cases did (the cost: extra reads).
6. `python main.py predict --gradcam` on one URGENT case - show where the model "looked".
7. `python main.py serve` - demo the upload page.

## Proof to capture
class distribution · sample images · preprocessing figure · training log + curves · confusion matrix · ROC ·
prediction gallery · error gallery · Grad-CAM · triage summary + worklist CSV head · terminal screenshot of `evaluate` · web demo screenshot.
