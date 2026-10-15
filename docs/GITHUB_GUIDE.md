# GitHub guide

**Repo name:** `AI-Medical-Image-Analysis`  
**Description:** Pneumonia detection from chest X-rays with CNN + MobileNetV2, Grad-CAM explainability, triage simulation and Flask API.  
**Topics:** `deep-learning` `medical-imaging` `computer-vision` `tensorflow` `keras` `opencv` `cnn` `transfer-learning` `grad-cam` `flask` `healthcare-ai` `python`

## First push
```bash
cd AI-Medical-Image-Analysis
git init
git add .
git commit -m "chore: initial project structure"
git branch -M main
git remote add origin https://github.com/<your-username>/AI-Medical-Image-Analysis.git
git push -u origin main
```
Create the empty repo on github.com first (no README/licence, since we already have them). Add an MIT `LICENSE` via GitHub's UI if you want.

## Never commit
datasets, `.keras` models (large), `venv/`. They are already in `.gitignore`. If a file is >100 MB GitHub rejects the push.

## Screenshots and demo assets
1. Run commands → files appear in `outputs/`.
2. Copy the best ones to `images/` using the names in the README (`cnn_training_curves.png`, `cnn_confusion_matrix.png`, `cnn_predictions.png`, `gradcam_example.png`, `pipeline.png`…).
3. `git add images && git commit -m "docs: add result screenshots" && git push`.
4. Short demo video (30-60 s screen recording of `serve`): upload to YouTube (unlisted) or drag into a GitHub issue/README to host, and link it at the top of the README.
5. Pin the repo on your profile; add the website/Colab link in the repo "About" box.

## 7-day proof plan (commit messages included)
| Day | Work | Commit message(s) | Proof |
|---|---|---|---|
| 1 | Environment, folders, config | `chore: project structure and config` · `docs: add data instructions` | repo + folder screenshot |
| 2 | Download data, `explore` | `feat: dataset summary and sample visualisation` | class_distribution.png, sample_images.png |
| 3 | Preprocessing + synthetic data | `feat: preprocessing utilities` · `feat: synthetic dataset generator` · `test: preprocessing tests` | preprocessing_steps.png |
| 4 | Models | `feat: custom CNN` · `feat: MobileNetV2 transfer learning` | model summary screenshot |
| 5 | Training | `feat: training pipeline with callbacks` · `results: CNN training curves` | training log + curves |
| 6 | Evaluation, Grad-CAM, triage | `feat: evaluation metrics and plots` · `feat: grad-cam` · `feat: triage simulation` · `results: add metrics` | confusion matrix, ROC, Grad-CAM, worklist |
| 7 | Flask API, README, polish | `feat: flask api` · `docs: README with results` · `docs: architecture and simulation` | demo screenshot/video, pinned repo |

Make commits on the days you actually do the work; honest history is better proof than a single giant commit.

## Final checklist
- [ ] README results table filled with **your** measured numbers
- [ ] 6+ images in `images/` render on the README
- [ ] Tests pass (`pytest -q`)
- [ ] Repo description, topics, pinned
- [ ] Disclaimer present
- [ ] LinkedIn post linking the repo
