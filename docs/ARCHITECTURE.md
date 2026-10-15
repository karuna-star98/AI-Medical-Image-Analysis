# Architecture

## Data flow
```
data/chest_xray/{train,test}/{NORMAL,PNEUMONIA}/*.jpeg
   │  data_loader.load_datasets()      grayscale, 224x224, batches, 80/20 train/val split
   ▼
tf.data pipeline (raw 0-255 floats)
   │  model.build_cnn() / build_mobilenet()
   │     Augmentation (rotation, zoom, shift, contrast - training only)
   │     Rescaling (inside the model => no train/serve mismatch)
   │     Feature extractor (Conv+BN+Pool blocks  OR  frozen MobileNetV2)
   │     GlobalAveragePooling → Dropout → Dense(1, sigmoid)
   ▼
train.train()   Adam + class weights + EarlyStopping(val_auc) + ModelCheckpoint
   ▼
models/<tag>_best.keras
   ├── evaluate.evaluate()   → metrics.json, confusion matrix, ROC, prediction galleries
   ├── predict.*  + explain  → label + probability + Grad-CAM
   ├── triage.run_triage()   → prioritised worklist CSV + summary chart
   └── app (Flask)           → POST /predict
```

## Modules
| File | Responsibility |
|---|---|
| `config.py` | paths, image size, threshold, priority bands |
| `data_loader.py` | dataset summary, class weights, tf.data loading |
| `preprocessing.py` | OpenCV load/resize, CLAHE demo, model-input shaping |
| `model.py` | CNN, MobileNetV2, compile, fine-tuning helper |
| `train.py` | training loop, callbacks, saving curves |
| `metrics.py` | clinical metrics (sensitivity, specificity, AUC…) |
| `evaluate.py` | test-set evaluation + artifacts |
| `visualize.py` | all plots |
| `predict.py` | single / batch inference |
| `explain.py` | Grad-CAM |
| `triage.py` | worklist simulation |
| `synthetic.py` | cartoon dataset for pipeline testing |
| `app.py` | Flask API |

## Key design decisions
1. **Rescaling inside the model** - one source of truth for normalisation.
2. **No horizontal flip** augmentation - heart position makes mirrored chest X-rays unrealistic.
3. **Validation split from train** - the official `val/` folder has only 16 images.
4. **Select by val AUC, report sensitivity/specificity** - accuracy alone hides missed cases when classes are imbalanced.
5. **Test set never used for training or model selection.**
