# Data

Datasets are **not** committed to Git (too large). Put them here instead.

## Option 1 - Real public data (recommended)
**Chest X-Ray Images (Pneumonia)** - Kaggle, by Paul Mooney  
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

1. Download and unzip.
2. Place the folder so you get exactly this layout:

```
data/chest_xray/
├── train/  NORMAL/  PNEUMONIA/
├── val/    NORMAL/  PNEUMONIA/     (only 16 images - we split validation from train instead)
└── test/   NORMAL/  PNEUMONIA/
```

Or with the Kaggle CLI:
```bash
pip install kaggle
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data --unzip
```
(Some downloads nest an extra `chest_xray/chest_xray/` folder - move the inner one up.)

## Option 2 - Built-in synthetic data (no download, for testing the pipeline)
```bash
python main.py synthetic
```
Creates `data/simulated/` with the same folder layout. These are **computer-generated
cartoon "X-rays"** - they exist only so you can verify the code end-to-end in minutes.
Never report their accuracy as a medical result.

## Optional extra datasets
- NIH ChestX-ray14: https://nihcc.app.box.com/v/ChestXray-NIHCC (large, multi-label)
- Brain MRI tumor datasets on Kaggle (same folder layout works with a tweak to class names)
