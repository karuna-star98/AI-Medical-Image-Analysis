# Interview Q&A (answer using YOUR measured results)

**What did you build?** A pipeline that classifies chest X-rays (normal vs pneumonia) with a CNN and MobileNetV2, evaluates it with clinical metrics, explains predictions with Grad-CAM, simulates worklist triage, and serves predictions through Flask.

**Why CNNs?** Convolutions learn local patterns (edges → textures → opacities) with shared weights, so they suit images.

**Why transfer learning?** Medical datasets are small; ImageNet features (edges, textures) transfer, so fewer epochs and often better results. I froze the base, then optionally fine-tuned the top layers at a low learning rate.

**Why not just accuracy?** The dataset is imbalanced (~3:1 pneumonia). I report sensitivity (missed cases are costly), specificity, ROC-AUC and the confusion matrix.

**How did you handle imbalance?** Class weights during training; metrics beyond accuracy.

**Why no horizontal flip?** Anatomy is not symmetric (heart side); flips create unrealistic images.

**How do you know the model is not cheating?** Grad-CAM to check it looks at the lungs; limitation: dataset shortcuts and no external validation.

**How would you deploy it?** Flask/FastAPI behind Docker; model versioning; input validation; logging; DICOM handling; and for real use, regulatory approval and clinical validation.

**What are the limitations?** Single public dataset, pediatric images, small test set, unverified labels, no patient-level split guarantee, not a medical device.

**What next?** External test set, calibration, multi-class, DICOM, detection/segmentation.
