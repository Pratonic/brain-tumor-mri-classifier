# 🧠 Brain Tumor MRI Classifier

EfficientNetB2 transfer learning classifier for 30-class brain tumor MRI dataset.

---

## Folder Structure

```
brain_tumor_project/
├── train.py            ← Run once to train and save the model
├── predict.py          ← Run anytime to predict on a new image
├── requirements.txt    ← Python dependencies
├── README.md
└── outputs/            ← Auto-populated after training
    ├── brain_tumor_model.keras       ← saved model (loaded on future runs)
    ├── class_names.txt               ← class list used by predict.py
    ├── checkpoint_phase1.keras       ← best checkpoint from phase 1
    ├── checkpoint_phase2.keras       ← best checkpoint from phase 2
    ├── training_curves.png
    ├── confusion_matrix.png
    └── classification_report.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Usage

### Train (run once)
```bash
python train.py --data "C:/path/to/your/dataset"
# Linux/Mac:
python train.py --data "/home/you/dataset"
```

- If `outputs/brain_tumor_model.keras` already exists, training is **skipped** and it loads directly.
- To force retrain from scratch:
```bash
python train.py --data "C:/path/to/dataset" --retrain
```

### Predict on a new image
```bash
python predict.py --image "path/to/mri_scan.jpg"

# With Grad-CAM heatmap overlay
python predict.py --image "path/to/mri_scan.jpg" --gradcam

# Show top 3 predictions instead of 5
python predict.py --image "path/to/mri_scan.jpg" --topk 3
```

---

## Dataset Structure Expected

```
your_dataset/
├── class_1/
│   ├── img001.jpg
│   └── ...
├── class_2/
└── ...
```

---

## Notes

- GPU strongly recommended. CPU will work but phase 2 fine-tuning will be slow.
- If you get OOM errors, lower `BATCH_SIZE` from 32 → 16 in `train.py`.
- If you're on an older GPU (pre-RTX 30xx), comment out the mixed precision line in `train.py`:
  ```python
  # tf.keras.mixed_precision.set_global_policy('mixed_float16')
  ```
