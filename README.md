# OCR Insurance ID Classifier (PyTorch)

## Overview

This project trains a **multi-input (multimodal) neural network using PyTorch** to classify OCR-extracted insurance ID codes as either:

- **Primary ID** (`0`)
- **Secondary ID** (`1`)

Each sample contains two inputs:

1. **A grayscale OCR image tensor** representing the scanned ID/code image  
2. **A one-hot insurance type vector** representing the policy category (`home`, `life`, `auto`, `health`, `other`)

The model processes the image using a **CNN** and the type vector using a small **MLP**, then **concatenates** both feature representations to make a final binary prediction.

This project demonstrates:

- CNN-based feature extraction for OCR-like images  
- Multimodal feature fusion (image + metadata)  
- Binary classification in PyTorch  
- Script-based ML workflow (train → evaluate → export CSV)

---

# Dataset

The dataset is provided as a pickle file:

- `ocr_insurance_dataset.pkl`

This file contains:
- `data`: list of `(image_tensor, type_tensor)` samples  
- `labels`: list of binary labels (`0/1`)  
- `label_mapping`: mapping of class names to integers  
- `type_mapping`: mapping of insurance types to indices  

> Note: The pickle was saved with a custom class name (`ProjectDataset`).  
> The scripts include a small compatibility stub so the dataset loads correctly without needing the original course utility files.

---

# Model Architecture

The model has two branches:

## 1) Image Branch (CNN)
Processes a grayscale OCR image tensor:

- Convolution → ReLU → Pooling (repeated)
- Adaptive average pooling to support different image sizes
- Flatten into a fixed-length embedding

## 2) Type Branch (MLP)
Processes a one-hot type vector (length 5) through fully-connected layers.

## 3) Fusion + Classifier
Concatenates image features + type features → dense layers → binary output logit.

The model is trained using:

- `BCEWithLogitsLoss` (binary classification best practice)
- `Adam` optimizer

---

# Installation and Dependencies

Install dependencies:

```bash
python -m pip install torch pandas
```

---

# How to Run

## Step 1 — Train

```bash
python train.py
```

This generates automatically:

- `outputs/ocr_model.pth`

## Step 2 — Evaluate + Export Predictions

```bash
python evaluate.py
```

This generates automatically:

- `outputs/predictions.csv`

---

# Output Files

The evaluation step creates a CSV file:

- `outputs/predictions.csv`

Columns include:

- `true_label` (0/1)
- `pred_label` (0/1)
- `prob_secondary` (model probability of class `1`)

---

# Files Included

| File | Description |
|------|-------------|
| `model.py` | Multimodal OCRModel (CNN for image + MLP for type + fusion head) |
| `train.py` | Loads dataset, trains model, saves weights (generated output) |
| `evaluate.py` | Loads saved weights, evaluates test split, exports CSV predictions |
| `ocr_insurance_dataset.pkl` | Dataset file (pickle) |
| `README.md` | Project documentation |

---

# Repository Notes

The following files are **generated automatically** when running the project and are **not intended to be committed to GitHub**:

- `outputs/ocr_model.pth` (trained model weights)
- `outputs/predictions.csv` (prediction output)

---

