# 🌾 Rice Leaf Disease Classification

### Deep Learning with Transfer Learning — ResNet50 vs EfficientNet-B0

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Accuracy](https://img.shields.io/badge/Best%20Accuracy-96.40%25-brightgreen)

A deep learning project that classifies **6 rice leaf diseases** from images using transfer learning with two pretrained CNN architectures — **ResNet50** and **EfficientNet-B0** — trained and compared on the Rice Leaf Disease dataset.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Disease Classes](#-disease-classes)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Training Curves](#-training-curves)
- [Confusion Matrices](#-confusion-matrices)
- [Model Comparison](#-model-comparison)
- [Setup & Usage](#-setup--usage)
- [Key Implementation Details](#-key-implementation-details)
- [Dependencies](#-dependencies)

---

## 🔍 Overview

This project tackles automated rice leaf disease detection using **transfer learning** on pretrained ImageNet models. Two architectures are trained, evaluated, and compared:

| Model               | Strategy             | Trainable Params | Epochs              | Val Accuracy | Training Time |
| ------------------- | -------------------- | ---------------- | ------------------- | ------------ | ------------- |
| **ResNet50**        | Layer4 + FC unfrozen | ~6.5M (25%)      | 6 _(early stopped)_ | **96.40%**   | ~43.5 min     |
| **EfficientNet-B0** | Classifier only      | ~30K (0.5%)      | 20                  | 77.27%       | ~49 min       |

**ResNet50** achieves significantly higher accuracy through partial fine-tuning, while **EfficientNet-B0** demonstrates how a classifier-only approach, though faster, is limited by frozen backbone features.

---

## 🌿 Disease Classes

The dataset contains **6 categories** of rice leaf conditions:

| #   | Class                   | Description                                     |
| --- | ----------------------- | ----------------------------------------------- |
| 1   | `bacterial_leaf_blight` | Bacterial infection causing yellowing & wilting |
| 2   | `brown_spot`            | Fungal disease with brown oval lesions          |
| 3   | `healthy`               | No disease present                              |
| 4   | `leaf_blast`            | Fungal disease causing diamond-shaped lesions   |
| 5   | `leaf_scald`            | Scalding symptoms along leaf margins            |
| 6   | `narrow_brown_spot`     | Narrow brown streaks along leaf veins           |

Each validation split contains **88 images per class** (528 total).

---

## 📁 Project Structure

```
rice-leaf-disease-classification/
│
├── 📓 ResNet50_training.ipynb          # ResNet50 training pipeline
├── 📓 EfficientNet-B0_training.ipynb   # EfficientNet-B0 training pipeline
│
├── dataset/
│   └── RiceLeafsDisease/
│       ├── train/                      # Training images (by class folder)
│       └── validation/                 # Validation images (by class folder)
│
├── outputs/
│   ├── plots & confusion_matrix/
│   │   ├── 1. Training vs Validation Accuracy (EfficientNet).png
│   │   ├── 2. Training vs Validation Loss (EfficientNet).png
│   │   ├── 3. Confusion Matrix (EfficientNet).png
│   │   ├── 4. EfficientNet - Training & Validation Accuracy Over Epochs.png
│   │   ├── 5. Training vs Validation Accuracy (ResNet).png
│   │   ├── 6. Training vs Validation Loss (ResNet).png
│   │   ├── 7. ResNet Confusion Matrix (ResNet).png
│   │   ├── 8. ResNet - Training & Validation Accuracy Over Epochs.png
│   │   └── 9. EfficientNet vs ResNet COMPARISON.png
│   │
│   ├── efficientnet_paddy_model.pth    # Saved EfficientNet-B0 weights
│   ├── resnet50_paddy_model.pth        # Saved ResNet50 weights
│   ├── efficientnet_results.pkl        # EfficientNet training history
│   └── resnet50_training_history.pkl   # ResNet50 training history
│
└── README.md
```

---

## 📊 Results

### ResNet50 — Classification Report

| Class                 | Precision | Recall   | F1-Score | Support |
| --------------------- | --------- | -------- | -------- | ------- |
| bacterial_leaf_blight | 1.00      | 1.00     | **1.00** | 88      |
| brown_spot            | 0.95      | 0.89     | 0.92     | 88      |
| healthy               | 0.97      | 0.99     | 0.98     | 88      |
| leaf_blast            | 0.88      | 0.95     | 0.92     | 88      |
| leaf_scald            | 0.99      | 0.99     | 0.99     | 88      |
| narrow_brown_spot     | 1.00      | 0.97     | 0.98     | 88      |
| **Macro Average**     | **0.97**  | **0.96** | **0.96** | 528     |
| **Accuracy**          |           |          | **96%**  | 528     |

### EfficientNet-B0 — Classification Report

| Class                 | Precision | Recall   | F1-Score | Support |
| --------------------- | --------- | -------- | -------- | ------- |
| bacterial_leaf_blight | 0.88      | 0.91     | 0.89     | 88      |
| brown_spot            | 0.72      | 0.66     | 0.69     | 88      |
| healthy               | 0.83      | 0.85     | 0.84     | 88      |
| leaf_blast            | 0.59      | 0.69     | 0.64     | 88      |
| leaf_scald            | 0.79      | 0.94     | 0.86     | 88      |
| narrow_brown_spot     | 0.86      | 0.58     | 0.69     | 88      |
| **Macro Average**     | **0.78**  | **0.77** | **0.77** | 528     |
| **Accuracy**          |           |          | **77%**  | 528     |

---

## 📈 Training Curves

### EfficientNet-B0

<table>
  <tr>
    <td><img src="./outputs/plots & confusion_matrix/1. Training vs Validation Accuracy (EfficientNet).png" alt="EfficientNet Accuracy" width="400"/></td>
    <td><img src="./outputs/plots & confusion_matrix/2. Training vs Validation Loss (EfficientNet).png" alt="EfficientNet Loss" width="400"/></td>
  </tr>
  <tr>
    <td align="center"><em>Training vs Validation Accuracy</em></td>
    <td align="center"><em>Training vs Validation Loss</em></td>
  </tr>
</table>

<img src="./outputs/plots & confusion_matrix/4. EfficientNet - Training & Validation Accuracy Over Epochs.png" alt="EfficientNet Accuracy Over Epochs" width="820"/>

> Best training accuracy: **81.52%** (Epoch 20) · Best validation accuracy: **80.68%** (Epoch 14) · Mild overfitting gap: **4.25%**

---

### ResNet50

<table>
  <tr>
    <td><img src="./outputs/plots & confusion_matrix/5. Training vs Validation Accuracy (ResNet).png" alt="ResNet Accuracy" width="400"/></td>
    <td><img src="./outputs/plots & confusion_matrix/6. Training vs Validation Loss (ResNet).png" alt="ResNet Loss" width="400"/></td>
  </tr>
  <tr>
    <td align="center"><em>Training vs Validation Accuracy</em></td>
    <td align="center"><em>Training vs Validation Loss</em></td>
  </tr>
</table>

<img src="./outputs/plots & confusion_matrix/8. ResNet - Training & Validation Accuracy Over Epochs.png" alt="ResNet Accuracy Over Epochs" width="820"/>

> Best training accuracy: **94.57%** · Best validation accuracy: **96.40%** (Epoch 6) · No overfitting (gap: **−1.83%**)

---

## 🔲 Confusion Matrices

<table>
  <tr>
    <td align="center"><strong>EfficientNet-B0</strong></td>
    <td align="center"><strong>ResNet50</strong></td>
  </tr>
  <tr>
    <td><img src="./outputs/plots & confusion_matrix/3. Confusion Matrix (EfficientNet).png" alt="EfficientNet Confusion Matrix" width="400"/></td>
    <td><img src="./outputs/plots & confusion_matrix/7. ResNet Confusion Matrix (ResNet).png" alt="ResNet Confusion Matrix" width="400"/></td>
  </tr>
</table>

ResNet50 achieves a near-perfect diagonal — **Bacterial Leaf Blight classified with 100% accuracy** (88/88). EfficientNet struggles most with `leaf_blast` and `narrow_brown_spot` due to the frozen backbone's inability to adapt high-level features.

---

## ⚖️ Model Comparison

<img src="./outputs/plots & confusion_matrix/9. EfficientNet vs ResNet COMPARISON .png" alt="Model Comparison" width="820"/>

| Metric                    | ResNet50                | EfficientNet-B0     |
| ------------------------- | ----------------------- | ------------------- |
| Final Validation Accuracy | **96.40%**              | 77.27%              |
| Best Validation Accuracy  | **96.40%** _(Epoch 6)_  | 80.68% _(Epoch 14)_ |
| Macro F1-Score            | **0.96**                | 0.77                |
| Training Time             | ~43.5 min               | ~49 min             |
| Overfitting               | ✅ None (−1.83%)        | ⚠️ Mild (+4.25%)    |
| Epochs to Converge        | **6** _(early stopped)_ | 20                  |

---

## 🚀 Setup & Usage

### 1. Clone the repository

```bash
git clone https://github.com/Rakin-Al-Mahin/Rice_Leaf_Disease_Classification
cd rice-leaf-disease-classification
```

### 2. Install dependencies

```bash
pip install torch torchvision timm scikit-learn matplotlib seaborn tqdm
```

### 3. Prepare the dataset

Organise your dataset in the following structure:

```
dataset/RiceLeafsDisease/
    train/
        bacterial_leaf_blight/
        brown_spot/
        healthy/
        leaf_blast/
        leaf_scald/
        narrow_brown_spot/
    validation/
        (same structure)
```

### 4. Train the models

Open and run the notebooks in order:

```bash
jupyter notebook EfficientNet-B0_training.ipynb
jupyter notebook ResNet50_training.ipynb
```

Or run both and compare results — the ResNet50 notebook automatically loads the saved EfficientNet results for side-by-side comparison plots.

---

## 🔧 Key Implementation Details

### Data Augmentation (Training only)

```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

### ResNet50 — Partial Fine-tuning

```python
resnet_model = models.resnet50(pretrained=True)

# Freeze all layers
for param in resnet_model.parameters():
    param.requires_grad = False

# Replace classifier head
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 6)

# Unfreeze Layer4 + FC for domain adaptation
for param in resnet_model.layer4.parameters():
    param.requires_grad = True
for param in resnet_model.fc.parameters():
    param.requires_grad = True
```

### EfficientNet-B0 — Classifier-only Fine-tuning

```python
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=6)

# Freeze entire backbone
for param in model.parameters():
    param.requires_grad = False

# Train only the final classifier
for param in model.classifier.parameters():
    param.requires_grad = True
```

### Optimizer & Scheduler

```python
optimizer = optim.Adam(trainable_params, lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
scaler    = torch.amp.GradScaler('cuda')  # Mixed precision (FP16)
```

### Early Stopping (ResNet50)

```python
if val_accuracy > 0.96:
    print(f"Target accuracy reached — stopping at epoch {epoch+1}")
    break
```

---

## 📦 Dependencies

| Library                  | Purpose                                     |
| ------------------------ | ------------------------------------------- |
| `torch` / `torchvision`  | Deep learning framework & pretrained models |
| `timm`                   | EfficientNet-B0 pretrained model            |
| `scikit-learn`           | Classification report & confusion matrix    |
| `matplotlib` / `seaborn` | Training curve & heatmap visualisations     |
| `tqdm`                   | Progress bars during training               |
| `numpy`                  | Numerical operations                        |
| `pickle`                 | Saving/loading training history             |

```bash
pip install torch torchvision timm scikit-learn matplotlib seaborn tqdm numpy
```

---

## 📌 Notes

- Both models use **batch size 8** and **mixed-precision training (FP16)** via `torch.amp`
- The ResNet50 notebook loads EfficientNet results from `efficientnet_results.pkl` for comparison — run EfficientNet first, or the comparison plots will be skipped gracefully
- GPU is strongly recommended (CUDA); CPU-only training will be significantly slower
- ImageNet normalisation statistics are used for both models as both are pretrained on ImageNet

---

## 📄 License

This project is licensed under the MIT License.
