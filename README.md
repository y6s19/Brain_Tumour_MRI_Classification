# Brain Tumour MRI Classification
### CSC-44112 | Advanced Applications of AI and ML — Assessment Part 2

> Automated classification of brain MRI scans into four categories using SVM, Custom CNN, and Transfer Learning (ResNet50).

---

## 📋 Project Overview

This project investigates automated brain tumour classification from T1-weighted MRI scans using three progressively sophisticated machine learning approaches:

| Model | Test Accuracy | Macro F1 |
|---|---|---|
| SVM + PCA (baseline) | 78.75% | 0.782 |
| Custom CNN (4-layer) | 36.75% | 0.256 |
| ResNet50 (Transfer Learning) | **93.00%** | **0.929** |

**Classes:** Glioma · Meningioma · Pituitary Tumour · No Tumour  
**Dataset:** 7,023 T1-weighted MRI images (Nickparvar, 2021)

---

## 📁 Repository Structure

```
├── Brain_Tumour_MRI_Classification_SVM_and_CNN.ipynb   # SVM + Custom CNN experiments
├── Brain_Tumour_ResNet50.ipynb                          # ResNet50 transfer learning
├── README.md                                            # This file
```

---

## 🗂️ Dataset

**Brain Tumour MRI Dataset** — Nickparvar (2021)  
🔗 https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

- 7,023 JPEG images split into `Training/` and `Testing/` folders
- Four classes: `glioma`, `meningioma`, `notumor`, `pituitary`
- Fully anonymised — no personally identifiable information

To download via code:
```python
import kagglehub
path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
```

---

## ⚙️ Setup & Requirements

### Environment
- Python 3.10+
- Google Colab (recommended) with NVIDIA T4 GPU, or any CUDA-enabled environment

### Install dependencies
```bash
pip install tensorflow==2.20 scikit-learn matplotlib seaborn kagglehub
```

### Key libraries
| Library | Purpose |
|---|---|
| TensorFlow 2.20 / Keras | CNN and ResNet50 model building & training |
| Scikit-learn | SVM, PCA, GridSearchCV, evaluation metrics |
| Matplotlib / Seaborn | Visualisations (KDE plots, confusion matrices, learning curves) |
| kagglehub | Dataset download |

---

## 🚀 How to Run

### Notebook 1 — SVM & Custom CNN
Open `Brain_Tumour_MRI_Classification_SVM_and_CNN.ipynb` in Google Colab.

Sections in the notebook follow the report structure:
1. **EDA** — class distribution, pixel intensity KDE plots, sample visualisations
2. **SVM Pipeline** — PCA reduction, GridSearchCV, confusion matrix
3. **Custom CNN** — architecture, training, learning curves, evaluation

### Notebook 2 — ResNet50 Transfer Learning
Open `Brain_Tumour_ResNet50.ipynb` in Google Colab.

1. **Preprocessing** — ImageNet `preprocess_input`, augmentation pipeline
2. **Phase 1** — Head warmup (base frozen, 15 epochs)
3. **Phase 2** — Selective fine-tuning (conv4+ unfrozen, 25 epochs)
4. **Evaluation** — Confusion matrix, Grad-CAM visualisations

> **Note:** Set `training=False` when calling the ResNet50 base in the Keras Functional API. This locks BatchNormalization layers in inference mode and is critical — omitting it causes Phase 2 accuracy to collapse to ~25% (random chance).

---

## 📊 Results Summary

### Model Comparison
```
SVM + PCA      ████████████████████░░░░░  78.75%
Custom CNN     █████████░░░░░░░░░░░░░░░░  36.75%
ResNet50       ███████████████████████░░  93.00%
```

### Per-class F1 — ResNet50
| Class | F1-Score |
|---|---|
| Glioma | 0.94 |
| Pituitary | 0.96 |
| No Tumour | 0.95 |
| Meningioma | 0.89 |

### Literature Comparison
| Study | Model | Accuracy |
|---|---|---|
| Deepak & Ameer (2019) | GoogLeNet | 97.1% |
| Swati et al. (2019) | VGG19 (fine-tuned) | 94.82% |
| **This project** | **ResNet50 (fine-tuned)** | **93.00%** |
| Abiwinanda et al. (2019) | Custom CNN | 84.19% |

---

## 🔑 Key Technical Notes

- **Stratified sampling** with `seed=42` used throughout for reproducibility
- **SVM**: RBF kernel, `C=10`, PCA reduced to 100 components (~95% variance retained)
- **Custom CNN**: 4 conv blocks (32→64→128→256 filters), GlobalAveragePooling, Dropout(0.5)
- **ResNet50**: `training=False` on base, GAP+GMP concatenation, label smoothing α=0.1
- **Grad-CAM** visualisations confirm anatomically meaningful activations for tumour classes

---

## 📚 Reference

Nickparvar, M. (2021) *Brain Tumor MRI Dataset*. Kaggle. Available at: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

