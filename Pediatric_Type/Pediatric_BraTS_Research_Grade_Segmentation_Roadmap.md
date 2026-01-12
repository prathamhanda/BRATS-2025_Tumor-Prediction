
# Research-Grade Pediatric Brain Tumor Segmentation (BraTS PEDs)
## End-to-End Methodology & Debugging Playbook

> **Purpose**  
This document is a **single source of truth** for implementing a **research-grade pediatric brain tumor segmentation pipeline** for the BraTS Pediatric (PEDs) dataset.  
It is designed so that **LLM agents, engineers, or researchers** can:
- Implement the pipeline step-by-step
- Debug failures systematically
- Achieve competitive Dice scores comparable to MICCAI/BraTS leaderboard methods

---

## 0. Core Principles (Read First)

### Pediatric ≠ Adult Glioma
- Pediatric tumors differ in:
  - Shape
  - Contrast enhancement
  - Size of enhancing regions
- Adult-trained glioma pipelines **will fail silently**
- Treat PEDs as a **new domain**, not a fine-tune problem

### Why PEDs Segmentation Fails
- Extremely small Enhancing Tumor (ET)
- Noisy labels
- Large inter-patient anatomical variance
- Class imbalance (ET << TC << WT)

---

## 1. Dataset & Directory Assumptions

### Expected Modalities
- T1
- T1ce
- T2
- FLAIR

### Expected Labels
- 0: Background
- 1: Necrotic / Non-enhancing tumor (NET)
- 2: Edema
- 4: Enhancing tumor (ET)

### Recommended Folder Structure
```
BraTS_PED/
 ├── imagesTr/
 │   ├── case_001_T1.nii.gz
 │   ├── case_001_T1ce.nii.gz
 │   ├── case_001_T2.nii.gz
 │   └── case_001_FLAIR.nii.gz
 ├── labelsTr/
 │   └── case_001_seg.nii.gz
 └── imagesTs/
```

---

## 2. Preprocessing (CRITICAL PHASE)

### 2.1 Orientation & Resolution
- Convert all volumes to **RAS orientation**
- Resample to:
  - 1×1×1 mm³ OR nnU-Net auto resolution

**Debugging**
- Verify shape consistency across modalities
- Visualize overlay to confirm alignment

---

### 2.2 Skull Stripping (Mandatory)
**Why**
- Pediatric skull morphology varies significantly
- Non-brain tissue causes false positives

**Recommended**
- nnU-Net skull-stripping model
- HD-BET (pediatric-tuned if possible)

**Debugging**
- Overlay brain mask on FLAIR
- Ensure no tumor tissue is removed

---

### 2.3 Intensity Normalization
- Z-score **per modality per subject**
- Normalize only inside brain mask

**Failure Signs**
- Model predicts tumor everywhere
- Dice collapses early in training

---

## 3. Baseline Model (Foundation)

### 3.1 nnU-Net v2 (Baseline)
- 3D full-resolution nnU-Net
- Default nnU-Net configuration

### Loss
```
DiceLoss + CrossEntropyLoss
```

### Augmentations
- Random flip (x, y, z)
- Rotation
- Scaling
- Gamma correction
- Gaussian noise
- Blur

**Expected Dice**
| Region | Dice |
|------|------|
| WT | ≥ 0.85 |
| TC | ≥ 0.75 |
| ET | ≥ 0.65 |

If not achieved → preprocessing is broken

---

## 4. Pediatric-Specific Optimization

### 4.1 Class-Imbalance-Aware Loss
```
Total Loss =
  Dice(WT)
+ Dice(TC)
+ 2.5 × Dice(ET)
+ FocalLoss(ET)
```

**Why**
- ET regions are tiny
- Dice alone under-trains ET

---

### 4.2 Two-Stage Segmentation (Highly Recommended)

#### Stage 1
- Binary segmentation: Whole Tumor (WT)

#### Stage 2
- Crop bounding box around WT
- Segment subregions: TC, ET

**Benefits**
- Massive ET Dice improvement
- Reduced false positives

---

## 5. Transformer-Based Model (Context Modeling)

### Recommended Architectures
- Swin-UNETR
- U-Mamba
- Transformer-enhanced nnU-Net

### Training Strategy
- Pretrain on adult BraTS (if available)
- Fine-tune on PEDs
- Same preprocessing as nnU-Net

**Debugging**
- Monitor overfitting (transformers overfit fast)
- Use stronger augmentation

---

## 6. Multi-Model Ensemble (Research-Grade)

### Models to Train Independently
1. nnU-Net (seed A)
2. nnU-Net (seed B)
3. Swin-UNETR
4. SegResNet / MedNeXt
5. Optional frequency-domain model

### Ensemble Strategy
- Softmax probability averaging
- ET-weighted fusion

Example:
```
ET_final = 0.5 * Transformer + 0.3 * nnU-Net + 0.2 * MedNeXt
```

---

## 7. Post-Processing (Free Dice)

### 7.1 Connected Component Filtering
- Remove ET regions < 10–20 voxels
- Keep largest WT component

### 7.2 Confidence-Based Suppression
- Suppress ET if max probability < threshold
- Prevent hallucinated ET

---

## 8. Evaluation & Debugging

### Metrics
- Dice (WT, TC, ET)
- Optional: Hausdorff 95

### Visual Debugging (Mandatory)
- Overlay predictions on FLAIR
- Focus on:
  - Missed ET
  - Over-segmentation
  - Boundary leakage

---

## 9. Expected Final Performance

| Region | Dice |
|------|------|
| WT | 0.88 – 0.92 |
| TC | 0.80 – 0.86 |
| ET | 0.72 – 0.80 |

These numbers are **BraTS-competitive**.

---

## 10. Research Polish (For Papers)

Include:
- Ablation studies
- Loss function comparison
- Ensemble vs single model
- Failure case analysis

---

## 11. Final Notes
- PED segmentation is **fragile**
- Most gains come from:
  - Preprocessing
  - Loss design
  - Ensembles
- Avoid premature architecture complexity

This pipeline reflects **current best practices from BraTS PED challenges** and is suitable for **MICCAI-level research**.

---
