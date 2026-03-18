
# BEST-OF-TWO-WORLDS — Pediatric Brain Tumor Segmentation
## Final, robust, LLM-agent-friendly implementation playbook
**Purpose:** combine your implemented nnU-Net v2 5-fold ensemble (baseline) with the literature-backed extensions (Swin UNETR, HFF-Net, widened ResEnc, radiomic stratification, specificity-driven loss, H100 optimizations) into a single, executable, debug-friendly plan that agents can implement cell-by-cell.

**This document merges and supersedes your pipeline analysis and the research recommendations.** See your pipeline analysis that motivated many design choices here: fileciteturn1file0

---

## Quick summary (one-line)
Start with your validated nnU-Net v2 baseline, add 2–3 complementary architectures (Swin UNETR, HFF-Net, optional SegResNet/MedNeXt), apply radiomic-stratified folds, use multi-init γ-scale nnU-Net variants, fine-tune transformer from BraTS adult data, enforce specificity-driven loss + ET upweighting, perform cluster-aware postprocessing (T1ce/T1 ratio + adaptive CC filtering), and ensemble with lesion-wise weighted fusion. Use H100 to dramatically increase batch & patch sizes and speed training.

---

# HOW TO USE THIS FILE (for agents)
- Implement **one CELL block** at a time.
- After each CELL: run, paste outputs (shapes, sample images, metrics).
- If any validation check fails, STOP and debug per "failure checklist".
- Seed everything: `seed = 42` (and set OS/PyTorch/NumPy seeds).

---

# DIRECTORY & data expectations (cell 0)
```
/data/BraTS_PED/
 ├── imagesTr/
 │    ├── case_00001_T1.nii.gz
 │    ├── case_00001_T1ce.nii.gz
 │    ├── case_00001_T2.nii.gz
 │    ├── case_00001_FLAIR.nii.gz
 ├── labelsTr/
 │    └── case_00001_seg.nii.gz
 └── imagesTs/
```
Modalities: T1, T1ce, T2, FLAIR. Labels use BraTS PED encoding.

CELL 0 — sanity checks (agents must run this)
```python
# print file counts, sample shapes, check nifti readability
import os, nibabel as nib
root="/data/BraTS_PED"
print(len(os.listdir(os.path.join(root,"imagesTr")))//4,"cases in imagesTr")
# sample file
img=nib.load(os.path.join(root,"imagesTr","case_00001_T1.nii.gz"))
print(img.shape, img.affine)
```
**EXPECTED OUTPUT**: file counts divisible by 4; shapes similar across modalities; affine not None.

---

# PREPROCESSING (cells 1–6)

## Design goals
- RAS orientation, resample to 1×1×1 mm if needed (or use nnU-Net's planner).
- Skull-strip everything (HD-BET or pediatric nnU-Net skullstrip).
- Z-score normalize inside brain mask **per modality**.
- Save preprocessed copies under `/data/preprocessed/{case}/` to avoid recompute.

### CELL 1 — orientation & resample
- Ensure orientation: use `nibabel.orientations` utilities or `dipy`/`SimpleITK`.
- If spacing != 1mm isotropic, resample to 1mm with `scipy.ndimage.zoom` or `SimpleITK.Resample`.

```python
# Cell 1 pseudocode
import nibabel as nib
from nilearn.image import resample_to_img
# load reference (e.g., FLAIR) and resample others to it
ref = nib.load(ref_path)
for mod in ["T1","T1ce","T2","FLAIR"]:
    img = nib.load(mod_path)
    if img.shape != ref.shape or not np.allclose(img.affine,ref.affine):
        # resample code (or use nnU-Net planner later)
```

**CHECK**: Print voxel spacing and shapes before/after. If >2% mismatch across modalities → fail.

### CELL 2 — skull strip
Options:
- Use HD-BET (recommended) or your pediatric nnU-Net skullstrip (`peds-brain-auto-skull-strip`).
- Save brain masks under `/data/preprocessed/masks/case_mask.nii.gz`.

**Validation**:
- Visualize overlay of mask on FLAIR for 3 slices (axial, coronal, sagittal).
- Compute percent of tumor voxels lost: check ground-truth overlap with mask (should be >99% of GT voxels inside mask). If not, adjust.

### CELL 3 — bias-field correction (optional but recommended)
- Run N4ITK bias correction on each modality (SimpleITK).
- Re-check intensities.

### CELL 4 — intensity normalization (must)
Per modality, inside brain mask:
```python
arr = img.get_fdata()
brain = mask.get_fdata().astype(bool)
mean = arr[brain].mean(); std = arr[brain].std()
arr[brain] = (arr[brain] - mean)/std
```
**CHECK**: mean ~ 0, std ~1 for brain voxels. Save normalized images.

### CELL 5 — label sanity & mapping
- Ensure labels mapping (BraTS style):
  - 0 background, 1 NET, 2 ED, 4 ET.
- Create derived maps:
  - WT = (labels>0)
  - TC = (labels==1 or labels==4 or label==?? include CC if present)
  - ET = (labels==4)
- Print voxel counts per region for a few cases. If some cases have 0 ET, note them for later handling.

### CELL 6 — radiomic features & stratified CV (important)
Use PyRadiomics to extract radiomic features (WT mask) per case (shape+texture features ~107 features).
- PCA reduce to retain 90% variance.
- KMeans clustering: choose k via silhouette; typical k=6–10 (papers found 9 clusters).
- Save cluster assignments to JSON.
- Create stratified 5-fold splits by cluster (ensure balanced distribution per fold).

**WHY**: avoids fold bias and stabilizes ensemble.

---

# AUGMENTATION (cell 7)
Implement augmentations using MONAI or TorchIO. Use on-the-fly 3D transforms.

Core transforms (apply with probabilities):
- Random flip (all axes) p=0.5
- Random rotation ±30 degrees
- Random scale 0.9–1.1
- Random elastic deformation (small sigma)
- Random gamma 0.7–1.3
- Random bias field simulation
- Gaussian noise, blur

**CELL 7** should visualize a small grid of augmented slices (6–8 examples) to ensure transforms are realistic.

---

# MODEL SETUP — OVERVIEW
We will train three paradigms:
- **Alpha**: Enhanced nnU-Net v2 family (baseline) — add γ-init variants, widen/res-enc optional.
- **Beta**: Swin UNETR (pretrained on BraTS adult then fine-tune).
- **Gamma**: HFF-Net (frequency decomposition + dual-branch).
- **Delta** (optional): SegResNet or MedNeXt.

Train each with 5-fold CV. Save best checkpoint per fold by validation composite score (primary objective: average Dice(WT,TC,ET)).

---

# nnU-Net ENHANCEMENTS (cells 8–14)

## Rationale
Your baseline nnU-Net is solid for WT but fails TC/ET due to inductive bias and initialization. We will:
- Train multi-init γ variants (γ ∈ {0.3,0.5,0.7,0.9})
- Optionally widen encoder (ResEnc), replace convs with depthwise-separable convs if memory allows
- Add SE blocks in encoder if using ResEnc planner
- Use composite loss with ET upweighting

### CELL 8 — nnU-Net Planner override for H100 (VERY IMPORTANT)
Force planner to use H100 VRAM:
```bash
# run in shell
nnUNetv2_plan_experiment -d <DATASET_ID> -pl nnUNetPlannerResEncM -gpu_memory_target 80 -overwrite_plans_name nnUNetResEncUNetPlans_80G
```
Then inspect generated plan files under `nnUNet_raw_data/.../nnUNetPlans...json` and adjust patch size / batch size.

**EXPECTED ACTION**: patch size increase and batch size → 8–16 depending on model.

### CELL 9 — implement γ-scale initialization
In model weight init code (PyTorch):
```python
def init_weights_scaled(m, gamma=0.7):
    if isinstance(m, (nn.Conv3d, nn.Linear)):
        fan_in = nn.init._calculate_correct_fan(m.weight, 'fan_in')
        std = (1.0/ (fan_in**0.5)) * gamma
        nn.init.normal_(m.weight, 0, std)
```
Train separate runs per γ. Log validation per γ.

### CELL 10 — Loss configuration
```python
# Dice + WeightedCE + Focal
loss = DiceLoss() + CrossEntropyLoss(weight=[w_bg,w_ET,w_TC,...]) + 0.5*FocalLoss(gamma=2)
# Alternatively: class-specific dice weights: give ET a factor 2-3
```

Tune ET weight based on validation ET recall. If ET recall too low, increase ET weight.

### CELL 11 — nnU-Net training cell
- Train per fold
- Use SGD/Nesterov or AdamW
- Use mixed precision (amp)
- Save best checkpoint by composite score

Log outputs: per-epoch Dice for WT/TC/ET.

---

# Swin UNETR (cells 15–20)

## Rationale
Transformer captures long-range context; pretrained on adult BraTS improves convergence.

### CELL 15 — get pre-trained weights
- Obtain BraTS2021 Swin UNETR pretrained weights (link in notes or repo).
- Verify architecture match (patch size etc).

### CELL 16 — model & finetune settings
- Patch: 128×128×128 recommended
- Initial LR: 1e-4 (or 5e-5 with AdamW)
- Disable deep supervision for finetune as per papers
- Epochs: 800–1000 (early stopping allowed)
- Batch size: as allowed (H100 large batch better)

### CELL 17 — finetuning cell
- Load pretrained weights
- Freeze early layers for 1–5 epochs (optional), then unfreeze
- Monitor val metrics vs training from scratch to confirm benefit

**CHECK**: improvement ≥1% overall Dice vs scratch; if not, re-evaluate LR and batch.

---

# HFF-NET (cells 21–28)

## Rationale
Frequency decomposition preserves high-frequency ET details.

### CELL 21 — implement frequency decomposition
- DTCWT (Dual-Tree Complex Wavelet Transform) for LF
- NSCT for HF directional bands
- Libraries: `dtcwt` python package (or implement via PyWavelets), for NSCT use repo or custom implementation.
- Output: 4 LF channels + 16 HF channels → feed into dual-branch network.

**Visualization**: show LF and HF images for 1 case (Figure with LF and HF1..HF4).

### CELL 22 — HFF-Net architecture skeleton
- Two encoders: LF encoder (shallower), HF encoder (deeper)
- ALC modules in HF path (Adaptive Laplacian Conv block)
- FDCA fusion blocks at bottleneck
- Decoder merges with skip-connections

### CELL 23 — HFF training tips
- Batch size may be 1–2 due to many channels; H100 may allow batch 2–4
- LR: 1e-1 for SGD as in literature? (tune; training authors used high LR with different schedules). Safer: AdamW 1e-3
- Train longer, strong ET weighting

**CHECK**: HFF should primarily raise ET Dice; expect +0.03–0.07 on ET if implemented right.

---

# Optional: SegResNet / MedNeXt (cells 29–32)
- Train one of these to add architectural diversity for ensemble
- Use standard configs (MONAI implementations)
- Fine-tune from adult BraTS weights if available

---

# ENSEMBLE (cells 33–36)

## Ensemble philosophy
- Use **lesion-wise weighted fusion**: compute model weights per lesion (WT, TC, ET) from internal CV ranking.
- Save softmax maps per model/fold to disk to enable rapid fusion experiments.

### CELL 33 — generate softmax maps
- For each model and fold, run inference on val/test, saving `prob_{model}_{fold}.npz` with shape (C,H,W,D).

### CELL 34 — compute internal ranking weights (per model per lesion)
- For each model, compute lesion-wise Dice on held-out folds.
- Convert to weight: `w_i = (1 / rank_i) / sum_j (1 / rank_j)` or normalized inverse-error.

### CELL 35 — fusion (voxel-wise)
- Weighted average of probabilities per class:
```python
prob_fused = sum(w_model * prob_model for model) / sum(w_model)
label = prob_fused.argmax(0)
```
- Optional region-wise overrides: e.g., apply higher weight to HFF for ET channel.

### CELL 36 — Post-fusion checks
- Visualize label overlays
- Compute Dice per region and per-case distributions

---

# POST-PROCESSING (cells 37–42)

## 1) Radiomic cluster aware CC filtering
- For each test case, compute its PCA radiomic signature, assign cluster
- For that cluster, use precomputed optimal CC size threshold (grid searched on CV) to remove tiny/large CCs adaptively.

## 2) T1CE/T1 ratio re-labeling (physiology-based)
- After fusion, compute `r = zscore(T1CE)/zscore(T1)` for brain voxels.
- Apply deterministic rules (from Li et al.):
  - If predicted NET but `r > 1.388` ⇒ relabel → ET
  - If predicted ET but `r < 0.766` ⇒ relabel → NET
- Thresholds must be tuned on your validation set (start with above values).

**Important**: Apply ratio check only within WT mask to avoid false conversions outside tumor.

## 3) CC & ET small component pruning
- Per-cluster threshold search (0:500 voxels step 25) done offline on CV: pick values maximizing internal CV metric.
- Remove components smaller than cluster threshold for ET.

## 4) Anatomical constraints
- Enforce: ET ⊆ TC ⊆ WT. If not, set union/intersection to ensure consistency.

---

# EVALUATION & VISUALIZATION (cells 43–47)

### CELL 43 — compute Dice metrics
- For each case and ensemble, compute Dice(WT,TC,ET).
- Report mean ± std across validation/test.

### CELL 44 — per-case reports & failure cases
- Identify bottom 10% cases by overall Dice and produce visualization grid (slices + overlays).

### CELL 45 — lesion-wise NSD, Hausdorff95 (optional)
- Compute boundary metrics to quantify boundary quality.

### CELL 46 — ablation experiments
- Ablation checklist (each as separate run):
  1. baseline nnU-Net only
  2. + γ-scale nnU-Net variants
  3. + Swin UNETR
  4. + HFF-Net
  5. + Radiomic stratification
  6. + Ratio-based postprocessing
  7. full ensemble + postprocessing

Record delta in each run.

---

# DEBUGGING CHECKLIST (run automatically after each major stage)
- Data: Are shapes, spacing, orientations consistent?
- Norms: Post-norm brain mean ~0 and std ~1?
- Masks: Skull-strip preserved >99% GT volumes?
- Aug: Visualized and realistic?
- Training: Loss curves decreasing; val Dice improving?
- ET: Is per-batch ET voxels present? If many batches have zero ET → increase batch size or sample patches centered on tumor.
- Ensemble: Do model errors decorrelate? (Compute pairwise IoU between model predictions; diversity >0.2 recommended)
- Postprocess: Are deterministic ratio thresholds harming recall? Run on few cases and visually inspect.

---

# HARDWARE & RUNTIME TIPS (H100 specifics)
- Use mixed precision (AMP) to accelerate.
- Use `nnUNetv2_plan_experiment` override to enlarge patch and batch sizes for H100.
- For heavy models (Swin UNETR), use gradient accumulation if batch > memory allows.
- Use distributed training across multiple H100s if available to speed up cross-validation folds.

---

# SENSIBLE DEFAULTS (starting hyperparameters)
- nnU-Net variants: epochs 1000, batch 4–8 (on H100), LR 1e-2 with poly decay, momentum 0.99
- Swin UNETR: epochs 800–1000, batch 2–4, LR 1e-4 to 5e-5 (AdamW)
- HFF-Net: epochs 450–800, batch 1–2 (or higher on H100), LR 1e-3 (AdamW)
- Loss: Dice + CE + Focal (ET weight = 2.0 initial)
- Post-process ratio thresholds: start with r_low=0.766, r_high=1.388 (tune)

---

# FINAL EXPECTED GAINS
- ET Dice: +0.05–0.12 (via HFF + ratio postprocess + ET upweight)
- TC Dice: +0.06–0.10 (via ResEnc + radiomic folds)
- WT Dice: +0.02–0.05 (ensemble stabilizes)
- Overall: push past 0.90.

---

# REQUIRED ARTIFACTS (what agents must produce & save)
- Preprocessed dataset folder
- Radiomic cluster JSON + fold split JSON
- Trained checkpoints per model/fold
- Softmax probability files per model/fold (.npz)
- Ensemble fusion scripts and saved final masks
- Evaluation CSV with per-case metrics
- Visualizations: slices for best/worst/median cases

---

# REPRODUCIBILITY
- Save all hyperparameters (YAML/JSON).
- Save git commit hash for all code.
- Seed randomness (torch, numpy, random).
- Document environment: python, pytorch, cuda, cudnn versions.

---

# SHORTCELL/TODO for agents (executable checklist)
1. Run CELL 0 → sanity checks
2. Run preprocessing cells 1–6 → confirm masks and normalization
3. Extract radiomics + create stratified folds
4. Implement augmentation cell 7 and visualize
5. Run nnU-Net planner override and implement γ-init runs (cells 8–14)
6. Implement/prepare Swin UNETR pretrain & finetune (cells 15–17)
7. Implement HFF-Net pipeline (cells 21–23) and validate HF/LF outputs
8. Train SegResNet optional
9. Save softmax maps, compute model ranking and weights
10. Fuse, postprocess (ratio + cluster-aware CC), evaluate
11. Ablation runs & produce final report

---

## CITATION & REFERENCES
This combined plan integrates your pipeline analysis and the top-papers recommendations: fileciteturn1file0

---

**END OF PLAYBOOK — start with CELL 0 and paste outputs back to me.**
