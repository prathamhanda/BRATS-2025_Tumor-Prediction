# Pediatric Brain Tumor Segmentation (BraTS PEDs) — Methodology (nnU-Net v2 Baseline)

This document describes the **exact end-to-end methodology we followed** to build our current pediatric tumor segmentation baseline in this repo (the “Pediatric tumor” workstream). It is written to be **paper- and supervisor-ready**: assumptions, pipeline theory, concrete implementation details, and the operational steps we used (including the disconnect-safe training pattern).

**Primary implementation notebook:** [Pediatric_Type/pediatric_5.2_publish.ipynb](Pediatric_Type/pediatric_5.2_publish.ipynb)

**Extended research roadmap / future upgrades:** [Pediatric_Type/Pediatric_BraTS_Research_Grade_Segmentation_Roadmap.md](Pediatric_Type/Pediatric_BraTS_Research_Grade_Segmentation_Roadmap.md)

---

## 1) Problem statement

Given multi-modal brain MRI volumes, we perform **voxel-wise semantic segmentation** of pediatric brain tumor subregions, following BraTS PEDs conventions.

### Input modalities (4 channels)
- T1
- T1ce (post-contrast)
- T2
- FLAIR

### Label ontology (BraTS PEDs raw)
BraTS-style labels on disk are typically:
- **0**: background
- **1**: necrotic / non-enhancing tumor core (NET)
- **2**: edema
- **4**: enhancing tumor (ET)

### Derived regions used in reporting
In BraTS-style reporting, three composite regions are common:
- **WT (Whole Tumor)** = NET ∪ Edema ∪ ET
- **TC (Tumor Core)** = NET ∪ ET
- **ET (Enhancing Tumor)** = ET only

In label-id notation (raw):
- WT = {1,2,4}
- TC = {1,4}
- ET = {4}

---

## 2) Repository scope and “what we have today”

### What is implemented and used (baseline)
- nnU-Net v2 **3D full-resolution** baseline training
- 5-fold cross-validation (folds **0–4**)
- Conversion/export of BraTS PEDs folder structure into **nnU-Net v2 raw dataset format**
- Two critical hygiene fixes that were required in practice:
  1) **Consecutive label remap** for nnU-Net compatibility (ET label **4 → 3**)
  2) Repair of incorrectly-named “`.nii.gz`” files that were **not actually gzip** (SimpleITK read failures)
- **Disconnect-safe training orchestration** using background runners (`nohup`) + on-disk logs and checkpoints

### What is not included in this repo (but exists on the training server)
- Full nnU-Net results directories (trained weights/checkpoints) are stored in `$nnUNet_results` on the server, not committed here.

---

## 3) End-to-end pipeline (high-level flow)

```mermaid
flowchart TD
  A[Raw BraTS PEDs dataset\n(Training + optional Validation folder)] --> B[Dataset integrity checks\nmodalities present, alignment sanity]
  B --> C[Export to nnU-Net v2 raw format\nimagesTr/labelsTr/imagesTs]
  C --> D[Label standardization\nET label 4→3 so labels are consecutive]
  D --> E[Fix malformed NIfTI gzip\nfor any *.nii.gz not actually gzip]
  E --> F[nnU-Net v2 plan & preprocess\ncreate fingerprint, plans, splits, preprocessed data]
  F --> G[5-fold CV training\nfolds 0..4, 3d_fullres]
  G --> H[Progress monitoring\nparse training logs + checkpoints]
  H --> I[Inference / prediction\noptionally ensemble folds]
  I --> J[Post-processing + evaluation\nDice WT/TC/ET, QC overlays]
```

---

## 4) Data layout, dataset IDs, and “numbers involved”

### Dataset identity used for nnU-Net
- **Dataset ID:** `501`
- **Dataset name:** `BraTS_PEDs2024`
- **nnU-Net dataset folder:** `Dataset501_BraTS_PEDs2024`

### Concrete case counts observed in our run
From the recorded notebook outputs (server mount):
- **Training cases folder count:** **261**
- **Validation cases folder count:** **91** (when validation folder existed)

Notes:
- Some BraTS releases provide a “validation” folder without labels or with different naming. The notebook includes logic to handle missing/withheld validation labels by relying on nnU-Net’s internal CV split file (`splits_final.json`) generated during preprocessing.

### Folds
- **Cross-validation folds:** 5
- **Fold indices:** 0, 1, 2, 3, 4

### Modalities
- **# modalities used:** 4 (T1, T1ce, T2, FLAIR)

### Reproducibility seed
- Notebook seed: **1337**

---

## 5) nnU-Net v2 raw dataset format (what we created)

nnU-Net v2 expects a standardized “raw” folder structure:

```
$nnUNet_raw/
  Dataset501_BraTS_PEDs2024/
    imagesTr/
      <case>_0000.nii.gz   # T1
      <case>_0001.nii.gz   # T1ce
      <case>_0002.nii.gz   # T2
      <case>_0003.nii.gz   # FLAIR
    labelsTr/
      <case>.nii.gz        # segmentation mask
    imagesTs/              # optional test/validation images without labels
```

### Channel ordering (critical)
We use nnU-Net’s conventional channel indices:
- `_0000` = T1
- `_0001` = T1ce
- `_0002` = T2
- `_0003` = FLAIR

Any swap here will silently destroy performance, so the notebook validates modality naming before export.

---

## 6) Label standardization and why ET 4→3 is required

### Why this is necessary
nnU-Net internally assumes **foreground labels are consecutive integers** (e.g., 0/1/2/3). BraTS uses label **4** for ET, creating a non-consecutive set {0,1,2,4}. Some tooling will work, but many components (and third-party readers/metrics) become brittle.

### Our fix
We remap:
- 0 → 0
- 1 → 1
- 2 → 2
- 4 → 3

After remap, the foreground set is {1,2,3}.

### Mapping table
| Region | BraTS raw label | nnU-Net training label |
|---|---:|---:|
| Background | 0 | 0 |
| NET | 1 | 1 |
| Edema | 2 | 2 |
| ET | 4 | 3 |

### Reporting metrics still use BraTS regions
Even after remap, we still report WT/TC/ET conceptually; the “ET” channel becomes label **3**.

---

## 7) NIfTI IO robustness: fixing “.nii.gz that isn’t gzip”

### Failure mode
We encountered at least one file named `*.nii.gz` that was **not actually gzipped**, which breaks SimpleITK/Nibabel reads and can stall preprocessing.

Example observed:
- `.../imagesTr/BraTS-PED-00255-000_0001.nii.gz`

### Our fix
The notebook includes a safety check:
- If a file ends with `.nii.gz` but does not have a gzip header, we **rename** it to `.nii` (or otherwise repair handling) so readers treat it correctly.

This is an operational “data hygiene” step; it does not change voxel values.

---

## 8) Preprocessing theory (what nnU-Net v2 does and why)

nnU-Net is designed to self-configure based on dataset statistics (“planning”), then generate preprocessed caches used during training.

### 8.1 Fingerprinting
During `plan_and_preprocess`, nnU-Net scans cases to compute:
- shapes / spacings per modality
- intensity statistics
- class frequencies

This produces metadata files such as:
- `dataset_fingerprint.json`
- `plans.json` / `nnUNetPlans.json`
- `splits_final.json` (CV folds)

### 8.3 Planned hyperparameters (numbers from our run)
From the notebook’s recorded `nnUNetPlans.json` summary for **`3d_fullres`**, nnU-Net selected:

| Item | Value (observed) |
|---|---|
| Target spacing | (1.0, 1.0, 1.0) |
| Patch size | (96, 160, 160) |
| Batch size | 2 |
| Normalization | Z-score per modality (all 4 channels) |

Interpretation:
- Patch size and batch size are the main memory drivers; nnU-Net chooses them to fit GPU memory while preserving 3D context.
- Target spacing defines the resolution used for training caches; this strongly affects ET visibility.

### 8.2 Resampling and normalization
nnU-Net automatically selects a target spacing and performs:
- resampling to the chosen spacing
- intensity normalization (typically z-scoring per case, modality-dependent)

Why it matters for pediatrics:
- Pediatric anatomy has larger variance and smaller enhancing regions; normalization and augmentation stability are major determinants of ET Dice.

---

## 9) Training theory (nnU-Net v2 baseline)

### Model family
nnU-Net v2 uses a 3D U-Net-style architecture with:
- encoder–decoder with skip connections
- deep supervision (auxiliary outputs at multiple scales)

### Objective (baseline)
The canonical nnU-Net baseline optimizes a hybrid loss:
$$\mathcal{L} = \mathcal{L}_{\text{Dice}} + \mathcal{L}_{\text{CE}}$$

Intuition:
- Dice improves overlap for imbalanced segmentation
- CE stabilizes voxel-wise classification and gradients

### Augmentations
nnU-Net applies strong spatial + intensity augmentation by default (flips/rotations/scales, gamma, noise/blur, etc.). This is important for generalization.

### Training schedule (numbers observed from logs)
From the fold log tail captured in the notebook:
- Training runs to **~1000 epochs** (we observed epochs 993–999 near completion).
- Near the end of training, the learning rate decayed to approximately **$2\times10^{-5}$**.
- Observed epoch time near the end was ~**55–58 seconds/epoch** (hardware-dependent).

---

## 10) Commands actually used (baseline)

### Environment variables
We used nnU-Net’s standard environment variables:
- `nnUNet_raw`
- `nnUNet_preprocessed`
- `nnUNet_results`

These are set in the notebook before preprocessing/training.

### Planning + preprocessing
Command (config-gated in the notebook for compatibility across nnU-Net builds):

- `nnUNetv2_plan_and_preprocess -d 501 -c 3d_fullres`

Notes:
- Some environments accept `-c` for config; the notebook detects help text to avoid CLI mismatch.

### Training (5 folds)
Canonical command per fold:

- `nnUNetv2_train 501 3d_fullres <FOLD> <RESUME_FLAG_IF_NEEDED>`

Where `<FOLD>` ∈ {0,1,2,3,4}.

Resume-flag nuance (important):
- Our environment required a nonstandard resume switch (`--c`), so the notebook **auto-detects** which resume flag is supported by parsing `nnUNetv2_train -h`.

---

## 11) Disconnect-safe sequential training (how we ran it)

### Why it was needed
Jupyter sessions on remote GPU servers can disconnect/reset. A naive cell-running strategy can stop training mid-fold.

### Our solution
We launch a single background “runner” script that:
1) trains folds sequentially (0→4)
2) writes a per-fold log file
3) checks for existing checkpoints and resumes if available

```mermaid
flowchart TD
  S[Start runner] --> P{Is another nnUNetv2_train active?}
  P -- Yes --> X[Exit (avoid double training)]
  P -- No --> F0[Fold 0]
  F0 --> F1[Fold 1]
  F1 --> F2[Fold 2]
  F2 --> F3[Fold 3]
  F3 --> F4[Fold 4]
  F4 --> E[Done]

  subgraph Each Fold
    A1[Detect resume flag\n(--c / -c / --continue)] --> A2{checkpoint_latest.pth exists?}
    A2 -- Yes --> A3[Resume training]
    A2 -- No --> A4[Fresh training]
  end
```

### Artifact locations (server)
- Trainer root:
  - `$nnUNet_results/Dataset501_BraTS_PEDs2024/nnUNetTrainer__nnUNetPlans__3d_fullres/`
- Per-fold folders:
  - `fold_0/ ... fold_4/`
- Key checkpoints per fold:
  - `checkpoint_latest.pth`
  - `checkpoint_best.pth`
  - `checkpoint_final.pth`

---

## 12) Monitoring and “progress numbers” we used

We monitored training using two signal sources:
1) **trainer logs** (`training_log_*.txt`)
2) **checkpoint modification times** (`checkpoint_best.pth`, `checkpoint_latest.pth`, `checkpoint_final.pth`)

This avoids relying on live notebook state.

Typical files observed in results (example from fold 0):
- `training_log_2026_1_8_02_17_59.txt`
- `progress.png`
- `checkpoint_best.pth`

---

## 13) Evaluation methodology

### Metrics
Primary metrics are Dice scores per BraTS region:
- Dice(WT)
- Dice(TC)
- Dice(ET)

Optionally:
- HD95 (Hausdorff 95th percentile)

### Practical QC (qualitative)
For pediatrics, visual overlay QC is mandatory because ET can be extremely small:
- overlay segmentation on FLAIR and T1ce
- verify no large false-positive ET islands

### Expected baseline ranges (from literature/experience)
These are **target ranges** for a healthy pipeline (not guaranteed on first run):
- WT: 0.85+
- TC: 0.75+
- ET: 0.65+

For a stronger, research-grade pipeline see the roadmap document.

---

## 14) Known failure modes and how we handled them

```mermaid
flowchart TD
  A[Training looks stuck] --> B{Is nnUNetv2_train process running?}
  B -- No --> C[Inspect last log lines\nlikely crash (workers/OOM)]
  B -- Yes --> D{Log timestamp still updating?}
  D -- Yes --> E[Not stuck\nwait / monitor]
  D -- No --> F[Likely deadlock or dataloader issue]
  C --> G[Mitigation: reduce DA workers\nset nnUNet_n_proc_DA lower]
  F --> G
  G --> H[Resume with detected resume flag\n(--c / -c / --continue)]
```

Key issues we explicitly solved in this project:
- **Augmentation worker crashes / deadlocks:** mitigated by lowering data augmentation worker count (`nnUNet_n_proc_DA`).
- **Resume flag mismatch:** some builds reject `-c` and require `--c`; we detect it automatically.
- **Bad `.nii.gz` files:** rename/repair to make IO robust.

---

## 15) How to reproduce (minimal)

1) Run [Pediatric_Type/pediatric_5.2_publish.ipynb](Pediatric_Type/pediatric_5.2_publish.ipynb)
2) Verify:
   - dataset is detected
   - case counts are printed
   - nnU-Net CLIs are found
3) Run export → label fix → gzip fix
4) Run planning/preprocess
5) Launch sequential fold training in background
6) Use the dashboard cell to monitor progress

---

## 16) Suggested “paper-ready” reporting checklist

For the final paper/report, we recommend capturing these exact items (the notebook already prints most of them):
- Dataset version + counts (train/val), modalities
- Label ontology + our remap 4→3
- nnU-Net version + PyTorch version + GPU model
- nnU-Net config (`3d_fullres`) and folds (0–4)
- Mean±std Dice for WT/TC/ET across folds
- Failure case analysis (especially ET)

If you want, I can also add a small script to parse the nnU-Net results folder on the training server and automatically generate a “Results Table” (mean/std Dice per region across folds) for this README.
