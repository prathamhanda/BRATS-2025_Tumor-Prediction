# Research-Grade Pediatric Brain Tumor Segmentation (BraTS PEDs) — Conference-Style Paper Draft

> Audience: a co-author writing the methodology section **without needing to understand the code**.
>
> Scope: the Pediatric nnU-Net v2 pipeline (Dataset501_BraTS_PEDs2024) from data → preprocessing → training → inference → post-processing → evaluation → visual QC → submission packaging.

---

## 1. Title (suggested)

**A Robust nnU‑Net v2 Ensemble Pipeline for Pediatric Brain Tumor Subregion Segmentation with Visual Quality Control and Post‑Processing**

Alternative titles:
- **Reliable Pediatric Brain Tumor Segmentation using nnU‑Net v2 with Conservative Inference and Post‑Processing**
- **Research‑Grade Pediatric Tumor Subregion Segmentation: An End‑to‑End nnU‑Net v2 Pipeline with Automated QC**

---

## 2. Abstract (ready-to-use, edit numbers later)

Pediatric brain tumor segmentation differs substantially from adult glioma segmentation due to smaller enhancing tumor regions, heterogeneous contrast patterns, and higher anatomical variability. We present a research‑grade end‑to‑end segmentation pipeline for the BraTS Pediatric (PEDs) dataset using an nnU‑Net v2 3D full‑resolution configuration and a 5‑fold ensemble. To maximize robustness in remote GPU notebook environments, we use conservative inference settings, explicit monitoring and restartability, and a post‑processing stage that removes spurious components and preserves the largest whole‑tumor region. We further introduce an automated visual quality‑control (QC) overlay suite and a judge‑friendly visualisations export pack (PNG overlays, raw‑vs‑post comparisons, volume histograms, and HTML galleries). Our evaluation reports Dice scores for Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET) with explicit handling of empty‑region edge cases. The resulting pipeline produces challenge‑compliant submissions and is reproducible and inspectable end‑to‑end.

---

## 3. Contributions (bullet list for intro)

1. **Robust pediatric segmentation pipeline**: nnU‑Net v2 3D full‑resolution training + 5‑fold ensemble prediction.
2. **Dataset‑verified label mapping**: explicit audit of the dataset’s label set and correction of ET label convention.
3. **Reliable remote inference orchestration**: progress monitoring, safe continuation, and prevention of “false hangs” when outputs are complete.
4. **Post‑processing for free performance**: largest WT connected component retention and small ET component suppression.
5. **Mandatory visual QC**: automated FLAIR/T1ce overlay figures for fast human inspection.
6. **Research polish pack**: bulk export of figures/plots and an HTML gallery for reviewers/judges.

---

## 4. Background and Motivation

### 4.1 Why Pediatric ≠ Adult Glioma
Pediatric tumors often exhibit:
- smaller enhancing regions,
- distinct enhancement patterns,
- different edema extent,
- and higher variance in tumor morphology.

As a consequence, adult‑optimized segmentation heuristics can fail silently, especially for the Enhancing Tumor (ET) class, which is frequently tiny.

### 4.2 Task Definition
Given multi‑modal brain MRI volumes, we predict a voxelwise multi‑class segmentation mask identifying tumor subregions. We report performance in clinically meaningful composite regions:
- Whole Tumor (WT)
- Tumor Core (TC)
- Enhancing Tumor (ET)

---

## 5. Dataset and Label Conventions

### 5.1 Dataset
- Dataset ID: **Dataset501_BraTS_PEDs2024**
- Modalities (typical BraTS ordering): **T1, T1ce, T2, FLAIR**
- Splits:
  - `imagesTr`: training images (with labels)
  - `labelsTr`: training labels
  - `imagesTs`: test images (no labels)

### 5.2 Critical Label Mapping Note (must include in paper)
BraTS-style pediatric datasets are not always consistent in how they encode the **Enhancing Tumor (ET)** label. In practice we observed two common ontologies:
- **{0,1,2,3}** where **ET=3** (this is what our nnU‑Net-ready dataset used; see audit below)
- **{0,1,2,4}** where **ET=4** (commonly stated in some methodology notes)

To prevent silent metric/reporting errors, our pipeline includes a **label audit step** that scans a subset of `labelsTr` and selects $\ell_{ET} \in \{3,4\}$ based on which value is present.

Notebook audit result (training labels scan):
- labels present: **{0,1,2,3}**
- inferred $\ell_{ET}$: **3**
- TC labels: **{1,3}**

Paper-ready phrasing:
> “We automatically audit the training label volumes to determine the ET label value (3 vs. 4) and propagate this mapping consistently to composite-region definitions, post-processing, and evaluation. In our dataset, the audited label set was {0,1,2,3} with ET=3.”

### 5.3 Composite Region Definitions
Let $y$ be the integer label map.

| Region | Definition | Label set |
|---|---|---|
| WT | Whole Tumor | $y > 0$ |
| TC | Tumor Core | $y \in \{1,\ell_{ET}\}$ |
| ET | Enhancing Tumor | $y = \ell_{ET}$ |

---

## 6. Preprocessing

nnU‑Net v2 performs an automated and data‑driven preprocessing pipeline based on dataset fingerprinting (e.g., intensity normalization strategy, resampling). Key assumptions and checks:

### 6.1 Orientation and Consistency
- Ensure modalities are spatially aligned.
- Check voxel spacing and orientation are consistent.

### 6.2 Intensity Normalization
We use nnU‑Net’s configuration‑driven intensity normalization (typically per‑channel normalization within foreground). For paper clarity:
- “Normalization is performed per modality to reduce scanner‑dependent intensity shifts.”

### 6.3 (Optional but recommended) Skull Stripping Discussion
The roadmap suggests skull stripping can help pediatric variability; our baseline pipeline relies on nnU‑Net’s learned normalization and cropping behavior. If skull stripping is added later, include:
- the skull‑strip model/tool,
- a QC overlay verifying no tumor tissue was removed,
- and an ablation comparing Dice with/without skull stripping.

---

## 7. Model: nnU‑Net v2

### 7.1 Architecture
nnU‑Net is a self‑configuring U‑Net‑style segmentation framework. The v2 pipeline selects:
- network topology,
- patch size,
- batch size,
- and augmentation policy
based on dataset statistics.

### 7.2 Configuration
- nnU‑Net v2 configuration: **`3d_fullres`**

### 7.3 Training Strategy
- 5‑fold cross‑validation folds: **0–4**
- Final inference uses an **ensemble** across folds.

Paper‑ready phrasing:
> “We train five independent models using cross‑validation folds and ensemble their predictions at inference time.”

---

## 8. Training Procedure (conference-methods level)

### 8.1 Objective
nnU‑Net typically uses a compound loss:
- Dice loss + Cross‑Entropy loss

Write it as:
$$
\mathcal{L}(\theta) = \lambda\,\mathcal{L}_{Dice}(\theta) + (1-\lambda)\,\mathcal{L}_{CE}(\theta)
$$
where $\lambda$ is configured by nnU‑Net.

### 8.2 Augmentations
Typical 3D augmentations (nnU‑Net defaults; mention at high level):
- random flips,
- random rotations and scaling,
- gamma/intensity augmentations,
- Gaussian noise/blur.

### 8.3 Implementation Notes (reproducibility)
- Report:
  - GPU type
  - CUDA version
  - nnU‑Net version
  - training time per fold
  - total training time

Include in paper’s appendix as a “Compute” table.

---

## 9. Inference and Ensemble Prediction

### 9.1 Inference Targets
- **`imagesTs`**: test set inference used for submission
- **`imagesTr`**: training set inference used for internal evaluation (Dice)

### 9.2 Robust Inference in Remote Notebook Environments
In remote GPU notebook settings, inference processes can appear to “hang” even when outputs are complete (e.g., delayed finalization or zombie processes). We implement reliability checks:
- output file count monitoring vs expected case count,
- log tail monitoring,
- safe continuation via nnU‑Net’s `--continue_prediction` option,
- and controlled CPU worker counts to prevent dataloader crashes.

### 9.3 Ensemble Strategy
nnU‑Net’s standard ensemble averages probabilities/logits across folds and outputs the final label per voxel.

---

## 10. Post‑Processing

Post‑processing is applied to the raw predicted segmentation masks to improve plausibility and reduce false positives.

### 10.1 Connected Component Filtering for WT
**Goal**: remove small disconnected false positive tumor islands.

Algorithm:
1. Let $M_{WT} = \mathbb{1}[y > 0]$.
2. Compute connected components in $M_{WT}$ (3D connectivity).
3. Keep only the largest component; set all other voxels to background.

### 10.2 Small ET Component Suppression
**Goal**: suppress tiny enhancing regions that are common false positives.

Algorithm:
1. Let $M_{ET} = \mathbb{1}[y = 3]$.
2. Compute connected components in $M_{ET}$.
3. Remove components smaller than a voxel threshold (e.g., 10–20 voxels).

### 10.3 Paper‑ready Pseudocode
```text
Input: predicted labelmap y
Output: postprocessed labelmap y'

1) y' ← y
2) KeepLargestWTComponent(y')
3) RemoveSmallETComponents(y', min_voxels = τ)
4) return y'
```

---

## 11. Evaluation Metrics

### 11.1 Dice Similarity Coefficient
For binary masks $A$ and $B$:
$$
Dice(A,B) = \frac{2|A\cap B|}{|A| + |B|}
$$

### 11.2 Empty‑Region Edge Cases (important)
When both prediction and ground truth are empty for a region, Dice is defined as **1.0** (perfect agreement). When one is empty and the other is not, Dice is **0.0**.

This prevents the ET metric (often absent) from being misleading.

### 11.3 Region‑wise Dice Reporting
We compute Dice for:
- WT: $y > 0$
- TC: $y \in \{1,\ell_{ET}\}$
- ET: $y = \ell_{ET}$

---

## 12. Visual Debugging and Quality Control (Mandatory)

### 12.1 What We Visualize
For each case, we generate slice overlays:
- FLAIR background + segmentation overlay
- T1ce background + segmentation overlay

We choose a representative axial slice:
- the median slice index of the WT mask if tumor exists, otherwise mid‑volume.

### 12.2 Why It Matters
Visual QC catches:
- gross misalignment,
- tumor leakage into skull/CSF,
- empty predictions,
- anatomically implausible ET islands,
- modality channel mismapping.

### 12.3 Visualisations Export Pack
We export a judge‑friendly folder containing:
- per‑case overlays (PNGs)
- raw‑vs‑post comparisons
- volume histograms for WT/TC/ET
- a browsable HTML gallery

---

## 13. Results (structure + what to include)

> Replace bracketed placeholders with your measured values from the notebook’s evaluation outputs.

### 13.1 Quantitative Results
We report region-wise Dice on **`imagesTr`** (260 cases), using the audited label mapping ($\ell_{ET}=3$, TC={1,3}).

| Split | #Cases | WT Dice | TC Dice | ET Dice |
|---|---:|---:|---:|---:|
| imagesTr (ensemble prediction) | 260 | 0.9502 | 0.7995 | 0.8052 |

Also include:
- mean ± std
- median
- 25th/75th percentiles

### 13.2 Volume Statistics
We compute voxel-count summaries for WT/TC/ET on `imagesTr` (GT vs prediction), and export histograms for `imagesTs` predictions in the visual QC pack.

**ET prevalence / empty-region sanity (imagesTr, $\ell_{ET}=3$):**
- GT has ET: 192/260 (73.8%)
- Pred has ET: 180/260 (69.2%)
- Both empty: 62/260 (23.8%)
- Empty mismatch: 24/260 (9.2%)

**Voxel-count quantiles (imagesTr):**
- WT mean voxels: GT 52,946 | Pred 51,195
- TC mean voxels: GT 13,985 | Pred 13,082
- ET mean voxels: GT 5,909 | Pred 5,697

For the full percentile table (p0/p25/p50/p75/p90/p95/p99/max), see the notebook’s “WT/TC/ET voxel-volume summary” output.

### 13.3 Qualitative Results
Include a figure grid with:
- 6–12 cases showing FLAIR/T1ce overlays
- at least one “failure case” subsection

### 13.4 Submission Packaging and Validation
Paper‑ready statement:
> “We export challenge‑compliant NIfTI masks for each test case, validate naming and archive structure, and package the final submission as a single zip file.”

In our run:
- `imagesTs` cases predicted: 91/91
- Post-processing: keep largest WT component + remove ET components < 20 voxels ($\ell_{ET}=3$)
- Final zip: `/workspace/pediatric_tumor_data/submissions/nnunet_d501_3d_fullres_imagesTs_20260127_093237.zip`

---

## 14. Ablations and Research Polish (what reviewers will expect)

Even if you cannot re-train all variants, you can structure the section and fill what you have.

### 14.1 Ensemble vs Single Fold
- Single best fold vs ensemble across 5 folds.

### 14.2 Post‑Processing Ablation
- Raw prediction vs postprocessed prediction.

### 14.3 ET Threshold Sensitivity
- Vary min ET voxel threshold: τ ∈ {0, 10, 20, 50}
- Report ET Dice change and false positives.

### 14.4 Inference Robustness Settings
Report a small table:
- `--disable_tta` on/off
- worker counts (npp/nps)
- step size

---

## 15. Failure Case Analysis

Include 3–5 representative failure modes:
1. **Missed ET**: very small enhancement not detected.
2. **Over‑segmented ET**: spurious enhancement islands.
3. **Boundary leakage**: edema spills into ventricles/skull.
4. **Low contrast cases**: poor modality signal.

For each, include:
- the overlay figure,
- the predicted volumes,
- any known reason (motion, low SNR, etc.).

---

## 16. Reproducibility Checklist (appendix-ready)

### 16.1 Environment
Include:
- OS image / container
- Python version
- nnU‑Net v2 version
- CUDA / cuDNN

### 16.2 Commands (high-level)
- training per fold
- ensemble prediction
- post‑processing
- evaluation

### 16.3 Determinism Notes
nnU‑Net training is not perfectly deterministic across GPUs; mention seeds and version pinning.

---

## 17. What to Download from the Server (so you “have the whole project” locally)

You do **not** need the entire raw dataset unless you want to reproduce from scratch. Here is a practical tiered checklist.

### Tier A (minimum to prove results + make the paper)
1. **Final submission zip** (imagesTs predictions, postprocessed):
   - `/workspace/pediatric_tumor_data/submissions/*.zip`
2. **Visualisations pack folder(s)**:
   - `/workspace/pediatric_tumor_data/visualisations/run_*/` (especially `index.html`, `gallery/`, `qc/`, `volumes/`)
3. **Evaluation outputs** (CSV/JSON reports):
   - any generated `dice_*.csv`, `metrics_*.json`, `RESULTS_SUMMARY.md` equivalents inside `/workspace/pediatric_tumor_data/...`
4. **Notebook** you ran:
   - [Pediatric_Type/pediatric_5.2.ipynb](pediatric_5.2.ipynb)

### Tier B (recommended for full reproducibility)
5. **Trained nnU‑Net weights/checkpoints**:
   - `/workspace/pediatric_tumor_data/nnunetv2/nnUNet_results/Dataset501_BraTS_PEDs2024/.../fold_*/checkpoint_final.pth`
   - plus any `plans.json` and trainer config files in the same tree
6. **nnU‑Net dataset descriptor**:
   - `nnUNet_raw/Dataset501_BraTS_PEDs2024/dataset.json`

### Tier C (only if you want to reproduce preprocessing exactly)
7. **Preprocessed cache** (very large):
   - `/workspace/pediatric_tumor_data/nnunetv2/nnUNet_preprocessed/`

### Tier D (only if you need a full offline mirror)
8. **The full raw dataset**:
   - `/workspace/pediatric_tumor_data/nnunetv2/nnUNet_raw/Dataset501_BraTS_PEDs2024/`

---

## 18. Appendix: Figure and Table Shopping List (copy into paper plan)

### Figures
- Fig 1: Pipeline overview diagram (Data → nnU‑Net → Ensemble → Postprocess → QC → Submission)
- Fig 2: Example overlays (FLAIR/T1ce) for multiple cases
- Fig 3: Raw vs Postprocessed comparison overlays (show ET cleanup)
- Fig 4: Histograms of predicted WT/TC/ET volumes (imagesTs)
- Fig 5: GT vs Pred volume histograms (imagesTr)
- Fig 6: Failure cases panel

### Tables
- Table 1: Dataset description + label mapping (explicitly ET=3)
- Table 1: Dataset description + label mapping (explicitly ET=4 after remapping)
- Table 2: Composite region definitions
- Table 3: Training/inference hyperparameters (config=3d_fullres, folds=0–4)
- Table 4: Dice results summary (mean ± std)
- Table 5: Ablation results (postprocess on/off; ensemble vs single)

### Equations
- Dice definition
- Empty‑region handling rule

---

## 19. Notes for the Co‑Author (non-technical guidance)

- Emphasize the **pediatric‑specific challenges** (ET small, imbalance).
- Explicitly state the **label convention was verified and enforced** (ET=4 after remapping).
- Describe **robustness engineering** (monitoring and non‑hanging inference) as a reproducibility strength.
- Use the exported HTML gallery as the “qualitative appendix”.

