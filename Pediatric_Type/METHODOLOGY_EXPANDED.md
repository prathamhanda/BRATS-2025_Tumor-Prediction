# Methods 

This section documents the end-to-end pediatric brain tumor subregion segmentation pipeline implemented in the accompanying notebook. It is written to be reproducible and to prevent silent errors caused by label-ontology ambiguity and extreme class imbalance (especially for enhancing tumor).

## 3.1 Dataset and Task Definition

We evaluate on the BraTS-PED 2025 pediatric brain tumor segmentation dataset comprising **438** multi-parametric MRI (mpMRI) cases split into **260 training**, **91 validation**, and **87 test** cases. Each case contains **four co-registered MRI sequences**: T1-weighted (**T1**), contrast-enhanced T1-weighted (**T1ce**), T2-weighted (**T2**), and T2-weighted fluid-attenuated inversion recovery (**FLAIR**). Images are skull-stripped and provided at **1.0 × 1.0 × 1.0 mm** isotropic voxel spacing.

The task is voxel-wise semantic segmentation of tumor tissue. Performance is reported on BraTS composite regions:

- **Whole Tumor (WT)**: union of all tumor and edema tissue.
- **Tumor Core (TC)**: necrotic/non-enhancing tumor core plus enhancing tumor.
- **Enhancing Tumor (ET)**: enhancing components only.

Pediatric segmentation is substantially affected by class imbalance and inter-patient heterogeneity: enhancing tumor may be **absent in 26.2%** of cases (equivalently, present in **73.8%**), and when present it is often very small (frequently **< 1,000 voxels**), which increases susceptibility to false negatives and boundary noise.

## 3.2 Preprocessing and Data Integrity Checks

Although nnU-Net v2 performs automated preprocessing, we explicitly enforce conservative, pediatric-safe checks to prevent silent failures.

### 3.2.1 Spatial consistency and orientation

For each case, we verify modality consistency and alignment:

- consistent shapes (or expected resampling behavior),
- consistent affine transforms across modalities,
- canonical orientation (RAS) for reliability of downstream geometric measurements.

If required, volumes are reoriented to RAS and resampled to ensure consistent geometry. Final resampling targets **1 mm isotropic** spacing to maintain interpretability of distance-based metrics.

### 3.2.2 Brain mask construction and validation

We compute a conservative brain mask as the union of non-zero voxels across modalities:

$$
M_{brain} = \bigvee_{c \in \{T1,T1ce,T2,FLAIR\}} \mathbb{1}[x_c \neq 0].
$$

This mask is optionally refined with light morphological cleanup (hole filling and removal of tiny disconnected components), while prioritizing tumor preservation.

A critical validation step ensures the brain mask does not exclude tumor tissue. We quantify the fraction of labeled tumor voxels outside the mask and require **> 99%** tumor preservation. If violated, preprocessing is treated as failed and debugged before training/evaluation proceeds.

### 3.2.3 Intensity normalization

Intensity normalization is performed per-subject, per-modality using z-score standardization inside the brain mask:

$$
\hat{x} = \frac{x - \mu_{mask}}{\sigma_{mask} + \epsilon},
$$

where $\mu_{mask}$ and $\sigma_{mask}$ are computed over voxels in $M_{brain}$, and $\epsilon$ is a small constant to prevent division by zero. Masked normalization reduces bias from large background regions and improves robustness when skull stripping is imperfect.

### 3.2.4 Data augmentation

To reduce overfitting on the 260-case training set, on-the-fly augmentation is applied during training via batchgenerators, including:

- random 3D rotations and elastic deformations,
- anisotropic scaling in the range **0.85× to 1.25×**,
- gamma intensity transforms and additive Gaussian noise.

## 3.3 Label Ontology Standardization and Automated Mapping

Multi-institution pediatric datasets may exhibit label-ontology ambiguity. To prevent silent semantic and metric errors, we implement two complementary steps:

### 3.3.1 ET label audit (index-level consistency)

We scan a subset of training labels to determine whether enhancing tissue is encoded using label **3** or **4**. In the current audit run, **50 cases** were scanned and the chosen ET label for evaluation was **3**. This audit result is propagated to composite region definitions and metric computation.

### 3.3.2 Intensity-based label mapping (semantic consistency)

When the dataset contains labels **{0,1,2,3,4}** with ambiguous tissue meaning, we infer a mapping into the BraTS semantic convention **{0,1,2,4}** using mpMRI intensity signatures.

After within-brain z-score normalization, we compute per-label mean intensities and define a contrast-enhancement signature:

$$
\Delta CE(\ell) = Z_{T1ce}(\ell) - Z_{T1}(\ell),
$$

where $Z_{mod}(\ell)$ denotes the mean z-scored intensity of voxels belonging to label $\ell$ for a modality.

We sample **N = 20** training volumes and derive a dataset-specific mapping using (i) $\Delta CE$ and (ii) volume priors:

- **ET**: label with highest median $\Delta CE$ and comparatively smaller volume (in our run: original label **1**, $\Delta CE = 0.508$),
- **Edema (ED)**: label with highest FLAIR intensity and typically larger volume (in our run: original label **4**, $Z_{FLAIR} = 2.588$),
- **Necrotic/non-enhancing core (NCR/NET)**: remaining tumor labels with low contrast enhancement (original labels **2** and **3**, grouped).

This yields the inferred mapping:

$$
\{0 \rightarrow 0,\; 1 \rightarrow ET(4),\; 4 \rightarrow ED(2),\; \{2,3\} \rightarrow NCR(1)\}.
$$

To satisfy nnU-Net’s requirement for consecutive label IDs, ET is converted from **4 → 3** internally prior to training, producing the internal label set **{0,1,2,3}** while maintaining BraTS semantic reporting in composite-region metrics.

## 3.4 Model Architecture (nnU-Net v2)

We use **nnU-Net v2** in the **3D full-resolution (3d_fullres)** configuration. nnU-Net automatically selects key design parameters (patch size, network depth, normalization, and augmentation policy) based on dataset fingerprinting.

The training objective follows the standard nnU-Net compound loss (conceptually):

$$
\mathcal{L}(\theta) = \lambda\,\mathcal{L}_{Dice}(\theta) + (1-\lambda)\,\mathcal{L}_{CE}(\theta),
$$

where $\mathcal{L}_{Dice}$ addresses class imbalance and $\mathcal{L}_{CE}$ stabilizes voxel-wise classification.

## 3.5 Training Protocol and Compute

Training is performed using **5-fold cross-validation (folds 0–4)**. The network is trained for **1,000 epochs** with **batch size = 2** under the 3d_fullres configuration.

Compute setup and timing (captured from execution logs):

- GPU: **NVIDIA H100**
- Training time: **~16 hours per fold**
- Inference time: **~20 seconds per case**

Cross-validation fold predictions are used to form the final inference outputs. Where multiple fold outputs are available, fold aggregation is applied following nnU-Net’s inference convention.

## 3.6 Post-processing (Anatomy-aware Filtering)

To reduce anatomically implausible false positives and stabilize tiny enhancing predictions, a conservative post-processing stage is applied.

### 3.6.1 Largest connected component for WT

Let $M_{WT} = \mathbb{1}[y > 0]$. We compute 3D connected components using **26-connectivity** and retain only the largest component. This suppresses disconnected false-positive tumor islands.

### 3.6.2 Small ET component suppression

Let $M_{ET} = \mathbb{1}[y = 3]$ in internal nnU-Net label space. We remove ET connected components smaller than a threshold $\tau$, set to **$\tau = 20$ voxels**, relabeling them as background (or TC depending on the chosen policy).

## 3.7 Tumor Parameter Extraction (Quantitative Biomarkers)

Beyond segmentation masks, we compute quantitative tumor descriptors from the predicted label map (and optionally from ground truth where available). Using voxel spacing metadata, voxel counts are converted to physical volumes:

$$
V_{cm^3} = \frac{N_{vox} \cdot (s_x s_y s_z)}{1000},
$$

where spacing is in mm and $1\,cm^3 = 1000\,mm^3$.

Extracted biomarkers include:

- **Volumes (cm³)**: WT, TC, ET (and edema/core components where defined),
- **Composition ratios**: ET/WT and TC/WT,
- **Maximum cross-sectional area**: largest axial WT area and corresponding slice index,
- **Centroids**: center-of-mass in voxel coordinates and physical (mm) coordinates,
- **Bounding box extents**: width × height × depth in mm,
- **Connected component counts**: number of WT and ET components as plausibility indicators.

Example quantitative output (representative case): WT volume **83.70 cm³**, TC **13.83 cm³**, ET **6.84 cm³**, ED **69.87 cm³**, with ET/WT **0.082** and TC/WT **0.165**.

## 3.8 Evaluation Metrics and Reporting

We evaluate segmentation accuracy using overlap and boundary metrics on the three BraTS composite regions.

### 3.8.1 Region definitions

- **WT**: $y > 0$
- **TC**: $y \in \{1,3\}$ in internal label space (corresponding to NCR/NET and ET)
- **ET**: $y = 3$ in internal label space (corresponding to BraTS ET = 4 before consecutive relabeling)

### 3.8.2 Dice Similarity Coefficient (DSC)

For binary masks $A$ and $B$:

$$
Dice(A,B) = \frac{2|A \cap B|}{|A| + |B|}.
$$

### 3.8.3 95th percentile Hausdorff Distance (HD95)

HD95 is computed in **millimeters** using voxel spacing and surface-to-surface distances to quantify boundary error while reducing sensitivity to extreme outliers.

### 3.8.4 Empty-region policy (critical for ET)

- If both prediction and ground truth are empty for a region: **Dice = 1.0**.
- If exactly one is empty: **Dice = 0.0**.

This policy prevents misleading averages in pediatric ET where absence is common.

### 3.8.5 Reported performance summary

We report the following reference values:

- **Training set (n = 260):** WT Dice **0.950**, TC Dice **0.800**, ET Dice **0.805**, Mean Dice **0.852**.
- **Labeled reference subset (n = 30):** WT Dice **0.882 ± 0.103**, TC Dice **0.542 ± 0.380**, ET Dice **0.590 ± 0.374**, with region-wise mean HD95 of **6.73 mm (WT)**, **59.07 mm (TC)**, **66.67 mm (ET)** and overall mean HD95 **44.16 mm**.

## 3.9 Automated Visual QC and 3D Model Export (Optional)

To support rapid inspection, we generate automated QC overlays (FLAIR/T1ce underlay with segmentation overlay) and export structured per-case summaries.

For physical interpretability, binary WT/TC/ET masks can be exported as NIfTI and converted into surface meshes via **Marching Cubes**, saved as **STL** files for 3D printing. Meshes may be refined (hole filling, smoothing, and repair) to ensure watertight printable geometry while maintaining **1:1 anatomical scale**.
