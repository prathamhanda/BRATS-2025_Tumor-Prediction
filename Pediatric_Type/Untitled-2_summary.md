# Summary of Pediatric Tumor Segmentation Notebook (`Untitled-2.ipynb`)

This document provides a cell-by-cell section summary of the Jupyter Notebook pipeline for the Pediatric Brain Tumor Segmentation (BraTS PEDs). The notebook follows a comprehensive research-grade roadmap using nnU-Net v2 and MONAI, ranging from data preprocessing to 3D mesh export and alternative model evaluation.

## 1. Environment Setup & Dataset Validation (Cells 1-11)
- **Cell 1**: Installs necessary initial packages (`nibabel`, `scipy`) and attempts to auto-discover the prediction directory.
- **Cells 2-3**: Sets up reproducibility guidelines (Python seed, Torch determinism).
- **Cells 4-11**: Check GPU server compatibility, locate the training and validation dataset splits, confirm BraTS naming conventions, and infer meaning of specific labels (e.g., Label 3 vs 4 mapping). Finalizes by visualizing mid-slices for sanity checks.

## 2. Preprocessing (Cells 12-15)
- **Cells 12-13**: Performs strict validation of orientation and resolution for a sample MRI case.
- **Cell 14**: Applies mandatory skull stripping. It attempts to use `HD-BET` and uses a fallback if it fails.
- **Cell 15**: Runs per-modality, per-subject intensity normalization isolated strictly to the brain mask.

## 3. nnU-Net v2 Setup and Training Preparation (Cells 16-33)
- **Cells 16-21**: Configures `nnU-Net v2` paths and converts the baseline BraTS dataset to the required `nnU-Net` raw directory format, adjusting labels if needed.
- **Cells 22-27**: Fixes file extensions, runs nnU-Net dataset fingerprints, planning, and preprocessing steps.
- **Cells 28-33**: Launches and monitors the first fold (fold 0) of the `3d_fullres` task as a background training process, including scripts to tail logs.

## 4. Inference, Metrics, and Post-processing (Cells 34-64)
- **Cells 34-43**: Verifies that training has been completed for models (folds 0-4) and initiates an ensemble prediction script utilizing the GPU. Includes process monitoring to track VRAM usage.
- **Cells 44-57**: Monitors inference progress, audits outputs vs ground truth (GT), and calculates key baseline metrics such as Dice similarity scores for WT/TC/ET (Whole Tumor, Tumor Core, Enhancing Tumor) and absolute voxel volumes.
- **Cells 58-64**: Executes optional but recommended advanced post-processing adjustments, packaging the final results as an `imagesTs` zip for the validation/challenge submission.

## 5. Visual Quality Control (QC) & Galleries (Cells 65-77)
- **Cells 65-67**: Produces overlays on FLAIR and T1ce modalities, contrasting the algorithm's predictions with anatomy.
- **Cells 68-72**: Generates a structured visuals folder featuring bulk exported overlays, summary plots, and an elegant HTML gallery to assist in manual radiological review.
- **Cells 73-77**: Creates a compressed archive of all predictions/visuals and handles Google Drive offsite backups using `rclone`.

## 6. Model Comparison (Cells 78-92)
- **Cells 78-92**: Compares the `3d_fullres` model against `3d_lowres` or `2d` architectures. Loads validation GT, calculates summary metrics (Dice + Hausdorff Distance 95), and saves a comparison table alongside visually compelling side-by-side output montages.

## 7. Batch Validation & 3D Model Export (Cells 93-107)
- **Cells 93-99**: Discovers available ground truth labels and exports labeled datasets together with summary tables across batched validation cases.
- **Cells 100-107**: Converts 3D predictions to printable STL meshes using the marching cubes algorithm. Packages everything into "showcase" bundles representing individual subjects, appending original NIfTIs and modality readmes for the resulting 3D models.

## 8. Alternative Pipeline: MONAI SwinUNETR (Cells 108-136)
- **Cells 108-115**: Prepares a standalone MONAI pipeline leveraging the `SwinUNETR` model architecture, mapping the existing nnU-Net directories to custom strict dataloaders.
- **Cells 116-126**: Writes and spawns asynchronous training scripts out of the notebook (`train_swin.py`). Includes terminal tracking functions (`tail`, `pkill`).
- **Cells 127-136**: Loads the alternative SwinUNETR configuration, calculates Before-and-After (B/A) representations, visualizes outcomes, and tests soft-ensemble inference mechanics between models.