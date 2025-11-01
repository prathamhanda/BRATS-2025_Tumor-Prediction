# 🧠 Source Code Directory

This directory contains the core source code for the SSA Brain Tumor Segmentation project.

## Files

### **ssa_model.py** 
3D U-Net architecture implementation for brain tumor segmentation. Contains:
- `SSABrainTumorUNet3D`: Core 3D U-Net model (22.58M parameters)
- `SSADataset`: PyTorch dataset for patch loading
- `SSAModelManager`: Model utilities and data loaders

### **ssa_trainer.py**
Complete training pipeline with:
- Mixed precision training (AMP + GradScaler)
- Learning rate scheduling (ReduceLROnPlateau)
- Comprehensive metrics tracking
- Early stopping implementation

### **ssa_inference_demo.py**
Inference and visualization module:
- Run inference on single cases
- Generate prediction visualizations
- Compare with ground truth
- Per-slice and 3D volume analysis

### **ssa_evaluation.py**
Comprehensive performance evaluation:
- Calculate per-class metrics (Dice, Precision, Recall)
- Generate performance dashboards
- Create research-grade reports

### **ssa_visualizer.py**
Visualization utilities:
- Plot training curves
- Create confusion matrices
- Generate segmentation overlays
- 3D volume rendering

### **ssa_preprocessor.py**
Data preprocessing pipeline:
- Intensity normalization
- Isotropic resampling
- Patch extraction

### **ssa_dataset_explorer.py**
Dataset analysis and exploration tools

### **verify_patches.py**
Data validation and verification utilities

## Quick Start

```python
# Import and train model
from ssa_model import SSABrainTumorUNet3D, SSAModelManager
from ssa_trainer import SSATrainer

# Initialize model and trainer
model = SSABrainTumorUNet3D()
trainer = SSATrainer(model)

# Run training
trainer.train(epochs=25)
```

## Requirements

- PyTorch 2.0+
- CUDA 11.8+
- NumPy, SciPy
- nibabel (for NIfTI I/O)

## Architecture

- **Input:** 4-channel (T1n, T1c, T2w, T2f) × 128³ voxels
- **Output:** 4-class segmentation (Background, Necrotic, Edema, Enhancing)
- **Parameters:** 22.58M
- **Features:** 32→64→128→256→512 (bottleneck)

## Performance

- **Validation Dice:** 0.8857
- **Training Time:** 1.85 hours (GTX 1650, 4GB VRAM)
- **GPU Memory:** 3.41GB (85.2% utilization)

---

See `../docs/RESEARCH_METHODOLOGY.md` for detailed technical approach.
