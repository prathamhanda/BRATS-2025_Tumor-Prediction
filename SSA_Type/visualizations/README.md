# 🎨 Visualizations Directory

This directory contains all publication-quality figures organized by category.

## Structure

### **training/** - Training Process Visualizations
Training-related figures showing model convergence and learning dynamics.

### **inference/** - Inference & Analysis Visualizations
Inference results, segmentation examples, and analytical visualizations.

## Training Visualizations

### **01_training_curves.png** ⭐
**Main Figure for Papers**

Shows training dynamics with 4 subplots:
1. **Loss Convergence** - Training vs. validation loss over 25 epochs
   - Train loss: 1.0777 → 0.0409 (96.2% reduction)
   - Val loss: 1.1648 → 0.0575 (95.1% reduction)
   - Best model marked at epoch 18

2. **Dice Score Evolution** - Per-epoch Dice improvement
   - Initial: 0.218 → Best: 0.8857 (305% improvement)
   - Clinical threshold (0.70) and excellence threshold (0.80) marked

3. **Learning Rate Schedule** - LR over 25 epochs
   - Shows ReduceLROnPlateau adjustments
   - Stable at 0.001 until epoch 20

4. **Performance Summary Table** - Key metrics and statistics

### **02_performance_dashboard.png** ⭐ PRIMARY PUBLICATION FIGURE
**Comprehensive Performance Overview**

Shows 6 subplots:
1. **Per-Class Dice Scores** - Bar chart with clinical threshold
2. **Precision vs. Recall** - Comparison across 4 classes
3. **F1-Scores** - Per-class F1 metrics
4. **Confusion Matrix** - Normalized 4×4 matrix
5. **Model Specifications** - Architecture info box
6. **Clinical Comparison** - vs. thresholds and SOTA

### **03_training_dynamics.png**
**Advanced Training Analysis**

Shows 4 subplots:
1. **Loss Gradient** - Rate of change in losses
2. **Epoch-to-Epoch Improvement** - Validation Dice improvement per epoch
3. **Overfitting Analysis** - Train-validation gap evolution
4. **Learning Stability** - Rolling variance of validation loss

### **ssa_training_history.png**
Training history visualization (complementary figure)

## Inference Visualizations

### **ssa_inference_demonstration.png**
Inference results on real SSA case with:
- Predicted segmentation
- Ground truth comparison
- Per-slice analysis

### **ssa_3d_volume_analysis.png**
3D tumor volume rendering showing:
- Different tumor components
- 3D spatial relationships

### **ssa_comprehensive_analysis.png**
Multi-panel comprehensive analysis with:
- Axial, coronal, sagittal views
- Different MRI modalities
- Segmentation overlays

### **ssa_comparison_analysis.png**
Prediction vs. ground truth comparison

### **ssa_sample_visualization.png**
Sample case visualization

### **ssa_dataset_overview.png**
Dataset statistics visualization

## Usage in Publications

### For Main Text
Use: `02_performance_dashboard.png` (Figure 1 or equivalent)
- Shows overall performance vs. clinical standards
- Demonstrates per-class metrics
- Model specifications included

### For Supplementary Material
Use: `01_training_curves.png` + `03_training_dynamics.png`
- Details training procedure
- Shows convergence and stability
- Demonstrates data-driven approach

### For Appendix
Include all inference visualizations showing qualitative results

## Figure Specifications

- **Resolution:** 300 DPI (publication-quality)
- **Format:** PNG
- **Color Space:** RGB
- **Fonts:** Legible at 8pt minimum

## Accessing Figures

All figures are immediately viewable:
```bash
# View in default image viewer
open training/02_performance_dashboard.png

# Generate new visualizations
python ../utils/visualization_suite.py
```

## Reproducibility

All visualizations are generated from:
- `../results/training_stats.json` - Training history
- `../results/detailed_metrics.json` - Comprehensive metrics
- Model training with fixed seed (42)

---

See `../docs/RESULTS.md` for figure interpretation.
See `../utils/visualization_suite.py` to regenerate visualizations.
