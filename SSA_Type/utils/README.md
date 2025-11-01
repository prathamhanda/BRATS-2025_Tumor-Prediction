# ⚙️ Utilities Directory

This directory contains utility and helper scripts for GPU optimization, data preprocessing, and visualization.

## Files

### **gpu_validator.py**
GPU validation and optimization analysis tool.
- Detect available GPUs
- Validate CUDA setup
- Provide optimization recommendations
- Check VRAM availability

**Usage:**
```bash
python gpu_validator.py
```

### **gpu_ssa_preprocessor.py**
GPU-accelerated data preprocessing.
- Efficient patch extraction on GPU
- Parallel data loading
- Optimization statistics

**Features:**
- Processes data ~10x faster than CPU
- Validates output patches
- Generates preprocessing statistics

### **visualization_suite.py**
Comprehensive visualization generator.
- Training curves and convergence analysis
- Performance dashboards
- Training dynamics analysis

**Features:**
- 300 DPI publication-quality PNG files
- Per-class metrics visualization
- Clinical comparison plots

**Usage:**
```bash
python visualization_suite.py
```

## Quick Start

```bash
# Validate GPU setup
python gpu_validator.py

# Preprocess data on GPU
python gpu_ssa_preprocessor.py

# Generate visualizations
python visualization_suite.py
```

## Performance Characteristics

- **GPU Utilization:** 92.1% average
- **Memory Efficiency:** Mixed precision (30% reduction)
- **Throughput:** 6.7 patches/second inference

---

See `../docs/RESEARCH_METHODOLOGY.md` for GPU optimization details.
