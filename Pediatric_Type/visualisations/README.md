# Pediatric visualisations pack

This folder contains utilities to generate a **judge-friendly visual QC pack** (PNGs + simple HTML gallery) for the Pediatric BraTS PEDs nnU-Net pipeline.

## Recommended (Notebook)
Use the export cells in:
- [Pediatric_Type/pediatric_5.2.ipynb](../pediatric_5.2.ipynb)

They create a timestamped folder like:
- `/workspace/pediatric_tumor_data/visualisations/run_<timestamp>/...`

and (Section 7.4) generate `index.html` + sub-galleries.

## CLI (optional)
If you want to generate overlays/plots outside the notebook:

- [export_visualisations_pack.py](export_visualisations_pack.py)
- [build_gallery.py](build_gallery.py)

Example:

```bash
python Pediatric_Type/visualisations/export_visualisations_pack.py \
  --images-dir /workspace/pediatric_tumor_data/nnunetv2/nnUNet_raw/Dataset501_BraTS_PEDs2024/imagesTs \
  --pred-dir   /workspace/pediatric_tumor_data/nnunetv2/nnUNet_results/notebook_runs/pred_d501_3d_fullres_imagesTs_*/ \
  --out-dir    /workspace/pediatric_tumor_data/visualisations/run_manual_imagesTs

python Pediatric_Type/visualisations/build_gallery.py \
  --run-dir /workspace/pediatric_tumor_data/visualisations/run_manual_imagesTs
```

Notes:
- The scripts assume nnU-Net-style filenames: `<CASE>_0000.nii.gz`… and predictions: `<CASE>.nii.gz`.
- Channel selection attempts to read `dataset.json` when provided.
