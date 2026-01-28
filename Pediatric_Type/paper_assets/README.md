# Pediatric Paper Assets (Generate figures/tables)

This folder is for **paper-ready figures and tables** generated from the artifacts you download from the GPU server (submission zips, prediction folders, and the exported visualisations pack).

## 1) What to download from the server
Minimum:
- `/workspace/pediatric_tumor_data/visualisations/run_*/` (QC PNGs + histograms + `index.html`)
- `/workspace/pediatric_tumor_data/nnunetv2/nnUNet_results/notebook_runs/pred_d501_3d_fullres_*` (predictions)
- `/workspace/pediatric_tumor_data/submissions/*.zip`

Optional (for full reproducibility):
- `/workspace/pediatric_tumor_data/nnunetv2/nnUNet_results/Dataset501_*/.../fold_*/checkpoint_final.pth`

## 2) Recommended local layout
Place the downloaded run folder(s) under:

```
Pediatric_Type/
  paper_assets/
    server_download/
      visualisations/
        run_YYYYMMDD_HHMMSS/
      predictions/
        pred_d501_3d_fullres_imagesTs_.../
        pred_d501_3d_fullres_imagesTr_.../
```

## 3) Generate extra plots + tables
Run:

```bash
python Pediatric_Type/paper_assets/generate_paper_assets.py \
  --run-dir Pediatric_Type/paper_assets/server_download/visualisations/run_YYYYMMDD_HHMMSS \
  --out-dir Pediatric_Type/paper_assets/out
```

Outputs (in `--out-dir`):
- `tables/volume_quantiles_*.md` and `tables/volume_quantiles_*.tex`
- additional plots (hist/box/violin/scatter) for WT/TC/ET voxel counts

Notes:
- The script is **safe to run** even if some files are missing; it will skip what it cannot find.
- Inputs come from the visualisations pack `volumes/predicted_voxel_counts_*.csv`.
- The exported nnU-Net predictions are not bundled here to keep the repo small.
