#!/usr/bin/env python3
"""Compute Dice + HD95 on SSA preprocessed *patches* for the best model.

This is meant for research reporting when training/validation is patch-based.
It evaluates a deterministic validation split and writes a JSON summary.

Outputs:
- SSA_Type/results/ssa_patch_metrics_<timestamp>.json

Run:
  python SSA_Type/src/ssa_patch_metrics_hd95.py --model SSA_Type/models/best_ssa_model.pth

Notes:
- Masks in patches are expected to use labels {0,1,2,4}; for model-compatibility
  SSADataset maps 4 -> 3.
- HD95 is reported in mm assuming 1mm isotropic spacing (as per preprocessing).
  If you later store spacing per patch, plug it in here.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy import ndimage

# Local imports
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(_THIS_DIR))

from ssa_model import SSABrainTumorUNet3D, SSADataset  # noqa: E402


def hd95_binary(a: np.ndarray, b: np.ndarray, spacing_mm: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> float:
    a = a.astype(bool)
    b = b.astype(bool)

    if not np.any(a) and not np.any(b):
        return 0.0
    if not np.any(a) or not np.any(b):
        return float("nan")

    structure = ndimage.generate_binary_structure(3, 1)
    a_er = ndimage.binary_erosion(a, structure=structure, iterations=1)
    b_er = ndimage.binary_erosion(b, structure=structure, iterations=1)
    a_surf = a ^ a_er
    b_surf = b ^ b_er

    dt_b = ndimage.distance_transform_edt(~b_surf, sampling=spacing_mm)
    dt_a = ndimage.distance_transform_edt(~a_surf, sampling=spacing_mm)
    d_ab = dt_b[a_surf]
    d_ba = dt_a[b_surf]
    if d_ab.size == 0 or d_ba.size == 0:
        return float("nan")

    all_d = np.concatenate([d_ab, d_ba]).astype(np.float64)
    return float(np.percentile(all_d, 95))


def dice_binary(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return float((2.0 * inter) / denom)


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description="SSA patch-based Dice + HD95 evaluation")
    parser.add_argument("--patch-dir", type=str, default="SSA_Type/data/ssa_preprocessed_patches")
    parser.add_argument("--model", type=str, default="SSA_Type/models/best_ssa_model.pth")
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="Limit #val patches (0 = no limit)")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    spacing_mm = (1.0, 1.0, 1.0)

    patch_dir = Path(args.patch_dir)
    files = sorted(str(p) for p in patch_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No patches found in {patch_dir}")

    # deterministic shuffle/split
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(files))
    rng.shuffle(idx)
    split = int(len(files) * args.train_split)
    val_files = [files[i] for i in idx[split:]]
    if args.limit and args.limit > 0:
        val_files = val_files[: args.limit]

    ds = SSADataset(val_files, cache_size=10)

    model = SSABrainTumorUNet3D(in_channels=4, out_channels=4)
    ckpt = torch.load(args.model, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    per_patch: List[Dict] = []

    t0 = time.perf_counter()
    for i in range(len(ds)):
        image, mask = ds[i]
        image_t = image.unsqueeze(0).to(device)
        logits = model(image_t)
        pred = torch.argmax(logits, dim=1).cpu().numpy()[0].astype(np.int32)
        gt = mask.numpy().astype(np.int32)

        # Class-wise (1..3)
        d = {}
        for cls in (1, 2, 3):
            d[f"dice_class_{cls}"] = dice_binary(pred == cls, gt == cls)
            d[f"hd95_class_{cls}_mm"] = hd95_binary(pred == cls, gt == cls, spacing_mm=spacing_mm)

        # BraTS-style regions
        pred_wt = pred > 0
        gt_wt = gt > 0
        pred_tc = np.logical_or(pred == 1, pred == 3)
        gt_tc = np.logical_or(gt == 1, gt == 3)
        pred_et = pred == 3
        gt_et = gt == 3

        d["dice_WT"] = dice_binary(pred_wt, gt_wt)
        d["dice_TC"] = dice_binary(pred_tc, gt_tc)
        d["dice_ET"] = dice_binary(pred_et, gt_et)
        d["hd95_WT_mm"] = hd95_binary(pred_wt, gt_wt, spacing_mm=spacing_mm)
        d["hd95_TC_mm"] = hd95_binary(pred_tc, gt_tc, spacing_mm=spacing_mm)
        d["hd95_ET_mm"] = hd95_binary(pred_et, gt_et, spacing_mm=spacing_mm)

        per_patch.append({"patch": Path(val_files[i]).name, **d})

    elapsed = time.perf_counter() - t0

    def _nanmean(key: str) -> float:
        vals = np.array([p[key] for p in per_patch], dtype=np.float64)
        return float(np.nanmean(vals))

    summary = {
        "val_patches": len(per_patch),
        "device": str(device),
        "spacing_mm": spacing_mm,
        "mean": {
            "dice_WT": _nanmean("dice_WT"),
            "dice_TC": _nanmean("dice_TC"),
            "dice_ET": _nanmean("dice_ET"),
            "hd95_WT_mm": _nanmean("hd95_WT_mm"),
            "hd95_TC_mm": _nanmean("hd95_TC_mm"),
            "hd95_ET_mm": _nanmean("hd95_ET_mm"),
            "dice_class_1": _nanmean("dice_class_1"),
            "dice_class_2": _nanmean("dice_class_2"),
            "dice_class_3": _nanmean("dice_class_3"),
            "hd95_class_1_mm": _nanmean("hd95_class_1_mm"),
            "hd95_class_2_mm": _nanmean("hd95_class_2_mm"),
            "hd95_class_3_mm": _nanmean("hd95_class_3_mm"),
        },
        "time_seconds": elapsed,
        "sec_per_patch": float(elapsed / max(1, len(per_patch))),
    }

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "patch_dir": args.patch_dir,
        "train_split": args.train_split,
        "seed": args.seed,
        "limit": args.limit,
        "summary": summary,
        "per_patch": per_patch,
    }

    out_dir = Path("SSA_Type/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"ssa_patch_metrics_{run_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("✅ Patch metrics written:", out_path)
    print("Summary (mean):")
    for k, v in summary["mean"].items():
        if "hd95" in k:
            print(f"- {k}: {v:.2f} mm")
        else:
            print(f"- {k}: {v:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
