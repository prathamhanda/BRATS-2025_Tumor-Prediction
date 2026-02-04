#!/usr/bin/env python3
"""Run a small train→test ablation matrix on patch datasets.

This script is designed to be *cheap and paper-friendly*: it evaluates one or more
checkpoints (trained with an SSA-compatible 4-class CE setup) on one or more
patch directories containing `.npz` files with keys:
- image: float array shaped (4, P, P, P)
- mask:  uint/int array shaped (P, P, P) using BraTS-style labels {0,1,2,4}

It reuses the SSA label compatibility rule (4 -> 3) via `SSADataset`.

Outputs:
- SSA_Type/results/ablation_patch_matrix_<timestamp>.json
- SSA_Type/results/ablation_patch_matrix_<timestamp>.csv

Example:
  python SSA_Type/src/ablation_patch_matrix.py \
    --model SSA=SSA_Type/models/best_ssa_model.pth \
    --model GLI=model/best_model.pth \
    --patch-dir SSA=SSA_Type/ssa_preprocessed_patches \
    --patch-dir GLI=archive/preprocessed_patches \
    --limit 200

Notes:
- If a checkpoint is architecture-incompatible, strict loading will fail.
  Use --allow-partial to measure "compatibility" and optionally still run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy import ndimage

try:
    from tqdm import tqdm

    _TQDM_AVAILABLE = True
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore
    _TQDM_AVAILABLE = False

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(_THIS_DIR))

from ssa_model import SSABrainTumorUNet3D, SSADataset  # noqa: E402


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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


@dataclass
class LoadInfo:
    strict_ok: bool
    partial_ok: bool
    missing_keys: int
    unexpected_keys: int


def _load_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device, allow_partial: bool) -> LoadInfo:
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    try:
        model.load_state_dict(state, strict=True)
        return LoadInfo(strict_ok=True, partial_ok=True, missing_keys=0, unexpected_keys=0)
    except Exception:
        if not allow_partial:
            return LoadInfo(strict_ok=False, partial_ok=False, missing_keys=-1, unexpected_keys=-1)

    incompatible = model.load_state_dict(state, strict=False)
    missing = len(getattr(incompatible, "missing_keys", []) or [])
    unexpected = len(getattr(incompatible, "unexpected_keys", []) or [])
    return LoadInfo(strict_ok=False, partial_ok=True, missing_keys=missing, unexpected_keys=unexpected)


@torch.no_grad()
def _eval_model_on_patches(
    model: torch.nn.Module,
    patch_dir: Path,
    device: torch.device,
    train_split: float,
    seed: int,
    limit: int,
    compute_hd95: bool,
    spacing_mm: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict:
    files = sorted(str(p) for p in patch_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No patches found in {patch_dir}")

    rng = np.random.default_rng(seed)
    idx = np.arange(len(files))
    rng.shuffle(idx)
    split = int(len(files) * train_split)
    val_files = [files[i] for i in idx[split:]]
    if limit and limit > 0:
        val_files = val_files[:limit]

    ds = SSADataset(val_files, cache_size=10)
    model.eval()

    per_patch: List[Dict] = []
    t0 = time.perf_counter()

    it = range(len(ds))
    if _TQDM_AVAILABLE:
        it = tqdm(it, total=len(ds), desc=f"eval {patch_dir.name} on {device}")

    for i in it:
        if not _TQDM_AVAILABLE and (i == 0 or (i + 1) % 5 == 0):
            elapsed_so_far = time.perf_counter() - t0
            rate = elapsed_so_far / max(1, i + 1)
            remaining = rate * (len(ds) - (i + 1))
            print(f"   ... {i+1}/{len(ds)} patches | {rate:.2f}s/patch | ETA {remaining/60:.1f} min")
        image, mask = ds[i]
        logits = model(image.unsqueeze(0).to(device))
        pred = torch.argmax(logits, dim=1).cpu().numpy()[0].astype(np.int32)
        gt = mask.numpy().astype(np.int32)

        d: Dict[str, float] = {}

        # BraTS-style regions (using CE-compatible labels 0..3, where ET=3)
        pred_wt = pred > 0
        gt_wt = gt > 0
        pred_tc = np.logical_or(pred == 1, pred == 3)
        gt_tc = np.logical_or(gt == 1, gt == 3)
        pred_et = pred == 3
        gt_et = gt == 3

        d["dice_WT"] = dice_binary(pred_wt, gt_wt)
        d["dice_TC"] = dice_binary(pred_tc, gt_tc)
        d["dice_ET"] = dice_binary(pred_et, gt_et)

        if compute_hd95:
            d["hd95_WT_mm"] = hd95_binary(pred_wt, gt_wt, spacing_mm=spacing_mm)
            d["hd95_TC_mm"] = hd95_binary(pred_tc, gt_tc, spacing_mm=spacing_mm)
            d["hd95_ET_mm"] = hd95_binary(pred_et, gt_et, spacing_mm=spacing_mm)

        per_patch.append({"patch": Path(val_files[i]).name, **d})

    elapsed = time.perf_counter() - t0

    def _nanmean(key: str) -> float:
        vals = np.array([p[key] for p in per_patch], dtype=np.float64)
        return float(np.nanmean(vals))

    mean: Dict[str, float] = {
        "dice_WT": _nanmean("dice_WT"),
        "dice_TC": _nanmean("dice_TC"),
        "dice_ET": _nanmean("dice_ET"),
    }
    if compute_hd95:
        mean.update(
            {
                "hd95_WT_mm": _nanmean("hd95_WT_mm"),
                "hd95_TC_mm": _nanmean("hd95_TC_mm"),
                "hd95_ET_mm": _nanmean("hd95_ET_mm"),
            }
        )

    return {
        "val_patches": len(per_patch),
        "mean": mean,
        "time_seconds": elapsed,
        "sec_per_patch": float(elapsed / max(1, len(per_patch))),
    }


def _parse_kv(items: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected NAME=PATH, got: {item}")
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            raise ValueError(f"Empty NAME in: {item}")
        if not v:
            raise ValueError(f"Empty PATH in: {item}")
        out[k] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Patch-based ablation matrix (SSA-compatible checkpoints)")
    ap.add_argument("--model", action="append", default=[], help="NAME=PATH (repeatable)")
    ap.add_argument("--patch-dir", action="append", default=[], help="NAME=PATH (repeatable)")
    ap.add_argument("--train-split", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="Limit #val patches per dataset (0 = no limit)")
    ap.add_argument("--allow-partial", action="store_true", help="Allow strict=False checkpoint loading")
    ap.add_argument(
        "--compute-hd95",
        action="store_true",
        help="Compute HD95 (slow: requires distance transforms). If omitted, only Dice is reported.",
    )
    args = ap.parse_args()

    models = _parse_kv(args.model)
    patch_dirs = _parse_kv(args.patch_dir)

    if not models:
        raise SystemExit("No --model provided")
    if not patch_dirs:
        raise SystemExit("No --patch-dir provided")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    spacing_mm = (1.0, 1.0, 1.0)

    print(f"\n🔧 Ablation runner device: {device}")
    if device.type != "cuda":
        print("⚠️ Running on CPU. 3D U-Net inference on 128³ patches can take minutes per patch.")
        print("   Tip: run on a CUDA machine, or start with `--limit 1` / `--limit 5`.")

    rows: List[Dict] = []

    for model_name, model_path_str in models.items():
        ckpt_path = Path(model_path_str)
        if not ckpt_path.exists():
            for patch_name, patch_dir_str in patch_dirs.items():
                rows.append(
                    {
                        "model": model_name,
                        "patch_dir": patch_name,
                        "status": "missing_checkpoint",
                        "checkpoint": str(ckpt_path),
                    }
                )
            continue

        model = SSABrainTumorUNet3D(in_channels=4, out_channels=4).to(device)
        load_info = _load_checkpoint(model, ckpt_path, device, allow_partial=args.allow_partial)

        for patch_name, patch_dir_str in patch_dirs.items():
            patch_dir = Path(patch_dir_str)
            if not patch_dir.exists():
                rows.append(
                    {
                        "model": model_name,
                        "patch_dir": patch_name,
                        "status": "missing_patch_dir",
                        "checkpoint": str(ckpt_path),
                        "patch_path": str(patch_dir),
                        "strict_ok": load_info.strict_ok,
                        "partial_ok": load_info.partial_ok,
                        "missing_keys": load_info.missing_keys,
                        "unexpected_keys": load_info.unexpected_keys,
                    }
                )
                continue

            if not load_info.partial_ok:
                rows.append(
                    {
                        "model": model_name,
                        "patch_dir": patch_name,
                        "status": "incompatible_checkpoint",
                        "checkpoint": str(ckpt_path),
                        "patch_path": str(patch_dir),
                        "strict_ok": load_info.strict_ok,
                    }
                )
                continue

            try:
                print(f"\n▶ Model={model_name} | Patches={patch_name} | strict_ok={load_info.strict_ok} | partial_ok={load_info.partial_ok}")
                summary = _eval_model_on_patches(
                    model=model,
                    patch_dir=patch_dir,
                    device=device,
                    train_split=float(args.train_split),
                    seed=int(args.seed),
                    limit=int(args.limit),
                    compute_hd95=bool(args.compute_hd95),
                    spacing_mm=spacing_mm,
                )
                rows.append(
                    {
                        "model": model_name,
                        "patch_dir": patch_name,
                        "status": "ok" if load_info.strict_ok else "partial_load_ok",
                        "checkpoint": str(ckpt_path),
                        "patch_path": str(patch_dir),
                        "strict_ok": load_info.strict_ok,
                        "partial_ok": load_info.partial_ok,
                        "missing_keys": load_info.missing_keys,
                        "unexpected_keys": load_info.unexpected_keys,
                        **summary["mean"],
                        "val_patches": summary["val_patches"],
                        "sec_per_patch": summary["sec_per_patch"],
                    }
                )
            except Exception as e:
                rows.append(
                    {
                        "model": model_name,
                        "patch_dir": patch_name,
                        "status": "eval_failed",
                        "checkpoint": str(ckpt_path),
                        "patch_path": str(patch_dir),
                        "strict_ok": load_info.strict_ok,
                        "partial_ok": load_info.partial_ok,
                        "missing_keys": load_info.missing_keys,
                        "unexpected_keys": load_info.unexpected_keys,
                        "error": repr(e),
                    }
                )

    out_dir = Path("SSA_Type/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = _timestamp()

    json_path = out_dir / f"ablation_patch_matrix_{run_id}.json"
    csv_path = out_dir / f"ablation_patch_matrix_{run_id}.csv"

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "spacing_mm": spacing_mm,
        "args": vars(args),
        "rows": rows,
    }

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # CSV for easy paper tables
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("✅ Wrote:", json_path)
    print("✅ Wrote:", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
