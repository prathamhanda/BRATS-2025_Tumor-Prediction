#!/usr/bin/env python3
"""Compare multiple SSA segmentation models on the same patch split.

Goal: generate a quick, reproducible 5–6 model comparison table (Dice, per-class Dice,
throughput) on the SSA preprocessed patch dataset.

This script is intentionally "ASAP-friendly":
- Evaluates the existing pretrained SSA model (if provided)
- Trains several lightweight baselines for a few epochs

Outputs:
- SSA_Type/results/ssa_model_comparison_<timestamp>.json
- SSA_Type/results/ssa_model_comparison_<timestamp>.csv

Run (from repo root):
  python SSA_Type/src/ssa_model_comparison.py --epochs 3

Notes:
- Uses deterministic split (seed) so all models see identical train/val patches.
- Assumes patch files are .npz with keys: image, mask.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# Ensure local imports work regardless of CWD
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(_THIS_DIR))

from ssa_model import SSABrainTumorUNet3D, SSADataset  # noqa: E402


@dataclass
class ModelResult:
    name: str
    params_m: float
    trained: bool
    epochs: int
    train_patches: int
    val_patches: int
    device: str
    mean_dice_all: float
    mean_dice_tumor: float
    dice_class_0: float
    dice_class_1: float
    dice_class_2: float
    dice_class_3: float
    sec_per_val_patch: float


class Simple3DCNN(nn.Module):
    """Tiny baseline (no encoder/decoder, no skip connections)."""

    def __init__(self, in_channels: int = 4, out_channels: int = 4, base: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base * 2, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AllBackgroundModel(nn.Module):
    """Non-learning baseline: predicts background everywhere."""

    def __init__(self, out_channels: int = 4):
        super().__init__()
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return zeros logits -> argmax = 0 (background)
        b, _, h, w, d = x.shape
        return torch.zeros((b, self.out_channels, h, w, d), dtype=x.dtype, device=x.device)


def _center_crop_3d(image: torch.Tensor, mask: torch.Tensor, crop: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Center-crop (C,H,W,D) and (H,W,D) tensors to crop^3."""
    _, h, w, d = image.shape
    if crop > h or crop > w or crop > d:
        return image, mask

    hs = (h - crop) // 2
    ws = (w - crop) // 2
    ds = (d - crop) // 2
    he, we, de = hs + crop, ws + crop, ds + crop

    image_c = image[:, hs:he, ws:we, ds:de]
    mask_c = mask[hs:he, ws:we, ds:de]
    return image_c, mask_c


def _count_params(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sorted_patch_files(patch_dir: Path) -> List[str]:
    files = sorted(str(p) for p in patch_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz patches found in: {patch_dir}")
    return files


def _split_files(files: Sequence[str], train_split: float, seed: int) -> Tuple[List[str], List[str]]:
    idx = np.arange(len(files))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    split = int(len(files) * train_split)
    train_idx = idx[:split]
    val_idx = idx[split:]
    train_files = [files[i] for i in train_idx]
    val_files = [files[i] for i in val_idx]
    return train_files, val_files


def _dice_per_class(pred_logits: torch.Tensor, target: torch.Tensor, num_classes: int = 4) -> torch.Tensor:
    """Returns dice per class as float tensor shape (C,)."""
    pred = torch.argmax(pred_logits, dim=1)  # (B, H, W, D)
    dices: List[torch.Tensor] = []

    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        intersection = (pred_cls * target_cls).sum(dim=(1, 2, 3))
        union = pred_cls.sum(dim=(1, 2, 3)) + target_cls.sum(dim=(1, 2, 3))
        # if both empty => 1.0
        dice = torch.where(union == 0, torch.ones_like(union), (2.0 * intersection) / union)
        dices.append(dice.mean())

    return torch.stack(dices)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()

    t0 = time.perf_counter()
    intersection_sum = torch.zeros(4, device=device)
    union_sum = torch.zeros(4, device=device)

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)

        pred = torch.argmax(logits, dim=1)
        for cls in range(4):
            pred_cls = (pred == cls).float()
            target_cls = (masks == cls).float()
            intersection_sum[cls] += (pred_cls * target_cls).sum()
            union_sum[cls] += pred_cls.sum() + target_cls.sum()

    elapsed = time.perf_counter() - t0

    dice_vals: List[float] = []
    for cls in range(4):
        if union_sum[cls].item() == 0:
            dice_vals.append(float("nan"))
        else:
            dice_vals.append(float((2.0 * intersection_sum[cls] / union_sum[cls]).item()))

    all_present = [d for d in dice_vals if not np.isnan(d)]
    tumor_present = [dice_vals[i] for i in (1, 2, 3) if not np.isnan(dice_vals[i])]

    mean_all = float(np.mean(all_present)) if all_present else float("nan")
    mean_tumor = float(np.mean(tumor_present)) if tumor_present else float("nan")

    return {
        "dice_class_0": float(dice_vals[0]),
        "dice_class_1": float(dice_vals[1]),
        "dice_class_2": float(dice_vals[2]),
        "dice_class_3": float(dice_vals[3]),
        "mean_dice_all": mean_all,
        "mean_dice_tumor": mean_tumor,
        "sec_per_val_patch": float(elapsed / max(1, len(loader.dataset))),
    }


def train_quick(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    amp: bool,
) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    model.train()

    for _ in range(epochs):
        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(images)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # quick sanity forward on val (keeps caching behavior warm)
        _ = next(iter(val_loader), None)


def _load_checkpoint(model: nn.Module, ckpt_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(str(ckpt_path), map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint
    model.load_state_dict(state)


def build_models() -> List[Tuple[str, nn.Module]]:
    return [
        (
            "unet_base",
            SSABrainTumorUNet3D(in_channels=4, out_channels=4, features=(32, 64, 128, 256), dropout=0.1, use_attention=False),
        ),
        (
            "unet_attention",
            SSABrainTumorUNet3D(in_channels=4, out_channels=4, features=(32, 64, 128, 256), dropout=0.1, use_attention=True),
        ),
        (
            "unet_lite",
            SSABrainTumorUNet3D(in_channels=4, out_channels=4, features=(16, 32, 64, 128), dropout=0.1, use_attention=False),
        ),
        (
            "unet_tiny",
            SSABrainTumorUNet3D(in_channels=4, out_channels=4, features=(8, 16, 32, 64), dropout=0.1, use_attention=False),
        ),
        (
            "cnn_baseline",
            Simple3DCNN(in_channels=4, out_channels=4, base=16),
        ),
        (
            "all_background",
            AllBackgroundModel(out_channels=4),
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run quick multi-model comparison on SSA patches")
    parser.add_argument(
        "--patch-dir",
        type=str,
        default="SSA_Type/data/ssa_preprocessed_patches",
        help="Directory containing .npz SSA patches",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="SSA_Type/models/best_ssa_model.pth",
        help="Path to pretrained checkpoint for unet_base (optional)",
    )
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Ultra-fast mode (CPU-friendly): fewer epochs and fewer patches",
    )

    args = parser.parse_args()

    patch_dir = Path(args.patch_dir)
    results_dir = Path("SSA_Type/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp = torch.cuda.is_available()

    # Keep CPU runs from over-threading (helps responsiveness on Windows)
    if device.type != "cuda":
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 4)))

    _set_seed(args.seed)

    files = _sorted_patch_files(patch_dir)
    train_files, val_files = _split_files(files, train_split=args.train_split, seed=args.seed)

    # CPU-friendly shortcut (still same split logic, but uses a smaller subset consistently)
    crop_size: Optional[int] = None
    if args.fast and device.type != "cuda":
        train_files = train_files[:6]
        val_files = val_files[:2]
        epochs = max(1, min(args.epochs, 1))
        crop_size = 64
    else:
        epochs = args.epochs

    transform = (lambda img, m: _center_crop_3d(img, m, crop_size)) if crop_size else None
    train_ds = SSADataset(train_files, transform=transform, cache_size=20)
    val_ds = SSADataset(val_files, transform=transform, cache_size=10)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    results: List[ModelResult] = []

    # 1) Evaluate pretrained unet_base if available
    pretrained_path = Path(args.pretrained) if args.pretrained else None
    if pretrained_path and pretrained_path.exists():
        model = SSABrainTumorUNet3D(in_channels=4, out_channels=4, features=(32, 64, 128, 256), dropout=0.1, use_attention=False)
        model = model.to(device)
        _load_checkpoint(model, pretrained_path, device)
        metrics = evaluate(model, val_loader, device)
        results.append(
            ModelResult(
                name="unet_base_pretrained",
                params_m=_count_params(model),
                trained=False,
                epochs=0,
                train_patches=len(train_files),
                val_patches=len(val_files),
                device=str(device),
                **metrics,
            )
        )

    # 2) Train and evaluate 5 baselines
    for name, model in build_models():
        model = model.to(device)

        # Train quick baselines (skip training for non-learning baselines)
        if name not in {"all_background"}:
            train_quick(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                amp=amp,
            )

        metrics = evaluate(model, val_loader, device)

        results.append(
            ModelResult(
                name=name,
                params_m=_count_params(model),
                trained=True,
                epochs=epochs,
                train_patches=len(train_files),
                val_patches=len(val_files),
                device=str(device),
                **metrics,
            )
        )

    # Sort by tumor dice primarily
    results_sorted = sorted(results, key=lambda r: (r.mean_dice_tumor, r.mean_dice_all), reverse=True)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "patch_dir": str(patch_dir),
        "train_split": args.train_split,
        "seed": args.seed,
        "device": str(device),
        "amp": amp,
        "epochs": epochs,
        "crop_size": crop_size,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "models": [asdict(r) for r in results_sorted],
    }

    json_path = results_dir / f"ssa_model_comparison_{run_id}.json"
    csv_path = results_dir / f"ssa_model_comparison_{run_id}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(asdict(results_sorted[0]).keys()),
        )
        writer.writeheader()
        for r in results_sorted:
            writer.writerow(asdict(r))

    print("\n✅ SSA multi-model comparison complete")
    print(f"- JSON: {json_path}")
    print(f"- CSV:  {csv_path}")

    print("\nTop models (by mean_dice_tumor):")
    for r in results_sorted[:6]:
        print(f"- {r.name}: tumor_dice={r.mean_dice_tumor:.4f}, all_dice={r.mean_dice_all:.4f}, params={r.params_m:.2f}M")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
