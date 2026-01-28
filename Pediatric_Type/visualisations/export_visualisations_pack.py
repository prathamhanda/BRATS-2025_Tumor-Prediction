from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


def _lazy_imports():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib

    return plt, nib


@dataclass(frozen=True)
class ChannelSelection:
    flair_idx: int
    t1ce_idx: int


@dataclass(frozen=True)
class LabelOntology:
    """Label semantics for BraTS-style subregions.

    Different BraTS releases encode ET as either 3 or 4. We infer this from the
    predictions to keep voxel-count plots consistent with the actual outputs.
    """

    et_label: int | None
    tc_labels: frozenset[int]


def _load_dataset_channel_map(dataset_json: Path | None) -> dict[int, str]:
    if not dataset_json or not dataset_json.exists():
        return {}
    ds = json.loads(dataset_json.read_text(encoding="utf-8"))
    ch = ds.get("channel_names") or ds.get("channel_names_dict") or ds.get("channels")
    if isinstance(ch, dict):
        out: dict[int, str] = {}
        for k, v in ch.items():
            try:
                out[int(k)] = str(v)
            except Exception:
                continue
        return out
    return {}


def _pick_channels(channel_map: dict[int, str]) -> ChannelSelection:
    # Fallback to common BraTS ordering: (0=T1, 1=T1ce, 2=T2, 3=FLAIR)
    flair = 3
    t1ce = 1

    if channel_map:
        norm = {k: v.strip().lower().replace("-", "").replace("_", "") for k, v in channel_map.items()}

        def find(names: Iterable[str], default: int) -> int:
            target = {n.strip().lower().replace("-", "").replace("_", "") for n in names}
            for idx, name in norm.items():
                if name in target:
                    return idx
            return default

        flair = find(["flair", "t2flair", "t2f"], flair)
        t1ce = find(["t1ce", "t1c", "t1gd", "t1post"], t1ce)

    return ChannelSelection(flair_idx=flair, t1ce_idx=t1ce)


def _infer_label_ontology(pred_dir: Path, max_cases: int = 25) -> LabelOntology:
    """Infer ET label value from prediction volumes (prefers 4 if present).

    Returns:
      - et_label: 4 if present in any scanned file, else 3 if present, else None.
      - tc_labels: {1, et_label} if et_label is known, else {1}.
    """
    _, nib = _lazy_imports()

    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    pred_files = pred_files[:max_cases]

    seen_3 = False
    seen_4 = False
    for pf in pred_files:
        try:
            arr = np.asanyarray(nib.load(str(pf)).dataobj)
        except Exception:
            continue
        uniq = set(int(x) for x in np.unique(arr))
        if 4 in uniq:
            seen_4 = True
        if 3 in uniq:
            seen_3 = True
        if seen_4:
            break

    et_label: int | None
    if seen_4:
        et_label = 4
    elif seen_3:
        et_label = 3
    else:
        et_label = None

    tc = {1}
    if et_label is not None:
        tc.add(et_label)

    return LabelOntology(et_label=et_label, tc_labels=frozenset(sorted(tc)))


def _case_ids_from_images_dir(images_dir: Path) -> list[str]:
    # nnU-Net convention: CASE_0000.nii.gz ...
    files = sorted(images_dir.glob("*_0000.nii.gz"))
    case_ids = [f.name.replace("_0000.nii.gz", "") for f in files]
    return case_ids


def _load_modalities(images_dir: Path, case_id: str, flair_idx: int, t1ce_idx: int) -> tuple[np.ndarray, np.ndarray]:
    plt, nib = _lazy_imports()
    flair_p = images_dir / f"{case_id}_{flair_idx:04d}.nii.gz"
    t1ce_p = images_dir / f"{case_id}_{t1ce_idx:04d}.nii.gz"

    if not flair_p.exists():
        raise FileNotFoundError(f"Missing FLAIR file: {flair_p}")
    if not t1ce_p.exists():
        raise FileNotFoundError(f"Missing T1ce file: {t1ce_p}")

    flair = nib.load(str(flair_p)).get_fdata().astype(np.float32)
    t1ce = nib.load(str(t1ce_p)).get_fdata().astype(np.float32)
    return flair, t1ce


def _load_seg(seg_path: Path) -> np.ndarray:
    plt, nib = _lazy_imports()
    if not seg_path.exists():
        raise FileNotFoundError(f"Missing seg: {seg_path}")
    return nib.load(str(seg_path)).get_fdata().astype(np.int16)


def _robust_norm(img2d: np.ndarray, p_lo=1, p_hi=99) -> np.ndarray:
    lo, hi = np.percentile(img2d, [p_lo, p_hi])
    if hi <= lo:
        return np.zeros_like(img2d, dtype=np.float32)
    x = (img2d - lo) / (hi - lo)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def _pick_slice(seg: np.ndarray) -> int:
    wt = seg > 0
    if wt.any():
        z = np.where(wt)[2]
        return int(np.median(z))
    return seg.shape[2] // 2


def _render_overlay(flair: np.ndarray, t1ce: np.ndarray, seg: np.ndarray, out_path: Path, title: str) -> None:
    plt, _ = _lazy_imports()

    z = _pick_slice(seg)
    f = _robust_norm(flair[:, :, z])
    t = _robust_norm(t1ce[:, :, z])
    s = seg[:, :, z]

    # color map: 0 bg, 1 NET/NCR, 2 ED, 3 ET (dataset-specific ET label may differ, but colors still OK)
    # Keep it simple/robust: show any non-zero mask in red edges and full color fill.

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=140)
    for ax, base, name in [(axes[0], f, "FLAIR"), (axes[1], t, "T1ce")]:
        ax.imshow(base.T, cmap="gray", origin="lower")
        ax.imshow(np.ma.masked_where(s.T == 0, s.T), cmap="viridis", origin="lower", alpha=0.35, vmin=0, vmax=max(3, int(s.max())))
        ax.set_title(name)
        ax.axis("off")

    fig.suptitle(title, fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _voxel_counts(seg: np.ndarray, ontology: LabelOntology) -> dict[str, int]:
    # WT always means any tumor label > 0.
    wt = int((seg > 0).sum())

    tc_mask = seg == 1
    for lv in ontology.tc_labels:
        if lv == 1:
            continue
        tc_mask = np.logical_or(tc_mask, seg == lv)
    tc = int(tc_mask.sum())

    if ontology.et_label is None:
        et = 0
    else:
        et = int((seg == ontology.et_label).sum())
    return {"WT": wt, "TC": tc, "ET": et}


def export_pack(
    *,
    images_dir: Path,
    pred_dir: Path,
    out_dir: Path,
    dataset_json: Path | None,
    split_name: str,
    max_cases: int | None,
) -> Path:
    if not images_dir.exists():
        raise FileNotFoundError(f"images_dir not found: {images_dir}")
    if not pred_dir.exists():
        raise FileNotFoundError(f"pred_dir not found: {pred_dir}")

    channel_map = _load_dataset_channel_map(dataset_json)
    sel = _pick_channels(channel_map)

    ontology = _infer_label_ontology(pred_dir)
    print("Inferred label ontology:")
    print("  ET label:", ontology.et_label)
    print("  TC labels:", sorted(ontology.tc_labels))

    case_ids = _case_ids_from_images_dir(images_dir)
    if max_cases is not None:
        case_ids = case_ids[:max_cases]

    qc_dir = out_dir / "qc" / split_name
    vols_dir = out_dir / "volumes"
    qc_dir.mkdir(parents=True, exist_ok=True)
    vols_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []

    for i, cid in enumerate(case_ids, start=1):
        flair, t1ce = _load_modalities(images_dir, cid, sel.flair_idx, sel.t1ce_idx)
        seg = _load_seg(pred_dir / f"{cid}.nii.gz")

        out_png = qc_dir / f"{cid}_overlay.png"
        _render_overlay(flair, t1ce, seg, out_png, title=f"{cid} ({split_name})")

        vc = _voxel_counts(seg, ontology)
        rows.append({"case_id": cid, **vc})

        if i % 25 == 0:
            print(f"Exported {i}/{len(case_ids)} overlays...")

    # write CSV summary
    csv_path = vols_dir / f"predicted_voxel_counts_{split_name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "WT", "TC", "ET"])
        w.writeheader()
        for r in rows:
            w.writerow(r)  # type: ignore[arg-type]

    # write JSON summary
    json_path = vols_dir / f"predicted_voxel_counts_{split_name}.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # hist plots
    plt, _ = _lazy_imports()
    for key in ["WT", "TC", "ET"]:
        vals = [int(r[key]) for r in rows]
        fig = plt.figure(figsize=(8, 4), dpi=140)
        plt.hist(vals, bins=30, color="#3b82f6", alpha=0.9)
        plt.title(f"Predicted {key} voxel counts ({split_name})")
        plt.xlabel("voxels")
        plt.ylabel("cases")
        fig.savefig(vols_dir / f"hist_{key}_{split_name}.png", bbox_inches="tight")
        plt.close(fig)

    # record a minimal manifest
    manifest = {
        "split": split_name,
        "images_dir": str(images_dir),
        "pred_dir": str(pred_dir),
        "dataset_json": str(dataset_json) if dataset_json else None,
        "channel_selection": {"flair": sel.flair_idx, "t1ce": sel.t1ce_idx},
        "label_ontology": {"et_label": ontology.et_label, "tc_labels": sorted(ontology.tc_labels)},
        "n_cases": len(case_ids),
        "outputs": {
            "qc_dir": str(qc_dir),
            "volumes_dir": str(vols_dir),
            "csv": str(csv_path),
            "json": str(json_path),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Done. Wrote:", out_dir)
    return out_dir


def main() -> int:
    ap = argparse.ArgumentParser(description="Export a visual QC pack (overlays + voxel-count plots) from nnU-Net style inputs.")
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--pred-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--dataset-json", required=False, type=Path, default=None)
    ap.add_argument("--split-name", required=False, default="imagesTs", choices=["imagesTs", "imagesTr"])
    ap.add_argument("--max-cases", required=False, type=int, default=None)
    args = ap.parse_args()

    export_pack(
        images_dir=args.images_dir,
        pred_dir=args.pred_dir,
        out_dir=args.out_dir,
        dataset_json=args.dataset_json,
        split_name=args.split_name,
        max_cases=args.max_cases,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
