from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _lazy_import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


@dataclass(frozen=True)
class SplitCounts:
    split: str
    case_ids: list[str]
    wt: np.ndarray
    tc: np.ndarray
    et: np.ndarray


def _read_counts_csv(path: Path) -> SplitCounts:
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        case_ids: list[str] = []
        wt: list[int] = []
        tc: list[int] = []
        et: list[int] = []
        for row in r:
            case_ids.append(str(row["case_id"]))
            wt.append(int(float(row["WT"])))
            tc.append(int(float(row["TC"])))
            et.append(int(float(row["ET"])))

    split = "imagesTs" if "imagests" in path.name.lower() else ("imagesTr" if "imagestr" in path.name.lower() else "unknown")
    return SplitCounts(split=split, case_ids=case_ids, wt=np.array(wt), tc=np.array(tc), et=np.array(et))


def _quantiles(x: np.ndarray) -> dict[str, float]:
    x = x.astype(np.float64)
    qs = np.percentile(x, [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100])
    return {
        "n": float(x.size),
        "nonzero": float(np.sum(x > 0)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(qs[0]),
        "p01": float(qs[1]),
        "p05": float(qs[2]),
        "p10": float(qs[3]),
        "p25": float(qs[4]),
        "p50": float(qs[5]),
        "p75": float(qs[6]),
        "p90": float(qs[7]),
        "p95": float(qs[8]),
        "p99": float(qs[9]),
        "max": float(qs[10]),
    }


def _write_md_table(title: str, rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cols = [
        "region",
        "n",
        "nonzero",
        "mean",
        "std",
        "min",
        "p25",
        "p50",
        "p75",
        "p90",
        "p95",
        "p99",
        "max",
    ]

    def fmt(v: Any) -> str:
        if isinstance(v, (int, np.integer)):
            return f"{int(v):d}"
        if isinstance(v, (float, np.floating)):
            if abs(float(v)) >= 1000:
                return f"{float(v):,.1f}"
            return f"{float(v):.3f}"
        return str(v)

    lines: list[str] = []
    lines.append(f"# {title}\n")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latex_table(caption: str, label: str, rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cols = ["region", "mean", "std", "p25", "p50", "p75", "p95", "max"]

    def fmt(v: Any) -> str:
        if isinstance(v, (float, np.floating, int, np.integer)):
            return f"{float(v):.3f}"
        return str(v)

    lines: list[str] = []
    lines.append("% Auto-generated. Copy into paper appendix or results section.")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{lrrrrrrr}")
    lines.append("\\toprule")
    lines.append("Region & Mean & Std & P25 & P50 & P75 & P95 & Max \\\\")
    lines.append("\\midrule")
    for r in rows:
        lines.append(
            "{region} & {mean} & {std} & {p25} & {p50} & {p75} & {p95} & {max} \\\\".format(
                region=str(r.get("region")),
                mean=fmt(r.get("mean")),
                std=fmt(r.get("std")),
                p25=fmt(r.get("p25")),
                p50=fmt(r.get("p50")),
                p75=fmt(r.get("p75")),
                p95=fmt(r.get("p95")),
                max=fmt(r.get("max")),
            )
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_histograms(split: SplitCounts, out_dir: Path) -> None:
    plt = _lazy_import_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)

    for region, arr, color in [
        ("WT", split.wt, "#2563eb"),
        ("TC", split.tc, "#16a34a"),
        ("ET", split.et, "#dc2626"),
    ]:
        for log in [False, True]:
            fig = plt.figure(figsize=(8, 4.5), dpi=160)
            x = arr.astype(np.float64)
            if log:
                x = np.log10(x + 1.0)
                plt.hist(x, bins=35, color=color, alpha=0.9)
                plt.xlabel("log10(voxels + 1)")
                suffix = "log"
            else:
                plt.hist(x, bins=35, color=color, alpha=0.9)
                plt.xlabel("voxels")
                suffix = "linear"

            plt.title(f"{split.split}: {region} voxel-count distribution ({suffix})")
            plt.ylabel("cases")
            plt.grid(True, alpha=0.25)
            fig.savefig(out_dir / f"hist_{region}_{split.split}_{suffix}.png", bbox_inches="tight")
            plt.close(fig)


def _plot_box_violin(split: SplitCounts, out_dir: Path) -> None:
    plt = _lazy_import_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = [split.wt.astype(np.float64), split.tc.astype(np.float64), split.et.astype(np.float64)]
    labels = ["WT", "TC", "ET"]

    fig = plt.figure(figsize=(7.5, 4.5), dpi=160)
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.title(f"{split.split}: voxel counts (boxplot)")
    plt.ylabel("voxels")
    plt.grid(True, axis="y", alpha=0.25)
    fig.savefig(out_dir / f"boxplot_voxels_{split.split}.png", bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(7.5, 4.5), dpi=160)
    plt.violinplot(data, showmeans=True, showmedians=True)
    plt.xticks([1, 2, 3], labels)
    plt.title(f"{split.split}: voxel counts (violin)")
    plt.ylabel("voxels")
    plt.grid(True, axis="y", alpha=0.25)
    fig.savefig(out_dir / f"violin_voxels_{split.split}.png", bbox_inches="tight")
    plt.close(fig)


def _plot_scatter(split: SplitCounts, out_dir: Path) -> None:
    plt = _lazy_import_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)

    def scat(x: np.ndarray, y: np.ndarray, xlab: str, ylab: str, name: str) -> None:
        fig = plt.figure(figsize=(5.5, 5.5), dpi=160)
        plt.scatter(np.log10(x + 1.0), np.log10(y + 1.0), s=10, alpha=0.45)
        plt.xlabel(f"log10({xlab}+1)")
        plt.ylabel(f"log10({ylab}+1)")
        plt.title(f"{split.split}: {ylab} vs {xlab} (log scale)")
        plt.grid(True, alpha=0.25)
        fig.savefig(out_dir / f"scatter_{name}_{split.split}.png", bbox_inches="tight")
        plt.close(fig)

    scat(split.wt, split.tc, "WT", "TC", "tc_vs_wt")
    scat(split.wt, split.et, "WT", "ET", "et_vs_wt")
    scat(split.tc, split.et, "TC", "ET", "et_vs_tc")


def _et_prevalence(split: SplitCounts) -> dict[str, float]:
    n = float(split.et.size)
    if n <= 0:
        return {"n": 0.0, "et_nonzero": 0.0, "et_nonzero_pct": 0.0}
    nz = float(np.sum(split.et > 0))
    return {"n": n, "et_nonzero": nz, "et_nonzero_pct": 100.0 * nz / n}


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate paper-ready plots + tables from an exported Pediatric visualisations run.")
    ap.add_argument("--run-dir", type=Path, required=True, help="Path to a visualisations pack run folder (contains volumes/, qc/, manifest.json)")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output folder for paper assets")
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    out_dir: Path = args.out_dir

    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}")

    volumes_dir = run_dir / "volumes"
    if not volumes_dir.exists():
        print(f"[WARN] volumes/ not found under: {run_dir}")
        print("This script needs the exported visualisations pack output.")
        return 0

    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            (out_dir / "manifest_used.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        except Exception:
            pass

    csv_files = sorted(volumes_dir.glob("predicted_voxel_counts_*.csv"))
    if not csv_files:
        print(f"[WARN] No predicted_voxel_counts_*.csv under: {volumes_dir}")
        return 0

    out_tables = out_dir / "tables"
    out_plots = out_dir / "plots"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_plots.mkdir(parents=True, exist_ok=True)

    for csv_path in csv_files:
        split = _read_counts_csv(csv_path)

        rows = []
        for region, arr in [("WT", split.wt), ("TC", split.tc), ("ET", split.et)]:
            q = _quantiles(arr)
            q["region"] = region
            rows.append(q)

        title = f"Voxel-count quantiles ({split.split})"
        _write_md_table(title, rows, out_tables / f"volume_quantiles_{split.split}.md")
        _write_latex_table(
            caption=f"Voxel-count summary for {split.split} predictions (WT/TC/ET).",
            label=f"tab:voxels_{split.split}",
            rows=rows,
            out_path=out_tables / f"volume_quantiles_{split.split}.tex",
        )

        prev = _et_prevalence(split)
        (out_tables / f"et_prevalence_{split.split}.json").write_text(json.dumps(prev, indent=2), encoding="utf-8")

        _plot_histograms(split, out_plots)
        _plot_box_violin(split, out_plots)
        _plot_scatter(split, out_plots)

        print(f"[OK] Generated tables/plots for {split.split} from {csv_path.name}")

    print("Done. Output:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
