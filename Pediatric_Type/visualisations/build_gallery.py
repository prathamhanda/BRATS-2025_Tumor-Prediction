from __future__ import annotations

import argparse
import html
from pathlib import Path


def write_gallery(title: str, image_paths: list[Path], out_html: Path, run_dir: Path) -> None:
    out_html.parent.mkdir(parents=True, exist_ok=True)

    rows: list[str] = []
    for path in image_paths:
        rel = path.relative_to(out_html.parent)
        rows.append(
            f"<div class='card'><a href='{rel.as_posix()}' target='_blank'>"
            f"<img src='{rel.as_posix()}' loading='lazy'/></a>"
            f"<div class='cap'>{html.escape(path.stem)}</div></div>"
        )

    page = f"""<!doctype html>
<html>
<head>
<meta charset='utf-8'/>
<title>{html.escape(title)}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 16px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 12px; }}
.card {{ border: 1px solid #ddd; border-radius: 8px; padding: 8px; }}
img {{ width: 100%; height: auto; border-radius: 6px; }}
.cap {{ margin-top: 6px; font-size: 12px; color: #444; word-break: break-all; }}
.small {{ color: #666; font-size: 12px; }}
</style>
</head>
<body>
<h2>{html.escape(title)}</h2>
<p class='small'>Generated from {html.escape(str(run_dir))}</p>
<div class='grid'>
{''.join(rows)}
</div>
</body>
</html>"""
    out_html.write_text(page, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a simple HTML gallery for an exported visualisations run.")
    ap.add_argument("--run-dir", required=True, type=Path, help="Run folder containing qc/ comparisons/ volumes/ etc")
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}")

    gallery_dir = run_dir / "gallery"
    gallery_dir.mkdir(exist_ok=True)

    qc_ts = sorted((run_dir / "qc" / "imagesTs").glob("*.png"))
    qc_tr = sorted((run_dir / "qc" / "imagesTr").glob("*.png"))
    cmp_ts = sorted((run_dir / "comparisons" / "imagesTs_raw_vs_post").glob("*.png"))
    plots = sorted((run_dir / "volumes").glob("*.png"))

    parts: list[str] = []

    if qc_ts:
        write_gallery("QC overlays — imagesTs", qc_ts, gallery_dir / "qc_imagesTs.html", run_dir)
        parts.append("<li><a href='gallery/qc_imagesTs.html'>QC overlays — imagesTs</a></li>")
    if qc_tr:
        write_gallery("QC overlays — imagesTr", qc_tr, gallery_dir / "qc_imagesTr.html", run_dir)
        parts.append("<li><a href='gallery/qc_imagesTr.html'>QC overlays — imagesTr</a></li>")
    if cmp_ts:
        write_gallery("Raw vs Post comparisons — imagesTs", cmp_ts, gallery_dir / "compare_imagesTs.html", run_dir)
        parts.append("<li><a href='gallery/compare_imagesTs.html'>Raw vs Post — imagesTs</a></li>")
    if plots:
        write_gallery("Summary plots", plots, gallery_dir / "plots.html", run_dir)
        parts.append("<li><a href='gallery/plots.html'>Summary plots</a></li>")

    index = run_dir / "index.html"
    index_html = f"""<!doctype html>
<html>
<head>
<meta charset='utf-8'/>
<title>Visualisations Pack</title>
<style>body {{ font-family: Arial, sans-serif; margin: 16px; }}</style>
</head>
<body>
<h2>Visualisations Pack</h2>
<p>Folder: <code>{html.escape(str(run_dir))}</code></p>
<ul>
{''.join(parts) if parts else '<li>(No PNGs found)</li>'}
</ul>
</body>
</html>"""
    index.write_text(index_html, encoding="utf-8")
    print("Wrote:", index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
