# tools/det_labels_to_seg.py
"""CM-112: convert a YOLO DETECT dataset into a SEGMENT dataset (Batch T2).

Why: lada detection models are SEGMENTATION models (v2 = YOLO11m-seg), and
ultralytics refuses to fine-tune a seg model on detect labels ("Segment
dataset requires equal numbers of boxes and segments" -- field event
2026-08-23, the lada-transfer smoke test). Our generated regions are
axis-aligned rectangles, so the segment polygon is simply the box's four
corners: a pure text transformation of the labels that already exist. No
image is re-rendered and the source dataset is untouched, which keeps the
from-COCO vs from-lada comparison apples-to-apples -- identical images,
identical regions, only the starting weights differ.

Images are HARD-LINKED into the new dataset when both trees sit on the same
NTFS volume (zero extra disk); if linking fails they are copied. Empty label
files (clean negatives) stay empty -- the seg trainer reads them as
background images, same as the detect trainer.

Usage (repo root, venv active):
    python tools/det_labels_to_seg.py --src det-dataset --out det-dataset-seg
    python tools/train_det_poc.py --data det-dataset-seg/data.yaml \
        --base models/lada_mosaic_detection_model_v2.pt \
        --name mosaic-det-lada-transfer

License note (Giants doctrine): a fine-tune of lada weights is an AGPL
derivative -- share the resulting model under AGPL with credit.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def _det_line_to_seg(line: str) -> str | None:
    """'cls cx cy w h' (normalized) -> 'cls x1 y1 x2 y1 x2 y2 x1 y2'.

    Returns None for lines that do not parse as a 5-field detect label.
    """
    parts = line.split()
    if len(parts) != 5:
        return None
    try:
        cls = int(float(parts[0]))
        cx, cy, w, h = (float(v) for v in parts[1:])
    except ValueError:
        return None
    x1 = max(0.0, cx - w / 2.0)
    y1 = max(0.0, cy - h / 2.0)
    x2 = min(1.0, cx + w / 2.0)
    y2 = min(1.0, cy + h / 2.0)
    pts = (x1, y1, x2, y1, x2, y2, x1, y2)  # clockwise rectangle
    return str(cls) + " " + " ".join(f"{p:.6f}" for p in pts)


def _place_image(src: Path, dst: Path) -> str:
    """Hard-link src -> dst if possible (same volume), else copy."""
    if dst.exists():
        return "kept"
    try:
        os.link(src, dst)
        return "linked"
    except OSError:
        shutil.copy2(src, dst)
        return "copied"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="CM-112: detect dataset -> segment dataset (rect polygons)")
    ap.add_argument("--src", default="det-dataset",
                    help="existing detect dataset (from make_det_dataset.py)")
    ap.add_argument("--out", default="det-dataset-seg",
                    help="segment dataset to create")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if not (src / "images").is_dir() or not (src / "labels").is_dir():
        print(f"[seg] ERROR: {src} does not look like a YOLO dataset "
              "(need images/ and labels/)")
        return 1

    n_img = 0
    n_lines = 0
    n_empty = 0
    n_bad = 0
    linked = 0
    copied = 0
    for split in ("train", "val"):
        img_dir = src / "images" / split
        lbl_dir = src / "labels" / split
        if not img_dir.is_dir():
            continue
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

        for img in sorted(img_dir.iterdir()):
            if not img.is_file():
                continue
            lbl = lbl_dir / (img.stem + ".txt")
            seg_lines: list[str] = []
            if lbl.exists():
                for raw in lbl.read_text(encoding="ascii").splitlines():
                    raw = raw.strip()
                    if not raw:
                        continue
                    seg = _det_line_to_seg(raw)
                    if seg is None:
                        n_bad += 1
                        print(f"[seg] WARNING: unparseable label line in "
                              f"{lbl.name}: {raw!r}")
                        continue
                    seg_lines.append(seg)
            how = _place_image(img, out / "images" / split / img.name)
            if how == "linked":
                linked += 1
            elif how == "copied":
                copied += 1
            (out / "labels" / split / (img.stem + ".txt")).write_text(
                "\n".join(seg_lines) + ("\n" if seg_lines else ""),
                encoding="ascii")
            n_img += 1
            n_lines += len(seg_lines)
            if not seg_lines:
                n_empty += 1

    yaml_path = out / "data.yaml"
    yaml_path.write_text(
        "# CM-112 SEGMENT dataset (rect polygons; tools/det_labels_to_seg.py)\n"
        f"path: {out.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: mosaic\n", encoding="ascii")

    print(f"[seg] DONE: {n_img} images ({n_empty} background/negative), "
          f"{n_lines} rect polygons -> {out}")
    print(f"[seg] images: {linked} hard-linked, {copied} copied "
          f"(source dataset untouched)")
    if n_bad:
        print(f"[seg] WARNING: {n_bad} label line(s) skipped as unparseable")
    print(f"[seg] data.yaml: {yaml_path}")
    print("[seg] next: python tools/train_det_poc.py --data "
          f"{yaml_path} --base models/lada_mosaic_detection_model_v2.pt "
          "--name mosaic-det-lada-transfer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())