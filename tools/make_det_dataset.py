# tools/make_det_dataset.py
"""CM-112 Phase-A PoC: generate a mosaic-detector training dataset (Batch T1).

The founding method of this field, automated: because WE place the mosaic,
the ground-truth boxes are exact -- no manual annotation, ever. Point this
at any videos (SFW for anything shareable), and it produces a YOLO-format
dataset (images + labels + data.yaml) ready for tools/train_det_poc.py.

Per sampled frame:
  - with --clean-fraction probability the frame is kept CLEAN (a negative
    sample with an empty label file -- teaches the detector what is NOT a
    mosaic);
  - otherwise 1..--max-regions rectangles are pixelated using ChitraMaya's
    own mosaic renderer (chitramaya.mosaic.add_mosaic, the gRestorer-lineage
    core the Add Mosaic feature ships), with the JAV block-size convention:
    block scales with region size (longest_side / U(12, 28), clamped 4..40);
  - the frame is then JPEG re-encoded at a random quality (55..95) so the
    mosaic carries compression artifacts, as every real-world mosaic does.

Labels are YOLO txt (class 0 = mosaic, normalized cx cy w h). Layout:

    <out>/images/{train,val}/*.jpg
    <out>/labels/{train,val}/*.txt
    <out>/data.yaml

Usage (repo root, venv active):
    python tools/make_det_dataset.py --input path\to\clip1.mp4 --input clip2.mp4 ^
        --out det-dataset --frames-per-video 400 --seed 7

Doctrine notes:
  - RESTORATION-ONLY track: this generates supervised DETECTION data from
    perfect pairs. No generative models anywhere in this pipeline.
  - Content never leaves the machine. Share trained WEIGHTS, not data.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

# Repo import: the SAME pixelate core the Add Mosaic feature uses, so the
# training data matches what ChitraMaya itself renders. Run from repo root.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch  # noqa: E402
from chitramaya.mosaic.add_mosaic import pixelate_roi_bgr_u8_inplace  # noqa: E402


def _rand_region(rng: random.Random, w: int, h: int):
    """One plausible mosaic region: 6-30%% of min dim, aspect 0.5-2.0."""
    base = min(w, h)
    size = rng.uniform(0.06, 0.30) * base
    aspect = rng.uniform(0.5, 2.0)
    rw = max(8, int(size * aspect))
    rh = max(8, int(size / aspect))
    rw = min(rw, w - 2)
    rh = min(rh, h - 2)
    l = rng.randint(0, max(0, w - rw - 1))
    t = rng.randint(0, max(0, h - rh - 1))
    return (t, l, t + rh - 1, l + rw - 1)   # inclusive (t, l, b, r)


def _textured_enough(frame, roi, min_std: float) -> bool:
    """Reject regions on flat/low-texture areas. A mosaic over a
    featureless patch is INVISIBLE -- labeling it 'mosaic' is label noise
    that teaches the detector to hallucinate (caught on the first sandbox
    sanity image: a region on flat background showed no visible change).
    Real mosaic sits over textured content; require the same of ours."""
    t, l, b, r = roi
    patch = frame[t:b + 1, l:r + 1]
    if patch.size == 0:
        return False
    return float(patch.std()) >= float(min_std)


def _block_for(rng: random.Random, roi) -> int:
    """JAV convention: block size scales with region size."""
    t, l, b, r = roi
    longest = max(b - t + 1, r - l + 1)
    return int(max(4, min(40, longest / rng.uniform(12.0, 28.0))))


def _yolo_line(roi, w: int, h: int) -> str:
    t, l, b, r = roi
    cx = (l + r + 1) / 2.0 / w
    cy = (t + b + 1) / 2.0 / h
    bw = (r - l + 1) / w
    bh = (b - t + 1) / h
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def _sample_indices(n_total: int, n_want: int, rng: random.Random):
    if n_total <= 0:
        return []
    n_want = min(n_want, n_total)
    return sorted(rng.sample(range(n_total), n_want))


def main() -> int:
    ap = argparse.ArgumentParser(description="CM-112 PoC: mosaic-detector dataset generator")
    ap.add_argument("--input", action="append", required=True,
                    help="source video (repeatable); SFW for anything shareable")
    ap.add_argument("--out", default="det-dataset", help="output dataset directory")
    ap.add_argument("--frames-per-video", type=int, default=400)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--clean-fraction", type=float, default=0.15,
                    help="fraction of frames kept clean (negative samples)")
    ap.add_argument("--max-regions", type=int, default=3)
    ap.add_argument("--jpeg-q-min", type=int, default=55)
    ap.add_argument("--jpeg-q-max", type=int, default=95)
    ap.add_argument("--min-texture-std", type=float, default=12.0,
                    help="reject regions whose pre-mosaic pixel stddev is "
                         "below this (flat areas make invisible mosaic = "
                         "label noise); 0 disables")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_img = 0
    n_boxes = 0
    n_clean = 0
    for vid_i, vid in enumerate(args.input):
        cap = cv2.VideoCapture(str(vid))
        if not cap.isOpened():
            print(f"[dataset] WARNING: cannot open {vid}; skipping")
            continue
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        picks = _sample_indices(n_total, args.frames_per_video, rng)
        print(f"[dataset] {Path(vid).name}: {n_total} frames, sampling {len(picks)}")
        stem = f"v{vid_i:02d}_{Path(vid).stem}"

        for k, idx in enumerate(picks):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            h, w = frame.shape[:2]

            rois = []
            if rng.random() >= args.clean_fraction:
                want = rng.randint(1, max(1, args.max_regions))
                tries = 0
                while len(rois) < want and tries < want * 8:
                    tries += 1
                    roi = _rand_region(rng, w, h)
                    if args.min_texture_std > 0 and not _textured_enough(
                            frame, roi, args.min_texture_std):
                        continue
                    rois.append(roi)

            if rois:
                fb = torch.from_numpy(frame)   # uint8 [H, W, 3] BGR, CPU
                for roi in rois:
                    pixelate_roi_bgr_u8_inplace(fb, roi=roi, block=_block_for(rng, roi))
                frame = fb.numpy()
            else:
                n_clean += 1

            # Real-world mosaic has been through an encoder: JPEG-cycle the
            # frame at a random quality so the detector learns compressed
            # mosaic, not pristine block edges.
            q = rng.randint(args.jpeg_q_min, args.jpeg_q_max)
            okj, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            if not okj:
                continue

            split = "val" if rng.random() < args.val_fraction else "train"
            name = f"{stem}_{idx:07d}"
            (out / "images" / split / f"{name}.jpg").write_bytes(buf.tobytes())
            lines = [_yolo_line(r, w, h) for r in rois]
            (out / "labels" / split / f"{name}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="ascii")
            n_img += 1
            n_boxes += len(rois)
            if (k + 1) % 100 == 0:
                print(f"[dataset]   {k + 1}/{len(picks)} frames done")
        cap.release()

    yaml_path = out / "data.yaml"
    yaml_path.write_text(
        "# CM-112 PoC dataset (generated by tools/make_det_dataset.py)\n"
        f"path: {out.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: mosaic\n", encoding="ascii")

    print(f"[dataset] DONE: {n_img} images ({n_clean} clean negatives), "
          f"{n_boxes} mosaic boxes -> {out}")
    print(f"[dataset] data.yaml: {yaml_path}")
    print("[dataset] next: python tools/train_det_poc.py --data "
          f"{yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
