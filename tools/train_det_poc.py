# tools/train_det_poc.py
"""CM-112 Phase-A PoC: fine-tune a YOLO mosaic detector (Batch T1).

Thin wrapper over ultralytics training so the whole PoC is two commands:

    python tools/make_det_dataset.py --input clip.mp4 --out det-dataset
    python tools/train_det_poc.py --data det-dataset\\data.yaml

Defaults are sized for an overnight run on an 8 GB card at imgsz 800 (the
detection resolution ChitraMaya runs at). The output best.pt is a standard
ultralytics model: drop it into models\\ and compile it from Manage Models
exactly like the lada detectors -- the whole point of the PoC is that a
user-trained model flows through the EXISTING pipeline untouched.

Base-model choice (--base):
  - yolo11s.pt (default): COCO-pretrained small; auto-downloads. Clean-room
    baseline for "does the generated data train a competitive detector".
  - a lada detection .pt: in-domain transfer (their detector already knows
    mosaic). License note: lada models are AGPL -- a fine-tune is a
    derivative; share it under AGPL with credit, per the Giants doctrine.

Doctrine: restoration-only track, supervised perfect-pair data, no
generative models. Content never leaves the machine; share WEIGHTS only.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="CM-112 PoC: train a mosaic detector")
    ap.add_argument("--data", required=True, help="path to data.yaml from make_det_dataset.py")
    ap.add_argument("--base", default="yolo11s.pt",
                    help="base model: yolo11s.pt (default) or a lada det .pt for transfer")
    ap.add_argument("--imgsz", type=int, default=800,
                    help="train resolution; 800 = ChitraMaya's detection default")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=-1,
                    help="-1 = ultralytics auto-batch (fits VRAM automatically)")
    ap.add_argument("--device", default="0", help="CUDA device id, or 'cpu'")
    ap.add_argument("--project", default="det-poc-runs")
    ap.add_argument("--name", default="mosaic-det-poc")
    args = ap.parse_args()

    data = Path(args.data)
    if not data.exists():
        print(f"[train] data.yaml not found: {data}")
        return 1

    from ultralytics import YOLO
    print(f"[train] base={args.base} data={data} imgsz={args.imgsz} "
          f"epochs={args.epochs} batch={args.batch} device={args.device}")
    model = YOLO(str(args.base))
    results = model.train(
        data=str(data),
        imgsz=int(args.imgsz),
        epochs=int(args.epochs),
        batch=int(args.batch),
        device=args.device,
        project=str(args.project),
        name=str(args.name),
        exist_ok=True,
        # PoC choices: default augment pipeline EXCEPT mosaic-augmentation
        # collage (confusing here in every sense -- it tiles 4 images and
        # would teach scale/context lies about our synthetic regions), and
        # no flips-ups (mosaic patterns are orientation-agnostic; keep
        # fliplr default).
        mosaic=0.0,
        flipud=0.0,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print("=" * 70)
    print(f"[train] best weights: {best}")
    print("[train] validation (held-out generated frames):")
    metrics = YOLO(str(best)).val(data=str(data), imgsz=int(args.imgsz),
                                  device=args.device)
    try:
        print(f"[train] mAP50={metrics.box.map50:.4f}  "
              f"mAP50-95={metrics.box.map:.4f}  "
              f"precision={metrics.box.mp:.4f}  recall={metrics.box.mr:.4f}")
    except Exception:
        pass
    print("=" * 70)
    print("[train] NEXT STEPS")
    print("  1. Copy best.pt into your ChitraMaya models\\ folder (rename it")
    print("     something meaningful, e.g. my_mosaic_det_v0.pt).")
    print("  2. Manage Models -> select it -> Compile (same flow as lada).")
    print("  3. Benchmark against v2/v4 on the SFW synthetic test video:")
    print("     same misses-JSON counting as every other detector A/B.")
    print("  4. Share the WEIGHTS (HF repo + Manage Models '+ Add source'),")
    print("     never the training content. AGPL + credit if you fine-tuned")
    print("     from a lada model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
