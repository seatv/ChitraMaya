"""
Scan ALL frames of a video, reporting per-frame max confidence for each
of the 3 expected mosaic spatial locations.

Hypothesis: Lada detects the side mosaics on SOME frame in the sequence
(scene tracking then propagates the detection across frames). Our standalone
on a single frame might be missing the high-confidence frames.

Usage:
    python diag_scan_frames.py `
        --model .\models\lada_mosaic_detection_model_v4_accurate.pt `
        --video D:\ToClean\PurpleRain.mp4
"""
import argparse
import numpy as np
import torch
import cv2
import ultralytics
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.augment import LetterBox
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils import nms, ops
from ultralytics.utils.checks import check_imgsz


# 3 spatial regions in model space (480x544 letterboxed), where we expect mosaics.
# Anchors within 100px of these centers count as "in" the cluster.
REGIONS = {
    "middle":      (470, 304),
    "lower_right": (727, 441),
    "upper_left":  (155, 134),
}
TOLERANCE = 100  # px


class DiagModel:
    def __init__(self, model_path, device, imgsz=960, fp16=True):
        yolo_model = YOLO(model_path)
        self.stride = 32
        self.imgsz = check_imgsz(imgsz, stride=self.stride, min_dim=2)
        self.letterbox = LetterBox(self.imgsz, auto=True, stride=self.stride)
        custom = {"conf": 0.25, "batch": 1, "save": False, "mode": "predict",
                  "device": device, "half": fp16, "iou": 0.45}
        args = {**yolo_model.overrides, **custom}
        self.args = get_cfg(DEFAULT_CFG, args)
        self.device = torch.device(device)
        self.model = AutoBackend(model=yolo_model.model, device=self.device,
                                  dnn=False, data=None, fp16=fp16, fuse=True, verbose=False)
        self.args.half = self.model.fp16
        self.model.eval()
        self.model.warmup(imgsz=(1, 3, *self.imgsz))
        self.dtype = torch.float16 if fp16 else torch.float32

    def preprocess(self, frames_np):
        im = np.stack([self.letterbox(image=f) for f in frames_np])
        im = im.transpose((0, 3, 1, 2))
        im = np.ascontiguousarray(im)
        return torch.from_numpy(im)

    def inference(self, imgs):
        with torch.inference_mode():
            inp = imgs.to(device=self.device).to(dtype=self.dtype).div_(255.0)
            return self.model(inp, augment=False, visualize=False, embed=None)


def get_per_region_max_conf(preds):
    """Return {region_name: max_conf_for_anchors_in_that_region} for batch[0]."""
    det = preds[0][0] if isinstance(preds[0], (tuple, list)) else preds[0]
    t = det[0]
    nc = t.shape[0] - 4 - 32
    class_scores = t[4:4 + nc, :]  # (nc, A)
    boxes_raw = t[:4, :]  # (4, A) — xywh
    max_conf = class_scores.max(dim=0).values  # (A,)

    cx = boxes_raw[0, :]
    cy = boxes_raw[1, :]

    result = {}
    for region_name, (rx, ry) in REGIONS.items():
        in_region = (torch.abs(cx - rx) < TOLERANCE) & (torch.abs(cy - ry) < TOLERANCE)
        if in_region.any():
            region_max = float(max_conf[in_region].max().item())
        else:
            region_max = 0.0
        result[region_name] = region_max
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--every", type=int, default=1, help="Only scan every Nth frame")
    args = ap.parse_args()

    print(f"=== Frame Scanner ===")
    print(f"ultralytics: {ultralytics.__version__}, torch: {torch.__version__}, cuDNN: {torch.backends.cudnn.version()}")

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {args.video} ({total} frames)")
    print(f"Scanning every {args.every} frame(s)")
    print()

    m = DiagModel(args.model, device="cuda:0", imgsz=args.imgsz, fp16=True)
    print()

    # Track per-region max conf across the video
    overall_max = {r: 0.0 for r in REGIONS}
    overall_max_frame = {r: -1 for r in REGIONS}
    # Histogram of how many frames had conf above thresholds for each region
    histogram = {r: {"above_50": 0, "above_25": 0, "above_15": 0, "above_05": 0, "scanned": 0}
                 for r in REGIONS}

    print(f"{'frame':>5} | {'middle':>8} | {'lower_R':>8} | {'upper_L':>8}")
    print("-" * 45)

    scanned = 0
    for frame_idx in range(0, total, args.every):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue

        imgs = m.preprocess([frame])
        preds = m.inference(imgs)
        confs = get_per_region_max_conf(preds)
        scanned += 1

        for r in REGIONS:
            c = confs[r]
            histogram[r]["scanned"] += 1
            if c > 0.50: histogram[r]["above_50"] += 1
            if c > 0.25: histogram[r]["above_25"] += 1
            if c > 0.15: histogram[r]["above_15"] += 1
            if c > 0.05: histogram[r]["above_05"] += 1
            if c > overall_max[r]:
                overall_max[r] = c
                overall_max_frame[r] = frame_idx

        # Only print "interesting" frames to keep output compact
        interesting = any(confs[r] > 0.05 for r in ("lower_right", "upper_left"))
        if interesting or frame_idx % max(1, total // 20) == 0:
            print(f"{frame_idx:>5} | {confs['middle']:>8.3f} | {confs['lower_right']:>8.3f} | {confs['upper_left']:>8.3f}")

    cap.release()

    print("-" * 45)
    print(f"Scanned {scanned} frames")
    print()
    print(f"=== Per-region MAX conf across all scanned frames ===")
    for r in REGIONS:
        print(f"  {r:>11}: max_conf={overall_max[r]:.4f} at frame {overall_max_frame[r]}")
    print()
    print(f"=== Histogram of frames above each threshold ===")
    print(f"  {'region':>11} | {'>0.50':>6} | {'>0.25':>6} | {'>0.15':>6} | {'>0.05':>6}")
    for r in REGIONS:
        h = histogram[r]
        print(f"  {r:>11} | {h['above_50']:>6} | {h['above_25']:>6} | {h['above_15']:>6} | {h['above_05']:>6}")


if __name__ == "__main__":
    main()
