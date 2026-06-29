# tools/diag_detection.py
"""
Standalone detection diagnostic.

Decodes ONE specified frame from a video file two different ways
(OpenCV VideoCapture and PyAV bgr24), then runs the SAME YOLO model
on each at multiple batch sizes, dumping pre-NMS top-15 confidences.

Now also reports confidence per spatial CLUSTER (groups nearby anchors)
so we can see what each of the 3 mosaic locations gets.

Use --fp32 to test fp32 inference (rules out fp16 numerical drift).

Usage:
    python tools/diag_detection.py `
        --model .\models\lada_mosaic_detection_model_v4_accurate.pt `
        --video D:\ToClean\PurpleRain.mp4 `
        --frame 5 `
        --imgsz 960
    # Then again with --fp32 to compare
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import cv2
import av

import ultralytics
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.augment import LetterBox
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils import nms, ops
from ultralytics.utils.checks import check_imgsz


def decode_opencv(video_path: str, frame_idx: int) -> np.ndarray:
    """OpenCV VideoCapture decode. Returns BGR HWC uint8."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"OpenCV failed to read frame {frame_idx}")
    return frame  # BGR HWC uint8


def decode_pyav(video_path: str, frame_idx: int) -> np.ndarray:
    """PyAV with bgr24 (Lada's exact decode). Returns BGR HWC uint8."""
    container = av.open(video_path, metadata_errors='ignore')
    stream = container.streams.video[0]
    stream.thread_type = 'AUTO'
    i = 0
    for packet in container.demux(stream):
        for frame in packet.decode():
            if i == frame_idx:
                arr = frame.to_ndarray(format='bgr24')
                container.close()
                return arr
            i += 1
    container.close()
    raise RuntimeError(f"PyAV did not find frame {frame_idx}")


class DiagModel:
    """Yolo11SegmentationModel — same code path as our LadaYoloDetector port."""

    def __init__(self, model_path, device, imgsz=960, fp16=True, conf=0.25, iou=0.45):
        yolo_model = YOLO(model_path)
        assert yolo_model.task == 'segment'
        self.stride = 32
        self.imgsz = check_imgsz(imgsz, stride=self.stride, min_dim=2)
        self.letterbox = LetterBox(self.imgsz, auto=True, stride=self.stride)

        custom = {"conf": conf, "batch": 1, "save": False, "mode": "predict",
                  "device": device, "half": fp16, "iou": iou}
        args = {**yolo_model.overrides, **custom}
        self.args = get_cfg(DEFAULT_CFG, args)

        self.device = torch.device(device)
        self.model = AutoBackend(
            model=yolo_model.model, device=self.device,
            dnn=self.args.dnn, data=self.args.data,
            fp16=self.args.half, fuse=True, verbose=False,
        )
        self.args.half = self.model.fp16
        self.model.eval()
        self.model.warmup(imgsz=(1, 3, *self.imgsz))
        self.dtype = torch.float16 if fp16 else torch.float32

    def preprocess(self, frames_np):
        """frames_np: list of BGR HWC uint8 numpy. Returns BCHW tensor."""
        im = np.stack([self.letterbox(image=f) for f in frames_np])
        im = im.transpose((0, 3, 1, 2))
        im = np.ascontiguousarray(im)
        return torch.from_numpy(im)

    def inference(self, imgs):
        """imgs: BCHW uint8 tensor. Returns preds tuple from model."""
        with torch.inference_mode():
            inp = imgs.to(device=self.device).to(dtype=self.dtype).div_(255.0)
            return self.model(inp, augment=False, visualize=False, embed=None)


def analyze_preds(preds, label, top_k=50):
    """
    Print pre-NMS analysis: top-K confidences, AND group anchors by spatial
    cluster (close box centers) to show what each mosaic location scores.
    """
    if isinstance(preds, (tuple, list)):
        if isinstance(preds[0], (tuple, list)):
            det = preds[0][0]
        else:
            det = preds[0]
    else:
        det = preds

    if not isinstance(det, torch.Tensor):
        print(f"  [{label}] cannot find det tensor, preds type={type(preds)}")
        return

    t = det[0]  # first batch elem
    if t.shape[0] < t.shape[1]:
        # nc may be 1 or 2 (mosaic models can be 2-class). Take max across known channels.
        # 4 box + nc class + 32 mask; deduce nc from shape.
        total_c = t.shape[0]
        nm = 32
        nc = total_c - 4 - nm
        class_scores = t[4:4 + nc, :]
        boxes_raw = t[:4, :]
    else:
        total_c = t.shape[1]
        nm = 32
        nc = total_c - 4 - nm
        class_scores = t[:, 4:4 + nc].T
        boxes_raw = t[:, :4].T

    max_conf = class_scores.max(dim=0).values
    n_top = min(top_k, max_conf.numel())
    top_vals, top_idx = max_conf.topk(n_top)
    top_vals_list = top_vals.cpu().tolist()
    top_centers = [(float(boxes_raw[0, int(top_idx[k])].item()),
                    float(boxes_raw[1, int(top_idx[k])].item()))
                   for k in range(n_top)]

    above_50 = int((max_conf > 0.50).sum().item())
    above_25 = int((max_conf > 0.25).sum().item())
    above_15 = int((max_conf > 0.15).sum().item())
    above_05 = int((max_conf > 0.05).sum().item())
    print(f"  [{label}] nc={nc} anchors >0.50:{above_50} >0.25:{above_25} >0.15:{above_15} >0.05:{above_05}")

    # Cluster top-K anchors by spatial proximity. Two anchors are in the same
    # cluster if their centers are within 80 px in model coords.
    clusters = []  # list of dicts: {center: (cx, cy), max_conf, count}
    for k in range(n_top):
        cx, cy = top_centers[k]
        conf = top_vals_list[k]
        assigned = False
        for c in clusters:
            ccx, ccy = c["center"]
            if abs(cx - ccx) < 80 and abs(cy - ccy) < 80:
                c["count"] += 1
                c["max_conf"] = max(c["max_conf"], conf)
                assigned = True
                break
        if not assigned:
            clusters.append({"center": (cx, cy), "max_conf": conf, "count": 1})

    # Sort clusters by max_conf desc
    clusters.sort(key=lambda c: -c["max_conf"])
    print(f"  [{label}] spatial clusters (top-{n_top} anchors, sorted by max conf):")
    for ci, c in enumerate(clusters[:6]):
        cx, cy = c["center"]
        print(f"      cluster#{ci}: center=({cx:6.1f},{cy:6.1f})  "
              f"max_conf={c['max_conf']:.3f}  anchor_count={c['count']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--frame", type=int, default=5)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fp32", action="store_true",
                    help="Force fp32 inference (default: fp16)")
    args = ap.parse_args()

    use_fp16 = not args.fp32
    precision_label = "fp16" if use_fp16 else "fp32"

    print(f"=== Diagnostic: detection on frame {args.frame} of {args.video} ===")
    print(f"model    = {args.model}")
    print(f"ultralytics version: {ultralytics.__version__}")
    print(f"torch version:       {torch.__version__}")
    print(f"CUDA:                {torch.version.cuda}")
    print(f"cuDNN:               {torch.backends.cudnn.version()}")
    print(f"cudnn.benchmark:     {torch.backends.cudnn.benchmark}")
    print(f"cudnn.deterministic: {torch.backends.cudnn.deterministic}")
    print(f"Inference precision: {precision_label}")
    print()

    # Decode the same frame two ways
    print("Decoding frame via OpenCV...")
    f_cv = decode_opencv(args.video, args.frame)
    print(f"  OpenCV  shape={f_cv.shape} dtype={f_cv.dtype}")

    print("Decoding frame via PyAV (Lada's exact path)...")
    f_av = decode_pyav(args.video, args.frame)
    print(f"  PyAV    shape={f_av.shape} dtype={f_av.dtype}")

    # Are they byte-identical?
    if f_cv.shape == f_av.shape:
        diff = np.abs(f_cv.astype(np.int16) - f_av.astype(np.int16))
        print(f"  OpenCV vs PyAV max abs pixel diff: {int(diff.max())}")
    print()

    # Build the model
    print(f"Loading model on {args.device}, fp16={use_fp16}...")
    m = DiagModel(args.model, device=args.device, imgsz=args.imgsz, fp16=use_fp16)
    print(f"  letterbox shape: {m.imgsz}")
    print(f"  model.fp16:      {m.model.fp16}")
    print()

    # Run on PyAV-decoded frame (Lada's exact decode), at batch=1 and batch=4
    # batch=4 matches Lada's default; batch=1 is the cleanest comparison.
    for decode_name, frame in [("PyAV", f_av), ("OpenCV", f_cv)]:
        print(f"--- Decode={decode_name} ---")
        for bs in (1, 4):
            frames_list = [frame.copy() for _ in range(bs)]
            imgs = m.preprocess(frames_list)
            preds = m.inference(imgs)
            label = f"{decode_name} b={bs} {precision_label}"
            analyze_preds(preds, label, top_k=50)
            print()


if __name__ == "__main__":
    main()
