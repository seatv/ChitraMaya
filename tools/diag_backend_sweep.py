"""
Sweep torch backend configurations on a single frame to find what makes
upper-left mosaic confidence rise above ~0.001.

If any config makes UL conf >> 0.001, we've found the bug.
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
from ultralytics.utils.checks import check_imgsz


REGIONS = {"middle": (470, 304), "lower_right": (727, 441), "upper_left": (155, 134)}
TOLERANCE = 100


def decode(video, frame_idx):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, f = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Can't decode frame {frame_idx}")
    return f


def load_model(model_path, device, imgsz, fp16):
    yolo_model = YOLO(model_path)
    stride = 32
    imgsz_t = check_imgsz(imgsz, stride=stride, min_dim=2)
    letterbox = LetterBox(imgsz_t, auto=True, stride=stride)
    custom = {"conf": 0.25, "batch": 1, "save": False, "mode": "predict",
              "device": device, "half": fp16, "iou": 0.45}
    args = {**yolo_model.overrides, **custom}
    args_cfg = get_cfg(DEFAULT_CFG, args)
    model = AutoBackend(model=yolo_model.model, device=torch.device(device),
                        dnn=False, data=None, fp16=fp16, fuse=True, verbose=False)
    model.eval()
    model.warmup(imgsz=(1, 3, *imgsz_t))
    return model, letterbox, imgsz_t, fp16


def preprocess(letterbox, frames_np):
    im = np.stack([letterbox(image=f) for f in frames_np])
    im = im.transpose((0, 3, 1, 2))
    im = np.ascontiguousarray(im)
    return torch.from_numpy(im)


def per_region_max(preds):
    det = preds[0][0] if isinstance(preds[0], (tuple, list)) else preds[0]
    t = det[0]
    nc = t.shape[0] - 4 - 32
    class_scores = t[4:4 + nc, :]
    boxes_raw = t[:4, :]
    max_conf = class_scores.max(dim=0).values
    cx, cy = boxes_raw[0, :], boxes_raw[1, :]
    out = {}
    for name, (rx, ry) in REGIONS.items():
        in_r = (torch.abs(cx - rx) < TOLERANCE) & (torch.abs(cy - ry) < TOLERANCE)
        out[name] = float(max_conf[in_r].max().item()) if in_r.any() else 0.0
    return out


def run_inference(model, letterbox, frame, device, fp16):
    imgs = preprocess(letterbox, [frame])
    with torch.inference_mode():
        inp = imgs.to(device=torch.device(device)).to(dtype=torch.float16 if fp16 else torch.float32).div_(255.0)
        preds = model(inp, augment=False, visualize=False, embed=None)
    return per_region_max(preds)


def report(label, confs, fp16):
    print(f"  {label:>55} | mid={confs['middle']:.4f}  LR={confs['lower_right']:.4f}  UL={confs['upper_left']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--frame", type=int, default=0,
                    help="Frame index Lada is known to detect UL on (default 0)")
    ap.add_argument("--imgsz", type=int, default=960)
    args = ap.parse_args()

    print(f"=== Backend sweep on frame {args.frame} of {args.video} ===")
    print(f"torch={torch.__version__}, cuDNN={torch.backends.cudnn.version()}, ultralytics={ultralytics.__version__}")
    print()

    frame = decode(args.video, args.frame)
    print(f"Frame decoded: shape={frame.shape}")
    print()

    device = "cuda:0"

    # ---- Baseline: as we've been running it ----
    print("[baseline: cudnn on, benchmark=False, det=False, tf32=default, fp16]")
    model, letterbox, imgsz_t, fp16 = load_model(args.model, device, args.imgsz, fp16=True)
    confs = run_inference(model, letterbox, frame, device, fp16=True)
    report("baseline fp16", confs, fp16=True)
    print()

    # ---- Config 1: cudnn disabled ----
    print("[Config 1: cudnn DISABLED]")
    torch.backends.cudnn.enabled = False
    model, letterbox, _, _ = load_model(args.model, device, args.imgsz, fp16=True)
    confs = run_inference(model, letterbox, frame, device, fp16=True)
    report("cudnn=disabled fp16", confs, fp16=True)
    torch.backends.cudnn.enabled = True  # restore
    print()

    # ---- Config 2: cudnn deterministic ----
    print("[Config 2: cudnn DETERMINISTIC]")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model, letterbox, _, _ = load_model(args.model, device, args.imgsz, fp16=True)
    confs = run_inference(model, letterbox, frame, device, fp16=True)
    report("cudnn.deterministic=True fp16", confs, fp16=True)
    torch.backends.cudnn.deterministic = False
    print()

    # ---- Config 3: cudnn benchmark on ----
    print("[Config 3: cudnn BENCHMARK on]")
    torch.backends.cudnn.benchmark = True
    model, letterbox, _, _ = load_model(args.model, device, args.imgsz, fp16=True)
    confs = run_inference(model, letterbox, frame, device, fp16=True)
    report("cudnn.benchmark=True fp16", confs, fp16=True)
    torch.backends.cudnn.benchmark = False
    print()

    # ---- Config 4: TF32 off ----
    print("[Config 4: TF32 OFF]")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, letterbox, _, _ = load_model(args.model, device, args.imgsz, fp16=True)
    confs = run_inference(model, letterbox, frame, device, fp16=True)
    report("tf32 OFF fp16", confs, fp16=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print()

    # ---- Config 5: TF32 ON explicitly ----
    print("[Config 5: TF32 ON explicit]")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model, letterbox, _, _ = load_model(args.model, device, args.imgsz, fp16=True)
    confs = run_inference(model, letterbox, frame, device, fp16=True)
    report("tf32 ON fp16", confs, fp16=True)
    print()

    # ---- Config 6: CPU inference (most extreme reference point) ----
    print("[Config 6: CPU inference fp32]")
    model, letterbox, _, _ = load_model(args.model, "cpu", args.imgsz, fp16=False)
    confs = run_inference(model, letterbox, frame, "cpu", fp16=False)
    report("CPU fp32", confs, fp16=False)
    print()


if __name__ == "__main__":
    main()
