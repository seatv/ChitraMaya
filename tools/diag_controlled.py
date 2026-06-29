"""
Controlled experiment: decode and detect through ChitraMaya's ACTUAL code path
(NVDEC decoder + rgbp->rgb->bgr conversion + LadaYoloDetector + Yolo11SegmentationModel
all imported from ChitraMaya).

The only variable is --imgsz. The aim is to confirm that with our exact pipeline,
at imgsz=640 the model finds 3 mosaics on PurpleRain frame 44 (matching Lada
CLI's behavior). Anything other than imgsz being the determining factor would
show up here.

Usage:
    python tools\diag_controlled.py `
        --model .\models\lada_mosaic_detection_model_v4_accurate.pt `
        --video D:\ToClean\PurpleRain.mp4 `
        --frame 44 `
        --imgsz 640

    # Compare against:
    python tools\diag_controlled.py ... --imgsz 960
"""
import argparse
import os
import sys
import numpy as np
import torch

# Import ChitraMaya's actual modules
from chitramaya.video.decoder import Decoder
from chitramaya.mosaic.pipeline_utils import (
    rgbp_chw_to_rgb_hwc_u8,
    rgb_hwc_to_bgr_hwc_u8,
    wrap_surface_as_tensor,
)
from chitramaya.mosaic.detector.lada_yolo import LadaYoloDetector, Yolo11SegmentationModel


REGIONS_960 = {
    "middle":      (470, 304),
    "lower_right": (727, 441),
    "upper_left":  (155, 134),
}
REGIONS_640 = {
    "middle":      (313, 213),
    "upper_left":  (104, 100),
    "third":       (485, 304),
}


def decode_one_frame_via_chitramaya(video_path: str, frame_idx: int, device: str = "cuda:0") -> torch.Tensor:
    """
    Use ChitraMaya's actual Decoder + conversion chain to produce a single
    BGR HWC uint8 tensor for the requested absolute frame index.
    Returns the frame still on its source device (GPU for NVDEC).
    """
    dec = Decoder(video_path, gpu_id=0, batch_size=8, output_format="RGBP")

    abs_idx = 0
    target_frame_bgr_u8 = None

    while True:
        batch = dec.read_batch()
        if not batch:
            break
        for item in batch:
            if abs_idx == frame_idx:
                # Convert through exact pipeline path: surface -> RGBP CHW -> RGB HWC -> BGR HWC
                if isinstance(item, torch.Tensor):
                    rgb = item
                    if rgb.device != torch.device(device):
                        rgb = rgb.to(device, non_blocking=True)
                else:
                    t = wrap_surface_as_tensor(item)
                    if t.ndim == 3 and t.shape[-1] == 3:
                        rgb = t
                    else:
                        rgb = rgbp_chw_to_rgb_hwc_u8(t)
                    if rgb.device != torch.device(device):
                        rgb = rgb.to(device, non_blocking=True)
                rgb = rgb.contiguous()
                bgr_u8 = rgb_hwc_to_bgr_hwc_u8(rgb)
                target_frame_bgr_u8 = bgr_u8.clone()  # detach from prefetch queue
                break
            abs_idx += 1
        if target_frame_bgr_u8 is not None:
            break

    dec.close()
    if target_frame_bgr_u8 is None:
        raise RuntimeError(f"Could not decode frame {frame_idx}")
    return target_frame_bgr_u8


def analyze_preds_per_region(preds, regions, label):
    """Print per-region max_conf and top-15 confs from raw preds[0]."""
    cand = preds[0]
    if isinstance(cand, (tuple, list)):
        cand = cand[0]
    if not isinstance(cand, torch.Tensor):
        print(f"  [{label}] cand is not Tensor: {type(cand)}")
        return

    t = cand[0]  # batch[0]
    total_c = t.shape[0] if t.shape[0] < t.shape[1] else t.shape[1]
    nm = 32
    nc = total_c - 4 - nm
    if t.shape[0] < t.shape[1]:
        class_scores = t[4:4 + nc, :]
        boxes_raw = t[:4, :]
    else:
        class_scores = t[:, 4:4 + nc].T
        boxes_raw = t[:, :4].T

    max_conf = class_scores.max(dim=0).values
    above_50 = int((max_conf > 0.50).sum().item())
    above_25 = int((max_conf > 0.25).sum().item())
    above_15 = int((max_conf > 0.15).sum().item())
    above_05 = int((max_conf > 0.05).sum().item())
    print(f"  [{label}] nc={nc} >0.50:{above_50} >0.25:{above_25} >0.15:{above_15} >0.05:{above_05}")
    top_vals, _ = max_conf.topk(min(15, max_conf.numel()))
    print(f"  [{label}] top-15 confs: {[f'{x:.3f}' for x in top_vals.cpu().tolist()]}")

    cx = boxes_raw[0, :]
    cy = boxes_raw[1, :]
    print(f"  [{label}] per-region max_conf:")
    for region_name, (rx, ry) in regions.items():
        in_r = (torch.abs(cx - rx) < 100) & (torch.abs(cy - ry) < 100)
        if in_r.any():
            region_max = float(max_conf[in_r].max().item())
            cnt = int(in_r.sum().item())
        else:
            region_max = 0.0
            cnt = 0
        print(f"      {region_name:>12}: max_conf={region_max:.4f}  n={cnt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--frame", type=int, default=44)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--iou", type=float, default=0.70)
    args = ap.parse_args()

    print(f"=== Controlled diag (ChitraMaya actual code path) ===")
    print(f"torch={torch.__version__}, cuDNN={torch.backends.cudnn.version()}")
    print(f"frame={args.frame}, imgsz={args.imgsz}, conf={args.conf}, iou={args.iou}")
    print()

    # Step 1: decode via ChitraMaya Decoder
    print("Decoding via ChitraMaya Decoder (NVDEC)...")
    bgr_u8 = decode_one_frame_via_chitramaya(args.video, args.frame, device=args.device)
    print(f"  Decoded: shape={tuple(bgr_u8.shape)} dtype={bgr_u8.dtype} device={bgr_u8.device}")
    for ch_idx, ch_name in enumerate(("B", "G", "R")):
        v = bgr_u8[..., ch_idx].float()
        print(f"  ch {ch_name}: min={v.min().item():.0f} max={v.max().item():.0f} "
              f"mean={v.mean().item():.3f} std={v.std().item():.3f}")
    print()

    # Step 2: build LadaYoloDetector via ChitraMaya's actual class
    print(f"Building LadaYoloDetector via ChitraMaya.mosaic.detector.lada_yolo...")
    det = LadaYoloDetector(
        model_path=args.model,
        device=args.device,
        imgsz=args.imgsz,
        fp16=True,
        conf_thres=args.conf,
        iou_thres=args.iou,
    )
    print(f"  Underlying Yolo11SegmentationModel.imgsz: {det.model.imgsz}")
    print(f"  args.conf={det.model.args.conf} args.iou={det.model.args.iou}")
    print()

    # Step 3: call detect_batch (ChitraMaya's actual code path)
    print(f"Running detector.detect_batch (ChitraMaya actual path)...")
    detections = det.detect_batch([bgr_u8])
    print(f"  Returned {len(detections)} Detection objects")
    if detections and detections[0].boxes is not None:
        boxes = detections[0].boxes
        scores = detections[0].scores
        classes = detections[0].classes
        print(f"  Post-NMS boxes: {len(boxes)}")
        for i in range(min(10, len(boxes))):
            x1, y1, x2, y2 = boxes[i].cpu().tolist()
            sc = float(scores[i].item()) if scores is not None else -1
            cl = int(classes[i].item()) if classes is not None else -1
            print(f"    box[{i}] conf={sc:.4f} cls={cl} xyxy=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
    else:
        print("  No detections returned")
    print()

    # Step 4: also dig in to the raw pre-NMS confidences for cross-comparison
    # We replicate the model.preprocess and inference path directly here
    print("Re-running through model.preprocess/inference for raw pre-NMS scores...")
    cpu_frame = bgr_u8.detach().to("cpu").contiguous() if bgr_u8.device.type != "cpu" else bgr_u8.contiguous()
    imgs = det.model.preprocess([cpu_frame])
    with torch.inference_mode():
        inp = imgs.to(device=det.model.device).to(dtype=det.model.dtype).div_(255.0)
        preds = det.model.inference(inp)
    print(f"  preprocessed input shape: {tuple(inp.shape)} dtype={inp.dtype}")

    # Use the region centers appropriate for the imgsz
    if args.imgsz == 640:
        regions = REGIONS_640
    else:
        # Scale 640-space centers to the actual imgsz space proportionally if not 960
        regions = REGIONS_960
    analyze_preds_per_region(preds, regions, f"imgsz={args.imgsz}")


if __name__ == "__main__":
    main()
