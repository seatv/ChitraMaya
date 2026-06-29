from __future__ import annotations

"""
GPU-first YOLO-seg mosaic detector for Ultralytics 8.3.243+.

Pinned API expectations (validated by your probe):
- non_max_suppression: ultralytics.utils.nms
- process_mask/scale_boxes: ultralytics.utils.ops

Contract:
- infer_batch() expects a list of RGB frames:
    torch.Tensor[H, W, 3], dtype=uint8, values 0..255
  (float [0,1] accepted, converted on-device)
- Preprocess + inference + mask generation stay on the inference device
  (CUDA/XPU preferred). We only copy *small* outputs (boxes/scores/classes) to CPU.
"""

import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression  # <- 8.3.243 puts NMS here


@dataclass
class FrameDetections:
    """Detections for a single frame.

    - boxes/scores/classes are CPU tensors (cheap, small copies)
    - masks are uint8 [N,H,W] on inference device by default (optional CPU)
    """

    boxes_xyxy: Optional[torch.Tensor] = None  # CPU float32 [N,4] xyxy
    scores: Optional[torch.Tensor] = None      # CPU float32 [N]
    classes: Optional[torch.Tensor] = None     # CPU int64   [N]
    masks: Optional[torch.Tensor] = None       # uint8 [N,H,W] on inference device (or CPU)
    orig_size: Optional[Tuple[int, int]] = None  # (W,H)


@dataclass(frozen=True)
class _LetterboxMeta:
    orig_hw: Tuple[int, int]                   # (H,W)
    lb_hw: Tuple[int, int]                     # (H,W) after resize+pad
    new_unpad_hw: Tuple[int, int]              # (H,W) resized (no pad)
    pad_tlbr: Tuple[int, int, int, int]        # (top, left, bottom, right)


def _to_u8(frame_rgb: torch.Tensor) -> torch.Tensor:
    """Convert an RGB frame to uint8 on the same device."""
    if frame_rgb.dtype == torch.uint8:
        return frame_rgb
    if frame_rgb.is_floating_point():
        # Assume float is [0,1]. Avoid sync-heavy range probes.
        return (frame_rgb * 255.0).clamp(0, 255).to(torch.uint8)
    return frame_rgb.to(torch.uint8)


def _stride_value(stride) -> int:
    """AutoBackend.stride can be int, list, tuple, tensor; normalize to int."""
    try:
        if isinstance(stride, (list, tuple)):
            return int(max(stride))
        if hasattr(stride, "max"):
            return int(stride.max())
        return int(stride)
    except Exception:
        return 32


def _compute_letterbox_meta(
    orig_hw: Tuple[int, int],
    new_shape: Tuple[int, int],
    stride: int = 32,
) -> _LetterboxMeta:
    """Ultralytics-style letterbox geometry (auto=True behavior)."""
    h, w = orig_hw
    new_h, new_w = new_shape

    r = min(new_h / h, new_w / w)
    new_unpad_w = int(round(w * r))
    new_unpad_h = int(round(h * r))

    dw = new_w - new_unpad_w
    dh = new_h - new_unpad_h

    # Ultralytics letterbox(auto=True) uses mod stride padding.
    dw = int(dw % stride)
    dh = int(dh % stride)

    left = dw // 2
    right = dw - left
    top = dh // 2
    bottom = dh - top

    lb_h = new_unpad_h + top + bottom
    lb_w = new_unpad_w + left + right

    return _LetterboxMeta(
        orig_hw=(h, w),
        lb_hw=(lb_h, lb_w),
        new_unpad_hw=(new_unpad_h, new_unpad_w),
        pad_tlbr=(top, left, bottom, right),
    )


def _letterbox_bchw(img_bchw: torch.Tensor, meta: _LetterboxMeta) -> torch.Tensor:
    """Resize + pad BCHW float image to letterboxed size."""
    new_unpad_h, new_unpad_w = meta.new_unpad_hw
    top, left, bottom, right = meta.pad_tlbr

    if img_bchw.shape[-2:] != (new_unpad_h, new_unpad_w):
        img_bchw = F.interpolate(
            img_bchw,
            size=(new_unpad_h, new_unpad_w),
            mode="bilinear",
            align_corners=False,
        )

    if any((top, left, bottom, right)):
        pad_val = 114.0 / 255.0
        img_bchw = F.pad(img_bchw, (left, right, top, bottom), value=pad_val)

    return img_bchw


def _unletterbox_masks(masks_nhw: torch.Tensor, meta: _LetterboxMeta) -> torch.Tensor:
    """Crop padding and scale masks back to original H,W (float)."""
    top, left, bottom, right = meta.pad_tlbr
    new_unpad_h, new_unpad_w = meta.new_unpad_hw
    orig_h, orig_w = meta.orig_hw

    if any((top, left, bottom, right)):
        masks_nhw = masks_nhw[:, top : top + new_unpad_h, left : left + new_unpad_w]

    if masks_nhw.shape[-2:] != (orig_h, orig_w):
        # interpolate doesn't support bilinear for uint8/bool; masks should be resized with nearest anyway
        if masks_nhw.dtype in (torch.uint8, torch.bool):
            masks_nhw = masks_nhw.to(torch.float32)
        masks_nhw = F.interpolate(
            masks_nhw.unsqueeze(1),
            size=(orig_h, orig_w),
            mode="nearest",
        ).squeeze(1)

    return masks_nhw


class MosaicDetectionModel:
    """YOLO segmentation wrapper with torch-letterbox + GPU-first behavior."""

    def __init__(
        self,
        model_path: str,
        device: str,
        imgsz: int = 640,
        fp16: bool = True,
        conf: float = 0.25,
        iou: float = 0.45,
        classes=None,
        max_det: int = 300,
    ) -> None:
        self.device = torch.device(device)

        self.fp16 = bool(fp16)
        self.debug = os.environ.get("GRESTORER_DET_DEBUG", "0") == "1"
        self.gpu_only = os.environ.get("GRESTORER_DET_GPU_ONLY", "0") == "1"
        self.output_masks_device = os.environ.get("GRESTORER_DET_MASKS", "gpu").lower().strip()
        if self.output_masks_device not in ("gpu", "cpu"):
            self.output_masks_device = "gpu"

        self.model = AutoBackend(
            model=model_path,
            device=self.device,
            fp16=self.fp16,
            fuse=True,
            verbose=False,
        )
        self.model.eval()

        self.stride = _stride_value(getattr(self.model, "stride", 32))
        self.imgsz = check_imgsz(imgsz, stride=self.stride, min_dim=2)

        self.conf = float(conf)
        self.iou = float(iou)
        self.classes = classes
        self.max_det = int(max_det)

        # names is typically dict[int->str] or list
        self.nc = len(self.model.names)

        # dtype: use AutoBackend's detected fp16, not the constructor param.
        # For .pt + fp16=True: AutoBackend casts model to half → self.model.fp16=True → dtype=float16.
        # For .engine: AutoBackend reads the engine's input binding dtype.
        # Ultralytics' fp16 engines declare FLOAT input bindings (the engine
        # accepts fp32 input and casts internally), so AutoBackend sets
        # self.model.fp16=False. Feeding fp16 to an FLOAT-input engine produces
        # silent garbage (no error, near-zero detections, occasional phantom
        # mask explosions that OOM at unletterbox time).
        effective_fp16 = bool(self.model.fp16)
        self.dtype = torch.float16 if (effective_fp16 and self.device.type in ("cuda", "xpu")) else torch.float32

        # Warmup
        with torch.inference_mode():
            _ = self.model(torch.zeros((1, 3, self.imgsz[0], self.imgsz[1]), device=self.device, dtype=self.dtype))

        if self.debug:
            print(f"[YOLO] device={self.device} imgsz={self.imgsz} stride={self.stride} "
                  f"fp16_req={self.fp16} fp16_effective={effective_fp16} masks={self.output_masks_device}")

    def infer_batch(self, frames_rgb: List[torch.Tensor]) -> List[FrameDetections]:
        if not frames_rgb:
            return []

        dev_in = frames_rgb[0].device
        if self.gpu_only and dev_in.type == "cpu":
            raise RuntimeError("GRESTORER_DET_GPU_ONLY=1 but detector received CPU frames")

        # Convert to uint8 on-device
        frames_u8 = [_to_u8(f) for f in frames_rgb]

        f0 = frames_u8[0]
        b = len(frames_u8)

        if f0.ndim != 3:
            raise RuntimeError(f"Expected 3D frame tensor, got shape={tuple(f0.shape)}")

        # Accept either HWC (H,W,3) or CHW (3,H,W).
        if f0.shape[-1] == 3:
            # HWC
            h, w = int(f0.shape[0]), int(f0.shape[1])
            batch = torch.stack(frames_u8, dim=0)  # [B,H,W,3]
            batch = batch.permute(0, 3, 1, 2).contiguous()  # [B,3,H,W]
        elif f0.shape[0] == 3:
            # CHW
            h, w = int(f0.shape[1]), int(f0.shape[2])
            batch = torch.stack(frames_u8, dim=0).contiguous()  # [B,3,H,W]
        else:
            raise RuntimeError(
                f"Unsupported frame layout, expected HWC[...,3] or CHW[3,...], got shape={tuple(f0.shape)}"
            )

        x = batch.to(dtype=self.dtype) / 255.0
        if self.debug:
            layout = "HWC" if f0.shape[-1] == 3 else "CHW"
            print(f"[DetYOLO] input_layout={layout} frame_shape={tuple(f0.shape)}")

        # Move to model device if needed
        if x.device != self.device:
            x = x.to(self.device, non_blocking=True)

        meta = _compute_letterbox_meta((h, w), (int(self.imgsz[0]), int(self.imgsz[1])), stride=self.stride)
        x = _letterbox_bchw(x, meta)

        if self.debug:
            t0 = time.perf_counter()

        with torch.inference_mode():
            out = self.model(x)

            # Seg models typically return (pred, proto) but newer Ultralytics may wrap proto in dict/list
            if isinstance(out, (list, tuple)) and len(out) >= 2:
                pred_logits = out[0]
                protos = out[1]
            elif isinstance(out, dict):
                # very defensive: handle dict output if it ever happens
                pred_logits = out.get("pred", out.get(0, None))
                protos = out.get("proto", out.get("protos", out.get(1, None)))
            else:
                raise RuntimeError("YOLO model output did not include protos (segmentation). Check weights/model type.")

            # Normalize protos to a tensor or list/tuple of tensors
            if isinstance(protos, dict):
                # common keys across versions
                for k in ("proto", "protos", "p"):
                    if k in protos:
                        protos = protos[k]
                        break
                else:
                    protos = next(iter(protos.values()))

            if isinstance(protos, (list, tuple)):
                # some builds return a list of proto tensors; last element is usually the proto feature map
                protos = protos[-1]

            # Ultralytics 8.3.243 NMS is in ultralytics.utils.nms and does not take `nm`,
            # it infers extra dims from `nc` (see docs).
            preds = non_max_suppression(
                pred_logits,
                conf_thres=self.conf,
                iou_thres=self.iou,
                classes=self.classes,
                agnostic=False,
                max_det=self.max_det,
                nc=self.nc,
                end2end=getattr(self.model, "end2end", False),
            )

        dets: List[FrameDetections] = []
        for i, pred in enumerate(preds):
            if pred is None or not len(pred):
                dets.append(FrameDetections(orig_size=(w, h)))
                continue

            # Defensive clones: some Ultralytics utilities do in-place ops on their inputs
            mask_coeff = pred[:, 6:].detach()
            boxes_for_mask = pred[:, :4].detach().clone()

            # Pick per-image proto safely (8.4.4 sometimes returns protos as dict/other container)
            proto_i = protos
            if isinstance(protos, torch.Tensor) and protos.ndim == 4:
                # protos shape: [B, C, H, W]
                proto_i = protos[i]
            elif isinstance(protos, (list, tuple)):
                proto_i = protos[i]
            elif isinstance(protos, dict):
                # try common keys, else first value
                if "proto" in protos:
                    proto_i = protos["proto"]
                elif "protos" in protos:
                    proto_i = protos["protos"]
                else:
                    proto_i = next(iter(protos.values()))
                # if proto is still batched, index it
                if isinstance(proto_i, torch.Tensor) and proto_i.ndim == 4:
                    proto_i = proto_i[i]

            masks_lb = ops.process_mask(proto_i, mask_coeff, boxes_for_mask, x.shape[2:], upsample=True)

            # Scale boxes from letterbox to original
            # ops.scale_boxes() does in-place math; clone to avoid inference-tensor restrictions
            boxes_in = pred[:, :4].detach().clone()
            boxes = ops.scale_boxes(x.shape[2:], boxes_in, (h, w, 3))

            keep = masks_lb.sum((-2, -1)) > 0
            if not bool(keep.any()):
                dets.append(FrameDetections(orig_size=(w, h)))
                continue

            pred = pred[keep]
            boxes = boxes[keep]
            masks_lb = masks_lb[keep]

            masks_full = _unletterbox_masks(masks_lb, meta)
            masks_u8 = (masks_full > 0.5).to(torch.uint8) * 255

            if self.output_masks_device == "cpu":
                masks_u8 = masks_u8.cpu()

            # Small outputs to CPU (avoid per-element .item() GPU syncs)
            boxes_cpu = boxes.detach().to(torch.float32).cpu()
            scores_cpu = pred[:, 4].detach().to(torch.float32).cpu()
            classes_cpu = pred[:, 5].detach().to(torch.int64).cpu()

            dets.append(
                FrameDetections(
                    boxes_xyxy=boxes_cpu,
                    scores=scores_cpu,
                    classes=classes_cpu,
                    masks=masks_u8,
                    orig_size=(w, h),
                )
            )

        if self.debug:
            if self.device.type == "cuda":
                torch.cuda.synchronize(device=self.device)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(
                f"[DetYOLO] batch={b} frame={w}x{h} dev_in={dev_in.type} dev_model={self.device.type} "
                f"lb={meta.lb_hw[1]}x{meta.lb_hw[0]} masks={self.output_masks_device} total={dt_ms:.2f}ms"
            )

        return dets
