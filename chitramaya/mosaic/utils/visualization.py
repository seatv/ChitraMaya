"""
GPU-based visualization utilities for gRestorer.

All drawing operations use pure PyTorch on GPU - NO CPU transfers.

Supports BOTH image ranges:
  - uint8  [0..255]
  - float  [0..1]
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch


def _color_tensor_for_img(
    img: torch.Tensor,
    color_bgr: Tuple[int, int, int],
) -> torch.Tensor:
    """
    Returns a color tensor shaped [1,1,3] on img.device, in img.dtype and img range.
    color_bgr is always specified in 0..255.
    """
    if img.dtype.is_floating_point:
        c = torch.tensor(color_bgr, device=img.device, dtype=img.dtype) / 255.0
    else:
        c = torch.tensor(color_bgr, device=img.device, dtype=torch.uint8)
    return c.view(1, 1, 3)


def _clamp_box(x1: int, y1: int, x2: int, y2: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(W, int(x1)))
    x2 = max(0, min(W, int(x2)))
    y1 = max(0, min(H, int(y1)))
    y2 = max(0, min(H, int(y2)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def draw_box_gpu(
    img: torch.Tensor,  # [H, W, 3] uint8 or float, on GPU
    x1: int, y1: int, x2: int, y2: int,
    color: Tuple[int, int, int],           # BGR 0..255
    thickness: int = 2,
) -> torch.Tensor:
    """
    Draw a rectangle outline on GPU.

    Modifies img in-place.
    """
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"draw_box_gpu expects [H,W,3], got {tuple(img.shape)}")

    H, W = int(img.shape[0]), int(img.shape[1])
    x1, y1, x2, y2 = _clamp_box(x1, y1, x2, y2, W, H)
    if x1 == x2 or y1 == y2:
        return img

    t = max(1, int(thickness))
    c = _color_tensor_for_img(img, color)

    # Top
    img[y1:min(y1 + t, y2), x1:x2, :] = c
    # Bottom
    img[max(y2 - t, y1):y2, x1:x2, :] = c
    # Left
    img[y1:y2, x1:min(x1 + t, x2), :] = c
    # Right
    img[y1:y2, max(x2 - t, x1):x2, :] = c

    return img


def fill_box_gpu(
    img: torch.Tensor,  # [H, W, 3] uint8 or float, on GPU
    x1: int, y1: int, x2: int, y2: int,
    color: Tuple[int, int, int],           # BGR 0..255
    opacity: float = 0.5,
) -> torch.Tensor:
    """
    Fill a rectangle with semi-transparent color on GPU.

    Modifies img in-place.
    """
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"fill_box_gpu expects [H,W,3], got {tuple(img.shape)}")

    H, W = int(img.shape[0]), int(img.shape[1])
    x1, y1, x2, y2 = _clamp_box(x1, y1, x2, y2, W, H)
    if x1 >= x2 or y1 >= y2:
        return img

    a = float(opacity)
    if a <= 0.0:
        return img
    if a >= 1.0:
        # solid fill
        c = _color_tensor_for_img(img, color)
        img[y1:y2, x1:x2, :] = c
        return img

    c = _color_tensor_for_img(img, color)

    roi = img[y1:y2, x1:x2, :]

    if img.dtype.is_floating_point:
        # In-place blend in float [0..1]
        roi.mul_(1.0 - a)
        roi.add_(c * a)
    else:
        # Blend in float then convert back to uint8
        blended = roi.float() * (1.0 - a) + c.float() * a
        roi.copy_(blended.to(torch.uint8))

    return img


def draw_detections(
    bgr_img: torch.Tensor,  # [H, W, 3] uint8 or float, on GPU
    detection: Any,
    box_color: Tuple[int, int, int] = (0, 255, 0),   # BGR 0..255
    box_thickness: int = 2,
    show_confidence: bool = True,
    show_class: bool = True,
    fill_color: Optional[Tuple[int, int, int]] = None,  # BGR 0..255
    fill_opacity: float = 0.5,
) -> torch.Tensor:
    """
    Draw detection boxes (and optional fill) on GPU tensor (NO CPU transfer).

    NOTE: Text rendering not implemented (show_confidence/show_class ignored).
    """
    # Expect detection.boxes to be Nx4 xyxy on GPU
    boxes = getattr(detection, "boxes", None)
    if boxes is None or not isinstance(boxes, torch.Tensor) or boxes.numel() == 0:
        return bgr_img

    # Ensure on same device
    if boxes.device != bgr_img.device:
        boxes = boxes.to(bgr_img.device)

    n = int(boxes.shape[0])

    for i in range(n):
        box = boxes[i]
        x1 = int(box[0].item())
        y1 = int(box[1].item())
        x2 = int(box[2].item())
        y2 = int(box[3].item())

        if fill_color is not None:
            fill_box_gpu(bgr_img, x1, y1, x2, y2, fill_color, float(fill_opacity))

        draw_box_gpu(bgr_img, x1, y1, x2, y2, box_color, int(box_thickness))

    return bgr_img
