# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, TypeAlias

from . import image_utils

Box: TypeAlias = Tuple[int, int, int, int]
Mask: TypeAlias = np.ndarray


def get_box(mask: Mask) -> Box:
    points = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(points)
    t = int(y)
    l = int(x)
    b = int(y + h - 1)
    r = int(x + w - 1)
    return t, l, b, r


def morph(mask: Mask, iterations=1, operator=cv2.MORPH_DILATE) -> Mask:
    if get_mask_area(mask) < 0.01:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    return cv2.morphologyEx(mask, operator, kernel, iterations=iterations)


def dilate_mask(mask: Mask, dilatation_size=11, iterations=2):
    if iterations == 0:
        return mask
    element = np.ones((dilatation_size, dilatation_size), np.uint8)
    mask_img = cv2.dilate(mask, element, iterations=iterations).reshape(mask.shape)
    return mask_img


def extend_mask(mask: Mask, value) -> Mask:
    # value between 0 and 3 -> higher values mean more extension of mask area. 0 does not change mask at all
    if value == 0:
        return mask

    # Dilations are slow when using huge kernels (which we would need for high-res masks). therefore we downscale mask to perform morph operations on much smaller pixel space with smaller kernels
    target_size = 256
    extended_mask = image_utils.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    extended_mask = morph(extended_mask, iterations=value, operator=cv2.MORPH_DILATE)
    extended_mask = image_utils.resize(extended_mask, mask.shape[:2], interpolation=cv2.INTER_NEAREST)
    extended_mask = extended_mask.reshape(mask.shape)
    assert mask.shape == extended_mask.shape
    return extended_mask


def clean_mask(mask: Mask, box: Box) -> tuple[Mask, Box]:
    t, l, b, r = box
    # Masks from YOLO prediction extend detection area in some cases. Let's crop
    mask[:t + 1, :, :] = 0
    mask[b:, :, :] = 0
    mask[:, :l + 1, :] = 0
    mask[:, r:, :] = 0

    # Mask from YOLO prediction can sometimes contain additional disconnected (tiny) segments. Keep only the largest
    edited_mask = np.zeros_like(mask, dtype=mask.dtype)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours) != 0
    if len(contours) > 1:
        contours = sorted(contours, key=lambda contour: cv2.contourArea(contour), reverse=True)[0]
    largest_contour = contours[0]
    cv_box = cv2.boundingRect(largest_contour)
    box = box_utils.convert_from_opencv(cv_box)
    cv2.drawContours(edited_mask, [largest_contour], 0, 255, thickness=cv2.FILLED)
    return edited_mask, box


def get_mask_area(mask: Mask) -> float:
    pixels = cv2.countNonZero(mask)
    return pixels / (mask.shape[0] * mask.shape[1])


def smooth_mask(mask: Mask, kernel_size: int) -> Mask:
    return cv2.medianBlur(mask, kernel_size).reshape(mask.shape)


def get_nonzero_box_torch(mask: torch.Tensor) -> Optional[Box]:
    """Return (top, left, bottom, right) for non-zero support, or None."""
    mask = mask.squeeze()
    if mask.ndim != 2:
        raise ValueError(f"Expected HW mask, got shape={tuple(mask.shape)}")

    nz = mask > 0
    if not bool(nz.any()):
        return None

    rows = torch.where(nz.any(dim=1))[0]
    cols = torch.where(nz.any(dim=0))[0]
    return int(rows[0]), int(cols[0]), int(rows[-1]), int(cols[-1])


def create_support_blend_mask(
    crop_mask: torch.Tensor,
    feather_px: Optional[int] = None,
    *,
    min_feather_px: int = 2,
    max_feather_ratio: float = 0.05,
    passes: int = 1,
) -> torch.Tensor:
    """Create an inward-only soft alpha from the actual mask support.

    This is intended for the mainline gRestorer compositor.

    Behavior:
      - derives support from the resized crop mask itself
      - feathers *inside* the support boundary (never outside)
      - renormalizes so the strongest interior remains 1.0

    Unlike the legacy create_blend_mask(), this does not build a synthetic inner
    rectangle first. That makes the alpha follow the real ROI support and allows
    the compositor to blend only the effective mask bounds.
    """
    mask = crop_mask.squeeze()
    if mask.ndim != 2:
        raise ValueError(f"Expected HW mask, got shape={tuple(mask.shape)}")

    support = (mask > 0).to(dtype=torch.float32)
    h, w = int(support.shape[0]), int(support.shape[1])

    if not bool(support.any()):
        return torch.zeros_like(support)

    if feather_px is None:
        feather_px = int(round(min(h, w) * float(max_feather_ratio)))
        feather_px = max(int(min_feather_px), feather_px)

    feather_px = int(max(0, feather_px))
    if feather_px <= 0:
        return support

    k = 2 * feather_px + 1
    alpha = support.unsqueeze(0).unsqueeze(0)
    for _ in range(max(1, int(passes))):
        alpha = F.avg_pool2d(alpha, kernel_size=k, stride=1, padding=feather_px)
    alpha = alpha.squeeze(0).squeeze(0)

    # Inward-only feather: do not let alpha extend outside the actual support.
    alpha = alpha * support

    # Re-normalize so valid interior remains strong even on small crops.
    amax = alpha.max()
    if float(amax) > 0.0:
        alpha = alpha / amax

    return alpha.clamp(0.0, 1.0)


def create_blend_mask(crop_mask: torch.Tensor):
    """Legacy LADA-like blend mask.

    Kept unchanged for compatibility and parity experiments.
    The mainline compositor should prefer create_support_blend_mask().
    """
    mask = crop_mask.squeeze()
    h, w = mask.shape
    border_ratio = 0.05
    h_inner, w_inner = int(h * (1.0 - border_ratio)), int(w * (1.0 - border_ratio))
    h_outer, w_outer = h - h_inner, w - w_inner
    border_size = min(h_outer, w_outer)
    if border_size < 5:
        return torch.ones_like(mask)
    blur_size = int(border_size)
    if blur_size % 2 == 0:
        blur_size += 1
    inner = torch.ones((h_inner, w_inner), device=mask.device, dtype=mask.dtype)
    pad_top = h_outer // 2
    pad_bottom = h_outer - pad_top
    pad_left = w_outer // 2
    pad_right = w_outer - pad_left
    blend = F.pad(inner, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
    mask4 = (mask > 0)
    blend = torch.maximum(mask4, blend)
    kernel = torch.tensor(1.0 / (blur_size**2), device=blend.device, dtype=blend.dtype).expand(1, blur_size, blur_size)
    blend = image_utils.filter2D(blend.unsqueeze(0).unsqueeze(0), kernel).squeeze(0).squeeze(0)
    assert blend.shape == mask.shape
    return blend


def apply_random_mask_extensions(mask: Mask) -> Mask:
    value = np.random.choice([0, 0, 1, 1, 2])
    return extend_mask(mask, value)


def box_to_mask(box: Box, shape, mask_value: int):
    mask = np.zeros((shape[0], shape[1], 1), np.uint8)
    t, l, b, r = box
    mask[t:b + 1, l:r + 1] = mask_value
    return mask
