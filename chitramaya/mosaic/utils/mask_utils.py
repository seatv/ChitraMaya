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


def box_filter_2d(x: torch.Tensor, k: int, *, pad_mode: str = "reflect") -> torch.Tensor:
    """k x k box (mean) filter, stride 1, same-size output -- WITHOUT conv2d / avg_pool2d.

    CM-172. The blend masks are built at the size of the region in the frame,
    so their (h, w, k) changes with every region on every frame. On CUDA that
    is free (cuDNN picks a kernel by heuristic); on ROCm/Windows every unseen
    shape sent to F.conv2d or F.avg_pool2d is a new MIOpen problem -- search,
    compile, cache -- costing seconds to minutes each. On a full title that
    became a compile storm: paste-back of one MCL-180 clip took 2-5 minutes
    with the GPU idle and one CPU core busy (9060 XT, 09-08). PurpleRain never
    showed it (3 fixed regions = 3 shapes, paid once).

    This computes the identical box mean from a summed-area table: reflect or
    zero pad, two cumulative sums, four gathers, one divide. Pure tensor
    arithmetic -- no cuDNN, no MIOpen, no oneDNN -- so it costs the same on
    every edition and never compiles anything. Sums are taken in float64 so a
    0/1 mask (the usual input) is exact; the result is cast back to x.dtype.

    Args:
        x: (H, W) tensor.
        k: window size (odd; k <= 1 returns a copy).
        pad_mode: "reflect" reproduces the legacy filter2D path
                  (create_blend_mask); "zeros" reproduces
                  F.avg_pool2d(..., stride=1, padding=k//2) with
                  count_include_pad=True (create_support_blend_mask).
    """
    if x.ndim != 2:
        raise ValueError(f"box_filter_2d expects an HW tensor, got shape={tuple(x.shape)}")
    k = int(k)
    if k <= 1:
        return x.clone()
    r = k // 2
    x4 = x.unsqueeze(0).unsqueeze(0)
    if pad_mode == "reflect":
        xp = F.pad(x4, (r, r, r, r), mode="reflect")
    elif pad_mode == "zeros":
        xp = F.pad(x4, (r, r, r, r), mode="constant", value=0.0)
    else:
        raise ValueError(f"box_filter_2d: unknown pad_mode {pad_mode!r}")
    xp = xp[0, 0].to(torch.float64)

    hp, wp = xp.shape
    sat = torch.zeros((hp + 1, wp + 1), dtype=torch.float64, device=xp.device)
    sat[1:, 1:] = xp.cumsum(0).cumsum(1)

    h, w = int(x.shape[0]), int(x.shape[1])
    # Window for output (i, j) covers padded rows i..i+k-1, cols j..j+k-1.
    s = (sat[k:k + h, k:k + w] - sat[0:h, k:k + w]
         - sat[k:k + h, 0:w] + sat[0:h, 0:w])
    return (s / float(k * k)).to(x.dtype)


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


# CM-214: auto feather as a share of the FRAME height (60 px at 2160,
# 30 px at 1080), so the blend band is the same size on screen whatever the
# region's size -- a small region no longer gets a 2-px seam. Idea from
# jasna's blend mask (kruk2, AGPL-3.0: dilation and falloff each 2.8 % of
# the frame height); applied here to our outward feather, where the same
# number is the solid margin AND the ramp width.
FRAME_FEATHER_RATIO = 0.028


def auto_feather_px(frame_h: Optional[int], crop_hw: Tuple[int, int], *,
                    min_feather_px: int = 2, region_ratio: float = 0.05) -> int:
    """CM-214: the feather the compositor uses when the panel says 0 (auto).
    With the frame height known: FRAME_FEATHER_RATIO of it. Without (unit
    tools, old callers): the pre-1.80 rule, 5 % of the region's short side."""
    if frame_h is not None and int(frame_h) > 0:
        px = int(round(int(frame_h) * FRAME_FEATHER_RATIO))
    else:
        px = int(round(min(int(crop_hw[0]), int(crop_hw[1])) * float(region_ratio)))
    return max(int(min_feather_px), px)


def create_support_blend_mask(
    crop_mask: torch.Tensor,
    feather_px: Optional[int] = None,
    *,
    min_feather_px: int = 2,
    max_feather_ratio: float = 0.05,
    passes: int = 1,
    frame_h: Optional[int] = None,
) -> torch.Tensor:
    """Shape-aware paste-back alpha with an OUTWARD feather (CM-086).

    Used by the "facefusion" blend mask (compositor + VR projection).

    Behavior:
      - support = the resized crop mask (the detector's segmentation outline)
      - alpha is exactly 1.0 on every support pixel AND on a margin of
        feather_px around it, then ramps to 0 over the next feather_px
      - the ramp therefore lies in the restorer's reconstruction of clean
        pixels around the mosaic, never on the mosaic itself
      - feather_px None = auto: FRAME_FEATHER_RATIO of frame_h when the
        caller passes the frame height (CM-214, the compositor does), else
        max_feather_ratio of the region's short side (the pre-1.80 rule)

    History: until 09-16 this feathered INWARD (blur, then alpha *= support),
    which left alpha == 0 on every mosaic pixel outside the segmentation
    outline -- a rim of untouched mosaic blocks around each region, moving
    frame to frame -- and cut hard along the outline's straight edges
    (CM-086, seen on the 09-11 AMD runs and the 09-16 4K cut). Segmentation
    masks are tight; mosaic blocks run past them. Growing the support first
    and blurring outward fixes both symptoms.

    Implementation: box dilation (summed-area box mean > 0) by
    feather_px * passes, then `passes` box means of the same width. Reflect
    padding keeps alpha at 1.0 where the support touches the crop edge.
    Same CM-172 box filter, so no per-shape MIOpen compile on ROCm.
    """
    mask = crop_mask.squeeze()
    if mask.ndim != 2:
        raise ValueError(f"Expected HW mask, got shape={tuple(mask.shape)}")

    support = (mask > 0).to(dtype=torch.float32)
    h, w = int(support.shape[0]), int(support.shape[1])

    if not bool(support.any()):
        return torch.zeros_like(support)

    if feather_px is None:
        feather_px = auto_feather_px(frame_h, (h, w), min_feather_px=int(min_feather_px),
                                     region_ratio=float(max_feather_ratio))

    feather_px = int(max(0, feather_px))
    # Reflect padding needs pad < dim: an explicit Feather larger than a small
    # crop (Feather 64 on a 40-px region) is clamped, not an error.
    feather_px = min(feather_px, max(0, (min(h, w) - 1) // 2))
    if feather_px <= 0:
        return support

    n_pass = max(1, int(passes))
    k = 2 * feather_px + 1

    # 1. Solid core: the support grown by feather_px per pass (a box mean is
    #    > 0 exactly where the box window touches the support).
    core = support
    for _ in range(n_pass):
        core = (box_filter_2d(core, k, pad_mode="reflect") > 0).to(dtype=torch.float32)

    # 2. Outward ramp: blur the core. Every support pixel sits at least
    #    feather_px * n_pass inside the core, so its window is all ones and
    #    alpha stays exactly 1.0 there; outside the core alpha falls to 0 over
    #    feather_px * n_pass.
    alpha = core
    for _ in range(n_pass):
        alpha = box_filter_2d(alpha, k, pad_mode="reflect")

    # Guard against float dust: the support itself is always fully pasted.
    alpha = torch.maximum(alpha, support)
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
    mask4 = (mask > 0).to(dtype=blend.dtype)
    blend = torch.maximum(mask4, blend)
    # CM-172: the legacy path was image_utils.filter2D (F.conv2d with a
    # uniform 1/k^2 kernel, reflect padding). Same box mean via a summed-area
    # table -- no conv2d, so no per-shape MIOpen compile on ROCm.
    blend = box_filter_2d(blend, blur_size, pad_mode="reflect")
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
