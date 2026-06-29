from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from chitramaya.mosaic.core.scene import Box, Pad
from chitramaya.mosaic.utils import image_utils, mask_utils


def _unpad_any(img: torch.Tensor, pad: Pad) -> torch.Tensor:
    # image_utils.unpad_image works on both numpy and torch (slicing).
    return image_utils.unpad_image(img, pad)


def _resize_img_u8(img_u8: torch.Tensor, shape_hw: Tuple[int, int]) -> torch.Tensor:
    # HWC uint8 -> resize to (h,w) with INTER_LINEAR
    return image_utils.resize(img_u8, size=shape_hw, interpolation=cv2.INTER_LINEAR)


def _resize_mask_u8(mask_u8: torch.Tensor, shape_hw: Tuple[int, int]) -> torch.Tensor:
    # HW uint8 -> resize to (h,w) nearest
    if isinstance(mask_u8, torch.Tensor):
        if mask_u8.ndim == 2:
            mask_ch = mask_u8.unsqueeze(-1)  # HWC(1)
            out = image_utils.resize(mask_ch, size=shape_hw, interpolation=cv2.INTER_NEAREST)
            return out[:, :, 0]
        return image_utils.resize(mask_u8, size=shape_hw, interpolation=cv2.INTER_NEAREST)

    # numpy path (unlikely in gRestorer, but keep parity)
    return cv2.resize(mask_u8, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)




def _blend_into_frame_lada(
    *,
    frame_bgr_u8: torch.Tensor,
    clip_img_u8: torch.Tensor,
    clip_mask_u8: torch.Tensor,
    orig_clip_box: Box,
    model_dtype: torch.dtype,
    border_ratio: float = 0.05,
) -> None:
    """
    Direct port of LADA frame_restorer.py blend:
      temp = clip - roi
      temp *= blend_mask
      temp += roi
      round+clamp (GPU path)
      CPU path uses numpy and uint8 cast (trunc)
    """
    t, l, b, r = map(int, orig_clip_box)
    frame_roi = frame_bgr_u8[t : b + 1, l : r + 1]

    # Create blend mask like LADA: create_blend_mask(mask.float())
    blend_mask = mask_utils.create_blend_mask(clip_mask_u8.float())

    if frame_bgr_u8.device.type != "cuda":
        # CPU/numpy path (matches LADA CPU semantics: astype(uint8) truncation)
        frame_roi_np = frame_roi.detach().cpu().numpy()  # view if contiguous
        roi_np = frame_roi_np.astype(np.float32, copy=False)

        clip_np = clip_img_u8.detach().cpu().numpy().astype(np.float32, copy=False)
        bm = blend_mask.detach().cpu().numpy().astype(np.float32, copy=False)
        if bm.ndim == 2:
            bm = bm[:, :, None]

        temp = (clip_np - roi_np) * bm + roi_np
        frame_roi_np[:] = temp.astype(np.uint8)

        # (Not strictly needed if numpy view shares memory, but safe)
        frame_roi[:] = torch.from_numpy(frame_roi_np)
        return

    # GPU path: LADA uses model dtype (fp16 on CUDA), then round+clamp
    target_dtype = model_dtype
    roi_f = frame_roi.to(dtype=target_dtype)
    temp = clip_img_u8.to(device=frame_roi.device, dtype=target_dtype)
    bm = blend_mask.to(device=frame_roi.device, dtype=target_dtype)
    if bm.ndim == 2:
        bm = bm.unsqueeze(-1)

    temp.sub_(roi_f)
    temp.mul_(bm)
    temp.add_(roi_f)
    temp.round_()
    temp.clamp_(0, 255)

    frame_roi[:] = temp  # torch will cast into uint8 ROI


def composite_clip_into_store(
    *,
    clip,
    restored_frames_u8: List[torch.Tensor],
    store_bgr_u8: Dict[int, torch.Tensor],
    model_dtype: torch.dtype,
    border_ratio: float = 0.05,
) -> None:
    """
    Port of LADA FrameRestorer._restore_frame applied over an entire clip.
    """
    n = min(len(restored_frames_u8), len(clip.frame_nums))
    for i in range(n):
        frame_num = int(clip.frame_nums[i])
        frame = store_bgr_u8.get(frame_num)
        if frame is None:
            continue

        clip_img = restored_frames_u8[i]
        clip_mask = clip.masks[i]
        orig_box: Box = clip.boxes[i]
        orig_shape_hw = clip.crop_shapes[i]
        pad: Pad = clip.pad_after_resizes[i]

        # Unpad back to resized crop dims
        clip_img = _unpad_any(clip_img, pad)
        clip_mask = _unpad_any(clip_mask, pad)

        # Resize back to original crop size
        clip_img = _resize_img_u8(clip_img, orig_shape_hw)
        clip_mask = _resize_mask_u8(clip_mask, orig_shape_hw)

        _blend_into_frame_lada(
            frame_bgr_u8=frame,
            clip_img_u8=clip_img,
            clip_mask_u8=clip_mask,
            orig_clip_box=orig_box,
            model_dtype=model_dtype,
            border_ratio=border_ratio,
        )

__all__ = ["composite_clip_into_store"]
