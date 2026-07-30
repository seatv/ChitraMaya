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
    # HWC uint8 -> resize to (h,w) with INTER_LINEAR (the CUDA resize path
    # supports only LINEAR/NEAREST, so this stays LINEAR for both grow and
    # shrink -- including the post-secondary downscale).
    return image_utils.resize(img_u8, size=shape_hw, interpolation=cv2.INTER_LINEAR)


def _scale_pad(pad: Pad, scale: int) -> Pad:
    # Pad offsets live in clip-crop space; after a secondary upscale the image
    # is scale x larger, so the unpad offsets scale with it.
    return tuple(int(p) * int(scale) for p in pad)  # type: ignore[return-value]


def _apply_secondary(
    clip_img_u8: torch.Tensor,
    orig_shape_hw: Tuple[int, int],
    secondary,
    frame_num: int | None = None,
) -> Tuple[torch.Tensor, int]:
    """CM-077: optionally upscale a restored (still padded) clip frame.

    Returns (image, scale). scale=1 means untouched (secondary off, gated
    off for small regions, or frame size unexpected).

    CM-077b: every offered crop is recorded into secondary.stats (seen /
    upscaled / gated-small / bad-geometry, plus the frame number when the
    upscale actually ran) so the run summary can report whether -- and
    exactly WHERE -- the scaler kicked in."""
    if secondary is None:
        return clip_img_u8, 1
    _st = getattr(secondary, "stats", None)
    if not secondary.should_apply(orig_shape_hw):
        if _st is not None:
            _st.note(frame_num, orig_shape_hw, "small")
        return clip_img_u8, 1
    if (clip_img_u8.shape[0] != secondary.min_apply_size
            or clip_img_u8.shape[1] != secondary.min_apply_size):
        if _st is not None:
            _st.note(frame_num, orig_shape_hw, "geom")
        return clip_img_u8, 1  # unexpected geometry -- keep the proven path
    out = secondary.upscale_frame_bgr_u8(clip_img_u8)
    if _st is not None:
        _st.note(frame_num, orig_shape_hw, "applied")
    return out, int(secondary.scale)


def _resize_mask_u8(mask_u8: torch.Tensor, shape_hw: Tuple[int, int]) -> torch.Tensor:
    # HW uint8 -> resize to (h,w) nearest
    if isinstance(mask_u8, torch.Tensor):
        if mask_u8.ndim == 2:
            mask_ch = mask_u8.unsqueeze(-1)  # HWC(1)
            out = image_utils.resize(mask_ch, size=shape_hw, interpolation=cv2.INTER_NEAREST)
            # CPU path goes through cv2, which drops the singleton channel.
            return out if out.ndim == 2 else out[:, :, 0]
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
    blendmask: str = "none",
    feather_radius: int = 0,
) -> None:
    """
    Direct port of LADA frame_restorer.py blend:
      temp = clip - roi
      temp *= blend_mask
      temp += roi
      round+clamp (GPU path)
      CPU path uses numpy and uint8 cast (trunc)

    blendmask selects how the per-pixel alpha is built:
      - "none"       : legacy LADA rectangular blend mask (create_blend_mask).
                       feather_radius does not apply here.
      - "facefusion" : alpha follows the mosaic's actual mask support
                       (create_support_blend_mask) for a softer, shape-aware
                       edge. feather_radius (px) sets the inward feather;
                       0 = auto (derived from crop size).
    """
    t, l, b, r = map(int, orig_clip_box)
    frame_roi = frame_bgr_u8[t : b + 1, l : r + 1]

    # Build the blend alpha per the selected mode.
    if str(blendmask).lower() == "facefusion":
        _feather = int(feather_radius) if int(feather_radius) > 0 else None
        blend_mask = mask_utils.create_support_blend_mask(
            clip_mask_u8.float(), feather_px=_feather,
        )
    else:
        # Legacy LADA rectangular blend mask.
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
    blendmask: str = "none",
    feather_radius: int = 0,
    secondary=None,
) -> None:
    """
    Port of LADA FrameRestorer._restore_frame applied over an entire clip.

    blendmask / feather_radius select the paste-back alpha (see
    _blend_into_frame_lada). Defaults ("none", 0) preserve the legacy behavior.

    secondary (CM-077): optional RTX Super-Res stage. When set, restored
    frames whose original region exceeds the clip size are upscaled 2x/4x
    before paste-back, so the final resize shrinks instead of stretching.
    Masks stay in clip space (they carry no detail worth upscaling).
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

        # CM-077: secondary upscale BEFORE unpad (fixed-size model input).
        clip_img, sec_scale = _apply_secondary(
            clip_img, orig_shape_hw, secondary, frame_num=frame_num)

        # Unpad back to resized crop dims (image offsets scale with the upscale)
        clip_img = _unpad_any(clip_img, _scale_pad(pad, sec_scale))
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
            blendmask=blendmask,
            feather_radius=feather_radius,
        )

__all__ = ["composite_clip_into_store"]
