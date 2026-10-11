from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import torch

from chitramaya.mosaic.core.scene import Box, Pad, Scene, crop_storage_scale, resample_capped_to_clip
from chitramaya.mosaic.utils import image_utils


def _box_size(b: Box) -> Tuple[int, int]:
    return (int(b[2]) - int(b[0]) + 1, int(b[3]) - int(b[1]) + 1)


@dataclass
class Clip:
    """
    LADA-compatible Clip representation:
      - frames are uint8 HWC (BGR), padded to clip_size
      - masks are uint8 HW, padded to clip_size
      - crop boxes are in original frame coordinates (t,l,b,r)
      - crop_shapes are original (h,w) prior to resize+pad
      - pad_after_resizes is (pt,pb,pl,pr)
    """

    clip_id: int
    frame_start: int
    frame_nums: List[int]
    clip_size: int

    # Per-frame data
    frames: List[torch.Tensor]
    masks: List[torch.Tensor]
    boxes: List[Box]
    crop_shapes: List[Tuple[int, int]]
    pad_after_resizes: List[Pad]
    # CM-227: the (h, w) each crop was STORED at; equals crop_shapes unless
    # the scene's Crop Cap downscaled it at capture (report + diagnostics).
    stored_shapes: Optional[List[Tuple[int, int]]] = None

    # CM-213: per-frame paste weight (0..1) for a continuation clip's seeded
    # frames; None = every frame weighs 1.0 (ordinary clip). The compositor
    # multiplies its blend alpha by this, so a 0 frame is skipped outright.
    weights: Optional[List[float]] = None
    # CM-213: how many leading frames this clip shares with its parent clip
    # (2 * overlap for a continuation; 0 otherwise). Restored twice by design.
    n_seeded: int = 0

    @classmethod
    def from_scene(
        cls,
        scene: Scene,
        clip_id: int,
        clip_size: int,
        pad_mode: str,
    ) -> "Clip":
        frame_start = scene.frame_nums[0]
        frame_nums = list(scene.frame_nums)

        # Determine max crop size (LADA uses box extents)
        sizes = [_box_size(b) for b in scene.crop_boxes]
        max_h = max(h for h, _ in sizes)
        max_w = max(w for _, w in sizes)

        scale_h = float(clip_size) / float(max_h)
        scale_w = float(clip_size) / float(max_w)

        frames: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        boxes: List[Box] = []
        crop_shapes: List[Tuple[int, int]] = []
        stored_shapes: List[Tuple[int, int]] = []
        pad_after_resizes: List[Pad] = []

        for crop_u8, mask_u8, box in zip(scene.crops, scene.masks, scene.crop_boxes):
            # crop_u8: uint8 HWC (BGR). CM-227: it may be STORED smaller than
            # the box (capped at capture); geometry comes from the box, the
            # pixels from whatever is stored.
            ch, cw = _box_size(box)
            if (int(crop_u8.shape[0]), int(crop_u8.shape[1])) != (ch, cw) and \
                    crop_storage_scale(ch, cw, getattr(scene, "crop_cap_px", 0)) >= 1.0:
                # not capped, yet a different shape: keep the pre-CM-227 rule
                ch, cw = int(crop_u8.shape[0]), int(crop_u8.shape[1])
            crop_shapes.append((ch, cw))
            stored_shapes.append((int(crop_u8.shape[0]), int(crop_u8.shape[1])))
            boxes.append(box)

            out_h = int(ch * scale_h)
            out_w = int(cw * scale_w)

            # Resize uint8 image (LADA: INTER_LINEAR)
            #
            # CM-227 T26: a CAPPED stored crop (shape != box) goes through
            # resample_capped_to_clip instead -- a 2-tap sample at the native
            # clip sample's position on the aligned storage grid, so the
            # capped chain reads the frame where the one-step native chain
            # does (bit-for-bit along an integer-ratio axis, unbiased
            # elsewhere) and the restorer sees the native path's detail with
            # no displacement. See the T26 note in scene.py for the why and
            # the numbers. Uncapped crops: the plain bilinear, unchanged.
            _sh, _sw = int(crop_u8.shape[0]), int(crop_u8.shape[1])
            _capped = (_sh, _sw) != (ch, cw)
            if _capped:
                img_rs = resample_capped_to_clip(crop_u8, (out_h, out_w), int(getattr(scene, "crop_cap_px", 0)))
            else:
                img_rs = image_utils.resize(crop_u8, size=(out_h, out_w), interpolation=cv2.INTER_LINEAR)

            # Pad to clip_size (LADA uses configured pad_mode)
            img_pad, pad = image_utils.pad_image(img_rs, clip_size, clip_size, mode=pad_mode)
            frames.append(img_pad)
            pad_after_resizes.append(pad)

            # Mask: uint8 HW, LADA: INTER_NEAREST, pad with zeros
            if mask_u8 is None:
                mask_u8 = torch.ones((ch, cw), device=crop_u8.device, dtype=torch.uint8) * 255

            # image_utils.resize expects HWC for torch, so add channel then squeeze.
            if _capped and tuple(mask_u8.shape[:2]) == (_sh, _sw):
                # CM-227 T26: the capped mask follows the image's aligned grid
                mask_rs = resample_capped_to_clip(mask_u8, (out_h, out_w), int(getattr(scene, "crop_cap_px", 0)),
                                                  is_mask=True).unsqueeze(-1)
            else:
                mask_ch = mask_u8.unsqueeze(-1)  # HWC with C=1
                mask_rs = image_utils.resize(mask_ch, size=(out_h, out_w), interpolation=cv2.INTER_NEAREST)
            mask_pad, _ = image_utils.pad_image(mask_rs, clip_size, clip_size, mode="zero")
            if mask_pad.ndim == 3 and mask_pad.shape[2] == 1:
                mask_pad = mask_pad[:, :, 0]
            masks.append(mask_pad)

        _pw = getattr(scene, "paste_weights", None)
        weights = _pw() if callable(_pw) else None
        n_seeded = min(len(getattr(scene, "seed_weights", []) or []), len(frame_nums))

        return cls(
            clip_id=clip_id,
            frame_start=frame_start,
            frame_nums=frame_nums,
            clip_size=int(clip_size),
            frames=frames,
            masks=masks,
            boxes=boxes,
            crop_shapes=crop_shapes,
            pad_after_resizes=pad_after_resizes,
            stored_shapes=stored_shapes,
            weights=weights,
            n_seeded=int(n_seeded),
        )


    def __len__(self) -> int:
        return len(self.frames)

    @property
    def crop_boxes(self) -> List[Box]:
        return self.boxes

    def weight_at(self, i: int) -> float:
        """CM-213: paste weight of clip frame i (1.0 for an ordinary clip)."""
        if self.weights is None or i >= len(self.weights):
            return 1.0
        return float(self.weights[i])


    def pop(self) -> Tuple[torch.Tensor, torch.Tensor, Box, Tuple[int, int], Pad]:
        """Pop the earliest clip element (LADA-compatible signature)."""
        if not self.frames:
            raise IndexError("pop from empty clip")
        img = self.frames.pop(0)
        m = self.masks.pop(0)
        box = self.boxes.pop(0)
        shape = self.crop_shapes.pop(0)
        pad = self.pad_after_resizes.pop(0)
        self.frame_nums.pop(0)
        if self.weights:
            self.weights.pop(0)
        if self.n_seeded > 0:
            self.n_seeded -= 1
        return img, m, box, shape, pad
