from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import torch

from chitramaya.mosaic.core.scene import Box, Pad, Scene
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
        pad_after_resizes: List[Pad] = []

        for crop_u8, mask_u8, box in zip(scene.crops, scene.masks, scene.crop_boxes):
            # crop_u8: uint8 HWC (BGR)
            ch, cw = int(crop_u8.shape[0]), int(crop_u8.shape[1])
            crop_shapes.append((ch, cw))
            boxes.append(box)

            out_h = int(ch * scale_h)
            out_w = int(cw * scale_w)

            # Resize uint8 image (LADA: INTER_LINEAR)
            img_rs = image_utils.resize(crop_u8, size=(out_h, out_w), interpolation=cv2.INTER_LINEAR)

            # Pad to clip_size (LADA uses configured pad_mode)
            img_pad, pad = image_utils.pad_image(img_rs, clip_size, clip_size, mode=pad_mode)
            frames.append(img_pad)
            pad_after_resizes.append(pad)

            # Mask: uint8 HW, LADA: INTER_NEAREST, pad with zeros
            if mask_u8 is None:
                mask_u8 = torch.ones((ch, cw), device=crop_u8.device, dtype=torch.uint8) * 255

            # image_utils.resize expects HWC for torch, so add channel then squeeze.
            mask_ch = mask_u8.unsqueeze(-1)  # HWC with C=1
            mask_rs = image_utils.resize(mask_ch, size=(out_h, out_w), interpolation=cv2.INTER_NEAREST)
            mask_pad, _ = image_utils.pad_image(mask_rs, clip_size, clip_size, mode="zero")
            if mask_pad.ndim == 3 and mask_pad.shape[2] == 1:
                mask_pad = mask_pad[:, :, 0]
            masks.append(mask_pad)

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
)


    def __len__(self) -> int:
        return len(self.frames)

    @property
    def crop_boxes(self) -> List[Box]:
        return self.boxes

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
        return img, m, box, shape, pad
