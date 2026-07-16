# chitramaya/mosaic/restorer/mosaic_clip_restorer.py
"""Mosaic (pixelate) clip restorer — the SFW auto-censor.

Same plug-in interface as the BasicVSR++ restorer and the pseudo (mask-preview)
restorer, but instead of *reconstructing* the region it *censors* it: the
detected mask area is replaced with an opaque blocky mosaic. This lets the whole
detect -> track -> composite -> encode pipeline (per-eye SBS, TRT detection,
ROI dilation, scene tracking) be reused unchanged to automatically censor
clean NSFW footage.

Runtime clip format (matches PseudoClipRestorer, which is field-proven — the
BaseClipRestorer docstring describing float[0,1] is stale):
  clip.frames: uint8 HWC (BGR) in [0,255] on the decode device
  clip.masks : uint8 HW (0..255), already ROI-dilated upstream when requested
Output: same list, same shapes/dtype, region pixelated where mask > 127.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F

from chitramaya.mosaic.core.clip import Clip
from chitramaya.mosaic.restorer.clip_restorer import BaseClipRestorer


class MosaicClipRestorer(BaseClipRestorer):
    def __init__(self, device: torch.device, block: int = 16) -> None:
        super().__init__(device=device)
        # Mosaic cell size in clip pixels (>=2). The compositor resizes the
        # clip back to the crop with INTER_LINEAR, so very large blocks read
        # as a classic coarse mosaic; small blocks soften slightly on paste.
        self.block = max(2, int(block))
        # Compositor reads this to pick blend math dtype; no real model here.
        self.model_dtype = torch.float32

    @torch.inference_mode()
    def restore_clip(self, clip: Clip) -> List[torch.Tensor]:
        out: List[torch.Tensor] = []
        for frm, m in zip(clip.frames, clip.masks):
            # No mask for this frame -> nothing to censor, pass through.
            if m is None:
                out.append(frm)
                continue

            h, w = int(frm.shape[0]), int(frm.shape[1])
            if h < 2 or w < 2:
                out.append(frm)
                continue

            # Pixelate the whole clip frame once (clips are small crops, so this
            # is cheap), then keep the mosaic only inside the mask.
            x = frm.permute(2, 0, 1).unsqueeze(0).to(torch.float32)  # 1,C,H,W
            sh = max(1, h // self.block)
            sw = max(1, w // self.block)
            small = F.interpolate(x, size=(sh, sw), mode="area")
            pix = F.interpolate(small, size=(h, w), mode="nearest")
            pix = pix.squeeze(0).permute(1, 2, 0).round().clamp_(0, 255).to(torch.uint8)

            keep = (m.to(device=frm.device) > 127).unsqueeze(-1)     # HW1 bool
            out.append(torch.where(keep, pix, frm))

        return out


__all__ = ["MosaicClipRestorer"]
