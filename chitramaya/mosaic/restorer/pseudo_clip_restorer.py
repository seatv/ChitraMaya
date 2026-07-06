from __future__ import annotations

from typing import List, Tuple

import torch

from chitramaya.mosaic.core.clip import Clip
from chitramaya.mosaic.restorer.clip_restorer import BaseClipRestorer


class PseudoClipRestorer(BaseClipRestorer):
    """A clip-based pseudo restorer.

    This does *not* run a neural model. It simply overlays a translucent color
    inside the clip mask, so we can validate:
      - scene/clip tracking
      - clip crop/resize/pad mapping
      - compositor paste-back ordering

    Output is clip-sized uint8 HWC (BGR) frames in [0,255], matching what the
    real BasicVSR++ restorer emits and what the compositor consumes. (Earlier
    this returned [0,1] floats, which the compositor read as ~0 => black fill.)
    """

    def __init__(
        self,
        device: torch.device,
        fill_color_bgr: Tuple[int, int, int] = (255, 0, 255),
        fill_opacity: float = 0.70,
    ) -> None:
        super().__init__(device=device)
        b, g, r = [int(x) for x in fill_color_bgr]
        # Keep the fill in [0,255] to match the compositor, which treats
        # restored frames as uint8-range. (Was /255 -> [0,1], which the
        # compositor then blended as ~0 and produced a near-black fill.)
        self.fill_color = torch.tensor([b, g, r], device=device, dtype=torch.float32)
        self.fill_opacity = float(fill_opacity)
        # Compositor reads this to know which dtype to do blending math in.
        # Pseudo doesn't have a real model; float32 is fine here.
        self.model_dtype = torch.float32

    def restore_clip(self, clip: Clip) -> List[torch.Tensor]:
        out: List[torch.Tensor] = []

        # Each clip frame is uint8 HWC (BGR) in [0,255] on the decode device.
        for frm, m in zip(clip.frames, clip.masks):
            # m: uint8 HW, 0..255
            if m is None:
                out.append(frm)
                continue

            f = frm.to(dtype=torch.float32)                         # [0,255]
            fill = self.fill_color.to(f.device).view(1, 1, 3)       # [0,255] BGR
            a = (m.to(device=f.device, dtype=torch.float32) / 255.0) * self.fill_opacity  # HW
            a3 = a.unsqueeze(-1)                                     # HW1

            # Blend within mask, then round+clamp back to uint8 like the real
            # restorer so the compositor receives [0,255] values.
            y = f * (1.0 - a3) + fill * a3
            y = y.round_().clamp_(0, 255).to(torch.uint8)
            out.append(y)

        return out


__all__ = ["PseudoClipRestorer"]