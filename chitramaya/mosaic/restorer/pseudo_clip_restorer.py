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

    Output is clip-sized float32 frames in [0,1].
    """

    def __init__(
        self,
        device: torch.device,
        fill_color_bgr: Tuple[int, int, int] = (255, 0, 255),
        fill_opacity: float = 0.70,
    ) -> None:
        super().__init__(device=device)
        b, g, r = [int(x) for x in fill_color_bgr]
        self.fill_color = torch.tensor([b, g, r], device=device, dtype=torch.float32) / 255.0
        self.fill_opacity = float(fill_opacity)
        # Compositor reads this to know which dtype to do blending math in.
        # Pseudo doesn't have a real model; float32 is fine here.
        self.model_dtype = torch.float32

    def restore_clip(self, clip: Clip) -> List[torch.Tensor]:
        out: List[torch.Tensor] = []

        # Each clip frame is float32 HWC in [0,1] on the decode device.
        for frm, m in zip(clip.frames, clip.masks):
            # m: uint8 HW, 0..255
            if m is None:
                out.append(frm)
                continue

            a = (m.to(dtype=torch.float32) / 255.0) * self.fill_opacity  # HW
            a3 = a.unsqueeze(-1)  # HW1

            # Blend within mask
            y = frm * (1.0 - a3) + self.fill_color.view(1, 1, 3) * a3
            out.append(y)

        return out


__all__ = ["PseudoClipRestorer"]
