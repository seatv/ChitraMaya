from __future__ import annotations

from typing import List

import torch

from chitramaya.mosaic.restorer.clip_restorer import BaseClipRestorer
from chitramaya.mosaic.models.basicvsrpp.inference import load_model


class BasicVSRPPClipRestorer(BaseClipRestorer):
    """
    Near-direct port of LADA's BasicvsrppMosaicRestorer behavior:
      - input: list of uint8 HWC frames
      - model input: float (0..1) via /255
      - output: uint8 HWC with round+clamp
      - optional chunking via max_frames
    """

    def __init__(
            self,
            model_path: str,
            device: torch.device,
            *,
            fp16: bool,
            max_frames: int = 32,
    ) -> None:
        super().__init__(device=device)

        # Define fp16 + dtype BEFORE loading
        self.fp16 = bool(fp16) and (self.device.type == "cuda")
        self.model_dtype = torch.float16 if self.fp16 else torch.float32
        self.max_frames = int(max_frames)

        # gRestorer load_model signature: (config, checkpoint_path, device, fp16=...)
        self.model = load_model(config=None, checkpoint_path=model_path, device=self.device, fp16=self.fp16)

    @torch.inference_mode()
    def restore_clip(self, clip) -> List[torch.Tensor]:
        frames = clip.frames  # uint8 HWC
        if not frames:
            return []

        out_frames: List[torch.Tensor] = []
        n = len(frames)
        dtype = self.model_dtype

        for start in range(0, n, self.max_frames):
            chunk = frames[start : start + self.max_frames]

            # TCHW uint8
            tchw_u8 = torch.stack([f.permute(2, 0, 1).contiguous() for f in chunk], dim=0)
            # 1,T,C,H,W float
            btchw = tchw_u8.to(device=self.device, dtype=dtype).div_(255.0).unsqueeze(0)

            out = self.model(inputs=btchw)  # -> BTCHW
            out_tchw = out.squeeze(0)

            # Back to uint8 HWC, with LADA's rounding+clamp
            out_u8 = (
                out_tchw.mul(255.0)
                .round_()
                .clamp_(0, 255)
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
                .contiguous()
            )
            out_frames.extend(list(torch.unbind(out_u8, dim=0)))

        return out_frames


__all__ = ["BasicVSRPPClipRestorer"]
