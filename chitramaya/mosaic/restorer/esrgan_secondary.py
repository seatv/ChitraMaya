# chitramaya/mosaic/restorer/esrgan_secondary.py
"""Secondary restoration: Real-ESRGAN upscale of restored crops (CM-139, Batch 68).

The cross-vendor sibling of rtx_secondary.py: same job (upscale the restored
256px crop BEFORE paste-back so large regions stop being bilinear-stretched),
same compositor interface, but the scaler is Real-ESRGAN's compact model
(realesr-general-x4v3, SRVGGNetCompact) running in-process through plain
torch -- which means it works on every edition: CUDA, XPU (Intel Arc), and
ROCm (AMD). Until this batch, secondary restoration was NVIDIA-only (RTX
Super-Res needs Maxine); this gives the Intel and AMD editions their first
patch scaler.

Model: realesr-general-x4v3 (4x), ~4.7MB, BUNDLED in the weights/ directory
next to the temporalfix checkpoints -- no download step, no network. A
user-placed copy in ./models overrides the bundled one (same precedence rule
as the temporal stabilizer weights).

Architecture port: SRVGGNetCompact -- 3x3 conv stem, 32 conv+PReLU blocks,
3x3 conv to out_ch*scale^2, PixelShuffle, plus a nearest-upsampled residual
of the input. Ported from Real-ESRGAN (Xintao Wang et al., BSD-3-Clause);
integration pattern studied from HolyWu's vs-realesrgan (BSD-3-Clause).
Both credited in the README Acknowledgements. Weights load with strict=True,
so any drift between this port and the published checkpoint fails loudly at
construction, never silently at quality.

Precision: fp16 on cuda/xpu (same policy as the restorer, CM-093), fp32 on
CPU. ROCm torch reports device.type == "cuda", so AMD rides the fp16 path.

Gating and stats are shared with the RTX secondary (SecondaryStats,
should_apply semantics): only crops whose ORIGINAL region exceeds 256px get
upscaled; the [SecStats] run summary and misses-JSON frame list work
unchanged because the compositor cannot tell the two scalers apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from chitramaya.mosaic.restorer.rtx_secondary import SecondaryStats

_INPUT_SIZE = 256          # must equal restoration clip_size; enforced by caller
_MODEL_FILE = "realesr-general-x4v3.pth"
_MODEL_SHA256 = "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292"


class SRVGGNetCompact(nn.Module):
    """Compact VGG-style SR network (Real-ESRGAN 'general-x4v3' architecture).

    Port of realesrgan's SRVGGNetCompact (BSD-3-Clause, Xintao Wang et al.).
    Module layout matches the published checkpoint EXACTLY (a flat `body`
    ModuleList of conv/PReLU, indices 0..2*num_conv+2) so the state dict
    loads with strict=True."""

    def __init__(self, num_in_ch: int = 3, num_out_ch: int = 3,
                 num_feat: int = 64, num_conv: int = 32,
                 upscale: int = 4) -> None:
        super().__init__()
        self.upscale = int(upscale)
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        # Residual over the nearest-upsampled input (the model learns the
        # correction, not the image -- Real-ESRGAN compact design).
        out = out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return out


def _find_weights(user_dirs: Optional[list] = None) -> Optional[Path]:
    """User-placed copy first (./models), bundled copy last -- the same
    precedence rule as the temporal stabilizer weights (Batch 29)."""
    candidates = []
    for d in (user_dirs or []):
        try:
            candidates.append(Path(d) / _MODEL_FILE)
        except Exception:
            pass
    candidates.append(Path(__file__).resolve().parent / "weights" / _MODEL_FILE)
    for c in candidates:
        try:
            if c.is_file():
                return c
        except Exception:
            pass
    return None


class EsrganSecondaryRestorer:
    """Real-ESRGAN compact scaler: 256 BGR-u8 crop in, 256*scale BGR-u8 out.

    Same interface as RtxSecondaryRestorer -- the compositor's
    _apply_secondary cannot tell them apart (by design)."""

    def __init__(self, *, device: torch.device, scale: int = 4,
                 input_size: int = _INPUT_SIZE,
                 user_weight_dirs: Optional[list] = None) -> None:
        if scale != 4:
            raise ValueError(
                f"Real-ESRGAN secondary supports scale 4 (realesr-general-x4v3); "
                f"got {scale}")
        if int(input_size) != _INPUT_SIZE:
            raise RuntimeError(
                f"Real-ESRGAN secondary expects clip size {_INPUT_SIZE}, "
                f"got {input_size} (custom restoration.clip_size is not supported)")

        self.device = torch.device(device)
        self.scale = int(scale)
        self.min_apply_size = _INPUT_SIZE  # gate: orig crop must exceed this
        self.out_size = _INPUT_SIZE * self.scale

        wpath = _find_weights(user_weight_dirs)
        if wpath is None:
            raise RuntimeError(
                f"model file {_MODEL_FILE} not found (bundled weights/ dir or "
                f"./models) -- broken install?")

        # fp16 on GPU (cuda covers ROCm too; xpu passes the fp16 battery,
        # CM-093), fp32 on CPU.
        self.dtype = (torch.float16
                      if self.device.type in ("cuda", "xpu") else torch.float32)

        sd = torch.load(str(wpath), map_location="cpu", weights_only=True)
        params = sd.get("params") or sd.get("params_ema") or sd
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64,
                                num_conv=32, upscale=4)
        model.load_state_dict(params, strict=True)  # drift fails HERE, loudly
        model.eval()
        self.model = model.to(device=self.device, dtype=self.dtype)
        self.weights_path = str(wpath)
        self.stats = SecondaryStats()  # CM-077b instrumentation, shared class

    # -- gating (identical semantics to the RTX secondary) ------------------

    def should_apply(self, orig_shape_hw: Tuple[int, int]) -> bool:
        """Upscale only when the paste-back target is LARGER than the crop."""
        return max(int(orig_shape_hw[0]), int(orig_shape_hw[1])) > self.min_apply_size

    # -- inference ----------------------------------------------------------

    @torch.inference_mode()
    def upscale_frame_bgr_u8(self, img_hwc_u8: torch.Tensor) -> torch.Tensor:
        """(256, 256, 3) BGR uint8 -> (1024, 1024, 3) BGR uint8, same device."""
        src_device = img_hwc_u8.device
        x = img_hwc_u8.to(device=self.device, non_blocking=True)
        # BGR u8 HWC -> RGB float [0,1] NCHW (Real-ESRGAN is trained on RGB)
        rgb = x[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0) \
               .to(self.dtype).div_(255.0)
        out = self.model(rgb)  # 1,3,1024,1024
        out_u8 = (out.squeeze(0)
                  .clamp_(0, 1)
                  .mul_(255.0)
                  .round_()
                  .to(torch.uint8)
                  .permute(1, 2, 0))          # HWC RGB
        # RGB -> BGR, back to the caller's device
        return out_u8[..., [2, 1, 0]].contiguous() \
            .to(device=src_device, non_blocking=True)

    def close(self) -> None:
        self.model = None


__all__ = ["EsrganSecondaryRestorer", "SRVGGNetCompact"]
