# chitramaya/mosaic/restorer/output_upscale.py
"""CM-219 / CM-228: Output Upscale -- every finished frame through a
super-resolution scaler before the encoder, so a 720p or 1080p title comes
out at 1080p / 1440p / 2160p (a target), or at 2x / 4x its size (a factor).

Two scalers (``OUTPUT_SCALERS``):
  esrgan -- Real-ESRGAN compact (realesr-general-x4v3, SRVGGNetCompact,
            BSD-3-Clause; port in esrgan_secondary.py), plain torch, every
            GPU vendor. The net is a fixed 4x; any other size is the 4x
            result area-resampled (2x = a 2x2 mean, exact). Tiled with a
            32-px halo -- bit-exact against an untiled pass, bounded VRAM.
  rtx    -- NVIDIA Maxine VideoSuperRes at whole-frame size (the RTX
            secondary's effect on the full frame; needs an RTX GPU and the
            nvidia-vfx package). Native factors 4/3, 1.5, 2, 3, 4, built for
            video streams and fast enough for titles; Scaler Quality and
            Scaler Denoise apply. When it cannot take the frame (a factor or
            size Maxine refuses, no nvvfx), the pipeline falls back to esrgan
            and says so.

This is a whole-frame stage, distinct from the esrgan-4x / rtx-4x SECONDARY
that upscales restored 256-px crops before paste-back. Placement:
``UpscalingEncoder`` wraps the encoder, so every path that encodes a frame
-- the frame-store drain, the CM-191 re-decode drain, the passthrough loop
(and with it the CM-228 upscale-only mode, where every frame is
passthrough), gap-fill duplicates, VR -- is upscaled without knowing it. The
encoder itself is created at the OUTPUT size by the pipeline.

Cost is honest and large for esrgan: the net runs on every output pixel of
every frame (~0.5 s per 1080p frame on a 3060). The pipeline prints the
per-frame wall at the end and the run report carries it. Limits: NVENC
H.264 stops at 4096 on a side, HEVC/AV1 at 8192; ``output_size_check``
refuses a request past them (the run then continues WITHOUT the upscale and
says so -- never fails).
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from chitramaya.mosaic.restorer.esrgan_secondary import SRVGGNetCompact, _find_weights, _MODEL_FILE

# factors and targets; the UI offers them in this order
OUTPUT_UPSCALE_MODES = ("none", "2x", "4x", "1080p", "1440p", "2160p")
OUTPUT_SCALERS = ("esrgan", "rtx")
MAX_SIDE_H264 = 4096
MAX_SIDE_HEVC = 8192
_TARGET_H = {"1080p": 1080, "1440p": 1440, "2160p": 2160}
# Maxine VideoSuperRes scale factors (NVIDIA Video Effects SDK, SuperRes)
RTX_FACTORS = (4.0 / 3.0, 1.5, 2.0, 3.0, 4.0)


def _even(v: float) -> int:
    return max(2, int(round(v / 2.0)) * 2)


def resolve_output_size(w: int, h: int, mode: str) -> Tuple[Optional[Tuple[int, int]], str]:
    """(out_w, out_h) for an Output Upscale mode on a w x h source, or (None,
    reason) when the mode is off or the source is already at/above the
    target. Targets keep the aspect ratio (width rounded to even)."""
    m = str(mode or "none").strip().lower()
    w, h = int(w), int(h)
    if m in ("none", "", "off"):
        return None, ""
    if m in ("2x", "4x"):
        s = int(m[0])
        return (w * s, h * s), f"{s}x"
    if m in _TARGET_H:
        th = _TARGET_H[m]
        if h >= th:
            return None, f"source is {w}x{h}, already {h}p or taller than the {m} target; ran without the upscale"
        tw = _even(w * th / float(h))
        return (tw, th), f"{m} ({th / float(h):.3g}x)"
    return None, f"unknown Output Upscale {m!r}; ran without the upscale"


def output_size_check(out_w: int, out_h: int, codec: str) -> Tuple[bool, str]:
    """(ok, reason). The encoder's hard side limits per codec."""
    ow, oh = int(out_w), int(out_h)
    lim = MAX_SIDE_H264 if str(codec).lower() in ("h264", "avc") else MAX_SIDE_HEVC
    if max(ow, oh) > lim:
        return False, (f"Output Upscale would make {ow}x{oh}; {str(codec).upper()} encoding stops at "
                       f"{lim} on a side" + (" -- switch to HEVC or AV1, or pick a smaller output"
                                             if lim == MAX_SIDE_H264 else " -- pick a smaller output"))
    return True, f"{ow}x{oh}"


def _resample(x: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
    """1,3,H,W float -> 1,3,oh,ow. Exact 2x2 mean when it is exactly a half;
    area (anti-aliased) for any other downscale; bicubic for an upscale."""
    h, w = int(x.shape[-2]), int(x.shape[-1])
    oh, ow = int(out_hw[0]), int(out_hw[1])
    if (oh, ow) == (h, w):
        return x
    if (2 * oh, 2 * ow) == (h, w):
        return F.avg_pool2d(x, kernel_size=2, stride=2)
    if oh <= h and ow <= w:
        return F.interpolate(x, size=(oh, ow), mode="area")
    return F.interpolate(x, size=(oh, ow), mode="bicubic", align_corners=False).clamp_(0, 1)


class _OutputUpscalerBase:
    """Shared bookkeeping: BGRA u8 in/out, timing, summary."""

    name = "?"

    def __init__(self, *, device: torch.device, in_hw: Tuple[int, int], out_hw: Tuple[int, int]) -> None:
        self.device = torch.device(device)
        self.in_hw = (int(in_hw[0]), int(in_hw[1]))
        self.out_hw = (int(out_hw[0]), int(out_hw[1]))
        self.frames = 0
        self.wall = 0.0

    @property
    def scale(self) -> float:
        return self.out_hw[0] / float(self.in_hw[0])

    def _rgb01(self, bgra_hwc_u8: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        x = bgra_hwc_u8.to(device=self.device, non_blocking=True)
        return x[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).to(dtype).div_(255.0)

    @staticmethod
    def _bgra_u8(out: torch.Tensor) -> torch.Tensor:
        out_u8 = out.squeeze(0).clamp_(0, 1).mul_(255.0).round_().to(torch.uint8).permute(1, 2, 0)   # HWC RGB
        h2, w2 = int(out_u8.shape[0]), int(out_u8.shape[1])
        res = torch.empty((h2, w2, 4), device=out_u8.device, dtype=torch.uint8)
        res[..., 0] = out_u8[..., 2]
        res[..., 1] = out_u8[..., 1]
        res[..., 2] = out_u8[..., 0]
        res[..., 3] = 255
        return res

    def _run(self, rgb: torch.Tensor) -> torch.Tensor:   # 1,3,h,w float -> 1,3,oh,ow float
        raise NotImplementedError

    @torch.no_grad()
    def upscale_bgra_u8(self, bgra_hwc_u8: torch.Tensor) -> torch.Tensor:
        """(H, W, 4) BGRA uint8 -> (oh, ow, 4) BGRA uint8, same device."""
        t0 = time.perf_counter()
        src_device = bgra_hwc_u8.device
        out = self._run(self._rgb01(bgra_hwc_u8, self.work_dtype))
        res = self._bgra_u8(out)
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()
        self.frames += 1
        self.wall += time.perf_counter() - t0
        return res.to(device=src_device, non_blocking=True) if src_device != self.device else res

    def summary(self) -> dict:
        ms = (1000.0 * self.wall / self.frames) if self.frames else 0.0
        return {"scaler": self.name, "scale": round(self.scale, 3), "frames": int(self.frames),
                "wall_s": round(self.wall, 2), "ms_per_frame": round(ms, 1),
                "input": f"{self.in_hw[1]}x{self.in_hw[0]}", "output": f"{self.out_hw[1]}x{self.out_hw[0]}"}

    def describe(self) -> str:
        return f"{self.name} {self.in_hw[1]}x{self.in_hw[0]} -> {self.out_hw[1]}x{self.out_hw[0]}"

    def close(self) -> None:
        pass


class OutputUpscaler(_OutputUpscalerBase):
    """Whole-frame Real-ESRGAN (compact x4) with tiling; other sizes via
    ``_resample`` of the 4x result (2x = exact 2x2 mean, as before)."""

    name = "Real-ESRGAN (realesr-general-x4v3)"

    def __init__(self, *, device: torch.device, in_hw: Tuple[int, int], out_hw: Tuple[int, int],
                 tile: int = 640, halo: int = 32, user_weight_dirs: Optional[list] = None,
                 scale: Optional[int] = None) -> None:
        # ``scale`` kept for callers that still say scale=2/4 with in_hw
        if scale is not None and out_hw is None:
            out_hw = (int(in_hw[0]) * int(scale), int(in_hw[1]) * int(scale))
        super().__init__(device=device, in_hw=in_hw, out_hw=out_hw)
        self.tile = max(128, int(tile))
        self.halo = max(0, int(halo))
        self.work_dtype = torch.float16 if self.device.type in ("cuda", "xpu") else torch.float32
        wpath = _find_weights(user_weight_dirs)
        if wpath is None:
            raise RuntimeError(f"model file {_MODEL_FILE} not found (bundled weights/ dir or ./models) -- broken install?")
        sd = torch.load(str(wpath), map_location="cpu", weights_only=True)
        params = sd.get("params") or sd.get("params_ema") or sd
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4)
        model.load_state_dict(params, strict=True)
        model.eval()
        self.model = model.to(device=self.device, dtype=self.work_dtype)
        self.weights_path = str(wpath)

    @torch.no_grad()
    def _net4(self, rgb: torch.Tensor) -> torch.Tensor:
        """1,3,h,w float -> 1,3,4h,4w float, tiled with a halo."""
        _, _, h, w = rgb.shape
        t, p = self.tile, self.halo
        out = torch.empty((1, 3, 4 * h, 4 * w), device=rgb.device, dtype=rgb.dtype)
        # Reflect-pad the whole frame once so every tile takes its halo, and
        # the frame border sees reflected picture instead of conv zeros. A
        # 32-px halo is bit-exact against an untiled pass on this net (32
        # 3x3 convs = 32-px receptive radius; measured in the sandbox).
        pad_h = min(p, h - 1)
        pad_w = min(p, w - 1)
        src = F.pad(rgb, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
        for y0 in range(0, h, t):
            y1 = min(h, y0 + t)
            for x0 in range(0, w, t):
                x1 = min(w, x0 + t)
                sy0, sy1 = y0, y1 + 2 * pad_h
                sx0, sx1 = x0, x1 + 2 * pad_w
                o = self.model(src[:, :, sy0:sy1, sx0:sx1])
                out[:, :, 4 * y0:4 * y1, 4 * x0:4 * x1] = o[:, :, 4 * pad_h:4 * pad_h + 4 * (y1 - y0),
                                                            4 * pad_w:4 * pad_w + 4 * (x1 - x0)]
        return out

    def _run(self, rgb: torch.Tensor) -> torch.Tensor:
        return _resample(self._net4(rgb), self.out_hw)

    def describe(self) -> str:
        return super().describe() + f" (net 4x, tiles {self.tile} px, halo {self.halo})"

    def close(self) -> None:
        self.model = None


def rtx_factor_for(in_hw: Tuple[int, int], out_hw: Tuple[int, int]) -> float:
    """The smallest Maxine factor whose output covers the target (else the
    largest); the remainder is an area resample down (or bicubic up)."""
    for f in RTX_FACTORS:
        if in_hw[0] * f + 1e-6 >= out_hw[0] and in_hw[1] * f + 1e-6 >= out_hw[1]:
            return f
    return RTX_FACTORS[-1]


class RtxOutputUpscaler(_OutputUpscalerBase):
    """CM-228 (was CM-219b): Maxine VideoSuperRes on the whole frame. Runs at
    the nearest native factor covering the target, then ``_resample``.
    Constructor runs one warm frame so an unsupported size/factor raises
    HERE and the pipeline can fall back to Real-ESRGAN before frame 0."""

    name = "RTX Super-Res (Maxine VideoSuperRes)"

    def __init__(self, *, device: torch.device, in_hw: Tuple[int, int], out_hw: Tuple[int, int],
                 quality: str = "high", denoise: str = "none") -> None:
        super().__init__(device=device, in_hw=in_hw, out_hw=out_hw)
        if self.device.type != "cuda":
            raise RuntimeError("RTX Super-Res needs an NVIDIA GPU (CUDA device)")
        from chitramaya.mosaic.restorer.rtx_secondary import _preload_pip_tensorrt, _LIVE_SECONDARIES
        _preload_pip_tensorrt()
        from nvvfx import VideoSuperRes  # import AFTER the preload
        self.work_dtype = torch.float32
        self.factor = rtx_factor_for(self.in_hw, self.out_hw)
        self.mid_hw = (int(round(self.in_hw[0] * self.factor)), int(round(self.in_hw[1] * self.factor)))
        qmap = {"low": VideoSuperRes.QualityLevel.LOW, "medium": VideoSuperRes.QualityLevel.MEDIUM,
                "high": VideoSuperRes.QualityLevel.HIGH, "ultra": VideoSuperRes.QualityLevel.ULTRA}
        self.quality = str(quality or "high").strip().lower()
        if self.quality not in qmap:
            raise ValueError(f"unknown Scaler Quality {self.quality!r}")
        self._stream_ptr = torch.cuda.current_stream(self.device).cuda_stream
        self._sr = VideoSuperRes(device=(self.device.index or 0), quality=qmap[self.quality])
        self._sr.output_width = int(self.mid_hw[1])
        self._sr.output_height = int(self.mid_hw[0])
        self._sr.load()
        self.denoise = str(denoise or "none").strip().lower()
        self._dn = None
        if self.denoise != "none":
            dmap = {"low": VideoSuperRes.QualityLevel.DENOISE_LOW, "medium": VideoSuperRes.QualityLevel.DENOISE_MEDIUM,
                    "high": VideoSuperRes.QualityLevel.DENOISE_HIGH, "ultra": VideoSuperRes.QualityLevel.DENOISE_ULTRA}
            if self.denoise not in dmap:
                raise ValueError(f"unknown Scaler Denoise {self.denoise!r}")
            self._dn = VideoSuperRes(device=(self.device.index or 0), quality=dmap[self.denoise])
            self._dn.output_width = int(self.mid_hw[1])
            self._dn.output_height = int(self.mid_hw[0])
            self._dn.load()
        self._layout: Optional[str] = None
        _LIVE_SECONDARIES.add(self)   # released at exit (nanobind audit)
        # warm frame: an input Maxine refuses raises now, not at frame 0
        warm = torch.zeros((self.in_hw[0], self.in_hw[1], 4), device=self.device, dtype=torch.uint8)
        _ = self._run(self._rgb01(warm, self.work_dtype))
        self.frames = 0
        self.wall = 0.0

    def _sr_run(self, eff, rgb_hwc: torch.Tensor) -> torch.Tensor:
        if self._layout == "HWC":
            return torch.from_dlpack(eff.run(rgb_hwc, stream_ptr=self._stream_ptr).image).clone()
        if self._layout == "CHW":
            return torch.from_dlpack(eff.run(rgb_hwc.permute(2, 0, 1).contiguous(),
                                             stream_ptr=self._stream_ptr).image).clone()
        last: Optional[Exception] = None
        for layout, tensor in (("HWC", rgb_hwc), ("CHW", rgb_hwc.permute(2, 0, 1).contiguous())):
            try:
                out = torch.from_dlpack(eff.run(tensor, stream_ptr=self._stream_ptr).image).clone()
                self._layout = layout
                return out
            except Exception as e:
                last = e
        raise RuntimeError(f"nvvfx rejected both HWC and CHW input layouts: {last}")

    def _run(self, rgb: torch.Tensor) -> torch.Tensor:
        hwc = rgb.squeeze(0).permute(1, 2, 0).contiguous()          # H,W,3 RGB float
        out = self._sr_run(self._sr, hwc)
        if self._dn is not None:
            out = torch.from_dlpack(self._dn.run(out.contiguous(), stream_ptr=self._stream_ptr).image).clone()
        if out.ndim == 3 and out.shape[0] in (1, 3):                 # CHW -> HWC
            out = out.permute(1, 2, 0)
        if out.shape[-1] == 1:
            out = out.repeat(1, 1, 3)
        x = out.permute(2, 0, 1).unsqueeze(0).contiguous()           # 1,3,mh,mw
        return _resample(x, self.out_hw)

    def describe(self) -> str:
        s = super().describe() + f" (native {self.factor:.3g}x to {self.mid_hw[1]}x{self.mid_hw[0]}"
        if self.mid_hw != self.out_hw:
            s += f", resampled to {self.out_hw[1]}x{self.out_hw[0]}"
        return s + f"; quality={self.quality}, denoise={self.denoise})"

    def close(self) -> None:
        for attr in ("_sr", "_dn"):
            eff = getattr(self, attr, None)
            if eff is not None:
                try:
                    eff.close()
                except Exception:
                    pass
                setattr(self, attr, None)


class UpscalingEncoder:
    """Encoder wrapper: upscale each BGRA frame, then hand it on. Attribute
    reads fall through to the wrapped encoder; ``underlying`` unwraps."""

    def __init__(self, encoder, upscaler: _OutputUpscalerBase) -> None:
        self._encoder = encoder
        self._up = upscaler

    @property
    def underlying(self):
        return self._encoder

    @property
    def upscaler(self) -> _OutputUpscalerBase:
        return self._up

    def encode_frame(self, bgra: torch.Tensor) -> None:
        self._encoder.encode_frame(self._up.upscale_bgra_u8(bgra))

    def close(self) -> None:
        try:
            self._encoder.close()
        finally:
            self._up.close()

    def __getattr__(self, name):
        # only reached when normal lookup fails: delegate to the wrapped encoder
        return getattr(self._encoder, name)


def unwrap_encoder(enc):
    """The raw Encoder / FfmpegEncoder under any AsyncEncoder / UpscalingEncoder layers."""
    seen = 0
    while hasattr(enc, "underlying") and seen < 4:
        enc = enc.underlying
        seen += 1
    return enc


__all__ = ["OUTPUT_UPSCALE_MODES", "OUTPUT_SCALERS", "RTX_FACTORS", "OutputUpscaler", "RtxOutputUpscaler",
           "UpscalingEncoder", "output_size_check", "resolve_output_size", "rtx_factor_for",
           "unwrap_encoder", "MAX_SIDE_H264", "MAX_SIDE_HEVC"]
