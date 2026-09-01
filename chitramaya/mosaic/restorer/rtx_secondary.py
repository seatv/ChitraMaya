# chitramaya/mosaic/restorer/rtx_secondary.py
"""Secondary restoration: NVIDIA Maxine RTX Super-Res upscale of restored crops (CM-077, Batch 20).

The primary restorer (BasicVSR++) works on a fixed clip-size crop (256px).
Regions LARGER than that get bilinear-stretched at paste-back and look soft --
worst on VR close-ups. This stage upscales the restored 256 crop to 512/1024
with NVIDIA's Maxine VideoSuperRes effect (the tech behind driver "RTX VSR")
BEFORE paste-back, so the final resize becomes a downscale or near-1:1.

Requirements at runtime: an RTX GPU, a recent NVIDIA driver, and the
`nvidia-vfx` pip package. Everything is probed gracefully -- if any piece is
missing, the pipeline logs a loud warning and runs exactly as before.

Field-validated 2026-07-21 (probe on 1440p restored frames, 1535-frame
sequence): "scaled is definitely better" -- Gman.

CRITICAL ordering note (the jasna lesson): Maxine bundles its OWN TensorRT
and loads it with global symbol visibility. Our BasicVSR++ engines are
serialized against the pip TensorRT; if Maxine's copy wins symbol resolution,
engine deserialization fails with a version mismatch. `_preload_pip_tensorrt`
loads the pip runtime first so torch_tensorrt binds to the matching one.
Import order inside the app: this module must be imported (and the preload
run) before the first nvvfx import -- the constructor handles it.

Gating: `should_apply()` returns True only when the original crop is larger
than the clip size -- upscaling a region that will be pasted back smaller
than 256 adds nothing and risks altering texture, so small regions keep the
proven path untouched. The stage applies only in real restoration mode
(never censor/preview -- pixelation and flat fills must not be "enhanced").
"""

from __future__ import annotations

import ctypes
import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# One dropdown value per mode; shared with pipeline/cli so lists cannot drift.
# Batch 68 (CM-139): esrgan-4x -- Real-ESRGAN compact scaler, all GPU vendors
# (see esrgan_secondary.py; the rtx-* modes remain NVIDIA/Maxine).
SECONDARY_MODES = ("none", "rtx-2x", "rtx-4x", "esrgan-4x")

_QUALITY = "high"          # matches the field-validated probe configuration
_INPUT_SIZE = 256          # must equal restoration clip_size; enforced by caller

_preloaded = False


# Batch 25: nanobind (nvvfx's binding layer) audits live bound objects at
# interpreter shutdown and prints "nanobind: leaked 1 instances
# (nvvfx._ext._Effect)" if the warm secondary is still alive at exit --
# harmless (the process is ending either way), but in the windowed build
# those lines are the last thing in ChitraMaya-console.log and read like a
# crash. Track live instances and release them via atexit so the audit
# finds nothing to complain about.
import atexit as _atexit
import gc as _gc
import weakref as _weakref

_LIVE_SECONDARIES = _weakref.WeakSet()


def _atexit_release_secondaries() -> None:
    for inst in list(_LIVE_SECONDARIES):
        try:
            inst.close()
        except Exception:
            pass
    try:
        _gc.collect()
    except Exception:
        pass


_atexit.register(_atexit_release_secondaries)


def _preload_pip_tensorrt() -> None:
    """Load the pip TensorRT runtime before nvvfx pulls in Maxine's bundled copy."""
    global _preloaded
    if _preloaded:
        return
    try:
        import tensorrt_libs  # type: ignore
        libs_dir = os.path.dirname(tensorrt_libs.__file__)
        if sys.platform == "win32":
            for name in ("nvinfer_10.dll", "nvinfer_plugin_10.dll"):
                p = os.path.join(libs_dir, name)
                if os.path.isfile(p):
                    ctypes.WinDLL(p)
        else:
            for name in ("libnvinfer.so.10", "libnvinfer_plugin.so.10"):
                p = os.path.join(libs_dir, name)
                if os.path.isfile(p):
                    ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
        _preloaded = True
    except ImportError:
        # No pip tensorrt_libs package (e.g. system TRT install) -- nothing to pin.
        _preloaded = True


class SecondaryStats:
    """CM-077b: per-run instrumentation for the secondary stage.

    Counts every crop the compositor offered to the secondary and what
    happened to it, and records WHICH frames actually got an upscale so a
    field run can be inspected at exactly those frames (seek + Test Frame
    with the scaler on/off). Attached as `.stats` on both restorer classes;
    the compositor's _apply_secondary records into it, the pipeline prints
    a [SecStats] line and writes the frame list into the misses JSON.

    largest_px is the answer to "why did it never kick in": if it never
    exceeds min_apply_size (256), no crop on this content was big enough
    to open the gate -- the scaler idle is correct, not a bug.
    """

    __slots__ = ("crops_seen", "crops_upscaled", "skipped_small",
                 "skipped_geom", "largest_px", "applied_frames",
                 "frame_max_px")

    def __init__(self) -> None:
        self.crops_seen = 0        # crops offered while secondary active
        self.crops_upscaled = 0    # gate open -> Maxine ran
        self.skipped_small = 0     # gate closed: orig crop <= min_apply_size
        self.skipped_geom = 0      # unexpected clip geometry (should be 0)
        self.largest_px = 0        # max(orig crop h,w) seen across the run
        self.applied_frames = set()  # frame numbers with >=1 upscaled crop
        # Per-frame biggest crop (longest side, px), every offered crop.
        # This is what makes largest_px FINDABLE: top_frames() ranks it, so
        # "largest_crop_px=2245" comes with the frame number to seek to.
        # Memory: one int per restored frame -- trivial even on 2.5hr runs.
        self.frame_max_px = {}     # frame_num -> max crop px seen there

    def note(self, frame_num, orig_shape_hw, outcome: str) -> None:
        self.crops_seen += 1
        dim = max(int(orig_shape_hw[0]), int(orig_shape_hw[1]))
        if dim > self.largest_px:
            self.largest_px = dim
        if frame_num is not None:
            fn = int(frame_num)
            if dim > self.frame_max_px.get(fn, 0):
                self.frame_max_px[fn] = dim
        if outcome == "applied":
            self.crops_upscaled += 1
            if frame_num is not None:
                self.applied_frames.add(int(frame_num))
        elif outcome == "small":
            self.skipped_small += 1
        else:
            self.skipped_geom += 1

    def top_frames(self, n: int = 20):
        """Frames ranked by their biggest crop, largest first.

        Returns [{"frame": int, "crop_px": int}, ...] -- the seek list for
        inspecting the scaler where it matters most. Ties break toward the
        earlier frame so the list is stable run-to-run."""
        ranked = sorted(self.frame_max_px.items(), key=lambda kv: (-kv[1], kv[0]))
        return [{"frame": f, "crop_px": px} for f, px in ranked[:max(0, int(n))]]


def scale_for_mode(mode: str) -> int:
    m = str(mode or "none").lower()
    if m == "rtx-2x":
        return 2
    if m in ("rtx-4x", "esrgan-4x"):
        return 4
    return 0


class RtxSecondaryRestorer:
    """Maxine VideoSuperRes wrapper: 256 BGR-u8 crop in, 256*scale BGR-u8 crop out.

    Layout note: nvvfx accepts HWC or CHW depending on version; we resolve it
    on the first frame and cache the answer (same approach the field probe
    validated on Gman's cards -- his stack resolved to CHW).
    """

    def __init__(self, *, device: torch.device, scale: int = 2,
                 input_size: int = _INPUT_SIZE) -> None:
        if scale not in (2, 4):
            raise ValueError(f"scale must be 2 or 4, got {scale}")
        if int(input_size) != _INPUT_SIZE:
            raise RuntimeError(
                f"RTX Super-Res secondary expects clip size {_INPUT_SIZE}, "
                f"got {input_size} (custom restoration.clip_size is not supported)"
            )
        _preload_pip_tensorrt()
        from nvvfx import VideoSuperRes  # import AFTER the preload

        self.device = torch.device(device)
        self.scale = int(scale)
        self.min_apply_size = _INPUT_SIZE  # gate: orig crop must exceed this
        self.out_size = _INPUT_SIZE * self.scale

        qmap = {
            "low": VideoSuperRes.QualityLevel.LOW,
            "medium": VideoSuperRes.QualityLevel.MEDIUM,
            "high": VideoSuperRes.QualityLevel.HIGH,
            "ultra": VideoSuperRes.QualityLevel.ULTRA,
        }
        self._stream_ptr = torch.cuda.current_stream(self.device).cuda_stream if \
            self.device.type == "cuda" else 0
        self._sr = VideoSuperRes(device=(self.device.index or 0), quality=qmap[_QUALITY])
        self._sr.output_width = self.out_size
        self._sr.output_height = self.out_size
        self._sr.load()
        self._layout: Optional[str] = None  # resolved on first frame
        self.stats = SecondaryStats()  # CM-077b: run instrumentation
        _LIVE_SECONDARIES.add(self)  # released at exit (nanobind audit)

    # -- gating ------------------------------------------------------------

    def should_apply(self, orig_shape_hw: Tuple[int, int]) -> bool:
        """Upscale only when the paste-back target is LARGER than the crop."""
        return max(int(orig_shape_hw[0]), int(orig_shape_hw[1])) > self.min_apply_size

    # -- inference ---------------------------------------------------------

    def _run(self, rgb_hwc_f32: torch.Tensor) -> torch.Tensor:
        if self._layout == "HWC":
            return torch.from_dlpack(
                self._sr.run(rgb_hwc_f32, stream_ptr=self._stream_ptr).image).clone()
        if self._layout == "CHW":
            t = rgb_hwc_f32.permute(2, 0, 1).contiguous()
            return torch.from_dlpack(
                self._sr.run(t, stream_ptr=self._stream_ptr).image).clone()
        last: Exception | None = None
        for layout, tensor in (("HWC", rgb_hwc_f32),
                               ("CHW", rgb_hwc_f32.permute(2, 0, 1).contiguous())):
            try:
                out = torch.from_dlpack(
                    self._sr.run(tensor, stream_ptr=self._stream_ptr).image).clone()
                self._layout = layout
                return out
            except Exception as e:  # try the other layout once
                last = e
        raise RuntimeError(f"nvvfx rejected both HWC and CHW input layouts: {last}")

    def upscale_frame_bgr_u8(self, img_hwc_u8: torch.Tensor) -> torch.Tensor:
        """(256, 256, 3) BGR uint8 -> (256*scale, 256*scale, 3) BGR uint8, same device."""
        src_device = img_hwc_u8.device
        x = img_hwc_u8.to(device=self.device, non_blocking=True)
        # BGR u8 HWC -> RGB float [0,1] HWC (Maxine is trained on RGB video)
        rgb = x[..., [2, 1, 0]].to(torch.float32).div_(255.0).contiguous()
        out = self._run(rgb)
        # Normalize output to HWC
        if out.ndim == 3 and out.shape[0] in (1, 3):
            out = out.permute(1, 2, 0)
        if out.shape[-1] == 1:
            out = out.repeat(1, 1, 3)
        out_u8 = out.clamp_(0, 1).mul_(255.0).round_().to(torch.uint8)
        # RGB -> BGR, back to the caller's device
        return out_u8[..., [2, 1, 0]].contiguous().to(device=src_device, non_blocking=True)

    def close(self) -> None:
        if getattr(self, "_sr", None) is not None:
            try:
                self._sr.close()
            except Exception:
                pass
            self._sr = None


class InterpSecondaryRestorer:
    """Dependency-free stand-in with the same interface (bicubic upscale).

    Used by the container test harness and available as a debugging fallback;
    NOT wired to any user-facing mode. Produces the same geometry (256 ->
    256*scale) so the compositor's scaled-unpad path can be exercised without
    an RTX GPU or nvvfx.
    """

    def __init__(self, *, device: torch.device, scale: int = 2,
                 input_size: int = _INPUT_SIZE) -> None:
        self.device = torch.device(device)
        self.scale = int(scale)
        self.min_apply_size = int(input_size)
        self.out_size = int(input_size) * self.scale
        self.stats = SecondaryStats()  # CM-077b: run instrumentation

    def should_apply(self, orig_shape_hw: Tuple[int, int]) -> bool:
        return max(int(orig_shape_hw[0]), int(orig_shape_hw[1])) > self.min_apply_size

    def upscale_frame_bgr_u8(self, img_hwc_u8: torch.Tensor) -> torch.Tensor:
        x = img_hwc_u8.permute(2, 0, 1).unsqueeze(0).float()
        y = F.interpolate(x, size=(self.out_size, self.out_size),
                          mode="bicubic", align_corners=False)
        return y.round_().clamp_(0, 255).to(torch.uint8).squeeze(0).permute(1, 2, 0).contiguous()

    def close(self) -> None:
        pass


__all__ = [
    "SECONDARY_MODES",
    "RtxSecondaryRestorer",
    "InterpSecondaryRestorer",
    "SecondaryStats",
    "scale_for_mode",
]
