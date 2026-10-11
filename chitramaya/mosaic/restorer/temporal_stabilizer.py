# chitramaya/mosaic/restorer/temporal_stabilizer.py
"""CM-078 temporal stabilizer (Batch 26): vs_temporalfix in-process.

Single-image restoration and super-resolution produce slightly different
results each frame -- visible as fizzle, shimmer, and wiggly fine detail.
Field motivation: the RTX Super-Res secondary (CM-077) passed its headset
A/B on detail, with possible flicker as the open question; this stage
answers it.

vs_temporalfix (pifroggi, Apache-2.0) is a tiny flow-gated model (~0.5M
params, 2 MB weights) that averages each frame with its 6 temporal
neighbors WHERE the content agrees (confidence-gated, so real motion and
lighting changes pass through untouched). Container-verified on the real
s2 weights: ~31% temporal-variance reduction on synthetic fizzle/jitter
while preserving content, and -- by design -- global brightness changes
are NOT smoothed (they are legitimate lighting).

ChitraMaya integration choices:
  - Runs on the RESTORED CROP SEQUENCES per clip (256-space), before
    paste-back and before the secondary upscale. Only restored pixels are
    ever touched -- the "never worse than the mosaic" principle holds.
    Pre-secondary keeps the tensors uniform (the secondary's output size
    varies per frame with the paste-back gate) and removes the per-frame
    variance the secondary would otherwise amplify.
  - Pure PyTorch via the vendored arch + official .pth weights (the same
    path vs_temporalfix's own CPU/CUDA backend uses). No VapourSynth, no
    new dependencies.
  - 7-frame sliding window (radius 3) with edge clamping, exactly like
    upstream's gen_shifts. Clips shorter than 4 frames pass through
    unchanged (upstream's own minimum; nothing perceptibly flickers in
    3 frames anyway).
  - Strength 1..3 selects the matching official checkpoint (s1/s2/s3).
    0 = off. Fractional strengths (upstream weight interpolation) are
    deliberately not exposed -- integer steps are enough of a dial.

Weights are looked up next to the restoration model, in ./models, and
finally in the BUNDLED copy shipped inside the package (Batch 29:
``chitramaya/mosaic/restorer/weights/`` -- Apache-2.0 permits
redistribution; upstream LICENSE ships alongside). The bundled copy means
a fresh install has temporal stability working out of the box; the two
user-facing dirs stay first in the search order so a user can override
with their own weights. Missing weights still degrade gracefully to a
WARNING + no stabilization, same pattern as the secondary -- but with the
bundle, that WARNING now indicates a broken install rather than a missing
download.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import torch

TEMPORAL_STABILITY_LEVELS = (0, 1, 2, 3)

_MODEL_FILES = {
    1: "temporalfix_s1_v1.1.pth",
    2: "temporalfix_s2_v1.pth",
    3: "temporalfix_s3_v1.pth",
}
_RADIUS = 3            # 7-frame window: [-3 .. +3]
_MIN_CLIP_FRAMES = 4   # upstream minimum; shorter clips pass through


def model_file_for_strength(strength: int) -> str:
    return _MODEL_FILES[int(strength)]


def bundled_weights_dir() -> str:
    """Directory of the weights shipped inside the package (Batch 29).

    Resolved relative to this module, which works in both source checkouts
    and the PyInstaller onedir build (the spec's datas entry recreates
    chitramaya/mosaic/restorer/weights/ under the frozen tree, and frozen
    module __file__ paths point into that same tree)."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")


def find_model_file(strength: int, search_dirs: Sequence[str]) -> Optional[str]:
    """First existing weights file for ``strength`` across ``search_dirs``."""
    name = _MODEL_FILES[int(strength)]
    for d in search_dirs:
        if not d:
            continue
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


class TemporalStabilizer:
    """Stabilize a clip's restored crop sequence (BGR-u8 HWC tensors in/out)."""

    def __init__(self, *, device: torch.device, strength: int,
                 model_path: str, clip_size: int = 256) -> None:
        from chitramaya.mosaic.restorer.temporalfix_arch import temporalfix_arch

        self.device = torch.device(device)
        self.strength = int(strength)
        self.clip_size = int(clip_size)

        state = torch.load(model_path, map_location="cpu", weights_only=True)
        # Constructor constants match upstream's _pytorch backend exactly.
        model = temporalfix_arch(
            fixed_hw=(self.clip_size, self.clip_size),
            conf_thresh=0.6, min_support=1,
            gate_slope=12.0, count_slope=4.0,
        )
        model.load_state_dict(state, strict=True)
        model.eval().to(self.device)
        # CM-093 channels_last experiment: default OFF (field-measured
        # SLOWER in eager on the restorer -- transpose tax around
        # NCHW-forcing ops; see basicvsrpp_clip_restorer.py). Env-gated
        # for future graph-compile experiments.
        import os as _os
        if (self.device.type == "xpu"
                and _os.environ.get("CM_XPU_CHANNELS_LAST", "") == "1"):
            from chitramaya.device import enable_channels_last_convs
            enable_channels_last_convs(model)
        # CM-093: device-aware fp16 (CUDA Volta+; XPU validated in phase 0).
        # T20c (CM-186, the real cause): on ROCm the fp16 stabilizer returns
        # NaN / garbage for part of every 7-frame window (9060 XT, 09-23: 207
        # of 524 frames all-black, the rest "wavy") -- fp32 there. ROCm
        # reports as device.type == "cuda", hence the explicit HIP test.
        from chitramaya.device import fp16_supported
        self._is_hip = bool(getattr(getattr(torch, "version", None), "hip", None))
        self._fp16 = fp16_supported(self.device) and not self._is_hip
        if self._fp16:
            model.half()
        self._model = model
        self._dtype = torch.float16 if self._fp16 else torch.float32
        # T20c: per-window output check (see stabilize_clip). A bad window
        # hands back the UNSTABILIZED restored frame -- the restoration is
        # kept, only the smoothing is skipped for that frame -- and the
        # first bad fp16 window switches the model to fp32 for the run.
        self.bad_windows: int = 0
        self.fp32_fallbacks: int = 0
        self._warned: bool = False

    def precision(self) -> str:
        return "fp16" if self._fp16 else "fp32"

    def _to_fp32(self) -> None:
        if not self._fp16:
            return
        self._model.float()
        self._fp16 = False
        self._dtype = torch.float32
        self.fp32_fallbacks += 1

    @torch.inference_mode()
    def stabilize_clip(self, frames: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """[H,W,3] BGR-u8 tensors in -> same out, temporally stabilized.

        Non-uniform shapes (should not happen for 256-space crops) pass
        through unchanged rather than guessing.
        """
        n = len(frames)
        if n < _MIN_CLIP_FRAMES:
            return list(frames)
        shape0: Tuple[int, ...] = tuple(frames[0].shape)
        if any(tuple(f.shape) != shape0 for f in frames):
            return list(frames)

        # BGR u8 HWC -> RGB float CHW in 0..1, whole clip staged once.
        stack = torch.stack([f.to(self.device, non_blocking=True) for f in frames])
        stack = stack[..., [2, 1, 0]].permute(0, 3, 1, 2).contiguous()
        stack = stack.to(self._dtype).div_(255.0)          # [N,3,H,W] RGB

        def _run(stk: torch.Tensor, which: Sequence[int]) -> List[torch.Tensor]:
            outs: List[torch.Tensor] = []
            for i in which:
                idxs = [min(max(i + k, 0), n - 1) for k in range(-_RADIUS, _RADIUS + 1)]
                outs.append(self._model(stk[idxs].unsqueeze(0))[0])   # [3,H,W] RGB float
            return outs

        def _bad(outs: List[torch.Tensor], stk: torch.Tensor, which: Sequence[int]) -> List[int]:
            """T20c: indices whose window came back non-finite, or black for a
            non-black input. One host sync per clip, not per frame."""
            if not outs:
                return []
            o = torch.stack(outs).float()                         # [m,3,H,W]
            finite = torch.isfinite(o).flatten(1).all(dim=1)
            black = o.abs().flatten(1).amax(dim=1) <= 1e-3
            src_dark = stk[list(which)].float().flatten(1).amax(dim=1) <= 1e-3
            flags = (~finite | (black & ~src_dark)).tolist()
            return [w for w, f in zip(which, flags) if f]

        all_idx = list(range(n))
        outs = _run(stack, all_idx)
        bad = _bad(outs, stack, all_idx)
        if bad and self._fp16:
            # T20c (CM-186, the real cause): fp16 on ROCm returned NaN / garbage
            # for part of every window. Switch to fp32 once for the run and redo
            # the failed windows; the restored frames themselves are untouched.
            self._to_fp32()
            stack = stack.to(torch.float32)
            if not self._warned:
                self._warned = True
                print(f"[TemporalFix] WARNING: {len(bad)} stabilized window(s) came back black / "
                      f"non-finite in fp16; the stabilizer runs in fp32 for the rest of this run "
                      f"(restored frames are unaffected).")
            redo = _run(stack, bad)
            for w, o in zip(bad, redo):
                outs[w] = o
            bad = _bad([outs[w] for w in bad], stack, bad)
        if bad:
            self.bad_windows += len(bad)
            if not self._warned:
                self._warned = True
                print(f"[TemporalFix] WARNING: {len(bad)} stabilized window(s) came back black / "
                      f"non-finite; those frames keep their restored pixels without smoothing "
                      f"(count in the run summary).")
        bad_set = set(bad)
        out_frames: List[torch.Tensor] = []
        for i in all_idx:
            if i in bad_set:
                out_frames.append(frames[i])                   # the restored frame, unsmoothed
                continue
            out = outs[i].clamp_(0.0, 1.0).mul_(255.0).round_().to(torch.uint8)
            out_frames.append(out.permute(1, 2, 0)[..., [2, 1, 0]].contiguous())
        return out_frames


def load_temporal_stabilizer(
    *, device: torch.device, strength: int,
    search_dirs: Sequence[str], clip_size: int = 256,
):
    """Build a TemporalStabilizer, or explain why not.

    Returns ``(stabilizer, None)`` on success, ``(None, reason)`` on any
    failure -- the caller prints the reason and runs without stabilization
    (missing weights must never break a run; same contract as the
    secondary)."""
    s = int(strength)
    if s <= 0:
        return None, None
    if s not in _MODEL_FILES:
        return None, f"invalid strength {strength} (valid: 0-3)"
    path = find_model_file(s, search_dirs)
    if path is None:
        looked = ", ".join(str(d) for d in search_dirs if d)
        return None, (f"weights file {_MODEL_FILES[s]} not found "
                      f"(looked in: {looked})")
    try:
        stab = TemporalStabilizer(device=device, strength=s,
                                  model_path=path, clip_size=clip_size)
        return stab, None
    except Exception as e:
        return None, f"failed to load {os.path.basename(path)}: {e}"


__all__ = [
    "TEMPORAL_STABILITY_LEVELS", "TemporalStabilizer",
    "load_temporal_stabilizer", "find_model_file", "model_file_for_strength",
    "bundled_weights_dir",
]
