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
        self._fp16 = (self.device.type == "cuda"
                      and torch.cuda.get_device_capability(self.device)[0] >= 7)
        if self._fp16:
            model.half()
        self._model = model
        self._dtype = torch.float16 if self._fp16 else torch.float32

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

        out_frames: List[torch.Tensor] = []
        for i in range(n):
            idxs = [min(max(i + k, 0), n - 1) for k in range(-_RADIUS, _RADIUS + 1)]
            inp = stack[idxs].unsqueeze(0)                 # [1,7,3,H,W]
            out = self._model(inp)[0]                      # [3,H,W] RGB
            out = out.clamp_(0.0, 1.0).mul_(255.0).round_().to(torch.uint8)
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
