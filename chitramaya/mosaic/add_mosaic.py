# chitramaya/mosaic/add_mosaic.py
"""Synthetic mosaic overlay (SFW censoring).

Pixelates up to a few user-specified rectangles across a video and encodes
the result -- the inverse of restoration. Used to produce shareable SFW
demo/documentation material from otherwise NSFW sources.

The pixelate core is ported from gRestorer's add_mosaic module (the first
generation of this project); decode/encode is re-scaffolded onto ChitraMaya's
own Decoder/Encoder so it inherits install-anywhere paths, the NVDEC
preflight with ffmpeg-cpu fallback, and the hardened remux behavior.

Coordinates are pixels, INCLUSIVE (t, l, b, r) -- the LADA box convention.
For SBS side-by-side content, rectangles are given in PER-EYE coordinates
(as seen in one eye's view); with sbs=True each rectangle is applied to
both eye halves automatically.
"""
from __future__ import annotations

import time
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from chitramaya.video.decoder import Decoder
from chitramaya.video.encoder import Encoder
from chitramaya.mosaic.pipeline_utils import (
    bgr_u8_to_bgra_u8,
    nv12_to_rgb_hwc_u8,
    rgb_hwc_to_bgr_hwc_u8,
    rgbp_chw_to_rgb_hwc_u8,
    wrap_surface_as_tensor,
)

Box = Tuple[int, int, int, int]  # (t, l, b, r) inclusive


def _clamp_roi_inclusive(t: int, l: int, b: int, r: int, h: int, w: int) -> Optional[Box]:
    """Clamp an inclusive ROI to frame bounds; None if it degenerates."""
    t2 = max(0, min(h - 1, t))
    b2 = max(0, min(h - 1, b))
    l2 = max(0, min(w - 1, l))
    r2 = max(0, min(w - 1, r))
    if b2 < t2 or r2 < l2:
        return None
    return (t2, l2, b2, r2)


@torch.inference_mode()
def pixelate_roi_bgr_u8_inplace(frame_bgr: torch.Tensor, roi: Box, block: int) -> None:
    """Pixelate one ROI in-place on the frame's device (GPU when available).

    frame_bgr: uint8 [H, W, 3]; roi: inclusive (t, l, b, r); block: mosaic
    cell size in pixels. Area-downsample then nearest-upsample -- the classic
    mosaic look. (Ported from gRestorer.)
    """
    if frame_bgr.dtype != torch.uint8 or frame_bgr.ndim != 3 or frame_bgr.shape[-1] != 3:
        raise ValueError(
            f"frame_bgr must be uint8 [H,W,3], got {frame_bgr.dtype} {tuple(frame_bgr.shape)}")

    h, w = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
    roi2 = _clamp_roi_inclusive(*roi, h=h, w=w)
    if roi2 is None:
        return
    t, l, b, r = roi2

    patch = frame_bgr[t: b + 1, l: r + 1, :]
    ph, pw = int(patch.shape[0]), int(patch.shape[1])
    if ph <= 1 or pw <= 1:
        return

    block = max(1, int(block))
    sh = max(1, (ph + block - 1) // block)
    sw = max(1, (pw + block - 1) // block)

    x = patch.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
    small = F.interpolate(x, size=(sh, sw), mode="area")
    up = F.interpolate(small, size=(ph, pw), mode="nearest")
    patch.copy_(up.squeeze(0).permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8))


def expand_rois_sbs(rois: Sequence[Box], frame_w: int) -> List[Box]:
    """Per-eye rectangles -> both halves of an SBS LR frame.

    Each (t, l, b, r) is interpreted in one eye's coordinate space
    (0 <= l,r < frame_w/2). Returns the left-half rect plus the same rect
    shifted by half the frame width, each clamped to its own half so a
    generous rect can never bleed across the eye seam.
    """
    half = frame_w // 2
    out: List[Box] = []
    for (t, l, b, r) in rois:
        l1 = max(0, min(half - 1, int(l)))
        r1 = max(0, min(half - 1, int(r)))
        out.append((int(t), l1, int(b), r1))                      # left eye
        out.append((int(t), l1 + half, int(b), r1 + half))        # right eye
    return out


def run_add_mosaic(
    input_path: str,
    output_path: str,
    rois: Sequence[Box],
    *,
    block: int = 16,
    sbs: bool = False,
    gpu_id: int = 0,
    codec: str = "hevc",
    preset: str = "P5",
    qp: int = 18,
    progress_cb: Optional[Callable[[int, int, float], None]] = None,
    cancel_flag=None,
) -> int:
    """Decode -> pixelate ROIs -> encode. Returns frames written.

    1:1 frame mapping (no detection, tracking or restoration); audio is
    muxed back from the input by the Encoder. progress_cb(frame, total, fps)
    is called periodically; cancel_flag (threading.Event) stops cleanly and
    the partial output is still flushed and remuxed.
    """
    if not rois:
        raise ValueError("No ROIs given -- need 1..3 rectangles (t,l,b,r)")

    decoder = Decoder(
        input_path=str(input_path),
        gpu_id=gpu_id,
        batch_size=8,
        trim_negative_pts=False,
    )
    w = int(decoder.metadata.width)
    h = int(decoder.metadata.height)
    fps = float(decoder.metadata.fps or 0.0) or 30.0
    total = int(decoder.metadata.num_frames or 0)

    device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device("cpu")

    applied: List[Box] = expand_rois_sbs(rois, w) if sbs else [tuple(map(int, r)) for r in rois]
    print(f"[AddMosaic] {w}x{h} @ {fps:.2f}fps  rects={len(rois)}"
          f"{' (SBS -> ' + str(len(applied)) + ' applied)' if sbs else ''}  block={block}")

    encoder = Encoder(
        output_path=str(output_path),
        width=w, height=h, fps=fps,
        codec=str(codec), preset=str(preset), qp=int(qp),
        gpu_id=gpu_id,
        input_path=str(input_path),
        mux_audio=True,
        mp4_faststart=True,
    )

    frames = 0
    t_start = time.perf_counter()
    try:
        while True:
            if cancel_flag is not None and cancel_flag.is_set():
                print("[AddMosaic] cancelled")
                break
            batch = decoder.read_batch()
            if not batch:
                break
            for item in batch:
                if isinstance(item, torch.Tensor):
                    # CPU lane (ffmpeg fallback): NV12 [H*3/2, W] or RGB HWC u8
                    is_nv12 = (item.ndim == 2 and item.dtype == torch.uint8
                               and int(item.shape[0]) == (h * 3 // 2)
                               and int(item.shape[1]) == w)
                    if is_nv12:
                        nv12 = item.to(device, non_blocking=True) if device.type != "cpu" else item
                        rgb = nv12_to_rgb_hwc_u8(nv12, width=w, height=h)
                    else:
                        rgb = item.to(device, non_blocking=True) if device.type != "cpu" else item
                else:
                    # NVDEC lane: PyNvVideoCodec surface (dlpack)
                    t = wrap_surface_as_tensor(item)
                    rgb = t if (t.ndim == 3 and t.shape[-1] == 3) else rgbp_chw_to_rgb_hwc_u8(t)
                    if device.type != "cpu" and rgb.device != device:
                        rgb = rgb.to(device, non_blocking=True)

                bgr = rgb_hwc_to_bgr_hwc_u8(rgb.contiguous())
                for roi in applied:
                    pixelate_roi_bgr_u8_inplace(bgr, roi=roi, block=block)
                encoder.encode_frame(bgr_u8_to_bgra_u8(bgr))
                frames += 1

            if progress_cb is not None:
                elapsed = max(1e-6, time.perf_counter() - t_start)
                progress_cb(frames, total, frames / elapsed)
    finally:
        encoder.close()
        decoder.close()

    print(f"[AddMosaic] done: frames={frames} -> {output_path}")
    return frames
