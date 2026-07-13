# ChitraMaya/mosaic/pipeline_utils.py
# --------------------------------------------------------------------------
# Pipeline utilities. Originally ported from gRestorer, evolved with:
#   [CHANGE 2] FrameStore: added max_frames cap + vram_bytes() + is_full()
#   [CHANGE 4] FrameStore: PTS tracking per frame
#   [CHANGE 4] drain_store_to_encoder: writes timecodes v2 file for PTS-aware remux
#   [Threading Step 1] AsyncEncoder class
#   [Threading Step 2] REVERTED — AsyncRestorer class removed
# --------------------------------------------------------------------------
from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path
import cv2
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch

Box = Tuple[int, int, int, int]  # (t,l,b,r) inclusive


def now_ms() -> int:
    return int(time.time() * 1000)


@contextlib.contextmanager
def timing(name: str, enabled: bool = True) -> Iterator[None]:
    if not enabled:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = (time.perf_counter() - t0) * 1000.0
        print(f"[Timing] {name}: {dt:.2f} ms")


def sync_device(device: torch.device) -> None:
    """Sync torch work before NVENC/NVDEC reads GPU buffers (CUDA/XPU)."""
    try:
        if device.type == "cuda":
            torch.cuda.synchronize(device=device)
        elif device.type == "xpu" and hasattr(torch, "xpu"):
            torch.xpu.synchronize(device=device)  # type: ignore[attr-defined]
    except Exception:
        # Best-effort; do not crash.
        pass


# Backwards-compatible alias (older code referenced this name)
_sync_device = sync_device


def cfg_first(cfg, paths: Sequence[Sequence[str]], default=None):
    """
    Return the first non-None config value among multiple key-paths.
    Example:
        cfg_first(cfg, [("decoder","gpu_id"), ("gpu_id",)], default=0)
    """
    for keys in paths:
        try:
            v = cfg.get(*keys, default=None)
        except Exception:
            v = None
        if v is not None:
            return v
    return default


def cfg_path(cfg, keys: Sequence[str], default: str = "") -> str:
    """Read a string path from config and normalize/expand it."""
    v = cfg.get(*keys, default=default)
    if v is None:
        return default
    s = str(v).strip()
    if not s:
        return default
    s = os.path.expandvars(os.path.expanduser(s))
    return s


def wrap_surface_as_tensor(surface) -> torch.Tensor:
    """
    PyNvVideoCodec surfaces support DLPack; torch.from_dlpack gives a tensor view.
    The returned tensor is usually uint8 and on GPU.
    """
    return torch.from_dlpack(surface)


def rgbp_chw_to_rgb_hwc_u8(x: torch.Tensor) -> torch.Tensor:
    """
    Decoder output is typically RGBP CHW uint8 on GPU.
    Return RGB HWC uint8 contiguous.
    """
    if x.ndim == 3 and x.shape[0] == 3:
        y = x.permute(1, 2, 0)
    elif x.ndim == 3 and x.shape[-1] == 3:
        y = x
    else:
        raise ValueError(f"Unexpected frame tensor shape: {tuple(x.shape)} (expected CHW or HWC RGB)")
    if y.dtype != torch.uint8:
        y = y.to(torch.uint8)
    return y.contiguous()


def rgb_hwc_to_bgr_hwc_u8(rgb: torch.Tensor) -> torch.Tensor:
    """
    RGB HWC -> BGR HWC. Returns contiguous uint8.
    """
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB, got {tuple(rgb.shape)}")
    if rgb.dtype != torch.uint8:
        rgb = rgb.to(torch.uint8)
    return rgb.flip(-1).contiguous()


def bgr_u8_to_bgra_u8(bgr: torch.Tensor) -> torch.Tensor:
    """
    Encoder expects BGRA uint8 HWC (ARGB format, little-endian).
    """
    if bgr.ndim != 3 or bgr.shape[-1] != 3:
        raise ValueError(f"Expected HWC BGR, got {tuple(bgr.shape)}")
    if bgr.dtype != torch.uint8:
        bgr = bgr.to(torch.uint8)
    h, w, _ = bgr.shape
    out = torch.empty((h, w, 4), device=bgr.device, dtype=torch.uint8)
    out[..., :3].copy_(bgr)
    out[..., 3].fill_(255)
    return out


def clip_box_to_bounds(box: Box, w: int, h: int) -> Box:
    t, l, b, r = box
    t = max(0, min(int(t), h - 1))
    b = max(0, min(int(b), h - 1))
    l = max(0, min(int(l), w - 1))
    r = max(0, min(int(r), w - 1))
    if b < t:
        t, b = b, t
    if r < l:
        l, r = r, l
    return (t, l, b, r)


def seam_split_boxes(
    boxes: Sequence[Box],
    seam_x: int,
    full_w: int,
    full_h: int,
    masks: Optional[Sequence[Optional[torch.Tensor]]] = None,
) -> Tuple[List[Box], Optional[List[Optional[torch.Tensor]]]]:
    """
    Ensure no box crosses the SBS seam. If a box spans the seam, split into up to two.
    Masks (if provided) are *not* precisely split; for seam-crossing boxes we drop the mask (None)
    so downstream uses rectangle masks safely.
    """
    out_boxes: List[Box] = []
    out_masks: Optional[List[Optional[torch.Tensor]]] = [] if masks is not None else None

    for i, box in enumerate(boxes):
        t, l, b, r = clip_box_to_bounds(box, full_w, full_h)
        m = masks[i] if masks is not None else None

        crosses = (l < seam_x) and (r >= seam_x)
        if not crosses:
            out_boxes.append((t, l, b, r))
            if out_masks is not None:
                out_masks.append(m)
            continue

        # Left part
        if l <= seam_x - 1:
            out_boxes.append((t, l, b, seam_x - 1))
            if out_masks is not None:
                out_masks.append(None)

        # Right part
        if r >= seam_x:
            out_boxes.append((t, seam_x, b, r))
            if out_masks is not None:
                out_masks.append(None)

    return out_boxes, out_masks


def split_frame_lr(rgb_hwc: torch.Tensor, layout: str = "lr") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split a full RGB frame into left/right halves.
    layout:
      - "lr": left half is left eye, right half is right eye
      - "rl": swapped (right eye on left half, left eye on right half)
    """
    if rgb_hwc.ndim != 3 or rgb_hwc.shape[-1] != 3:
        raise ValueError("split_frame_lr expects RGB HWC")
    h, w, _ = rgb_hwc.shape
    half = w // 2
    left = rgb_hwc[:, :half, :]
    right = rgb_hwc[:, half : half * 2, :]
    if layout == "lr":
        return left, right
    if layout == "rl":
        return right, left
    raise ValueError(f"Unknown sbs layout: {layout!r}")


def unsplit_boxes_layout(
    boxes_left: Sequence[Box],
    boxes_right: Sequence[Box],
    half_w: int,
    layout: str = "lr",
) -> List[Box]:
    """
    Merge per-half detections back into full-frame coordinates, honoring layout.
    For layout="lr": left boxes stay, right boxes shift +half_w.
    For layout="rl": (because we swapped frames before detection) we invert: first list maps to right half.
    """
    out: List[Box] = []
    if layout == "lr":
        out.extend(boxes_left)
        out.extend([(t, l + half_w, b, r + half_w) for (t, l, b, r) in boxes_right])
        return out
    if layout == "rl":
        out.extend([(t, l + half_w, b, r + half_w) for (t, l, b, r) in boxes_left])
        out.extend(boxes_right)
        return out
    raise ValueError(f"Unknown sbs layout: {layout!r}")


def unsplit_masks_layout(
    masks_left: Optional[Sequence[Optional[torch.Tensor]]],
    masks_right: Optional[Sequence[Optional[torch.Tensor]]],
    full_w: int,
    half_w: int,
    layout: str = "lr",
) -> Optional[List[Optional[torch.Tensor]]]:
    """
    Pad half-width masks back to full width.
    Each mask is HW (or 1xHW); output is full-width HW mask.
    """
    if masks_left is None and masks_right is None:
        return None

    def pad_mask(m: Optional[torch.Tensor], offset_x: int) -> Optional[torch.Tensor]:
        if m is None:
            return None
        if m.ndim == 2:
            hw = m
        elif m.ndim == 3 and m.shape[0] == 1:
            hw = m[0]
        else:
            hw = m
        h = int(hw.shape[-2])
        out = torch.zeros((h, full_w), device=hw.device, dtype=hw.dtype)
        out[:, offset_x : offset_x + half_w].copy_(hw[..., :half_w])
        return out

    out: List[Optional[torch.Tensor]] = []

    if layout == "lr":
        for m in masks_left or []:
            out.append(pad_mask(m, 0))
        for m in masks_right or []:
            out.append(pad_mask(m, half_w))
        return out

    if layout == "rl":
        for m in masks_left or []:
            out.append(pad_mask(m, half_w))
        for m in masks_right or []:
            out.append(pad_mask(m, 0))
        return out

    raise ValueError(f"Unknown sbs layout: {layout!r}")


# ---------------------------------------------------------------------------
# [CHANGE 2] FrameStore: backpressure via max_frames cap
# [CHANGE 4] FrameStore: PTS tracking per frame
# ---------------------------------------------------------------------------
@dataclass
class FrameStore:
    """In-memory store of full frames that may be modified later by clip compositing.

    Changes vs original:
    - Tracks per-frame PTS (presentation timestamp) from the decoder.
    - Supports a max_frames watermark for backpressure.
    - Provides vram_bytes() estimate for monitoring.
    """
    frames_bgr_u8: Dict[int, torch.Tensor]
    frame_pts: Dict[int, Optional[int]]           # [CHANGE 4] frame_num -> PTS (nanoseconds or codec ticks)
    max_frames: int                                 # [CHANGE 2] 0 = unlimited
    _frame_bytes: int                               # cached per-frame byte size

    def __init__(self, max_frames: int = 0) -> None:
        self.frames_bgr_u8 = {}
        self.frame_pts: Dict[int, Optional[int]] = {}
        self.max_frames = int(max_frames)
        self._frame_bytes = 0

    def put(self, frame_num: int, frame_bgr_u8: torch.Tensor, pts: Optional[int] = None) -> None:
        k = int(frame_num)
        self.frames_bgr_u8[k] = frame_bgr_u8
        self.frame_pts[k] = pts                     # [CHANGE 4]
        if self._frame_bytes == 0 and frame_bgr_u8.numel() > 0:
            self._frame_bytes = int(frame_bgr_u8.nelement() * frame_bgr_u8.element_size())

    def pop(self, frame_num: int) -> torch.Tensor:
        k = int(frame_num)
        self.frame_pts.pop(k, None)                  # [CHANGE 4]
        return self.frames_bgr_u8.pop(k)

    def pop_with_pts(self, frame_num: int) -> Tuple[torch.Tensor, Optional[int]]:
        """Pop frame and its PTS together."""         # [CHANGE 4]
        k = int(frame_num)
        pts = self.frame_pts.pop(k, None)
        frm = self.frames_bgr_u8.pop(k)
        return frm, pts

    def keys_sorted(self) -> List[int]:
        return sorted(self.frames_bgr_u8.keys())

    def __len__(self) -> int:
        return len(self.frames_bgr_u8)

    # [CHANGE 2] -------------------------------------------------------
    def is_full(self) -> bool:
        """True when backpressure should pause decoding."""
        if self.max_frames <= 0:
            return False
        return len(self.frames_bgr_u8) >= self.max_frames

    def vram_bytes(self) -> int:
        """Estimated VRAM held by stored frames (bytes)."""
        if self._frame_bytes <= 0:
            return 0
        return len(self.frames_bgr_u8) * self._frame_bytes

    def vram_mb(self) -> float:
        return self.vram_bytes() / (1024.0 * 1024.0)
    # -------------------------------------------------------------------


_DEBUG_FRAME_SET: Optional[set[int]] = None
_DEBUG_DUMP_DIR: Optional[Path] = None
_DEBUG_ENABLED: Optional[bool] = None


def _get_debug_frames() -> set[int]:
    global _DEBUG_FRAME_SET
    if _DEBUG_FRAME_SET is not None:
        return _DEBUG_FRAME_SET

    raw = str(os.getenv("GR_CORRUPT_DUMP_FRAMES", "") or "").strip()
    out: set[int] = set()
    if raw:
        for part in raw.replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            try:
                out.add(int(part))
            except Exception:
                pass

    _DEBUG_FRAME_SET = out
    return out


def _get_debug_dump_dir() -> Optional[Path]:
    global _DEBUG_DUMP_DIR
    if _DEBUG_DUMP_DIR is not None:
        return _DEBUG_DUMP_DIR

    raw = str(os.getenv("GR_CORRUPT_DUMP_DIR", "") or "").strip()
    if not raw:
        _DEBUG_DUMP_DIR = None
        return None

    p = Path(raw)
    p.mkdir(parents=True, exist_ok=True)
    _DEBUG_DUMP_DIR = p
    return p


def _debug_enabled() -> bool:
    global _DEBUG_ENABLED
    if _DEBUG_ENABLED is not None:
        return _DEBUG_ENABLED

    frames = _get_debug_frames()
    dump_dir = _get_debug_dump_dir()
    _DEBUG_ENABLED = bool(frames and dump_dir is not None)
    return _DEBUG_ENABLED


def _maybe_dump_preencode_frame(frame_num: int, frm_bgr: torch.Tensor, pts: Optional[int]) -> None:
    if not _debug_enabled():
        return
    if int(frame_num) not in _get_debug_frames():
        return

    dump_dir = _get_debug_dump_dir()
    if dump_dir is None:
        return

    x = frm_bgr.detach()
    if x.device.type != "cpu":
        x = x.cpu()
    x = x.contiguous()
    arr = x.numpy()

    png_path = dump_dir / f"preencode_f{int(frame_num):06d}.png"
    txt_path = dump_dir / f"preencode_f{int(frame_num):06d}.txt"

    cv2.imwrite(str(png_path), arr)

    checksum = int(arr.astype("uint64").sum() % (1 << 32))
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"frame_num={int(frame_num)}\n")
        f.write(f"pts={pts}\n")
        f.write(f"shape={tuple(arr.shape)}\n")
        f.write(f"dtype={arr.dtype}\n")
        f.write(f"device={frm_bgr.device}\n")
        f.write(f"checksum32={checksum}\n")
        f.write(f"min={int(arr.min())} max={int(arr.max())}\n")

def drain_store_to_encoder(
    *,
    store: FrameStore,
    safe_before: int,
    encoder,
    device: torch.device,
    sync_before_encode: bool = True,
    pts_log: Optional[List[Tuple[int, Optional[int]]]] = None,
) -> int:

    """
    Encode and remove all frames with frame_num < safe_before.
    Returns number of frames encoded.

    Notes:
      - Keeps optional one-sync-per-drain behavior.
      - Per-frame ownership/safety is handled inside Encoder.encode_frame().
    """

    keys = store.keys_sorted()
    drain_keys = [k for k in keys if k < safe_before]
    if not drain_keys:
        return 0

    if sync_before_encode:
        sync_device(device)

    count = 0
    for k in drain_keys:
        frm_bgr, pts = store.pop_with_pts(k)
        _maybe_dump_preencode_frame(k, frm_bgr, pts)
        if pts_log is not None:
            pts_log.append((k, pts))

        bgra = bgr_u8_to_bgra_u8(frm_bgr).contiguous()
        encoder.encode_frame(bgra)
        count += 1

    return count


# ---------------------------------------------------------------------------
# Async encoder helpers — Step 1 of threading.
#
# Why this is safe (unlike T3c's stream-aware decoder attempt):
#  - The async path only moves encoder.encode_frame() off the main thread.
#    Decoder, detector, restorer, compositor all stay sequential.
#  - NVENC consumes BGRA frames READ-ONLY; main thread does the BGR→BGRA
#    conversion and a single sync_device() per drain BEFORE queueing the
#    frame, so when the encoder thread pulls a frame, it's fully valid.
#  - FrameStore mutation stays single-threaded (main thread only). No locks.
#  - Frame order is preserved naturally: drain pops in sorted-key order,
#    queues in order, encoder consumes in queue order.
# ---------------------------------------------------------------------------

import queue
import threading


class AsyncEncoder:
    """
    Background encoder worker. Drop-in replacement for direct encoder
    calls — main thread submits BGRA frames, the worker thread runs
    `underlying.encode_frame()` in the background.

    Public surface mirrors what the main pipeline uses on Encoder:
      - encode_frame(bgra)   — submit (non-blocking unless queue full)
      - flush_and_join()     — wait for queue drain + worker exit
      - close()              — flush + close underlying encoder (triggers remux)
      - underlying           — reference to wrapped Encoder for attr setters
                                (e.g. encoder.underlying._pts_fps = ...)
    """

    _SENTINEL = object()

    def __init__(self, encoder, device=None, queue_size: int = 16) -> None:
        self._encoder = encoder
        self._device = device  # may be None for legacy callers; sync skipped if so
        self._queue: queue.Queue = queue.Queue(maxsize=int(queue_size))
        self._error: Optional[BaseException] = None
        self._stopped: bool = False
        self._frames_encoded: int = 0
        self._worker_wall: float = 0.0
        self._thread = threading.Thread(
            target=self._worker, name="AsyncEncoderWorker", daemon=True
        )
        self._thread.start()

    @property
    def underlying(self):
        return self._encoder

    @property
    def frames_encoded(self) -> int:
        return self._frames_encoded

    @property
    def worker_wall(self) -> float:
        """Total wall time the worker spent inside encode_frame() calls."""
        return self._worker_wall

    def _worker(self) -> None:
        try:
            while True:
                item = self._queue.get()
                if item is self._SENTINEL:
                    return
                bgra = item
                # No sync needed here — the producer (main thread) did a
                # stream-sync on its own stream INSIDE encode_frame() before
                # queueing this tensor, so the kernel that wrote bgra is
                # guaranteed complete. NVENC can DMA from bgra safely.
                t0 = time.perf_counter()
                self._encoder.encode_frame(bgra)
                self._worker_wall += (time.perf_counter() - t0)
                self._frames_encoded += 1
        except BaseException as e:
            self._error = e

    def encode_frame(self, bgra) -> None:
        if self._stopped:
            raise RuntimeError("AsyncEncoder already stopped")
        # CRITICAL: sync the PRODUCER'S (main thread's) own stream so the
        # kernel that just wrote bgra (typically bgr_u8_to_bgra_u8) is
        # guaranteed complete before NVENC reads via DMA on the worker.
        # Per-stream sync, not cudaDeviceSynchronize — blocks main only
        # for its own stream, not other threads' GPU work.
        if self._device is not None:
            import torch
            torch.cuda.current_stream(self._device).synchronize()
        # Bounded put + error re-check loop (queue full → natural
        # backpressure). A plain blocking put() can deadlock the whole run:
        # if the worker dies (e.g. NVENC error 8 under 4K load — see
        # video/encoder.py) while this thread is already parked on a full
        # queue, nothing ever consumes and we hang forever holding the op
        # lock. Re-checking _error between short waits guarantees the
        # worker's exception surfaces here instead of hanging.
        while True:
            if self._error is not None:
                raise self._error
            try:
                self._queue.put(bgra, timeout=0.25)
                return
            except queue.Full:
                continue

    def flush_and_join(self) -> None:
        """Drain the queue, exit the worker thread, surface any exception."""
        if self._stopped:
            return
        self._stopped = True
        # Same bounded-put pattern as encode_frame: a blocking
        # put(SENTINEL) hangs close() if the worker is dead and the queue
        # is full. If the worker has already exited (error or otherwise),
        # skip the sentinel — join() returns immediately.
        while self._error is None and self._thread.is_alive():
            try:
                self._queue.put(self._SENTINEL, timeout=0.25)
                break
            except queue.Full:
                continue
        self._thread.join()
        if self._error is not None:
            raise self._error

    def close(self) -> None:
        self.flush_and_join()
        self._encoder.close()


def drain_store_to_async_encoder(
    *,
    store: FrameStore,
    safe_before: int,
    async_encoder: AsyncEncoder,
    device: torch.device,
    sync_before_encode: bool = True,
    pts_log: Optional[List[Tuple[int, Optional[int]]]] = None,
) -> int:
    """
    Like drain_store_to_encoder but submits BGRA frames to an AsyncEncoder
    instead of calling encode_frame() inline. The sync_device() still
    happens once per drain on the main thread so the BGRA tensors are
    fully written before they hit the worker thread.
    """
    keys = store.keys_sorted()
    drain_keys = [k for k in keys if k < safe_before]
    if not drain_keys:
        return 0

    if sync_before_encode:
        sync_device(device)

    count = 0
    for k in drain_keys:
        frm_bgr, pts = store.pop_with_pts(k)
        _maybe_dump_preencode_frame(k, frm_bgr, pts)
        if pts_log is not None:
            pts_log.append((k, pts))

        bgra = bgr_u8_to_bgra_u8(frm_bgr).contiguous()
        async_encoder.encode_frame(bgra)
        count += 1

    return count


def write_timecodes_v2(
    pts_log: List[Tuple[int, Optional[int]]],
    output_path: str,
    fps: float,
    time_base_den: int = 1_000_000_000,   # NVDEC PTS are typically in nanoseconds
) -> Optional[str]:
    """Write a Matroska-compatible 'timecodes v2' file from collected PTS data.

    Returns the path to the timecodes file, or None if PTS data is unavailable/uniform.
    The file format is:
        # timecode format v2
        0
        33.333
        66.667
        ...
    Each line is a timestamp in milliseconds for the corresponding frame.
    """
    if not pts_log:
        return None

    # Check if we have usable PTS data
    valid_pts = [(fn, p) for fn, p in pts_log if p is not None]
    if len(valid_pts) < 2:
        return None

    # Sort by frame number (should already be sorted, but be safe)
    valid_pts.sort(key=lambda x: x[0])

    # Compute deltas to detect VFR
    pts_values = [p for _, p in valid_pts]
    deltas = [pts_values[i + 1] - pts_values[i] for i in range(len(pts_values) - 1)]

    if not deltas:
        return None

    median_delta = sorted(deltas)[len(deltas) // 2]

    if median_delta <= 0:
        return None

    # Check if VFR: if max delta differs from median by >2%, it's VFR
    min_d = min(deltas)
    max_d = max(deltas)
    is_vfr = (max_d - min_d) / max(1, median_delta) > 0.02

    # Compute PTS-derived FPS for informational purposes
    total_time_ns = pts_values[-1] - pts_values[0]
    if total_time_ns > 0:
        pts_fps = (len(valid_pts) - 1) / (total_time_ns / 1_000_000_000.0)
    else:
        pts_fps = fps

    # Write the timecodes file
    tc_path = output_path + ".timecodes.txt"
    base_pts = pts_values[0]

    with open(tc_path, "w", encoding="utf-8") as f:
        f.write("# timecode format v2\n")
        # For frames with PTS, use actual PTS; for gaps, interpolate
        all_frames = sorted(pts_log, key=lambda x: x[0])
        for fn, p in all_frames:
            if p is not None:
                ms = (p - base_pts) / 1_000_000.0   # ns -> ms
            else:
                # Interpolate from frame number
                ms = (fn - all_frames[0][0]) * (1000.0 / fps)
            f.write(f"{ms:.3f}\n")

    vfr_str = "VFR" if is_vfr else "CFR"
    print(f"[PTS] Wrote timecodes: {tc_path} ({len(all_frames)} frames, {vfr_str}, pts_fps={pts_fps:.3f})")

    return tc_path


def compute_pts_fps(
    pts_log: List[Tuple[int, Optional[int]]],
    fallback_fps: float,
) -> Tuple[float, bool]:
    """Compute actual average FPS from PTS data.

    Returns (fps, is_vfr).
    If PTS data is insufficient, returns (fallback_fps, False).
    """
    valid = [(fn, p) for fn, p in pts_log if p is not None]
    if len(valid) < 10:
        return fallback_fps, False

    valid.sort(key=lambda x: x[0])
    pts_vals = [p for _, p in valid]
    total_ns = pts_vals[-1] - pts_vals[0]

    if total_ns <= 0:
        return fallback_fps, False

    fps = (len(valid) - 1) / (total_ns / 1_000_000_000.0)

    # VFR detection
    deltas = [pts_vals[i + 1] - pts_vals[i] for i in range(len(pts_vals) - 1)]
    median_d = sorted(deltas)[len(deltas) // 2]
    if median_d <= 0:
        return fallback_fps, False

    min_d = min(deltas)
    max_d = max(deltas)
    is_vfr = (max_d - min_d) / max(1, median_d) > 0.02

    return fps, is_vfr


def nv12_to_rgb_hwc_u8(
    nv12: torch.Tensor,
    *,
    width: int,
    height: int,
    matrix: str = "auto",
    full_range: bool = False,
) -> torch.Tensor:
    """Convert an NV12 frame to RGB HWC uint8.

    nv12 is expected to be uint8 shaped [H*3/2, W] (or flattenable to that).
    Conversion runs on nv12.device (CPU/CUDA/XPU).

    Notes:
      - Default assumes limited-range YUV (typical video). Set full_range=True if needed.
      - matrix="auto" picks bt709 for HD-ish frames, else bt601.
    """
    h = int(height)
    w = int(width)
    if nv12.dtype != torch.uint8:
        raise TypeError(f"nv12 must be uint8, got {nv12.dtype}")

    if nv12.ndim == 1:
        nv12 = nv12.view(h * 3 // 2, w)
    elif nv12.ndim != 2:
        raise ValueError(f"nv12 must be 1D or 2D, got shape={tuple(nv12.shape)}")

    y = nv12[:h, :].to(torch.float32)
    uv = nv12[h:, :].contiguous().view(h // 2, w // 2, 2).to(torch.float32)
    u = uv[..., 0]
    v = uv[..., 1]

    # Nearest upsample to full res
    u = u.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
    v = v.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)

    if matrix == "auto":
        matrix = "bt709" if (w >= 1280 or h >= 720) else "bt601"

    if full_range:
        c = y
    else:
        # limited-range luma: [16..235] -> scale by 1.164...
        c = (y - 16.0) * 1.164383

    d = u - 128.0
    e = v - 128.0

    if matrix == "bt709":
        r = c + 1.792741 * e
        g = c - 0.213249 * d - 0.532909 * e
        b = c + 2.112402 * d
    else:
        # bt601
        r = c + 1.402000 * e
        g = c - 0.344136 * d - 0.714136 * e
        b = c + 1.772000 * d

    rgb = torch.stack([r, g, b], dim=-1)
    return rgb.round().clamp(0, 255).to(torch.uint8)
