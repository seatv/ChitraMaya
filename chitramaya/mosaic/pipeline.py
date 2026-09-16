# ChitraMaya/mosaic/pipeline.py
# --------------------------------------------------------------------------
# Mosaic restoration pipeline. Originally ported from gRestorer, evolved with:
#   [CHANGE 2] FrameStore backpressure: store_max_frames config + is_full() check
#   [CHANGE 3] max_clip_length default: 9 -> 30  (better temporal stability)
#   [CHANGE 4] PTS preservation: read_batch_with_pts, PTS in FrameStore,
#              timecodes file generation, PTS-derived fps for remux
#   [CHANGE 5] face detector backend + restorer blendmask mode plumbed through
#   [Threading Step 1] AsyncEncoder wrap
#   [Threading Step 2] REVERTED — async restorer net-negative on 3060 Ti 8GB
# --------------------------------------------------------------------------
from __future__ import annotations

import datetime as _dt
import os
import queue as _queue
import sys as _sys
import threading as _threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm

from chitramaya.mosaic.core.scene_tracker import SceneTracker, TrackerConfig
from chitramaya.mosaic.detector.core import Detection, Detector as YoloDetector
from chitramaya.mosaic.core.clip import Clip
from chitramaya.mosaic.restorer.basicvsrpp_clip_restorer import BasicVSRPPClipRestorer
from chitramaya.mosaic.restorer.compositor import composite_clip_into_store
from chitramaya.mosaic.redecode import LagDecoder, RedecodeStore, drain_plan_to_encoder, normalize_patch_home
from chitramaya.run_report import RunReport, format_vram, vram_snapshot
from chitramaya.mosaic.vr_projection import composite_clip_into_store_projected
from chitramaya.mosaic.utils.config_util import Config
from chitramaya.video.decoder import Decoder
from chitramaya.video.encoder import Encoder, FfmpegEncoder, nvenc_available

def _fmt_hms(seconds: float) -> str:
    """0:12:34 style for console checkpoints (T9b)."""
    try:
        s = max(0, int(round(float(seconds))))
    except Exception:
        return "?"
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}"


from .pipeline_utils import (
    Box,
    FrameStore,
    bgr_u8_to_bgra_u8,
    clip_box_to_bounds,
    cfg_first,
    cfg_path,
    compute_pts_fps,
    pts_head_skip,
    drain_store_to_encoder,
    drain_store_to_async_encoder,
    AsyncEncoder,
    PtsGapFiller,
    nv12_to_rgb_hwc_u8,
    rgb_hwc_to_bgr_hwc_u8,
    rgbp_chw_to_rgb_hwc_u8,
    seam_split_boxes,
    split_frame_lr,
    sync_device,
    unsplit_boxes_layout,
    unsplit_masks_layout,
    wrap_surface_as_tensor,
    write_timecodes_v2,
)


class RegionStats:
    """CM-196 T11 report nit: the size of every restored region, whether or
    not a secondary upscaler is on (the SecStats crop sizes only existed
    with one). largest_px + the top frames are the seek list for the
    biggest regions -- the 256/512 clip-size study and the VRAM ledger
    both need them on plain runs."""

    __slots__ = ("crops", "largest_px", "frame_max_px")

    def __init__(self) -> None:
        self.crops = 0
        self.largest_px = 0
        self.frame_max_px: Dict[int, int] = {}

    def note(self, frame_num, orig_shape_hw) -> None:
        try:
            dim = max(int(orig_shape_hw[0]), int(orig_shape_hw[1]))
        except Exception:
            return
        self.crops += 1
        if dim > self.largest_px:
            self.largest_px = dim
        if frame_num is not None:
            fn = int(frame_num)
            if dim > self.frame_max_px.get(fn, 0):
                self.frame_max_px[fn] = dim

    def top_frames(self, n: int = 20):
        ranked = sorted(self.frame_max_px.items(), key=lambda kv: (-kv[1], kv[0]))
        return [{"frame": f, "crop_px": px} for f, px in ranked[:max(0, int(n))]]


@dataclass
class DetStats:
    frames_total: int = 0
    frames_with_det: int = 0
    total_boxes: int = 0
    total_roi_area_px: float = 0.0
    frame_area_px: int = 0
    # Set of frame_nums where the detector found at least one box.
    # Used to diagnose "detected but not restored" precisely.
    frames_with_det_set: set = field(default_factory=set)
    # Batch 44 (ab_eval support): when dump_rois is on, keep every
    # frame's final boxes (post-dilate/clip/seam-split -- exactly what
    # the restorer sees) for the misses JSON, so offline analysis tools
    # can mask metrics to the true detected regions instead of
    # reverse-engineering them from output divergence.
    dump_rois: bool = False
    rois: dict = field(default_factory=dict)

    def add(self, boxes, w: int, h: int, frame_num: int = -1) -> None:
        self.frames_total += 1
        if self.frame_area_px == 0:
            self.frame_area_px = int(w) * int(h)
        if not boxes:
            return

        self.frames_with_det += 1
        if frame_num >= 0:
            self.frames_with_det_set.add(int(frame_num))
            if self.dump_rois:
                self.rois[int(frame_num)] = [
                    [int(t), int(l), int(b), int(r)] for (t, l, b, r) in boxes
                ]
        self.total_boxes += len(boxes)

        a = 0.0
        for (t, l, b, r) in boxes:
            ww = max(0, int(r) - int(l) + 1)
            hh = max(0, int(b) - int(t) + 1)
            a += float(ww * hh)
        self.total_roi_area_px += a

    def summary(self):
        avg_area = (self.total_roi_area_px / max(1, self.frames_with_det))
        pct = (avg_area / max(1, self.frame_area_px)) * 100.0
        return self.frames_with_det, self.frames_total, self.total_boxes, avg_area, pct



class _DiscardEncoder:
    """No-op encoder used by the FOI preview.

    The FOI run only needs the captured target frame, not an output video, so
    the compositor's frames are consumed and thrown away — no NVENC encode, no
    .hevc elementary stream, no two-step remux. Satisfies the drain/finalize
    interface (encode_frame + close); it is deliberately NOT an AsyncEncoder,
    so run()'s async flush/join path is skipped.
    """

    def encode_frame(self, *args, **kwargs) -> None:
        pass

    def flush(self, *args, **kwargs) -> None:
        pass

    def close(self, *args, **kwargs) -> None:
        pass


@dataclass
class PipelineMetrics:
    processed_frames: int = 0
    early_passthrough_frames: int = 0

    t_decode: float = 0.0
    t_det: float = 0.0
    t_track: float = 0.0
    t_restore: float = 0.0
    t_encode: float = 0.0
    t_mux: float = 0.0

    t_queue_wait: float = 0.0
    t_prepare: float = 0.0
    t_upload: float = 0.0
    t_csc: float = 0.0

    wall_start: _dt.datetime | None = None
    wall_end: _dt.datetime | None = None

    det_stats: DetStats = field(default_factory=DetStats)

    # Tracking: which frame_nums actually had at least one restored clip
    # composited into them. Compare with det_stats.frames_with_det to
    # diagnose detection-vs-restoration miss rates.
    frames_restored: set = field(default_factory=set)

    # Restorer workload: one entry per restored clip = its frame count. Restore
    # cost tracks clips x clip-length far better than box count, so this is
    # what explains restore-time differences between detection configs.
    clip_lengths: list = field(default_factory=list)

    # Tracking: frames with no detection AND no active scene AND no new
    # clips. These are LEGITIMATE passthroughs — clean frames the eye
    # would expect to see passed through untouched. NOT a detection miss.
    # The "visible miss" count is: total - restored - legit_passthrough.
    frames_legit_passthrough: set = field(default_factory=set)

    # [CHANGE 2] backpressure stats
    backpressure_waits: int = 0

    # CM-191: re-decode frame source stats (0 when a FrameStore was used)
    t_redecode: float = 0.0
    redecode_frames: int = 0
    redecode_skipped: int = 0
    redecode_missing: int = 0
    redecode_peak_patch_mb: float = 0.0

    # CM-186: restored clip frames the black-output guard refused (all-zero
    # restorer output for a non-black source crop); the source crop stays.
    guard_black_frames: list = field(default_factory=list)

    def sum_parts(self) -> float:
        return self.t_decode + self.t_det + self.t_track + self.t_restore + self.t_encode


def _pick_device(gpu_id: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    if hasattr(torch, "xpu") and getattr(torch.xpu, "is_available", lambda: False)():
        return torch.device(f"xpu:{gpu_id}")
    return torch.device("cpu")


def _tensor_boxes_to_list_xyxy(
    boxes_xyxy: Optional[torch.Tensor],
    *,
    w: Optional[int] = None,
    h: Optional[int] = None,
) -> List[Box]:
    """Convert YOLO xyxy boxes -> List[Box] (t,l,b,r) using LADA-style quantization.

    - No rounding: int() truncation (after optional clamp) to avoid 0.5 ping-pong.
    - Optional clamp to [0..w] / [0..h] before truncation for stability.
    """
    if boxes_xyxy is None or boxes_xyxy.numel() == 0:
        return []

    w_f = float(w) if w is not None else None
    h_f = float(h) if h is not None else None

    out: List[Box] = []
    for row in boxes_xyxy.tolist():
        x1, y1, x2, y2 = row

        if w_f is not None:
            x1 = max(0.0, min(float(x1), w_f))
            x2 = max(0.0, min(float(x2), w_f))
        if h_f is not None:
            y1 = max(0.0, min(float(y1), h_f))
            y2 = max(0.0, min(float(y2), h_f))

        # NOTE: int() truncates toward zero (LADA-style after clamp).
        l = int(x1)
        t = int(y1)
        r = int(x2)
        b = int(y2)

        out.append((t, l, b, r))
    return out


def _extract_masks_list(det: Detection) -> Optional[List[Optional[torch.Tensor]]]:
    m = det.masks
    if m is None:
        return None
    if isinstance(m, torch.Tensor):
        if m.ndim == 3:
            return [m[i] for i in range(m.shape[0])]
        if m.ndim == 2:
            return [m]
    return None


# ---------------------------------------------------------------------------
# [CHANGE 2] Default FrameStore size computation
# ---------------------------------------------------------------------------
def _compute_default_store_max(width: int, height: int, max_clip_length: int) -> int:
    frame_bytes = width * height * 3
    if frame_bytes <= 0:
        return 300

    vram_budget = 1.5 * 1024 * 1024 * 1024
    budget_frames = int(vram_budget / frame_bytes)

    # Floor: just enough to hold ONE active clip's source frames plus a
    # modest in-flight buffer (decoder ahead of composite, detection
    # batching headroom). Was `mcl * 2 + 32` historically — that 2× factor
    # was VRAM-blowing overhead that's not actually needed for correctness.
    # Backpressure + the emergency bump cover transient overflow cases.
    min_frames = max(max_clip_length + 32, 64)

    return max(min_frames, min(budget_frames, 600))


# ---------------------------------------------------------------------------
# [CHANGE 2+] Emergency FrameStore cap bump
# ---------------------------------------------------------------------------
def _compute_emergency_store_max(
    width: int,
    height: int,
    max_clip_length: int,
    current_max_frames: int,
    device: torch.device,
) -> int:
    """Compute an emergency (temporary) max_frames cap when an active scene blocks draining.

    The goal is to avoid runaway growth ("decode anyway") while still allowing long-running
    active clips to make progress.

    Strategy:
      - target = max(current_max_frames, max_clip_length + 64)  — modest bump only
      - apply a hard ceiling (default 600)
      - on CUDA/XPU, also cap by a fraction of total device memory (best-effort)
    """
    base = int(current_max_frames)
    if base <= 0:
        return base

    # Smaller bump than the historical 4× — we just want enough headroom to
    # let one more clip's worth of frames buffer if a scene blocks the drain.
    # Larger bumps just consume VRAM without helping correctness.
    target = max(base, int(max_clip_length) + 64)

    # Keep the same absolute guardrail as the default computation unless you
    # explicitly set store_max_frames higher.
    abs_cap = 600

    frame_bytes = int(width) * int(height) * 3  # BGR u8
    if frame_bytes <= 0:
        return max(base, min(target, abs_cap))

    cap_by_mem = abs_cap

    if device.type == "cuda":
        try:
            total = int(torch.cuda.get_device_properties(device).total_memory)
            # Let FrameStore use up to ~55% of total VRAM. Remaining VRAM is for
            # model weights, activations, scratch, and NVENC surfaces.
            budget = int(total * 0.55)
            cap_by_mem = max(base, int(budget // frame_bytes))
        except Exception:
            cap_by_mem = abs_cap

    elif device.type == "xpu" and hasattr(torch, "xpu"):
        # Best-effort: torch.xpu.get_device_properties exists on some builds.
        try:
            getp = getattr(torch.xpu, "get_device_properties", None)  # type: ignore[attr-defined]
            if getp is not None:
                idx = getattr(device, "index", 0) or 0
                props = getp(int(idx))
                total = int(getattr(props, "total_memory", 0))
                if total > 0:
                    budget = int(total * 0.55)
                    cap_by_mem = max(base, int(budget // frame_bytes))
        except Exception:
            cap_by_mem = abs_cap

    ceiling = max(base, min(abs_cap, cap_by_mem if cap_by_mem > 0 else abs_cap))
    return max(base, min(target, ceiling))


# ---------------------------------------------------------------------------
# VRAM oversubscription check (warn up-front instead of paging silently)
# ---------------------------------------------------------------------------
# Base headroom (MB) the run needs BEYOND the FrameStore and the async-encoder
# queue, for restore activations, NVDEC surfaces, and torch scratch allocated
# during the loop (not yet allocated when we measure). The caller ADDS the
# async NVENC queue's real footprint (queue_size x W x H x 4 BGRA — ~600 MB at
# 4K/queue=16) on top. Raise this if it under-warns on your hardware, lower it
# if it over-warns.
_VRAM_BASE_RESERVE_MB = 512


def _vram_free_total(device: torch.device) -> Tuple[Optional[int], Optional[int]]:
    """Best-effort (free_bytes, total_bytes) for the device, or (None, None).

    Uses the driver-level free (torch.cuda.mem_get_info), which accounts for
    TensorRT-managed allocations too (they live outside torch's caching
    allocator), so it reflects what NVDEC/NVENC/the FrameStore can actually
    draw from.
    """
    try:
        if device.type == "cuda":
            free, total = torch.cuda.mem_get_info(device)
            return int(free), int(total)
        if device.type == "xpu":
            # CM-093: driver-level free where the build offers it, else
            # (None, total) -- callers already treat free=None as "cannot
            # measure" and skip the warning rather than guess.
            from chitramaya.device import mem_get_info as _dev_mgi
            return _dev_mgi(device)
    except Exception:
        pass
    return None, None


def _vram_plan(
    *,
    width: int,
    height: int,
    max_clip_length: int,
    requested_frames: int,
    free_bytes: int,
    total_bytes: int,
    reserve_bytes: int = _VRAM_BASE_RESERVE_MB * 1024 * 1024,
) -> Tuple[int, bool, Optional[str]]:
    """Decide the FrameStore cap and whether the run will likely oversubscribe.

    Measured AFTER the models are built (so free_bytes already excludes the
    detector/restorer contexts) and BEFORE the loop (FrameStore still empty).

    Returns ``(final_frames, reduced, warning)``:
      - final_frames : store cap to use. For an oversubscribing AUTO store, it
        is lowered toward what fits (never below a one-clip floor); callers
        should apply this only when the cap was auto-computed.
      - reduced      : True if final_frames < requested_frames.
      - warning      : a message if the *requested* config won't fit free VRAM
        (i.e. paging to system RAM is likely), else None.
    """
    mb = 1024 * 1024
    thin_bytes = 512 * mb  # "headroom is thin" band above a hard shortfall
    frame_bytes = max(1, int(width) * int(height) * 3)
    req = int(requested_frames)
    floor = max(int(max_clip_length) + 8, 32)  # hold ~one clip + small buffer

    req_bytes = req * frame_bytes
    headroom = int(free_bytes) - req_bytes - int(reserve_bytes)

    # Only reduce an oversubscribing AUTO store — down to what fits, never
    # below the one-clip floor. A store that merely leaves thin headroom is
    # left alone (it fits); we just note it.
    final = req
    reduced = False
    if headroom < 0:
        fit_frames = max(0, (int(free_bytes) - int(reserve_bytes))) // frame_bytes
        final = max(floor, int(fit_frames))
        reduced = final < req

    _lev = "Levers: lower --rest-max-clip-length or --store-max-frames, use " \
           "PyTorch or smaller --det-imgsz detection, or a larger-VRAM GPU."
    # SEVERE means "does not fit" — reserve it for when the minimum one-clip
    # store ALONE cannot fit free VRAM (the store frames physically don't have
    # room). The old test (floor + full surface reserve > free) over-fired: on
    # a 6 GB card the store floor fit with ~150 MB to spare and the run was
    # healthy, yet it printed "does not fit this GPU" — crying wolf on exactly
    # the modest hardware the warning most needs to be trusted on. The surface
    # reserve is a padded estimate (transient scratch/surfaces), so it drives
    # the softer OVERSUBSCRIBED tier, not the hard verdict.
    store_floor_bytes = floor * frame_bytes
    warning: Optional[str] = None
    if store_floor_bytes > int(free_bytes):
        warning = (
            f"VRAM SEVERELY oversubscribed at {width}x{height}: even the minimum "
            f"one-clip FrameStore (~{store_floor_bytes // mb} MB for "
            f"max_clip_length={int(max_clip_length)}) exceeds the "
            f"{int(free_bytes) // mb} MB free after models "
            f"({int(total_bytes) // mb} MB total) — before any surfaces or "
            f"activations. This configuration does not fit this GPU; expect "
            f"extreme paging and possible CUDA errors. " + _lev
        )
    elif headroom < 0:
        warning = (
            f"VRAM tight at {width}x{height}: FrameStore ~{(final * frame_bytes) // mb} MB "
            f"+ ~{int(reserve_bytes) // mb} MB surfaces vs {int(free_bytes) // mb} MB "
            f"free ({int(total_bytes) // mb} MB total after models). May page to "
            f"system RAM if scratch exceeds the ~{max(0, (int(free_bytes) - final * frame_bytes)) // mb} MB "
            f"headroom — watch throughput. " + _lev
        )
    elif headroom < thin_bytes:
        warning = (
            f"VRAM headroom is thin (~{headroom // mb} MB free after a "
            f"~{req_bytes // mb} MB FrameStore + ~{int(reserve_bytes) // mb} MB surfaces "
            f"at {width}x{height}); watch for paging. " + _lev
        )
    return final, reduced, warning


# ---------------------------------------------------------------------------
# CM-111 (Batch 50): host-store RAM plan
# ---------------------------------------------------------------------------
# Field-calibrated on the 2026-08-19 The-Idol forensics run (the first NVENC
# error-8 death captured end-to-end by nvGPUMonitor): a 4K60 MCL-300 run on a
# 23.3 GB machine filled its 7.9 GB HOST store on top of an 11 GB baseline,
# system RAM peaked at 90.7%, and nvEncLockBitstream -- which needs lockable
# (pinned, non-pageable) host memory -- failed with error 8. The old check
# ("warn if store > 70% of available-at-startup") passed that run at 58%.
# The two constants below encode what the telemetry actually showed:
#
#   _RAM_GROWTH_ALLOWANCE_MB: non-store host growth during the run. Measured
#   ~3.0 GB on the forensics run (encoder queue pages, SR/TF host buffers,
#   torch host allocations, other-process creep between startup and peak).
#
#   _RAM_FLOOR_*: physical free RAM that must survive AT PEAK for the OS and
#   NVENC's pinned allocations. The dead run bottomed out at ~2.2 GB free;
#   completed runs never went below ~4 GB.
_RAM_GROWTH_ALLOWANCE_MB = 3072
_RAM_FLOOR_MIN_MB = 3072
_RAM_FLOOR_FRACTION = 0.125     # 1/8 of total RAM, if that is larger


def _ram_plan_host(
    *,
    width: int,
    height: int,
    max_clip_length: int,
    requested_frames: int,
    avail_bytes: int,
    total_bytes: int,
) -> Tuple[bool, int, int, int]:
    """Judge whether a HOST FrameStore of ``requested_frames`` fits system
    RAM with the headroom a full run actually needs (see constants above).

    Returns ``(fits, safe_frames, suggested_mcl, floor_bytes)``:
      - fits          : the requested store stays inside the safe budget.
      - safe_frames   : largest store (frames) the safe budget covers.
      - suggested_mcl : largest Max Clip Length whose one-clip store floor
                        (mcl + 32) fits, rounded DOWN to a multiple of 10
                        (0 when even MCL 30 does not fit).
      - floor_bytes   : the free-RAM floor -- also the runtime guard level.

    NOTE: the store cap is NOT silently reduced on a miss. A host store
    smaller than the MCL holdback window just jams against backpressure and
    the emergency bump re-raises it -- the honest lever is a lower MCL, and
    the caller prints exactly that.
    """
    mb = 1024 * 1024
    frame_bytes = max(1, int(width) * int(height) * 3)
    floor_b = max(_RAM_FLOOR_MIN_MB * mb, int(int(total_bytes) * _RAM_FLOOR_FRACTION))
    safe_store_b = int(avail_bytes) - _RAM_GROWTH_ALLOWANCE_MB * mb - floor_b
    safe_frames = max(0, safe_store_b // frame_bytes)
    fits = int(requested_frames) <= safe_frames
    suggested_mcl = max(0, (int(safe_frames) - 32) // 10 * 10)
    return fits, int(safe_frames), int(suggested_mcl), int(floor_b)


@dataclass
class Pipeline:
    cfg: Config

    def __post_init__(self) -> None:
        self.input_path = str(self.cfg.get("input"))
        self.output_path = str(self.cfg.get("output"))
        self.max_frames: Optional[int] = self.cfg.get("max_frames", default=None)

        self.debug: bool = bool(self.cfg.get("debug_enabled", default=False))

        # Batch size drives decode + detection batching. Read the top-level
        # `batch_size` first (CLI --batch-size), falling back to
        # `detection.batch_size` — the key the UI ("Detection Batch") and CLI
        # --det-batch-size actually write. Without the fallback, both of those
        # controls were silently ignored and every UI run used 8.
        _bs = self.cfg.get("batch_size", default=None)
        if _bs is None:
            _bs = self.cfg.get("detection", "batch_size", default=8)
        self.batch_size: int = int(_bs)

        self.dec_gpu_id: int = int(cfg_first(self.cfg, [("decoder", "gpu_id")], default=0))
        self.enc_gpu_id: int = int(cfg_first(self.cfg, [("encoder", "gpu_id")], default=self.dec_gpu_id))
        self.device: torch.device = _pick_device(self.dec_gpu_id)

        self.mode: str = str(self.cfg.get("mode", default="real")).lower()
        self.restorer_name: str = str(self.cfg.get("restorer", default="basicvsrpp")).lower()

        # Decoder extra knobs
        self.dec_output_format: str = str(self.cfg.get("decoder", "output_format", default="RGBP")).upper()
        self.dec_ffmpeg_input_args: str = str(self.cfg.get("decoder", "ffmpeg_input_args", default="") or "")

        self.det_model: str = cfg_path(self.cfg, ("detection", "model_path"), default="")
        self.det_imgsz: int = int(self.cfg.get("detection", "imgsz", default=640))
        self.det_conf: float = float(self.cfg.get("detection", "conf_threshold", default=0.30))
        self.det_iou: float = float(self.cfg.get("detection", "iou_threshold", default=0.70))
        self.det_fp16: bool = bool(self.cfg.get("detection", "fp16", default=True))
        # Batch 44: opt-in per-frame ROI box dump into the misses JSON
        # (tools/ab_eval.py consumes it for region-masked metrics).
        self.det_dump_rois: bool = bool(
            self.cfg.get("detection", "dump_rois", default=False)
        )

        self.roi_dilate: int = int(self.cfg.get("roi_dilate", default=0))
        self.use_seg_masks: bool = bool(self.cfg.get("use_seg_masks", default=True))

        # --- Scene tracking (stabilization + TTL gap-fill) ---
        # See TrackerConfig docstring for the rationale on each. Defaults
        # mirror TrackerConfig defaults exactly (TTL=3, no crop quant/sticky,
        # 8px match pad).
        self.trk_ttl_after_end: int = int(self.cfg.get("scene_tracking", "ttl_after_end", default=3))
        self.trk_crop_quant_px: int = int(self.cfg.get("scene_tracking", "crop_quant_px", default=0))
        self.trk_crop_sticky: bool = bool(self.cfg.get("scene_tracking", "crop_sticky", default=False))
        self.trk_match_pad_px: int = int(self.cfg.get("scene_tracking", "match_pad_px", default=8))

        self.sbs_enabled: bool = bool(self.cfg.get("sbs_enabled", default=False))
        self.sbs_layout: str = str(self.cfg.get("sbs_layout", default="lr")).lower()
        self.sbs_det_split: bool = bool(self.cfg.get("sbs_det_split", default=False))

        # CM-045 (Batch 19): optional per-eye hequirect->fisheye projection.
        # "fisheye" warps each SBS eye before detection/tracking/restoration
        # (for studios that mosaic in viewing space, so blocks arrive warped
        # in the raw frame), then inverse-warps ONLY the restored regions
        # back onto the pristine original frames at composite time.
        self.vr_projection: str = str(
            self.cfg.get("vr_projection", default="none") or "none"
        ).lower()
        from chitramaya.mosaic.vr_projection import VR_PROJECTION_MODES
        if self.vr_projection not in VR_PROJECTION_MODES:
            raise ValueError(
                f"Invalid vr_projection: {self.vr_projection!r} "
                f"(expected one of {VR_PROJECTION_MODES})"
            )
        # Built lazily on the first frame (needs frame dims + device).
        self._vrproj = None

        # CM-077 (Batch 20): optional secondary restoration -- RTX Super-Res
        # upscale of restored crops before paste-back. Validated against the
        # shared mode list; the effect itself is built in run() (needs device)
        # with graceful fallback if nvvfx/driver support is missing.
        self.secondary_restoration: str = str(
            self.cfg.get("secondary_restoration", default="none") or "none"
        ).lower()
        # CM-078 (Batch 26): temporal stabilization strength (0=off, 1..3).
        self.temporal_stability: int = int(
            self.cfg.get("temporal_stability", default=0) or 0
        )
        from chitramaya.mosaic.restorer.temporal_stabilizer import (
            TEMPORAL_STABILITY_LEVELS,
        )
        if self.temporal_stability not in TEMPORAL_STABILITY_LEVELS:
            raise ValueError(
                f"Invalid temporal_stability: {self.temporal_stability!r} "
                f"(valid: {TEMPORAL_STABILITY_LEVELS})"
            )
        self._stabilizer = None

        from chitramaya.mosaic.restorer.rtx_secondary import SECONDARY_MODES
        # Batch 70 (CM-146): optional Maxine denoise pass chained after the
        # RTX secondary's upscale (jasna-ported approach; no effect on the
        # Real-ESRGAN secondary).
        self.secondary_denoise: str = str(
            self.cfg.get("secondary_denoise", default="none") or "none"
        ).strip().lower()
        if self.secondary_restoration not in SECONDARY_MODES:
            raise ValueError(
                f"Invalid secondary_restoration: {self.secondary_restoration!r} "
                f"(expected one of {SECONDARY_MODES})"
            )
        self._secondary = None

        self.rest_model: str = cfg_path(self.cfg, ("restoration", "rest_model_path"), default="")
        self.rest_fp16: bool = bool(self.cfg.get("restoration", "fp16", default=True))
        self.rest_max_clip_length: int = int(self.cfg.get("restoration", "max_clip_length", default=30))
        # Restoration backend: 'auto' (use TRT engines if present, else PyTorch),
        # 'trt' (require TRT engines, fail if absent), 'pytorch' (force PyTorch).
        self.rest_backend: str = str(
            self.cfg.get("restoration", "backend", default="auto") or "auto"
        ).lower()
        if self.rest_backend not in ("auto", "trt", "pytorch"):
            raise ValueError(
                f"Invalid restoration.backend: {self.rest_backend!r} "
                f"(expected 'auto', 'trt', or 'pytorch')"
            )

        # Batch 42: PyTorch-path temporal window cap. 0 (default) = feed
        # each clip to BasicVSR++ whole, matching lada's pipeline semantics
        # (clip length IS the temporal window); a positive value caps the
        # per-forward window (low-VRAM safety valve; 32 = the pre-Batch-42
        # behavior). Tensor path unaffected (engines already run whole
        # clips up to their compiled length).
        self.rest_chunk_frames: int = int(
            self.cfg.get("restoration", "chunk_frames", default=0) or 0
        )
        self.rest_clip_size: int = int(self.cfg.get("restoration", "clip_size", default=256))
        self.rest_border_ratio: float = float(self.cfg.get("restoration", "border_ratio", default=0.06))
        self.rest_pad_mode: str = str(self.cfg.get("restoration", "pad_mode", default="reflect"))
        self.feather_radius: int = int(self.cfg.get("restoration", "feather_radius", default=0))
        self.rest_blendmask: str = str(self.cfg.get("restoration", "blendmask", default="none") or "none").lower()
        if self.rest_blendmask not in ("none", "facefusion"):
            raise ValueError(f"Invalid restoration.blendmask: {self.rest_blendmask!r}")

        # Fixed-box analysis mode (use synth_mosaic.rois instead of detector boxes)
        self.analysis_use_synth_rois: bool = bool(
            self.cfg.get("restoration", "analysis_use_synth_rois", default=False)
        )
        _raw_synth_rois = self.cfg.get("synth_mosaic", "rois", default=[]) or []
        self.analysis_synth_rois: List[Tuple[int, int, int, int]] = []
        try:
            for _roi in _raw_synth_rois:
                if isinstance(_roi, (list, tuple)) and len(_roi) == 4:
                    t, l, b, r = [int(v) for v in _roi]
                    self.analysis_synth_rois.append((t, l, b, r))
        except Exception:
            self.analysis_synth_rois = []

        # [CHANGE 2] FrameStore backpressure
        # 0 = auto-compute from resolution + max_clip_length;  -1 = unlimited
        self.store_max_frames: int = int(self.cfg.get("store_max_frames", default=0))
        # CM-084 (Batch 36): FrameStore backend. "auto" (default) keeps
        # today's device-resident store whenever it fits free VRAM and
        # flips to system RAM only when the projected store would not fit
        # (the long-MCL enabler); "device"/"host" force the choice.
        # CM-191 (v1.71): "redecode" stores no frames at all -- a second
        # decoder on the same source re-produces each frame when it is due
        # for paste-back and encoding. "auto" resolves to it on the NVDEC
        # path (see run()); "device"/"host" keep the FrameStore.
        self.store_backend: str = str(
            self.cfg.get("store_backend", default="auto") or "auto"
        ).strip().lower()
        if self.store_backend not in ("auto", "device", "host", "redecode"):
            print(f"[FrameStore] WARNING: invalid store_backend "
                  f"{self.store_backend!r}; using 'auto'.")
            self.store_backend = "auto"
        # CM-196: where the re-decode path parks pending patches while their
        # clip is open: "host" (pinned RAM, default) | "device" (VRAM, the
        # T10c behaviour, for A/B). Config key redecode_patches / flat
        # redecodePatches / CLI --redecode-patches. No panel control.
        self.redecode_patches: str = normalize_patch_home(
            self.cfg.get("redecode_patches", default="host"))
        # CM-196 T10f: torch.cuda.empty_cache() after drains is OPT-IN. On an
        # 8 GB card at the ceiling the released space is taken by NVDEC/NVENC
        # surfaces and torch's next allocation has to evict someone -- the
        # T10e Dell run (died at 20 %, vs 59 % on T10d-2) is the suspect case.
        # Config key vram_cache_release / flat vramCacheRelease / CLI flag.
        self.vram_cache_release: bool = bool(self.cfg.get("vram_cache_release", default=False))
        # CM-196 T10f: per-stage VRAM ledger -- what each build step costs on
        # the card, so `other` in the [VRAM] lines gets names. Filled by
        # _vram_stage(); the warm UI build (MosaicPipeline) records the
        # detector/restorer here before any run exists.
        self._vram_ledger: list = []        # this run's stages (reset per run)
        # The UI's warm detector/restorer build. process_file() re-runs
        # __post_init__ per file on the same host, so keep what is there.
        self._vram_ledger_warm: list = list(getattr(self, "_vram_ledger_warm", []) or [])
        self._vram_prev = None

        # CM-180: run report (replaces the misses JSON). runReport = beside |
        # temp | off; run_panel = the control panel exactly as the UI
        # submitted it (None on CLI runs).
        self.run_report_mode: str = str(self.cfg.get("runReport", default="beside") or "beside").strip().lower()
        _rp = self.cfg.get("run_panel", default=None)
        self.run_panel = _rp if isinstance(_rp, dict) else None
        # What was ASKED, captured before any forcing (CM-169 mutates det_fp16
        # in _build_detector; the secondary may fall back; TRT may fall back).
        self._asked = {
            "det_fp16": bool(self.det_fp16),
            "rest_fp16": bool(self.rest_fp16),
            "store_backend": str(self.store_backend),
            "secondary": str(self.secondary_restoration),
            "secondary_denoise": str(getattr(self, "secondary_denoise", "none")),
            "rest_backend": str(self.cfg.get("restoration", "backend", default="auto") or "auto"),
            "blendmask": str(self.rest_blendmask),
            "feather_radius": int(self.feather_radius),
            "max_clip_length": int(self.rest_max_clip_length),
            "det_imgsz": int(self.det_imgsz),
            "redecode_patches": str(self.redecode_patches),
        }

        # Encoder base
        self.enc_codec: str = str(self.cfg.get("encoder", "codec", default="hevc")).lower()
        self.enc_preset: str = str(self.cfg.get("encoder", "preset", default="P7"))
        self.enc_qp: int = int(self.cfg.get("encoder", "qp", default=15))
        self.enc_sync_before_encode: bool = bool(self.cfg.get("encoder", "sync_before_encode", default=True))

        # NOTE: per-knob NVENC flags (tune/spatial_aq/bf/lookahead/multipass/etc.)
        # are baked into ChitraMaya.video.encoder.Encoder defaults as a safe subset
        # of Lada's hevc-nvidia-gpu-hq preset. A free-form ffmpeg-style override
        # CLI (`--enc-options`) is reserved for a future drop.

        # Mux/remux
        self.mux_audio: str = str(self.cfg.get("encoder", "mux_audio", default="auto") or "auto").lower()
        self.mux_keep_subs: bool = bool(self.cfg.get("encoder", "mux_keep_subs", default=False))
        self.mux_extra_args: str = str(self.cfg.get("encoder", "mux_extra_args", default="") or "")
        self.mp4_faststart: bool = bool(self.cfg.get("encoder", "mp4_faststart", default=True))

        # Async encoder thread: overlap NVENC encode with the main thread's
        # decode/detect/restore/composite pass. DEFAULT OFF (opt-in): the
        # async path has produced two distinct field failure modes on
        # VRAM/NVENC-pressured cards — worker deadlock-hangs (fixed) and
        # NVENC error 8 (nvEncLockBitstream) under 4K load — while the sync
        # path has been reliable. Sync also skips the ~queue_size x W x H x 4
        # BGRA queue (~600 MB at 4K/queue16), which matters on 8 GB cards.
        # Opt in on healthy/large-VRAM machines for up to ~20-25% wall-time
        # savings on encode-heavy runs (--async-encoder, or the UI checkbox).
        self.async_encoder: bool = bool(self.cfg.get("encoder", "async_encoder", default=False))
        self.async_encoder_queue: int = int(self.cfg.get("encoder", "async_encoder_queue", default=16))

    def _vram_stage(self, label: str, frame=None) -> None:
        """CM-196 T10f: one ledger line per build step: the driver's used
        VRAM now and the delta since the previous stage (= what this step
        allocated, torch and non-torch alike). Printed and kept for the run
        report's `vram.samples` (label, used, delta_mb, torch_*, other)."""
        try:
            snap = vram_snapshot(self.device)
        except Exception:
            snap = None
        if not snap:
            return
        prev = self._vram_prev
        delta = (int(snap["used"]) - int(prev["used"])) if prev else None
        self._vram_prev = dict(snap)
        row = {"label": str(label), "delta_mb": delta}
        row.update(snap)
        if frame is not None:
            row["frame"] = int(frame)
        self._vram_ledger.append(row)
        d = f" ({'+' if delta >= 0 else ''}{delta} MB this step)" if delta is not None else ""
        print(f"[VRAM] {label}: used {snap['used']} / {snap['total']} MB{d}; "
              f"torch reserved {snap['torch_reserved']}, other {snap['other']}")
        rep = getattr(self, "_report", None)
        if rep is not None:
            try:
                rep.vram(str(label), snap, frame=frame, delta_mb=delta)
            except Exception:
                pass

    def _build_detector(self):
        if self.mode == "none":
            return None
        if not self.det_model:
            raise FileNotFoundError("Detector model path is empty (check config.json or --det-model)")

        det_type = str(self.cfg.get("detection", "det_type", default="yolo") or "yolo").lower()
        # CM-169 (measured 2026-09-06, RX 9060 XT, Adrenalin 26.9.1, ROCm
        # 7.2): FP16 detection on ROCm returns detections for the first
        # few frames and then nothing -- PurpleRain 810 -> 18 boxes
        # (6/270 frames), Test Frame 3/167 -- the signature of the FP16
        # kernel path going bad after warm-up (NaN/garbage below the
        # confidence threshold), with t_det 5x SLOWER than FP32 on top.
        # Two field reports said the same before we owned the hardware.
        # Detection runs in FP32 on this edition regardless of the toggle;
        # the request line and the misses JSON echo the effective value.
        # Restoration FP16 is NOT fenced: its output was clean (810/810).
        from chitramaya.device import is_rocm as _is_rocm
        if self.det_fp16 and _is_rocm():
            self.det_fp16 = False
            print("[Detector] FP16 detection is disabled on the AMD (ROCm) "
                  "edition: measured on an RX 9060 XT (2026-09-06), FP16 "
                  "detection stops finding anything after the first frames. "
                  "Detection runs in FP32 here; restoration FP16 is unaffected.")
        print(
            f"[Detector] type={det_type} imgsz={self.det_imgsz} "
            f"conf={self.det_conf} iou={self.det_iou} fp16={self.det_fp16}"
        )
        # CM-201: say what shapes this engine was built for. The ledger line
        # right after the build shows what it costs; an engine without a
        # sidecar was built before CM-201 (ultralytics' 2x-imgsz, batch-8
        # profile) and is the 3 GB item on an 8 GB card.
        try:
            _mp = str(self.det_model)
            if _mp.lower().endswith(".engine"):
                import json as _json
                _sc = Path(_mp + ".json")
                if _sc.is_file():
                    _pi = _json.loads(_sc.read_text(encoding="utf-8"))
                    _mx = _pi.get("shape_max") or []
                    print(f"[Detector] engine profile: {_pi.get('profile', '?')} -- batch 1..{_pi.get('max_batch', '?')}, "
                          f"H,W up to {(_mx[2] if len(_mx) > 2 else '?')} (built {_pi.get('built', '?')}"
                          f"{', ' + str(_pi.get('gpu')) if _pi.get('gpu') else ''})")
                    if str(_pi.get("profile")) == "legacy":
                        print("[Detector] NOTE: legacy profile (max H,W = 2x imgsz, batch 8): TensorRT sizes its "
                              "context for shapes that never run. Recompile from Manage Models to reclaim ~2-3 GB (CM-201).")
                else:
                    print("[Detector] engine profile: unknown (built before CM-201: batch 8, H,W up to 2x imgsz -- "
                          "~3 GB of VRAM on this card). Recompile from Manage Models to reclaim it.")
        except Exception:
            pass

        common = dict(
            model_path=self.det_model,
            device=self.device,
            imgsz=self.det_imgsz,
            conf_thres=self.det_conf,
            iou_thres=self.det_iou,
            fp16=self.det_fp16,
        )

        if det_type == "yolo":
            return YoloDetector(**common)
        elif det_type in ("lada-yolo", "lada_yolo"):
            from chitramaya.mosaic.detector.lada_yolo import LadaYoloDetector
            return LadaYoloDetector(**common)
        else:
            raise ValueError(f"Unknown detector_type: {det_type}")

    def _build_restorer(self):
        # Reset the engine-set VRAM hint; set only when a TRT set is chosen
        # and smaller compiled sets exist (see below).
        self._rest_engine_note = ""

        if self.mode == "none":
            return None

        if self.restorer_name in ("none", "noop"):
            return None

        # Pseudo (viz-only) mode: replaces BasicVSR++ with a flat-color
        # overlay inside the mask. Useful for confirming detection coverage
        # visually — any frame where the original mosaic shows through is
        # a YOLO miss.
        if self.mode == "pseudo" or self.restorer_name == "pseudo":
            from chitramaya.mosaic.restorer.pseudo_clip_restorer import PseudoClipRestorer
            fill = self.cfg.get("visualization", "fill_color", default=[255, 0, 255])
            op = float(self.cfg.get("visualization", "fill_opacity", default=0.70))
            r, g, b = [int(x) for x in fill]  # config is RGB
            return PseudoClipRestorer(device=self.device, fill_color_bgr=(b, g, r), fill_opacity=op)

        # Mosaic (auto-censor) mode: pixelate the detected mask instead of
        # restoring it. Reuses the entire detect/track/composite path; no
        # restoration model required.
        if self.mode == "mosaic" or self.restorer_name == "mosaic":
            from chitramaya.mosaic.restorer.mosaic_clip_restorer import MosaicClipRestorer
            block = int(self.cfg.get("visualization", "block", default=16))
            return MosaicClipRestorer(device=self.device, block=block)

        if not self.rest_model:
            raise FileNotFoundError("Restoration model path is empty (check config.json or --rest-model)")

        # Prefer TensorRT sub-engines if they exist for this checkpoint +
        # precision (v1.60: engines are clip-size independent — any complete
        # set serves any Max Clip). Falls back to PyTorch on any failure
        # so a partial/incompatible engine set never blocks restoration.
        if str(self.rest_backend).lower() != "pytorch":
            try:
                from chitramaya.mosaic.models.basicvsrpp.engine_paths import (
                    _basicvsrpp_sub_engine_dir,
                    basicvsrpp_engines_usable,
                    find_basicvsrpp_engine_files,
                )

                # v1.60 (CM-104): engines are CLIP-SIZE INDEPENDENT. The
                # loader picks the best files on disk (canonical fixed-batch
                # set, or a legacy clip-sized set from pre-1.60 compiles) and
                # the runtime chunks preprocess/upsample within per-file
                # batch caps. Max Clip Length is purely the RUNTIME temporal
                # window now — no snapping to compiled sizes, no ceiling from
                # the engine layer; the dial costs activation memory only.
                req_mcl = int(self.rest_max_clip_length)
                info = find_basicvsrpp_engine_files(
                    self.rest_model, self.rest_fp16,
                )
                self._rest_engine_note = ""
                if info is not None:
                    if info["fixed"]:
                        _set_desc = (
                            f"fixed-batch b{info['preprocess_b']}/"
                            f"b{info['upsample_b']} (clip-size independent)"
                        )
                    else:
                        _set_desc = (
                            f"LEGACY preprocess_b{info['preprocess_b']} + "
                            f"upsample_b{info['upsample_b']}"
                        )
                        # One recompile converts a ladder-era set into the
                        # ~1 GB-class fixed-batch set. Not forced: legacy
                        # files run correctly through the batched runtime,
                        # they just keep their old (larger) reservation.
                        self._rest_engine_note = (
                            f" Note: legacy clip-sized engine files are in "
                            f"use (preprocess_b{info['preprocess_b']}, "
                            f"upsample_b{info['upsample_b']}). Recompiling "
                            f"once (Manage Models -> Compile) builds the "
                            f"v1.60 fixed-batch set, which reserves several "
                            f"GB less VRAM and serves every Max Clip value."
                        )
                        print(f"[Restorer] {self._rest_engine_note.strip()}")
                    # CM-103 (reworked for v1.60): the reservation follows
                    # the largest COMPILED batch among the loaded files. A
                    # legacy b120+ preprocess/upsample file still reserves
                    # near-whole-card scratch on small GPUs (field-proven:
                    # NVENC error 8 mid-encode). The fixed-batch set makes
                    # this caution structurally silent.
                    try:
                        _max_file_b = max(
                            int(info["preprocess_b"]), int(info["upsample_b"]),
                        )
                        if (self.device.type == "cuda"
                                and not info["fixed"] and _max_file_b >= 120):
                            _tot_b = torch.cuda.get_device_properties(
                                self.device).total_memory
                            if _tot_b < 12 * (1024 ** 3):
                                print(
                                    f"[Restorer] CAUTION: legacy engine file"
                                    f"(s) compiled at b{_max_file_b} on a "
                                    f"{_tot_b // (1024**3)} GB GPU typically "
                                    f"leave no VRAM for the encoder — runs "
                                    f"tend to fail mid-encode (NVENC error "
                                    f"8) regardless of the Max Clip value, "
                                    f"because the reservation follows the "
                                    f"COMPILED size, not the dial. Fix: "
                                    f"recompile once (Manage Models -> "
                                    f"Compile) — v1.60 engines are small "
                                    f"and clip-size independent."
                                )
                    except Exception:
                        pass

                    from chitramaya.mosaic.restorer.basicvsrpp_trt_clip_restorer import (
                        BasicVSRPPTRTClipRestorer,
                    )
                    print(
                        f"[Restorer] Using TensorRT sub-engines "
                        f"({_set_desc}, max_clip_length={req_mcl}, "
                        f"fp16={self.rest_fp16})"
                    )
                    return BasicVSRPPTRTClipRestorer(
                        model_path=self.rest_model,
                        device=self.device,
                        fp16=self.rest_fp16,
                        max_clip_size=req_mcl,
                    )
                elif str(self.rest_backend).lower() == "trt":
                    # User explicitly asked for TRT but no complete engine
                    # set exists for this precision. Say exactly what was
                    # looked for, where, and what IS on disk.
                    eng_dir = _basicvsrpp_sub_engine_dir(self.rest_model)
                    if basicvsrpp_engines_usable(
                        self.rest_model, not self.rest_fp16,
                    ):
                        hint = (
                            f" Engines for fp16={not self.rest_fp16} DO "
                            f"exist — match the compiled precision with "
                            f"{'--rest-fp16' if not self.rest_fp16 else '--no-rest-fp16'}, "
                            f"or recompile."
                        )
                    else:
                        hint = f" No compiled sub-engines found in {eng_dir}."
                    raise FileNotFoundError(
                        f"--rest-backend trt requested but no sub-engines found "
                        f"for fp16={self.rest_fp16} alongside {self.rest_model} "
                        f"(looked in {eng_dir})." + hint +
                        f" Compile with `ChitraMaya -compile-rest --rest-model "
                        f"{self.rest_model}` (one compile serves every Max "
                        f"Clip setting) or pass "
                        f"--rest-backend pytorch to use the PyTorch path."
                    )
                # else: backend=auto, nothing compiled → PyTorch fallback,
                # but say so loudly enough that "it's slow" is diagnosable.
                print(
                    f"[Restorer] No compiled TRT sub-engines for "
                    f"{self.rest_model} (fp16={self.rest_fp16}); "
                    f"falling back to PyTorch."
                )
            except (ImportError, ModuleNotFoundError) as e:
                if str(self.rest_backend).lower() == "trt":
                    raise
                print(f"[Restorer] TRT path unavailable ({e}); using PyTorch")

        # Batch 42: the temporal window follows the clip (lada semantics)
        # unless restoreChunkFrames caps it. Say which, loudly -- the
        # window length is the quality/VRAM trade the user is making.
        _chunk = int(self.rest_chunk_frames)
        if _chunk > 0:
            print(f"[Restorer] Using PyTorch BasicVSR++ (no sub-engines); "
                  f"temporal window CAPPED at {_chunk} frames per forward "
                  f"(restoreChunkFrames)")
        else:
            print(f"[Restorer] Using PyTorch BasicVSR++ (no sub-engines); "
                  f"temporal window = whole clip (up to Max Clip "
                  f"{self.rest_max_clip_length}; VRAM scales with it)")
        return BasicVSRPPClipRestorer(
            model_path=self.rest_model,
            device=self.device,
            fp16=self.rest_fp16,
            max_frames=_chunk,
        )

    def _stabilize_restored(self, restored):
        """CM-078: apply the temporal stabilizer to a clip's restored crops.

        A stabilization failure must never abort a run: on the first error
        we warn once, disable the stabilizer for the remainder, and return
        the frames unmodified."""
        if self._stabilizer is None:
            return restored
        try:
            return self._stabilizer.stabilize_clip(restored)
        except Exception as e:
            print(f"[TemporalFix] WARNING: stabilization failed ({e}); "
                  f"disabled for the rest of this run.")
            self._stabilizer = None
            return restored

    def run(
        self,
        *,
        detector_override=None,
        restorer_override=None,
        progress_cb=None,
        cancel_flag=None,
        foi_capture=None,
    ):
        """Run the pipeline over the configured input/output.

        All keyword arguments are optional and additive; called with none (as
        the CLI does via ``tools/process_mosaic.py``) the behavior is identical
        to before this bridge was added:

          - detector_override / restorer_override: inject pre-built (warm)
            models instead of building per-run. Used by ``MosaicPipeline`` so
            repeated UI previews don't reload models. When None, models are
            built here exactly as before.
          - progress_cb: optional callable invoked once per consumed batch with
            keyword args (frame_num, total_frames, detections, restorations,
            fps_win, fps_avg, buffered, mode). When None, nothing is emitted.
          - cancel_flag: optional threading.Event; checked at the read-loop top.
            When set, the run stops early and finalizes cleanly. When None,
            never checked.

        Returns the PipelineMetrics object (the CLI ignores the return value,
        so this is non-breaking; the UI bridge reads frame/detection counts
        off it).
        """
        # CM-107 (Batch 50): reset the salvage record FIRST. It is written
        # by the main finally below; a failure during setup (before that
        # try is even entered) must not leave a PREVIOUS run's record for
        # the server to misreport as this run's partial output.
        self.last_run_salvage = None

        # Batch 32: hold the system awake for the whole run (model builds /
        # TRT compiles included). Long unattended runs otherwise die to the
        # idle-sleep timer -- field: two Arc soak deaths at ~21:30 on
        # consecutive nights, minutes after the user's RDP disconnect
        # stopped resetting the idle timer. Thread-scoped claim; released
        # in the finally below on this same thread.
        from chitramaya.keep_awake import acquire as _ka_acquire, \
            release as _ka_release
        _ka_acquire(label=Path(self.input_path).name)
        _audio_sidecar = None   # CM-187, set once the encoder exists
        # A sidecar from a run that died before its main loop (model load
        # failure etc.) would otherwise linger next to that output.
        _prev_sc = getattr(self, "_audio_sidecar_live", None)
        if _prev_sc is not None:
            try:
                _prev_sc.cancel()
                _prev_sc.cleanup()
            except Exception:
                pass
            self._audio_sidecar_live = None

        inp = Path(self.input_path)
        out = Path(self.output_path)

        # FOI preview capture: when foi_capture is a dict with 'target_frame',
        # we snapshot that frame's boxes, the pre-composite (original) frame,
        # and the post-composite (paste-back) frame during the normal run. The
        # encode/mux still happens (to a temp output the caller discards) so the
        # detect/track/restore/composite path is byte-identical to a real run.
        _foi_target = None
        if foi_capture is not None:
            try:
                _foi_target = int(foi_capture.get("target_frame"))
            except (TypeError, ValueError):
                _foi_target = None
        out.parent.mkdir(parents=True, exist_ok=True)

        # CM-180: the run report starts NOW so the per-run console log
        # covers model builds and warnings printed before the first frame.
        # Test Frame (FOI) runs write no report.
        self._report = None
        if foi_capture is None:
            try:
                from chitramaya import __version__ as _cm_version
                from chitramaya.device import is_rocm as _is_rocm_rr
                _ed = "rocm" if _is_rocm_rr() else ("xpu" if self.device.type == "xpu"
                                                    else "cuda" if self.device.type == "cuda" else "cpu")
                self._report = RunReport(
                    mode=self.run_report_mode, output_path=str(out), input_path=str(inp),
                    panel=self.run_panel,
                    config=({k: v for k, v in self.cfg.data.items() if k != "run_panel"}
                            if isinstance(getattr(self.cfg, "data", None), dict) else None),
                    version=str(_cm_version), edition=_ed,
                    temp_dir=str(self.cfg.get("temp_dir", default="") or ""),
                )
                # CM-196 T10f: the warm build's ledger lines (detector,
                # restorer built before any run) go into this run's report.
                try:
                    for _row in list(self._vram_ledger_warm):
                        self._report.vram(str(_row.get("label")), {k: _row[k] for k in ("used","free","total","torch_allocated","torch_reserved","other") if k in _row},
                                          frame=_row.get("frame"), delta_mb=_row.get("delta_mb"))
                except Exception:
                    pass
                if self._report.mode == "off":
                    print("[Pipeline] run report: off (runReport) -- no .run.json / .log will be written")
            except Exception as _rr_e:
                print(f"[Pipeline] run report unavailable ({_rr_e}); continuing without it")
                self._report = None

        metrics = PipelineMetrics()
        metrics.det_stats.dump_rois = bool(self.det_dump_rois)
        metrics.wall_start = _dt.datetime.now()
        t0_all = time.perf_counter()

        # Batch VRAM hygiene (Batch 23c): in a warm server / folder batch the
        # PREVIOUS file's FrameStore + activations linger in PyTorch's caching
        # allocator after their tensors are freed. Torch itself could reuse
        # that cache, but NVDEC, NVENC, the Maxine secondary, and TRT contexts
        # allocate OUTSIDE torch and cannot see it -- in the field (5060 Ti
        # 8 GB, 6-file 4K batch) the hoarded cache read as "free VRAM after
        # models: ~0 MB" from file 2 onward and starved NVENC mid-run
        # (nvEncLockBitstream error 8). Hand the cache back to the driver
        # before building this file's decoder/encoder.
        if self.device.type in ("cuda", "xpu"):
            import gc as _gc
            _gc.collect()
            from chitramaya.device import empty_cache as _dev_empty_cache
            _dev_empty_cache(self.device)   # CM-093: device-generic

        self._vram_ledger = []
        self._vram_prev = None
        self._vram_stage("run start (context + warm models)")
        decoder = Decoder(
            input_path=str(inp),
            gpu_id=self.dec_gpu_id,
            batch_size=self.batch_size,
            output_format=self.dec_output_format,
            ffmpeg_input_args=self.dec_ffmpeg_input_args,
            trim_negative_pts=False,
        )
        self._vram_stage("primary decoder")

        w = int(decoder.metadata.width)
        h = int(decoder.metadata.height)
        fps = float(decoder.metadata.fps or 0.0) or 0.0
        total_frames = int(decoder.metadata.num_frames or 0)

        if self.analysis_use_synth_rois:
            print(f"[Analysis] Fixed synth ROIs enabled: {len(self.analysis_synth_rois)} boxes/frame")

        if self.sbs_enabled:
            if w < 2 or w % 2 != 0:
                print(f"[SBS] Warning: width {w} is not even; splitting by floor(w/2).")
            if self.sbs_layout not in ("lr", "rl"):
                raise ValueError(f"Invalid --sbs-layout: {self.sbs_layout!r}")
            _sbs_mode = "ON (per-eye)" if self.sbs_det_split else "OFF (whole frame)"
            print(f"[SBS] Enabled: layout={self.sbs_layout}, per-eye detection {_sbs_mode}")

        # CM-045: VR projection needs SBS geometry (an eye = half the frame).
        if self.vr_projection != "none" and not self.sbs_enabled:
            print("[VRProj] Warning: VR Projection requires Split SBS; "
                  "disabling projection for this run (enable Split SBS to use it).")
            self.vr_projection = "none"
        if self.vr_projection == "fisheye":
            from chitramaya.mosaic.vr_projection import VRProjection
            self._vrproj = VRProjection(w // 2, h, self.device)
            # Build grids + scratch canvases EAGERLY so the FrameStore's
            # VRAM-aware sizing (below) sees their allocation and budgets
            # around them instead of overcommitting.
            self._vrproj.grid_fwd()
            self._vrproj.grid_inv()
            self._vrproj._ensure_canvases()
            print(f"[VRProj] Fisheye projection ACTIVE: per-eye hequirect->fisheye "
                  f"(FOV 180, eye {w // 2}x{h}, grids ~{self._vrproj.vram_estimate_mb():.0f} MB). "
                  f"Detection/tracking/restoration run in fisheye space; restored "
                  f"regions are inverse-warped onto the original frames.")

        # CM-077: build the secondary restorer BEFORE the FrameStore sizes
        # itself, so Maxine's allocation is measured into the VRAM budget.
        # Applies only in real restoration -- pixelation (censor) and flat
        # fills (preview) must never be "enhanced".
        if self.secondary_restoration != "none" and self._secondary is None:
            from chitramaya.mosaic.restorer.rtx_secondary import scale_for_mode
            _sec_scale = scale_for_mode(self.secondary_restoration)
            if self.mode != "real":
                print("[Secondary] Note: secondary restoration applies only to real "
                      "restoration; ignored in preview/censor mode.")
            elif self.rest_clip_size != 256:
                print(f"[Secondary] WARNING: secondary restoration requires clip "
                      f"size 256 (configured: {self.rest_clip_size}); running "
                      f"without secondary.")
            elif self.secondary_restoration == "esrgan-4x":
                # Batch 68 (CM-139): Real-ESRGAN compact scaler -- plain torch,
                # so it runs on EVERY edition (CUDA, XPU, ROCm). Bundled
                # weights; a copy in ./models overrides.
                try:
                    from chitramaya.mosaic.restorer.esrgan_secondary import (
                        EsrganSecondaryRestorer,
                    )
                    self._secondary = EsrganSecondaryRestorer(
                        device=self.device, scale=_sec_scale,
                        input_size=self.rest_clip_size,
                        user_weight_dirs=["models"],
                    )
                    print(f"[Secondary] Real-ESRGAN ACTIVE: {_sec_scale}x "
                          f"(256 -> {256 * _sec_scale}, model "
                          f"realesr-general-x4v3, all GPU vendors). Restored "
                          f"regions larger than 256 px are upscaled before "
                          f"paste-back; smaller regions keep the standard path.")
                except Exception as _sec_err:
                    print(f"[Secondary] WARNING: Real-ESRGAN secondary "
                          f"unavailable ({_sec_err}); running without "
                          f"secondary.")
                    self._secondary = None
            else:
                try:
                    from chitramaya.mosaic.restorer.rtx_secondary import (
                        RtxSecondaryRestorer,
                    )
                    self._secondary = RtxSecondaryRestorer(
                        device=self.device, scale=_sec_scale,
                        input_size=self.rest_clip_size,
                        denoise=self.secondary_denoise,  # Batch 70 (CM-146)
                    )
                    print(f"[Secondary] RTX Super-Res ACTIVE: {_sec_scale}x "
                          f"(256 -> {256 * _sec_scale}, quality=high, "
                          f"denoise={self.secondary_denoise}). Restored regions "
                          f"larger than 256 px are upscaled before paste-back; smaller "
                          f"regions keep the standard path.")
                except Exception as _sec_err:
                    print(f"[Secondary] WARNING: RTX Super-Res unavailable "
                          f"({_sec_err}); running without secondary. It needs an RTX "
                          f"GPU, a recent NVIDIA driver, and the nvidia-vfx package "
                          f"(pip install nvidia-vfx).")
                    self._secondary = None
        # CM-077b: fresh stats every run. The server keeps this pipeline
        # (and its warm secondary) alive across runs and across every file
        # of a folder batch; without a reset the [SecStats] numbers would
        # accumulate over files and lie about the current one.
        if self._secondary is not None:
            from chitramaya.mosaic.restorer.rtx_secondary import SecondaryStats
            self._secondary.stats = SecondaryStats()

        # CM-078 (Batch 26): temporal stabilizer -- smooths per-frame
        # restoration/VSR shimmer across each clip's restored crops BEFORE
        # paste-back (and before the secondary, whose per-frame variance it
        # removes at the source). Real restoration only; only restored
        # pixels are ever touched. Weights degrade gracefully to a WARNING.
        self._vram_stage("secondary upscaler")
        if self.temporal_stability > 0 and self._stabilizer is None:
            if self.mode != "real":
                print("[TemporalFix] Note: temporal stabilization applies only "
                      "to real restoration; ignored in preview/censor mode.")
            else:
                from chitramaya.mosaic.restorer.temporal_stabilizer import (
                    bundled_weights_dir, load_temporal_stabilizer,
                    model_file_for_strength,
                )
                # Batch 29: bundled weights LAST so user-placed copies (next
                # to the restoration model, or in ./models) still override.
                _ts_dirs = [
                    os.path.dirname(self.rest_model) if self.rest_model else "",
                    "models",
                    bundled_weights_dir(),
                ]
                self._stabilizer, _ts_err = load_temporal_stabilizer(
                    device=self.device, strength=self.temporal_stability,
                    search_dirs=_ts_dirs, clip_size=self.rest_clip_size,
                )
                if self._stabilizer is not None:
                    print(f"[TemporalFix] Temporal stabilization ACTIVE: "
                          f"strength {self.temporal_stability} "
                          f"({model_file_for_strength(self.temporal_stability)}, "
                          f"7-frame window over restored regions only).")
                else:
                    print(f"[TemporalFix] WARNING: temporal stabilization "
                          f"unavailable ({_ts_err}); running without it. The "
                          f"weights normally ship inside ChitraMaya (Batch 29) "
                          f"-- if the bundled copy is missing, reinstall, or "
                          f"place the vs_temporalfix .pth files next to the "
                          f"restoration model or in models/.")

        # ChitraMaya's Encoder has a slimmer signature than gRestorer's.
        self._vram_stage("temporal stabilizer")
        # Convert mux_audio (str "auto/copy/aac/none") to ChitraMaya's bool flag.
        _mux_audio_bool = str(self.mux_audio).lower() != "none"
        if foi_capture is not None:
            # FOI preview: capture-only. Never encode or mux (see _DiscardEncoder).
            encoder = _DiscardEncoder()
        else:
            # CM-093 X3: NVENC on NVIDIA (unchanged path); ffmpeg backend
            # (Intel QSV / software) everywhere else.
            _EncCls = Encoder if nvenc_available() else FfmpegEncoder
            encoder = _EncCls(
                output_path=str(out),
                width=w,
                height=h,
                fps=fps,
                codec=self.enc_codec,
                preset=self.enc_preset,
                qp=self.enc_qp,
                gpu_id=self.enc_gpu_id,
                input_path=str(inp),
                mux_audio=_mux_audio_bool,
                mp4_faststart=self.mp4_faststart,
                mux_extra_args=self.mux_extra_args,
            )

            # CM-187: extract the source audio ONCE, now, in the background
            # (stream copy, low priority). The finalize muxes from this
            # sidecar instead of re-reading the whole source, and the
            # RECOVER script prefers it too -- the source drive is never
            # needed again after this point.
            if _mux_audio_bool and hasattr(encoder, "attach_audio_sidecar"):
                try:
                    from chitramaya.video.finalize import AudioSidecar
                    from chitramaya.video.encoder import _derive_ffprobe
                    _sc = AudioSidecar(
                        ffmpeg=encoder.ffmpeg_path,
                        ffprobe=_derive_ffprobe(encoder.ffmpeg_path),
                        src=str(inp), out_path=encoder.sidecar_path)
                    _sc.start()
                    encoder.attach_audio_sidecar(_sc)
                    _audio_sidecar = _sc
                    self._audio_sidecar_live = _sc
                except Exception as _sc_e:
                    print(f"[Encoder] Audio sidecar not started "
                          f"({type(_sc_e).__name__}: {_sc_e}); the finalize "
                          f"will read the source file.")

            # Wrap in AsyncEncoder if enabled — runs encode_frame() on a
            # background thread, overlapping NVENC work with the main thread's
            # decode/detect/restore/composite pass.
            if self.async_encoder:
                print(f"[Encoder] Async encoder thread enabled (queue_size={self.async_encoder_queue})")
                encoder = AsyncEncoder(encoder, device=self.device, queue_size=self.async_encoder_queue)

        self._vram_stage("encoder")
        # Warm-model injection (additive): the UI bridge passes pre-built
        # models so repeated previews don't reload. When not provided, build
        # exactly as before. analysis_use_synth_rois still forces no detector.
        if self.analysis_use_synth_rois:
            detector = None
        elif detector_override is not None:
            detector = detector_override
        else:
            detector = self._build_detector()
            self._vram_stage("detector (built in run)")
        if restorer_override is not None:
            restorer = restorer_override
        else:
            restorer = self._build_restorer()
            self._vram_stage("restorer (built in run)")

        tracker = None
        if self.mode != "none":
            tracker_cfg = TrackerConfig(
                clip_size=self.rest_clip_size,
                max_clip_length=self.rest_max_clip_length,
                pad_mode=self.rest_pad_mode,
                border_size=self.rest_border_ratio,
                debug=self.debug,
                use_seg_masks=self.use_seg_masks,
                ttl_after_end=self.trk_ttl_after_end,
                crop_quant_px=self.trk_crop_quant_px,
                crop_sticky=self.trk_crop_sticky,
                match_pad_px=self.trk_match_pad_px,
            )
            tracker = SceneTracker(cfg=tracker_cfg, seg_mask_only=True)

        # [CHANGE 2] FrameStore with backpressure.
        # Determine the requested cap first (auto / explicit / unlimited).
        if self.store_max_frames == 0:
            requested_cap = _compute_default_store_max(w, h, self.rest_max_clip_length)
            store_is_auto = True
        elif self.store_max_frames < 0:
            requested_cap = 0            # unlimited
            store_is_auto = False
        else:
            requested_cap = int(self.store_max_frames)
            store_is_auto = False

        # VRAM oversubscription check (best-effort; models are already built so
        # free VRAM reflects their contexts, and the store is still empty). For
        # an AUTO store we lower the cap toward what fits; either way we warn
        # up-front if the config is likely to page instead of failing silently.
        #
        # CM-084 (Batch 36): the store backend is resolved HERE, from the same
        # measurement. auto -> "device" whenever the requested store fits
        # (byte-identical to pre-CM-084 behavior), "host" when it would not
        # fit -- in host mode the cap is NOT reduced (system RAM holds it),
        # which is what makes MCL-180-class runs possible on 8GB cards.
        store_backend = self.store_backend
        # CM-111: runtime RAM-guard level; armed by the host-branch RAM plan
        # below, None (guard off) for device stores / when psutil is absent.
        # Reset per run -- the server reuses this Pipeline object across
        # files and a stale floor from a previous host run must not throttle
        # a device-store run.
        self._ram_floor_bytes = None

        # CM-191 (v1.71): the re-decode frame source. Resolved BEFORE the
        # store plan because when it is in use there is no store to plan:
        # no full frame is kept, so Max Clip Length no longer costs RAM or
        # VRAM, and the host-store GPU->host->GPU round trip is gone (the
        # Dell/OCuLink 4-lane case: ~5 GB/s of PCIe traffic, measured
        # 2026-09-12). The second decoder is opened HERE, before the first
        # frame is processed, so a failure to open still leaves the run on
        # the frame store with nothing lost.
        _lag: Optional[LagDecoder] = None
        _redecode_wanted = store_backend in ("auto", "redecode")
        _redecode_note = ""
        if store_backend == "auto" and decoder.backend != "nvdec":
            # ffmpeg-decode editions (AMD/Intel/CPU): a second software decode
            # is not free; the default stays the frame store until measured.
            _redecode_wanted = False
        if _redecode_wanted and self._vrproj is not None:
            _redecode_wanted = False
            _redecode_note = "VR projection composites into stored frames"
        if _redecode_wanted and foi_capture is not None:
            _redecode_wanted = False
            _redecode_note = "Test Frame runs keep the frame store"
        if _redecode_wanted and tracker is None:
            _redecode_wanted = False   # mode none: frames never enter the store
        if _redecode_wanted:
            try:
                _lag = LagDecoder(decoder, batch_size=2)   # CM-196: 4-frame queue
                self._vram_stage("lag decoder")
            except Exception as _lag_e:
                _lag = None
                print(f"[FrameStore] re-decode source unavailable "
                      f"({type(_lag_e).__name__}: {_lag_e}); using the frame "
                      f"store for this run.")
        elif store_backend == "redecode" and _redecode_note:
            print(f"[FrameStore] NOTE: {_redecode_note}; using the frame store for this run.")
        if _lag is not None:
            requested_cap = 0        # no store to plan
            print(f"[FrameStore] backend: REDECODE -- no frame store. Frames are "
                  f"decoded a second time ({_lag.backend}) when they are due "
                  f"for paste-back and encoding; Max Clip Length costs no "
                  f"VRAM, and no full frame crosses the PCIe bus. Pending "
                  f"patches wait in "
                  f"{'pinned RAM' if self.redecode_patches == 'host' else 'VRAM'} "
                  f"(redecodePatches={self.redecode_patches}).")
            # CM-103's encoder-headroom preflight still matters here (NVENC
            # surfaces live in VRAM); the store-plan block below is skipped.
            try:
                if self.device.type in ("cuda", "xpu"):
                    from chitramaya.device import empty_cache as _dev_empty_cache
                    _dev_empty_cache(self.device)
                _free_b0, _total_b0 = _vram_free_total(self.device)
                _enc_obj0 = (encoder.underlying if isinstance(encoder, AsyncEncoder) else encoder)
                if (_free_b0 is not None and type(_enc_obj0).__name__ == "Encoder"
                        and _free_b0 < max(192 * 1024 * 1024, int(w * h * 36))):
                    print(f"[Pipeline] WARNING: only ~{_free_b0 // (1024*1024)} MB VRAM "
                          f"free after models, but the NVENC encoder needs roughly "
                          f"{max(192 * 1024 * 1024, int(w * h * 36)) // (1024*1024)} MB "
                          f"at {w}x{h}. This run will likely FAIL mid-encode (NVENC error 8)."
                          f"{getattr(self, '_rest_engine_note', '') or ' Levers: a smaller restoration engine set, restore Use Tensor off, or smaller --det-imgsz.'}")
            except Exception:
                pass
        if store_backend == "host" and self._vrproj is not None:
            print("[FrameStore] NOTE: VR projection composites into device "
                  "frames; host offload disabled for this run.")
            store_backend = "device"
        if store_backend != "device" and self.device.type not in ("cuda", "xpu"):
            store_backend = "device"   # cpu device: frames are host-resident anyway
        final_cap = requested_cap
        if requested_cap > 0:
            # Batch 23c: measure AFTER releasing torch's cache, so the sizing
            # sees true availability instead of the previous file's freed-but-
            # cached frames (which read as "0 MB free" and shrank the store
            # for no reason in warm batches).
            if self.device.type in ("cuda", "xpu"):
                from chitramaya.device import empty_cache as _dev_empty_cache
                _dev_empty_cache(self.device)   # CM-093: device-generic
            _free_b, _total_b = _vram_free_total(self.device)
            if _free_b is not None:
                # Reserve = base + the async NVENC queue's real BGRA footprint
                # (the queue can hold up to async_encoder_queue full frames on
                # the GPU). Scales with resolution and the queue setting.
                _queue_frames = int(self.async_encoder_queue) if self.async_encoder else 2
                _reserve_b = (_VRAM_BASE_RESERVE_MB * 1024 * 1024) + _queue_frames * (w * h * 4)
                _plan_frames, _reduced, _warn = _vram_plan(
                    width=w, height=h, max_clip_length=self.rest_max_clip_length,
                    requested_frames=requested_cap,
                    free_bytes=_free_b, total_bytes=_total_b,
                    reserve_bytes=_reserve_b,
                )
                # CM-084: auto backend decision rides the SAME plan. _reduced
                # means "the requested store does not fit device memory" --
                # exactly the condition under which offloading beats shrinking.
                if store_backend == "auto":
                    store_backend = "host" if _reduced else "device"
                if store_backend == "host":
                    _need_mb = requested_cap * (w * h * 3) // (1024 * 1024)
                    _ram_note = ""
                    # CM-111 (Batch 50): real RAM plan instead of the old
                    # "70% of available at startup" check, which PASSED the
                    # run that died (58% of available was still 90.7% of
                    # total at peak once baseline usage and run growth were
                    # counted). See _ram_plan_host for the calibration.
                    try:
                        import psutil
                        _vm = psutil.virtual_memory()
                        _avail_b, _total_ram_b = int(_vm.available), int(_vm.total)
                        _fits, _safe_frames, _sugg_mcl, _floor_b = _ram_plan_host(
                            width=w, height=h,
                            max_clip_length=self.rest_max_clip_length,
                            requested_frames=requested_cap,
                            avail_bytes=_avail_b, total_bytes=_total_ram_b,
                        )
                        # Arm the runtime guard (checked in the decode loop).
                        self._ram_floor_bytes = _floor_b
                        _safe_mb = _safe_frames * (w * h * 3) // (1024 * 1024)
                        _ram_note = (f"; system RAM available: "
                                     f"{_avail_b // (1024*1024)} MB, safely "
                                     f"usable for the store: ~{_safe_mb} MB")
                        if not _fits:
                            _mcl_lever = (
                                f"lower Max Clip Length to ~{_sugg_mcl} or below"
                                if _sugg_mcl >= 30 else
                                "free system RAM (even MCL 30 does not fit right now)"
                            )
                            print(
                                f"[FrameStore] WARNING: RAM plan does NOT fit: "
                                f"the host store for Max Clip Length "
                                f"{int(self.rest_max_clip_length)} at {w}x{h} "
                                f"needs ~{_need_mb} MB of system RAM, but only "
                                f"~{_safe_mb} MB is safely usable "
                                f"({_avail_b // (1024*1024)} MB available minus "
                                f"~{_RAM_GROWTH_ALLOWANCE_MB} MB run growth and a "
                                f"{_floor_b // (1024*1024)} MB free-RAM floor the "
                                f"OS and NVENC pinned memory need at peak). Runs "
                                f"in this state die mid-encode with NVENC error 8 "
                                f"when RAM tops out in a dense section "
                                f"(field-traced 2026-08-19). Levers: "
                                f"{_mcl_lever}, close other applications, or "
                                f"add RAM. The RAM guard will pause decode-ahead "
                                f"when free RAM hits the floor, which slows the "
                                f"run instead of killing it -- but a fitting "
                                f"Max Clip Length is the real fix."
                            )
                    except Exception:
                        pass
                    print(f"[FrameStore] backend: HOST -- up to {requested_cap} "
                          f"frames (~{_need_mb} MB) held in system RAM instead "
                          f"of VRAM (free VRAM after models: "
                          f"~{_free_b // (1024*1024)} MB{_ram_note})")
                else:
                    if store_is_auto and _reduced:
                        final_cap = _plan_frames
                        # Honest wording: the reduction may have bottomed out at
                        # the one-clip floor, which does NOT necessarily fit free
                        # VRAM — the warning below states the real situation.
                        print(
                            f"[FrameStore] VRAM-aware: reduced auto max_frames "
                            f"{requested_cap}->{final_cap} (free VRAM after models: "
                            f"~{_free_b // (1024*1024)} MB; backpressure can still bump "
                            f"if a scene needs it)"
                        )
                    if _warn is not None:
                        print(f"[Pipeline] WARNING: {_warn}"
                              f"{getattr(self, '_rest_engine_note', '')}")
                # CM-103 (v1.50.00): encoder-headroom preflight. The host
                # store can rescue the FRAME STORE from a full card, but the
                # NVENC encoder's surfaces and bitstream buffers must live in
                # VRAM — with ~0 MB free after models, the first big drain
                # dies with nvEncLockBitstream error 8 (field-calibrated:
                # ~410-460 MB free encoded 4K fine; ~0 MB free always died).
                # Warn up front with the fix instead of failing 5 minutes in.
                try:
                    _enc_obj = (encoder.underlying
                                if isinstance(encoder, AsyncEncoder)
                                else encoder)
                    _enc_is_nvenc = type(_enc_obj).__name__ == "Encoder"
                    _enc_headroom_b = max(192 * 1024 * 1024, int(w * h * 36))
                    if _enc_is_nvenc and _free_b < _enc_headroom_b:
                        print(
                            f"[Pipeline] WARNING: only "
                            f"~{_free_b // (1024*1024)} MB VRAM free after "
                            f"models, but the NVENC encoder needs roughly "
                            f"{_enc_headroom_b // (1024*1024)} MB at "
                            f"{w}x{h}. This run will likely FAIL mid-encode "
                            f"(NVENC error 8)."
                            f"{getattr(self, '_rest_engine_note', '') or ' Levers: a smaller restoration engine set, restore Use Tensor off, or smaller --det-imgsz.'}"
                        )
                except Exception:
                    pass
        if store_backend == "auto":
            store_backend = "device"   # no measurement possible -> today's behavior
        if _lag is not None:
            store = RedecodeStore(patch_home=self.redecode_patches)    # CM-191: records PTS + patches, keeps no frame
        else:
            store = FrameStore(max_frames=final_cap, backend=store_backend)
        # CM-120: arm decoder-drop compensation. Both drain helpers pick the
        # filler up from the store, so every encode path (sync + async, all
        # backpressure/final drains) is covered from one construction point.
        # Inert when per-frame PTS are unavailable (see Decoder._frame_pts)
        # or uniform (the ffmpeg fallback synthesizes exact-CFR PTS).
        store.gap_filler = PtsGapFiller()

        # CM-180: what actually ran, for every field that can differ from
        # the panel. Recorded here because by now every stage is built.
        if self._report is not None:
            try:
                _rr = self._report
                _ak = getattr(self, "_asked", {})
                _rr.note("store_backend", _ak.get("store_backend"), store.backend,
                         "auto resolves to redecode on NVDEC; VR projection / Test Frame keep the store"
                         if _ak.get("store_backend") == "auto" else "")
                _rr.note("decoder", "auto", decoder.backend,
                         ("MPEG-TS remuxed to a CFR temp first (CM-120)"
                          if getattr(decoder, "_ts_remux_path", None) or getattr(_lag, "_dec", None) and getattr(_lag._dec, "_ts_remux_path", None)
                          else ""))
                _enc_obj = encoder.underlying if isinstance(encoder, AsyncEncoder) else encoder
                _rr.note("encoder", str(self.enc_codec),
                         f"{type(_enc_obj).__name__}:{getattr(_enc_obj, 'codec', self.enc_codec)}"
                         f" preset={getattr(_enc_obj, 'preset', '')} qp={getattr(_enc_obj, 'qp', '')}"
                         f" async={isinstance(encoder, AsyncEncoder)}", "")
                _rr.note("det_fp16", _ak.get("det_fp16"), bool(self.det_fp16),
                         "forced off on the ROCm edition (CM-169)"
                         if _ak.get("det_fp16") and not self.det_fp16 else "")
                _rr.note("rest_fp16", _ak.get("rest_fp16"), bool(self.rest_fp16), "")
                _rr.note("detector", "tensorrt" if self.cfg.get("detection", "trt", default=None) else "auto",
                         type(detector).__name__ if detector is not None else "none", "")
                _rr.note("restorer", _ak.get("rest_backend"),
                         type(restorer).__name__ if restorer is not None else "none",
                         getattr(self, "_rest_engine_note", "") or "")
                _sec_ran = ("none" if self._secondary is None
                            else f"{type(self._secondary).__name__}:{getattr(self._secondary, 'scale', '')}x")
                _rr.note("secondary", _ak.get("secondary"), _sec_ran,
                         "requested secondary unavailable on this GPU/edition; ran without one (CM-184 pending)"
                         if (_ak.get("secondary") not in ("none", "", None) and self._secondary is None) else "")
                _rr.note("secondary_denoise", _ak.get("secondary_denoise"),
                         str(getattr(self, "secondary_denoise", "none")), "")
                _rr.note("blendmask", _ak.get("blendmask"), str(self.rest_blendmask), "")
                _rr.note("feather_radius", _ak.get("feather_radius"), int(self.feather_radius),
                         "0 = auto (derived from crop size) with the facefusion mask" if self.feather_radius == 0 else "")
                _rr.note("max_clip_length", _ak.get("max_clip_length"), int(self.rest_max_clip_length), "")
                _rr.note("temporal_stability", int(self.cfg.get("temporal_stability", default=0) or 0),
                         int(self.cfg.get("temporal_stability", default=0) or 0), "")
                _rr.note("watchdog_stall_seconds", None,
                         float(self.cfg.get("monitoring", "watchdog_stall_seconds", default=0) or 0) or None,
                         "from ChitraMaya-config.json when set; None = built-in default")
                _rr.note("keep_awake", True, True, "released after finalize (CM-179)")
                _rr.note("frame_store_max_frames", int(self.store_max_frames),
                         int(store.max_frames),
                         "0 = no store on the redecode path" if store.backend == "redecode" else "")
                # CM-196: the panel copy is gathered before the CM-148 gate can
                # snap the Image Size dial to the engine's compiled size; the
                # UI now patches the panel too, but say it here regardless.
                _panel_imgsz = None
                try:
                    if isinstance(self.run_panel, dict) and self.run_panel.get("ctrlMosaicDetImgsz") not in (None, ""):
                        _panel_imgsz = int(self.run_panel.get("ctrlMosaicDetImgsz"))
                except Exception:
                    _panel_imgsz = None
                _rr.note("det_imgsz", _panel_imgsz if _panel_imgsz is not None else _ak.get("det_imgsz"),
                         int(self.det_imgsz),
                         "ran at the TensorRT engine's compiled size (CM-148 'Use engine size')"
                         if (_panel_imgsz is not None and _panel_imgsz != int(self.det_imgsz)) else "")
                _rr.note("run_files", str(self.cfg.get("runReport", default="beside") or "beside"),
                         str(getattr(self._report, "mode", "beside")),
                         "the .run.json / .log / .timecodes.txt set: beside the video | Temp folder | off "
                         "(panel 'Run Files' wins over the runReport key; CM-202)")
                _rr.note("vram_cache_release", bool(self.vram_cache_release), bool(self.vram_cache_release),
                         "torch cache handed back to the driver after drains; OFF by default since T10f (CM-196)")
                _rr.note("redecode_patches", _ak.get("redecode_patches"),
                         getattr(store, "patch_home", None) if store.backend == "redecode" else None,
                         "pending patches wait in pinned RAM (host) or VRAM (device); redecode path only (CM-196)")
            except Exception as _rr_e:
                print(f"[Pipeline] run report: effective-values note failed ({_rr_e})")

        # CM-196 VRAM ledger, line 1: every stage is built, no frame yet.
        # "other" is what torch cannot see (TensorRT engines, RTX SS, the
        # two NVDEC sessions, NVENC, the driver) -- on an 8 GB card this is
        # the number that decides whether the run survives its largest
        # regions, and until now it was a guess.
        try:
            _snap0 = vram_snapshot(self.device)
            if _snap0:
                print(format_vram("after models", _snap0))
                if self._report is not None:
                    self._report.vram("after_models", _snap0, frame=0)
        except Exception:
            pass

        # CM-084: NVENC consumes device tensors, so a host-backed store pays
        # one H2D upload per frame at drain time. The ffmpeg encoder path
        # takes CPU frames as-is (it pipes CPU bytes into ffmpeg anyway --
        # host mode actually REMOVES its per-frame download). FOI runs use
        # a discard encoder: nothing to upload.
        store_upload_device = (
            self.device
            if (store.backend == "host" and nvenc_available()
                and foi_capture is None)
            else None
        )

        if store.max_frames > 0:
            est_mb = (store.max_frames * w * h * 3) / (1024.0 * 1024.0)
            # v1.50.00: say where the budget actually lives -- a host-backed
            # store holds frames in system RAM, and calling that "VRAM
            # budget" confused the very run it was designed to enable.
            _budget_kind = "RAM" if store.backend == "host" else "VRAM"
            print(f"[FrameStore] max_frames={store.max_frames} "
                  f"(~{est_mb:.0f} MB {_budget_kind} budget)")
        elif _lag is None:
            print("[FrameStore] max_frames=unlimited")

        # CM-111 (Batch 50): runtime RAM guard for HOST-backed stores. The
        # startup plan predicts; this enforces. Checked alongside is_full()
        # in the decode loops: when physical free RAM drops to the floor,
        # decode-ahead pauses and the loop drains instead -- converting the
        # Idol failure mode (RAM tops out -> nvEncLockBitstream error 8)
        # into a slower-but-alive run. Sampled at most every 2s (psutil
        # virtual_memory is cheap but not free at 70 it/s). NEVER allowed
        # to deadlock: when nothing is drainable (an open clip needs MORE
        # decode to close), the loop proceeds -- see the drain sites.
        _ram_guard_floor = (self._ram_floor_bytes
                            if store.backend == "host" else None)
        _ram_guard = {"next": 0.0, "low": False, "warned": 0.0, "psutil": None}
        if _ram_guard_floor is not None:
            try:
                import psutil as _ram_psutil
                _ram_guard["psutil"] = _ram_psutil
            except Exception:
                _ram_guard_floor = None

        def _ram_low() -> bool:
            if _ram_guard_floor is None:
                return False
            _now = time.monotonic()
            if _now < _ram_guard["next"]:
                return _ram_guard["low"]
            _ram_guard["next"] = _now + 2.0
            try:
                _avail = int(_ram_guard["psutil"].virtual_memory().available)
            except Exception:
                return False
            _low = _avail < _ram_guard_floor
            if _low and not _ram_guard["low"] and (_now - _ram_guard["warned"]) > 30.0:
                _ram_guard["warned"] = _now
                print(f"[FrameStore] RAM guard: {_avail // (1024*1024)} MB free "
                      f"is below the {_ram_guard_floor // (1024*1024)} MB floor "
                      f"-- pausing decode-ahead while the encoder drains "
                      f"(CM-111). Throughput will dip; the run survives.")
            _ram_guard["low"] = _low
            return _low

        # Echo the effective batch size (decode + detection). Surfaced so the
        # "Detection Batch" control / --batch-size / --det-batch-size is
        # visible in the log — previously there was no way to confirm it took.
        print(f"[Pipeline] batch_size={self.batch_size}")

        # [CHANGE 4] PTS log: collects (frame_num, pts_ns) for all encoded frames
        pts_log: List[Tuple[int, Optional[int]]] = []

        pbar_total = (self.max_frames if self.max_frames is not None else (total_frames if total_frames > 0 else None))
        pbar = tqdm(total=pbar_total, disable=self.debug)

        frame_num = 0

        # Prefetch threading is only safe/valuable for ffmpeg-cpu lane.
        # NVDEC/PyNvVideoCodec decode is not reliably thread-safe across threads.
        use_thread_prefetch = getattr(decoder, "_ffmpeg_proc", None) is not None

        stop = _threading.Event()
        prod_exc: dict[str, BaseException] = {}
        prod: Optional[_threading.Thread] = None
        q: Optional[_queue.Queue[Optional[List[object]]]] = None

        # [CHANGE 4] batch-level PTS storage: populated by read_batch_with_pts
        batch_pts_cache: List[Optional[int]] = []

        # CM-093 X5: periodic VRAM telemetry on non-NVIDIA accelerators.
        # nvGPUMonitor is NVML-based and cannot see an Intel Arc, so a
        # multi-day xpu soak would otherwise run blind -- and the one Arc
        # crash so far (UR_RESULT_ERROR_UNKNOWN, day-two-class endurance
        # question) needs a memory trace to be diagnosable. Every ~5 min:
        # "[Pipeline] xpu VRAM: free N MB / total M MB". Memory pressure
        # shows as free walking down for hours before death; a driver
        # reset shows healthy free right up to the end. Silent on CUDA
        # (nvGPUMonitor owns that) and on backends with no free-memory API.
        _mem_beat_last = [0.0]

        def _mem_beat(period_s: float = 300.0) -> None:
            if self.device.type != "xpu":
                return
            now = time.perf_counter()
            if now - _mem_beat_last[0] < period_s:
                return
            _mem_beat_last[0] = now
            try:
                from chitramaya.device import mem_get_info as _mgi
                free_b, total_b = _mgi(self.device)
                if free_b is not None and total_b:
                    print(f"[Pipeline] xpu VRAM: free {free_b // (1 << 20)} MB "
                          f"/ total {total_b // (1 << 20)} MB "
                          f"(frame {frame_num})", flush=True)
                elif total_b:
                    print(f"[Pipeline] xpu VRAM: free n/a / total "
                          f"{total_b // (1 << 20)} MB (this torch build has "
                          f"no mem_get_info)", flush=True)
            except Exception:
                pass

        def _item_to_rgb(item: object) -> torch.Tensor:
            """One decoder item (NVDEC surface or CPU tensor) -> RGB HWC u8 on
            the pipeline device. Shared by consume_batch and, under CM-191,
            by the re-decode drain, so both decoders' frames take one path."""
            # CPU lane returns torch.Tensor (either NV12 2D or RGB HWC 3D)
            if isinstance(item, torch.Tensor):
                t_cpu = item

                # NV12 heuristic: [H*3/2, W] uint8
                is_nv12 = (
                    t_cpu.ndim == 2
                    and t_cpu.dtype == torch.uint8
                    and int(t_cpu.shape[0]) == (h * 3 // 2)
                    and int(t_cpu.shape[1]) == w
                )

                if is_nv12:
                    # Upload NV12 then CSC on device
                    if self.device.type != "cpu":
                        t0_up = time.perf_counter()
                        nv12_dev = t_cpu.to(self.device, non_blocking=True)
                        metrics.t_upload += (time.perf_counter() - t0_up)
                    else:
                        nv12_dev = t_cpu

                    t0_csc = time.perf_counter()
                    rgb = nv12_to_rgb_hwc_u8(nv12_dev, width=w, height=h)
                    metrics.t_csc += (time.perf_counter() - t0_csc)
                else:
                    # Assume RGB HWC u8
                    rgb = t_cpu
                    if self.device.type != "cpu":
                        t0_up = time.perf_counter()
                        rgb = rgb.to(self.device, non_blocking=True)
                        metrics.t_upload += (time.perf_counter() - t0_up)

                return rgb.contiguous()

            # NVDEC lane returns a PyNvVideoCodec surface (dlpack)
            t = wrap_surface_as_tensor(item)
            # t is usually RGBP CHW u8 on GPU; convert to RGB HWC u8
            if t.ndim == 3 and t.shape[-1] == 3:
                rgb = t
            else:
                rgb = rgbp_chw_to_rgb_hwc_u8(t)

            # Ensure on pipeline device (normally already correct for cuda)
            if self.device.type != "cpu" and rgb.device != self.device:
                rgb = rgb.to(self.device, non_blocking=True)

            return rgb.contiguous()

        def _item_to_bgr(item: object) -> torch.Tensor:
            """CM-191: the re-decoded frame, as the store would have held it
            (BGR HWC u8 on the pipeline device, own memory)."""
            # flip() copies, so the blend's in-place writes never touch the
            # decoder's surface -- the same tensor the store used to hold.
            return rgb_hwc_to_bgr_hwc_u8(_item_to_rgb(item))

        _ledger = {"first_drain": False, "last_release": 0.0}

        def _after_drain(n_done: int) -> None:
            """CM-196: (1) the ledger line after the first drain -- the point
            where every stage incl. NVENC, the lag decoder and the secondary
            has run once; (2) hand torch's idle cache back to the driver now
            and then, so a burst (a 2400-px clip's resize/blend temporaries)
            does not stay reserved for the rest of the run while NVENC and
            the decoders fight the WDDM pager for the same 8 GB."""
            try:
                if not _ledger["first_drain"]:
                    _ledger["first_drain"] = True
                    _snap = vram_snapshot(self.device)
                    if _snap:
                        print(format_vram(f"after first drain ({n_done} frames encoded)", _snap))
                        if self._report is not None:
                            self._report.vram("after_first_drain", _snap, frame=int(metrics.processed_frames))
                if self.device.type == "cuda" and self.vram_cache_release:
                    _now = time.perf_counter()
                    if _now - _ledger["last_release"] >= 30.0:
                        _idx = self.device.index if self.device.index is not None else 0
                        _idle = int(torch.cuda.memory_reserved(_idx)) - int(torch.cuda.memory_allocated(_idx))
                        if _idle > 256 * 1024 * 1024:
                            torch.cuda.empty_cache()
                            _ledger["last_release"] = _now
            except Exception:
                pass

        def _drain(safe_before: int) -> int:
            """Encode every frame below the tracker watermark, from the store
            (legacy) or by re-decoding it (CM-191)."""
            if _lag is not None:
                n_done = drain_plan_to_encoder(
                    store=store,
                    lag=_lag,
                    to_bgr=_item_to_bgr,
                    safe_before=int(safe_before),
                    encoder=encoder,
                    device=self.device,
                    sync_before_encode=self.enc_sync_before_encode,
                    pts_log=pts_log,
                    foi_target=_foi_target,
                    foi_capture=foi_capture,
                )
                if n_done > 0:
                    _after_drain(n_done)
                return n_done
            return drain_store_to_encoder(
                store=store,
                upload_device=store_upload_device,  # CM-084
                safe_before=int(safe_before),
                encoder=encoder,
                device=self.device,
                sync_before_encode=self.enc_sync_before_encode,
                pts_log=pts_log,
            )

        def _guard_restored(clip, restored: List[torch.Tensor]) -> List[Optional[torch.Tensor]]:
            """CM-186: a restorer that returns an all-zero frame for a source
            crop that is not black has failed (NaN/garbage -> 0 at uint8; the
            black rectangles of the 09-11 AMD overnight runs). Never paste
            that: drop the frame's restoration (the source crop stays) and
            record where it happened, so a recurrence carries a frame number
            and a wall-clock time instead of a mystery."""
            out: List[Optional[torch.Tensor]] = list(restored)
            n = min(len(out), len(clip.frame_nums), len(clip.frames))
            hits: List[int] = []
            for i in range(n):
                r = out[i]
                if r is None or r.numel() == 0:
                    continue
                try:
                    if int(r.max()) != 0:
                        continue
                    if int(clip.frames[i].max()) == 0:
                        continue     # black source -> black output is right
                except Exception:
                    continue
                out[i] = None
                hits.append(int(clip.frame_nums[i]))
            if hits:
                metrics.guard_black_frames.extend(hits)
                print(f"[Restorer] WARNING (CM-186 guard): the restorer returned an "
                      f"all-black result for {len(hits)} frame(s) "
                      f"({hits[0]}..{hits[-1]}) of a non-black region at "
                      f"{_dt.datetime.now().strftime('%H:%M:%S')}; those frames keep "
                      f"their source pixels. If this repeats, note the time: it "
                      f"points at a GPU/driver event, not at the settings.")
            return out

        _region_stats = RegionStats()

        def _composite_closed_clip(clip, restored: List[Optional[torch.Tensor]]) -> None:
            """Paste-back for one closed clip: into the store (legacy / VR) or
            into the CM-191 patch plan."""
            try:
                for _i, _shp in enumerate(getattr(clip, "crop_shapes", []) or []):
                    _region_stats.note(clip.frame_nums[_i] if _i < len(clip.frame_nums) else None, _shp)
            except Exception:
                pass
            if _lag is not None:
                store.add_clip(
                    clip, restored,
                    model_dtype=restorer.model_dtype,
                    blendmask=self.rest_blendmask,
                    feather_radius=self.feather_radius,
                    secondary=self._secondary,
                )
            elif self._vrproj is not None:
                composite_clip_into_store_projected(
                    clip=clip,
                    restored_frames_u8=restored,
                    store_bgr_u8=store.frames_bgr_u8,
                    vrproj=self._vrproj,
                    model_dtype=restorer.model_dtype,
                    blendmask=self.rest_blendmask,
                    feather_radius=self.feather_radius,
                    secondary=self._secondary,
                )
            else:
                composite_clip_into_store(
                    clip=clip,
                    restored_frames_u8=restored,
                    store_bgr_u8=store.frames_bgr_u8,
                    model_dtype=restorer.model_dtype,
                    blendmask=self.rest_blendmask,
                    feather_radius=self.feather_radius,
                    secondary=self._secondary,
                )

        def consume_batch(batch: List[object], batch_pts: Optional[List[Optional[int]]] = None) -> None:
            nonlocal frame_num
            _mem_beat()

            # If we're stopping early, trim batch to remaining frames.
            if self.max_frames is not None:
                remaining = self.max_frames - frame_num
                if remaining <= 0:
                    return
                if len(batch) > remaining:
                    batch = batch[:remaining]
                    if batch_pts is not None:
                        batch_pts = batch_pts[:remaining]

            # [CHANGE 4] Ensure batch_pts list is correctly sized
            if batch_pts is None:
                batch_pts = [None] * len(batch)
            while len(batch_pts) < len(batch):
                batch_pts.append(None)

            # -----------------------------
            # Prepare: surface/tensor -> RGB HWC u8 on pipeline device
            # + NV12 CPU lane support
            # -----------------------------
            t0_prep = time.perf_counter()
            batch_rgb: List[torch.Tensor] = [_item_to_rgb(item) for item in batch]
            metrics.t_prepare += (time.perf_counter() - t0_prep)

            # Convert RGB -> BGR uint8 once per frame (LADA parity + reuse everywhere)
            batch_bgr_u8: List[torch.Tensor] = [rgb_hwc_to_bgr_hwc_u8(rgb) for rgb in batch_rgb]

            # CM-045: with VR projection active, analysis (detection/tracking/
            # restoration crops) runs on the fisheye-warped frames while the
            # store keeps the ORIGINAL frames — restored regions are
            # inverse-warped back at composite time.
            if self._vrproj is not None:
                t0_warp = time.perf_counter()
                batch_analysis_u8: List[torch.Tensor] = [
                    self._vrproj.warp_frame_to_fisheye(bgr) for bgr in batch_bgr_u8
                ]
                metrics.t_prepare += (time.perf_counter() - t0_warp)
            else:
                batch_analysis_u8 = batch_bgr_u8

            # -----------------------------
            # Detect (optional)
            # -----------------------------
            detections: List[Detection] = []
            if self.analysis_use_synth_rois:
                detections = [Detection(boxes=None, scores=None, classes=None, masks=None) for _ in batch_rgb]
            elif detector is not None:
                if self.sbs_enabled and self.sbs_det_split:
                    if not getattr(self, "_sbs_split_logged", False):
                        print(f"[SBS] Per-eye detection path ACTIVE: splitting {w}x{h} into L|R halves "
                              f"(layout={self.sbs_layout}), detecting each half, merging boxes/masks.")
                        self._sbs_split_logged = True
                    left_frames: List[torch.Tensor] = []
                    right_frames: List[torch.Tensor] = []
                    half_w = w // 2
                    for bgr in batch_analysis_u8:
                        l, r = split_frame_lr(bgr, layout=self.sbs_layout)
                        left_frames.append(l.contiguous())
                        right_frames.append(r.contiguous())

                    t0 = time.perf_counter()
                    det_l = detector.detect_batch(left_frames)
                    det_r = detector.detect_batch(right_frames)
                    metrics.t_det += (time.perf_counter() - t0)

                    for dl, dr in zip(det_l, det_r):
                        boxes_l = _tensor_boxes_to_list_xyxy(dl.boxes, w=half_w, h=h)
                        boxes_r = _tensor_boxes_to_list_xyxy(dr.boxes, w=half_w, h=h)
                        masks_l = _extract_masks_list(dl) if self.use_seg_masks else None
                        masks_r = _extract_masks_list(dr) if self.use_seg_masks else None

                        merged_boxes = unsplit_boxes_layout(boxes_l, boxes_r, half_w=half_w, layout=self.sbs_layout)
                        merged_masks = unsplit_masks_layout(masks_l, masks_r, full_w=w, half_w=half_w, layout=self.sbs_layout)

                        det = Detection(
                            boxes=torch.tensor([[b[1], b[0], b[3], b[2]] for b in merged_boxes], dtype=torch.float32, device="cpu") if merged_boxes else None,
                            scores=None,
                            classes=None,
                            masks=None,
                        )

                        if merged_masks is not None:
                            try:
                                mm = [m for m in merged_masks if m is not None]
                                if len(mm) == len(merged_masks) and len(mm) > 0:
                                    det.masks = torch.stack(mm, dim=0)
                            except Exception:
                                det.masks = None

                        detections.append(det)
                else:
                    t0 = time.perf_counter()
                    detections = detector.detect_batch(batch_analysis_u8)
                    metrics.t_det += (time.perf_counter() - t0)
            else:
                detections = [Detection(boxes=None, scores=None, classes=None, masks=None) for _ in batch_rgb]

            # -----------------------------
            # Consumer: track/restore/composite/encode
            # Drain encode ONCE per batch.
            # -----------------------------
            if tracker is None or self.mode == "none":
                # No tracker: encode directly, but sync once per batch (not per frame).
                if self.enc_sync_before_encode:
                    sync_device(self.device)
                t0 = time.perf_counter()
                for i, bgr_u8 in enumerate(batch_bgr_u8):
                    # [CHANGE 4] track PTS for passthrough frames too
                    frame_pts = batch_pts[i] if i < len(batch_pts) else None
                    pts_log.append((frame_num, frame_pts))
                    encoder.encode_frame(bgr_u8_to_bgra_u8(bgr_u8))
                    frame_num += 1
                    metrics.processed_frames += 1
                    pbar.update(1)
                metrics.t_encode += (time.perf_counter() - t0)
                return

            safe_before_batch: int = frame_num
            for i, bgr_u8 in enumerate(batch_bgr_u8):
                if self.max_frames is not None and frame_num >= self.max_frames:
                    break

                det = detections[i] if i < len(detections) else Detection(boxes=None, scores=None, classes=None, masks=None)

                if self.analysis_use_synth_rois and self.analysis_synth_rois:
                    boxes = [clip_box_to_bounds(bx, w=w, h=h) for bx in self.analysis_synth_rois]
                    masks_list = None
                else:
                    boxes = _tensor_boxes_to_list_xyxy(det.boxes, w=w, h=h)
                    masks_list = _extract_masks_list(det) if (self.use_seg_masks and det.masks is not None) else None

                    if self.roi_dilate > 0 and boxes:
                        dil = self.roi_dilate
                        boxes = [(t - dil, l - dil, b + dil, r + dil) for (t, l, b, r) in boxes]

                    if boxes:
                        boxes = [clip_box_to_bounds(bx, w=w, h=h) for bx in boxes]

                if self.sbs_enabled and boxes:
                    seam_x = w // 2
                    boxes, masks_list = seam_split_boxes(boxes, seam_x=seam_x, full_w=w, full_h=h, masks=masks_list)

                metrics.det_stats.add(boxes, w=w, h=h, frame_num=frame_num)

                # FOI: snapshot the target frame's finalized boxes/masks.
                if _foi_target is not None and frame_num == _foi_target:
                    foi_capture["boxes"] = list(boxes)
                    foi_capture["masks"] = masks_list
                    foi_capture["frame_w"] = int(w)
                    foi_capture["frame_h"] = int(h)

                # [CHANGE 4] Store frame with PTS
                frame_pts = batch_pts[i] if i < len(batch_pts) else None
                store.put(frame_num, bgr_u8, pts=frame_pts)

                # FOI: snapshot the ORIGINAL (pre-composite) target frame. The
                # compositor mutates store frames in place, so clone now.
                if _foi_target is not None and frame_num == _foi_target:
                    foi_capture["original"] = bgr_u8.detach().clone()
                    # CM-045: also snapshot the fisheye analysis frame so the
                    # detection overlay pane can render boxes/masks in the
                    # space they were detected in.
                    if self._vrproj is not None:
                        foi_capture["analysis"] = (
                            batch_analysis_u8[i].detach().clone()
                        )

                # CM-045: with projection active, the tracker (and therefore
                # clip crops + gap-fill buffer) consumes the fisheye frame;
                # boxes/masks are already in fisheye coords.
                analysis_u8 = batch_analysis_u8[i]

                t0 = time.perf_counter()
                step = tracker.step_frame(frame_num, analysis_u8, boxes, masks_list)
                metrics.t_track += (time.perf_counter() - t0)

                if step.new_clips and restorer is not None:
                    for clip in step.new_clips:
                        t0 = time.perf_counter()
                        restored = restorer.restore_clip(clip)
                        restored = self._stabilize_restored(restored)  # CM-078
                        restored = _guard_restored(clip, restored)     # CM-186
                        _composite_closed_clip(clip, restored)         # store or CM-191 plan
                        metrics.frames_restored.update(int(fn) for fn in clip.frame_nums)
                        metrics.clip_lengths.append(int(len(clip.frame_nums)))
                        metrics.t_restore += (time.perf_counter() - t0)

                        # FOI: snapshot the target frame AFTER paste-back. If a
                        # later overlapping clip touches it again, this updates
                        # to the last composite (correct final state).
                        if (_foi_target is not None
                                and int(_foi_target) in clip.frame_nums
                                and int(_foi_target) in store.frames_bgr_u8):
                            foi_capture["composited"] = (
                                store.frames_bgr_u8[int(_foi_target)].detach().clone()
                            )

                # safe_before = min(tracker watermark, frame_num+1)
                min_start = tracker.min_active_start()
                tracker_safe = int(min_start) if min_start is not None else int(frame_num + 1)
                safe_before = min(tracker_safe, int(frame_num + 1))
                safe_before_batch = safe_before

                if (len(boxes) == 0) and (min_start is None) and (not step.new_clips):
                    metrics.early_passthrough_frames += 1
                    metrics.frames_legit_passthrough.add(int(frame_num))

                frame_num += 1
                metrics.processed_frames += 1
                pbar.update(1)

            # Drain once per batch (sync once per drain happens inside the drain helper)
            t0 = time.perf_counter()
            _drain(int(safe_before_batch))
            metrics.t_encode += (time.perf_counter() - t0)

            # T9b: honest console checkpoint. tqdm's it/s is the INSTANTANEOUS
            # rate between clip flushes; on a full title it read "29 it/s,
            # ETA 5h" while the true average was 14 fps and 22 h (9060 XT,
            # 2026-09-08). The UI modal has had completed-frame fps/ETA since
            # CM-135; the console log never did. Every 500 completed frames
            # print the average over completed frames and the ETA from it --
            # this line also survives into the log file, which tqdm's
            # in-place redraws do not.
            try:
                _ck_done = min(len(metrics.frames_restored)
                               + int(metrics.early_passthrough_frames),
                               int(metrics.processed_frames))
                _ck_last = int(getattr(self, "_ck_last_done", 0))
                if _ck_done // 500 > _ck_last // 500:
                    self._ck_last_done = _ck_done
                    _ck_el = time.perf_counter() - t0_all
                    _ck_fps = (_ck_done / _ck_el) if _ck_el > 1e-6 else 0.0
                    _ck_tot = int(total_frames) if total_frames and total_frames > 0 else 0
                    if _ck_tot > 0 and _ck_fps > 1e-6:
                        _ck_eta = (_ck_tot - _ck_done) / _ck_fps
                        _ck_line = (f"[Pipeline] frame {_ck_done}/{_ck_tot}  "
                                    f"avg {_ck_fps:.1f} fps  elapsed {_fmt_hms(_ck_el)}  "
                                    f"ETA {_fmt_hms(_ck_eta)}")
                    else:
                        _ck_line = (f"[Pipeline] frame {_ck_done}  avg {_ck_fps:.1f} fps  "
                                    f"elapsed {_fmt_hms(_ck_el)}")
                    # CM-196: the driver's VRAM reading rides on every
                    # checkpoint (console + report column 3) -- one driver
                    # call per 500 frames, and the run's own memory timeline.
                    _ck_vram = None
                    try:
                        if self.device.type == "cuda":
                            _cf, _ct = torch.cuda.mem_get_info(
                                self.device.index if self.device.index is not None else 0)
                            _ck_vram = int((_ct - _cf) // (1024 * 1024))
                            _ck_line += f"  vram {_ck_vram} MB"
                    except Exception:
                        _ck_vram = None
                    try:
                        pbar.write(_ck_line)
                    except Exception:
                        print(_ck_line, flush=True)
                    if self._report is not None:
                        self._report.checkpoint(_ck_done, _ck_el, vram_used_mb=_ck_vram)
            except Exception:
                pass

            # Progress emit (additive): once per batch, only if a callback was
            # provided. Reads existing metrics/state; does not affect behavior.
            if progress_cb is not None:
                _now = time.perf_counter()
                _pf = int(metrics.processed_frames)
                # CM-135 (Batch 69): fps and ETA are computed over COMPLETED
                # frames -- restored (clip closed + composited) plus legit
                # passthrough -- not over ingested frames. The old counter
                # measured the front of the pipeline (decode+detect racing
                # ahead into the store), which on short clips reads near
                # DOUBLE the delivered speed while restoration lags behind
                # (field: the Codeberg "fake fps" report -- he was right).
                # Frames sitting in open clips are backlog, not progress.
                _done = (len(metrics.frames_restored)
                         + int(metrics.early_passthrough_frames))
                _done = min(_done, _pf)  # safety: never report ahead of ingest
                # Batch 77: windowed fps is measured PER DELIVERY CYCLE, not
                # per emit. The chunked TRT path delivers in bursts (a full
                # MCL clip lands at once), so per-emit windows read 0.0 for
                # ~seconds then a huge spike for one poll -- the user only
                # ever sees the zero (field request 2026-09-01). Instead:
                # when new completions land, compute delta/dt since the LAST
                # delivery and hold that value between deliveries. The held
                # value is the true recent delivered throughput (~steady at
                # ~avg for long clips). fps_stale_s says how old the held
                # value is -- the UI marks it as refreshing when it ages.
                _prev_f = getattr(self, "_pcb_prev_frames", 0)
                _prev_t = getattr(self, "_pcb_prev_time", t0_all)
                if _done > _prev_f:
                    _dt_win = _now - _prev_t
                    _fps_win = ((_done - _prev_f) / _dt_win) if _dt_win > 1e-6 else 0.0
                    self._pcb_last_fps_win = _fps_win
                    self._pcb_prev_frames = _done
                    self._pcb_prev_time = _now
                else:
                    _fps_win = float(getattr(self, "_pcb_last_fps_win", 0.0))
                _fps_stale_s = _now - float(getattr(self, "_pcb_prev_time", t0_all))
                _dt_all = _now - t0_all
                _fps_avg = (_done / _dt_all) if _dt_all > 1e-6 else 0.0
                # CM-135: finalize (video-container pass + remux) estimate for
                # the ETA. Field data: SONE-174 (2h41m source) finalized in
                # ~125s; short clips in ~5-10s -- ~1.3% of source duration
                # with a 10s floor models both.
                _fin_est = max(10.0, 0.013 * (float(total_frames) / max(fps, 1e-6)))
                _mode = "detect-only" if (restorer is None and not self.analysis_use_synth_rois) else "restore"
                try:
                    progress_cb(
                        frame_num=_pf,
                        total_frames=int(total_frames),
                        detections=int(metrics.det_stats.frames_with_det),
                        restorations=int(len(metrics.frames_restored)),
                        fps_win=float(_fps_win),
                        fps_avg=float(_fps_avg),
                        buffered=int(len(store.frames_bgr_u8)),
                        mode=str(_mode),
                        completed=int(_done),          # CM-135
                        finalize_est_s=float(_fin_est),  # CM-135
                        fps_stale_s=float(_fps_stale_s),  # Batch 77
                    )
                except TypeError:
                    # Older callback signature (no CM-135 kwargs) -- emit the
                    # legacy shape so an out-of-step server still gets updates.
                    try:
                        progress_cb(
                            frame_num=_pf,
                            total_frames=int(total_frames),
                            detections=int(metrics.det_stats.frames_with_det),
                            restorations=int(len(metrics.frames_restored)),
                            fps_win=float(_fps_win),
                            fps_avg=float(_fps_avg),
                            buffered=int(len(store.frames_bgr_u8)),
                            mode=str(_mode),
                        )
                    except Exception:
                        pass
                except Exception:
                    # A misbehaving UI callback must never crash the pipeline.
                    pass

        # CM-081 (Batch 23): stall watchdog + PCIe link canary. Armed only
        # around the main loop -- model builds / TRT compiles (which can
        # legitimately take minutes) happen before this point, and the EOF
        # flush of a final long clip happens after stop(). stall_seconds<=0
        # disables it (config: monitoring.watchdog_stall_seconds).
        from chitramaya.mosaic.watchdog import StallWatchdog
        from chitramaya.device import is_rocm as _is_rocm_wd
        # CM-168 (T9b): on ROCm the first clip of a run legitimately sits for
        # minutes while MIOpen compiles kernels for shapes it has not seen
        # (first run after install/update; first time a clip size or model is
        # used). A 120 s threshold dumped stacks at every such compile and read
        # as a hang to users. Default 300 s there; the config key still wins.
        _wd_default = 300 if _is_rocm_wd() else 120
        _wd_stall = float(self.cfg.get("monitoring", "watchdog_stall_seconds",
                                       default=_wd_default))
        if _is_rocm_wd():
            print("[ROCm] First run after install or update compiles GPU kernels: the first "
                  "clip can take several minutes with no visible progress (a stack dump from "
                  "the watchdog during that wait is a diagnosis, not a crash). Later runs, "
                  "and later clips of the same size, start immediately.")
        _watchdog = StallWatchdog(
            lambda: int(metrics.processed_frames),
            stall_seconds=_wd_stall,
            gpu_index=self.dec_gpu_id,
            label=str(Path(self.input_path).name),
        )
        _watchdog.start()

        try:
            if use_thread_prefetch:
                # -----------------------------
                # 2-batch producer/consumer (ffmpeg-cpu only)
                # -----------------------------
                q = _queue.Queue(maxsize=2)

                def _q_put(item: Optional[List[object]]) -> bool:
                    assert q is not None
                    while True:
                        if stop.is_set():
                            return False
                        try:
                            q.put(item, timeout=0.10)
                            return True
                        except _queue.Full:
                            continue

                def producer() -> None:
                    try:
                        while not stop.is_set():
                            t0 = time.perf_counter()
                            batch0 = decoder.read_batch()
                            metrics.t_decode += (time.perf_counter() - t0)

                            if not batch0:
                                _q_put(None)
                                return

                            if not _q_put(list(batch0)):
                                return
                    except BaseException as e:
                        prod_exc["e"] = e
                        _q_put(None)

                prod = _threading.Thread(target=producer, name="decode-producer", daemon=True)
                prod.start()

                while True:
                    if cancel_flag is not None and cancel_flag.is_set():
                        break
                    if self.max_frames is not None and frame_num >= self.max_frames:
                        break

                    # [CHANGE 2+] Backpressure: enforce cap. If an active scene blocks draining,
                    # temporarily raise the cap (bounded) instead of letting the store grow unbounded.
                    while store.is_full() or _ram_low():
                        metrics.backpressure_waits += 1

                        # Drain frames that are guaranteed not to be touched by any active clip.
                        min_start = tracker.min_active_start() if tracker is not None else None
                        tracker_sb = int(min_start) if min_start is not None else int(frame_num + 1)
                        sb = tracker_sb

                        _drained = _drain(sb)

                        if not (store.is_full() or _ram_low()):
                            break

                        # CM-111: RAM low but the store is not full and nothing
                        # was drainable (an open clip needs MORE decoded frames
                        # before it can close). Waiting here would deadlock the
                        # run -- proceed with decode; the guard re-engages as
                        # soon as frames become drainable again.
                        if (not store.is_full()) and _drained == 0:
                            break

                        # If we cannot drain because the oldest stored frame is still within an
                        # active clip, bump the cap up to an emergency ceiling.
                        if store.is_full() and min_start is not None and len(store.frames_bgr_u8) > 0:
                            oldest = min(store.frames_bgr_u8.keys())
                            if sb <= oldest:
                                new_max = _compute_emergency_store_max(w, h, self.rest_max_clip_length, store.max_frames, self.device)
                                if new_max > store.max_frames:
                                    old_max = store.max_frames
                                    store.max_frames = new_max
                                    try:
                                        mb = store.vram_mb()
                                        print(f"[FrameStore] backpressure: active scene blocks drain (oldest={oldest}, safe_before={sb}); raising max_frames {old_max}->{new_max} (~{mb:.0f} MB est in-use)")
                                    except Exception:
                                        print(f"[FrameStore] backpressure: active scene blocks drain (oldest={oldest}, safe_before={sb}); raising max_frames {old_max}->{new_max}")
                                    continue

                        # Otherwise, block briefly and retry. This prevents "decode anyway" runaway.
                        time.sleep(0.001)

                    t0 = time.perf_counter()
                    batch = q.get()
                    metrics.t_queue_wait += (time.perf_counter() - t0)

                    if batch is None:
                        break

                    consume_batch(batch)

            else:
                # -----------------------------
                # NVDEC path: decode on main thread (fast + correct)
                # -----------------------------
                while True:
                    if cancel_flag is not None and cancel_flag.is_set():
                        break
                    if self.max_frames is not None and frame_num >= self.max_frames:
                        break

                    # [CHANGE 2+] Backpressure: enforce cap. If an active scene blocks draining,
                    # temporarily raise the cap (bounded) instead of letting the store grow unbounded.
                    while store.is_full() or _ram_low():
                        metrics.backpressure_waits += 1

                        # Drain frames that are guaranteed not to be touched by any active clip.
                        min_start = tracker.min_active_start() if tracker is not None else None
                        tracker_sb = int(min_start) if min_start is not None else int(frame_num + 1)
                        sb = tracker_sb

                        _drained = _drain(sb)

                        if not (store.is_full() or _ram_low()):
                            break

                        # CM-111: RAM low but the store is not full and nothing
                        # was drainable (an open clip needs MORE decoded frames
                        # before it can close). Waiting here would deadlock the
                        # run -- proceed with decode; the guard re-engages as
                        # soon as frames become drainable again.
                        if (not store.is_full()) and _drained == 0:
                            break

                        # If we cannot drain because the oldest stored frame is still within an
                        # active clip, bump the cap up to an emergency ceiling.
                        if store.is_full() and min_start is not None and len(store.frames_bgr_u8) > 0:
                            oldest = min(store.frames_bgr_u8.keys())
                            if sb <= oldest:
                                new_max = _compute_emergency_store_max(w, h, self.rest_max_clip_length, store.max_frames, self.device)
                                if new_max > store.max_frames:
                                    old_max = store.max_frames
                                    store.max_frames = new_max
                                    try:
                                        mb = store.vram_mb()
                                        print(f"[FrameStore] backpressure: active scene blocks drain (oldest={oldest}, safe_before={sb}); raising max_frames {old_max}->{new_max} (~{mb:.0f} MB est in-use)")
                                    except Exception:
                                        print(f"[FrameStore] backpressure: active scene blocks drain (oldest={oldest}, safe_before={sb}); raising max_frames {old_max}->{new_max}")
                                    continue

                        # Otherwise, block briefly and retry. This prevents "decode anyway" runaway.
                        time.sleep(0.001)

                    t0 = time.perf_counter()
                    # [CHANGE 4] Use read_batch_with_pts for PTS extraction
                    batch0, batch_pts_raw = decoder.read_batch_with_pts()
                    metrics.t_decode += (time.perf_counter() - t0)

                    if not batch0:
                        break

                    consume_batch(list(batch0), batch_pts=batch_pts_raw)

        finally:
            # True when this finally was entered by an exception propagating
            # out of the try-body (e.g. a CUDA illegal-memory-access mid-run).
            # In that state the GPU context is typically poisoned, so cleanup
            # steps that touch CUDA will ALSO fail — those failures must be
            # logged-and-skipped rather than allowed to MASK the original
            # exception and abort the rest of cleanup (remux of the partial
            # output, reports). On a clean exit, cleanup errors still raise.
            _inflight_exc = _sys.exc_info()[0] is not None
            # CM-196: keep the exception's own words for the run report's
            # run_error event (the console has them; the report did not).
            _inflight_exc_text = ""
            if _inflight_exc:
                try:
                    _et, _ev = _sys.exc_info()[0], _sys.exc_info()[1]
                    _inflight_exc_text = f"{getattr(_et, '__name__', 'Exception')}: {_ev}"
                    _inflight_exc_text = " ".join(_inflight_exc_text.split())[:400]
                except Exception:
                    _inflight_exc_text = ""

            # Watchdog off first: the EOF flush below can legitimately spend
            # a long time restoring one final long clip -- not a stall.
            try:
                _watchdog.stop()
            except Exception:
                pass

            # CM-179 (v1.71): the keep-awake claim is now released AFTER the
            # finalize below. It used to be dropped here, on the theory that
            # the EOF flush + remux "finish within minutes" -- a 31 GB 4K
            # remux does not (field: the machine slept mid-remux on two
            # boxes, mux=2316 s on both; the disk watchdog read the sleep
            # as a stall and killed it). Released in the try/finally that
            # wraps encoder.close(), so it still cannot outlive a failure.

            # Stop producer safely and avoid deadlock if it is blocked on a full queue.
            stop.set()
            if prod is not None:
                try:
                    prod.join(timeout=2.0)
                except Exception:
                    pass

            try:
                decoder.close()
            except Exception:
                pass

            # NOTE: a producer failure is re-raised at the END of this
            # finally block, not here. Raising here aborted the rest of the
            # cleanup — flush_eof, the final drain, and critically
            # encoder.close() — stranding an open NVENC session, the raw
            # bitstream file handle, and the AsyncEncoder worker thread on
            # every failed ffmpeg-CPU-lane run (they accumulate in the
            # long-lived UI server; NVENC sessions are a scarce resource).

            try:
                if tracker is not None and restorer is not None:
                    for clip in tracker.flush_eof():
                        # v1.50.00: attribute EOF-flush restore time to
                        # t_restore. A single clip that closes at EOF used to
                        # report t_restore=0.00s with the whole restore hiding
                        # in "Overhead" (field artifact: the 226-frame
                        # Flower&Fruit runs), which made the timing summary
                        # lie about where the run's time went.
                        _t_eof = time.perf_counter()
                        restored = restorer.restore_clip(clip)
                        restored = self._stabilize_restored(restored)  # CM-078
                        restored = _guard_restored(clip, restored)     # CM-186
                        # Batch 26 fix: the EOF flush previously omitted
                        # secondary= entirely, so end-of-video clips silently
                        # skipped the RTX Super-Res upscale. One helper now
                        # serves the main loop and the flush (store, VR
                        # projection, or the CM-191 patch plan).
                        _composite_closed_clip(clip, restored)
                        metrics.frames_restored.update(int(fn) for fn in clip.frame_nums)
                        metrics.clip_lengths.append(int(len(clip.frame_nums)))
                        metrics.t_restore += (time.perf_counter() - _t_eof)
                        if (_foi_target is not None
                                and int(_foi_target) in clip.frame_nums
                                and int(_foi_target) in store.frames_bgr_u8):
                            foi_capture["composited"] = (
                                store.frames_bgr_u8[int(_foi_target)].detach().clone()
                            )

                _drain(10**18)
            except Exception as _flush_e:
                if not _inflight_exc:
                    raise
                # Original exception is propagating through this finally; the
                # GPU context is likely dead and these CUDA ops fail too. Log
                # and continue so the encoder still closes/remuxes the frames
                # encoded so far, and the ORIGINAL error reaches the caller.
                print(
                    f"[Pipeline] WARNING: final flush/drain failed after run "
                    f"error ({type(_flush_e).__name__}: {_flush_e}); continuing "
                    f"cleanup with frames encoded so far"
                )
            finally:
                # CM-191: the second decoder is done once the last planned
                # frame is out (it also owns the MPEG-TS remux temp now).
                if _lag is not None:
                    try:
                        metrics.t_redecode = float(_lag.t_decode)
                        metrics.redecode_frames = int(_lag.frames_read)
                        metrics.redecode_skipped = int(_lag.skipped)
                        metrics.redecode_missing = int(_lag.missing)
                        metrics.redecode_peak_patch_mb = (
                            store.plan.peak_patch_bytes / (1024.0 * 1024.0))
                        print(_lag.summary()
                              + f"; peak pending patches {metrics.redecode_peak_patch_mb:.0f} MB"
                              + f" across {store.clips_planned} clip(s)")
                    except Exception:
                        pass
                    try:
                        _lag.close()
                    except Exception:
                        pass

            t_total_no_mux = time.perf_counter() - t0_all

            # [CHANGE 4] Write timecodes file and compute PTS-derived fps
            tc_path: Optional[str] = None
            pts_fps: float = fps
            is_vfr: bool = False
            if pts_log and foi_capture is None:
                pts_fps, is_vfr = compute_pts_fps(pts_log, fallback_fps=fps)
                # CM-202: the timecodes file follows the run-files switch:
                # beside (next to the video, as before) | temp (in the Temp
                # folder) | off (written to Temp for the finalize step only
                # and deleted after it, VFR or not).
                _tc_anchor = str(self.output_path)
                _rr_mode_now = str(getattr(self, "run_report_mode", "beside") or "beside").lower()
                if _rr_mode_now in ("temp", "off"):
                    try:
                        import tempfile as _tf
                        _td_ = str(self.cfg.get("temp_dir", default="") or "") or _tf.gettempdir()
                        Path(_td_).mkdir(parents=True, exist_ok=True)
                        _tc_anchor = str(Path(_td_) / Path(self.output_path).name)
                    except Exception:
                        _tc_anchor = str(self.output_path)
                tc_path = write_timecodes_v2(pts_log, _tc_anchor, fps=fps)

                if abs(pts_fps - fps) / max(fps, 0.001) > 0.002:
                    print(f"[PTS] FPS mismatch: metadata={fps:.3f}  pts_derived={pts_fps:.3f}")
                if is_vfr:
                    print("[PTS] WARNING: Variable frame rate detected. Timecodes file written for accurate remux.")
                # Set PTS metadata on the underlying Encoder (works whether
                # encoder is a bare Encoder or wrapped in AsyncEncoder).
                _real_encoder = encoder.underlying if isinstance(encoder, AsyncEncoder) else encoder
                _real_encoder._pts_fps = pts_fps
                _real_encoder._pts_timecodes_path = tc_path
                _real_encoder._pts_is_vfr = is_vfr

                # CM-120r2: HEAD-SKIP detection. Stream captures often start
                # mid-GOP and the hardware decode path silently discards the
                # reference-broken head frames (field case: output frame 0
                # showed source content from t=1.03s -> video led audio by
                # ~1 s from the very first frame). No PTS DELTA can see a
                # loss that precedes the first delivered frame, but the
                # first frame's ABSOLUTE timestamp vs the stream's
                # start_time measures it exactly. The remux compensates by
                # delaying the video by the same amount (same -itsoffset
                # machinery as the CM-121 source-offset restoration).
                try:
                    _stream_start = getattr(decoder.metadata, "start_time", None)
                    _head_s, _first_pts, _spu = pts_head_skip(
                        pts_log, fps, _stream_start)
                    if _first_pts is not None and _spu is not None:
                        print(f"[PTS] first delivered frame at source "
                              f"t={_first_pts * _spu:.3f}s (stream start "
                              f"{(_stream_start if _stream_start is not None else 0.0):.3f}s)")
                    if _head_s > 0:
                        print(f"[Pipeline] head-skip detected: decoder "
                              f"discarded ~{_head_s:.3f}s (~{int(round(_head_s * fps))} "
                              f"frames) at the stream head; the remux will "
                              f"delay the video to keep audio in sync.")
                        _real_encoder._head_skip_seconds = float(_head_s)
                except Exception as _hs_e:
                    print(f"[PTS] head-skip check failed ({_hs_e}); continuing")

            # CM-120: gap-fill / decoder-drop honesty report. Never end a
            # run with a silently short video: either the filler kept the
            # sync (say what it did), or frames are missing and we could
            # not see where (say THAT, loudly).
            _filler = getattr(store, "gap_filler", None)
            if _filler is not None and _filler.total_filled > 0:
                print(f"[Pipeline] gap-fill: inserted {_filler.total_filled} "
                      f"duplicate frame(s) across {len(_filler.gaps)} gap(s) "
                      f"to keep A/V sync (decoder skipped source frames).")
            if _filler is not None and _filler.skipped_gaps:
                print(f"[Pipeline] WARNING: {len(_filler.skipped_gaps)} "
                      f"timeline jump(s) were too large to fill -- audio "
                      f"sync shifts at those points (source timeline is "
                      f"discontinuous).")
            try:
                _promised = int(decoder.num_frames or 0)
                _filled = int(_filler.total_filled) if _filler else 0
                _short = _promised - int(metrics.processed_frames) - 0
                if (_promised > 0 and _short > 0 and _filled == 0
                        and self.max_frames is None
                        and not (cancel_flag is not None and cancel_flag.is_set())):
                    _sec = _short / fps if fps > 0 else 0.0
                    _head_known = 0.0
                    try:
                        _head_known = float(getattr(_real_encoder, "_head_skip_seconds", 0.0) or 0.0)
                    except Exception:
                        _head_known = 0.0
                    _head_frames = int(round(_head_known * fps)) if fps > 0 else 0
                    if tc_path is not None and _head_frames >= _short:
                        # The whole shortfall is the measured head skip and
                        # the remux is compensating -- informational only.
                        print(f"[Pipeline] NOTE: decoder delivered "
                              f"{metrics.processed_frames} of the {_promised} "
                              f"frames the stream reports; the missing "
                              f"~{_sec:.1f}s is the stream HEAD the decoder "
                              f"discarded, and the remux delays the video "
                              f"to compensate. A/V sync is preserved.")
                    elif tc_path is not None:
                        print(f"[Pipeline] WARNING: decoder delivered "
                              f"{metrics.processed_frames} of the {_promised} "
                              f"frames the stream reports ({_short} missing, "
                              f"~{_sec:.1f}s). Frame timestamps show a "
                              f"CONTINUOUS timeline, so the loss is at the "
                              f"stream head/tail or the decoder renumbers "
                              f"frames and hides interior gaps -- AUDIO MAY "
                              f"BE OUT OF SYNC. A software-decode run "
                              f"(ffmpeg) of the source recovers all frames.")
                    else:
                        print(f"[Pipeline] WARNING: decoder delivered "
                              f"{metrics.processed_frames} of the {_promised} "
                              f"frames the stream reports ({_short} missing, "
                              f"~{_sec:.1f}s) and no per-frame timestamps "
                              f"were available to locate or fill the gap -- "
                              f"AUDIO MAY BE OUT OF SYNC from the missing "
                              f"span onward. A software-decode run (ffmpeg) "
                              f"of the source will recover all frames.")
            except Exception:
                pass

            t0 = time.perf_counter()
            try:
                pbar.refresh()
                pbar.close()
            except Exception:
                pass

            # If async, flush + join the encoder worker before close()
            # triggers the remux. close() does this internally too, but
            # making it explicit lets us report worker stats. The inner
            # try/finally guarantees encoder.close() runs even when
            # flush_and_join() surfaces a worker exception — close() calls
            # flush_and_join() again, which is a no-op once _stopped is set,
            # so the underlying encoder/remux is never stranded.
            try:
                try:
                    if isinstance(encoder, AsyncEncoder):
                        encoder.flush_and_join()
                        print(
                            f"[Encoder] Async worker: encoded={encoder.frames_encoded} frames "
                            f"in {encoder.worker_wall:.2f}s wall (overlapping with main thread)"
                        )
                finally:
                    try:
                        encoder.close()
                    finally:
                        # CM-179: machine may sleep only once the output is
                        # finalized (or its finalize has failed).
                        try:
                            _ka_release()
                        except Exception:
                            pass
                        # A sidecar extraction still running at this point
                        # belongs to an aborted run: stop it.
                        try:
                            if _audio_sidecar is not None and not _audio_sidecar.decided:
                                _audio_sidecar.cancel()
                        except Exception:
                            pass
                        self._audio_sidecar_live = None
            except Exception as _enc_e:
                if not _inflight_exc:
                    raise
                # Same rationale as the flush guard: with the original error
                # in flight (dead GPU context), the async worker likely died
                # with the same CUDA error — surface it as a log line, keep
                # the original exception, and let the remaining cleanup run.
                # Encoder.close()'s remux of the partial bitstream is CPU-side
                # and has already been attempted by the inner finally.
                print(
                    f"[Pipeline] WARNING: encoder finalize failed after run "
                    f"error ({type(_enc_e).__name__}: {_enc_e})"
                )

            metrics.t_mux += (time.perf_counter() - t0)

            # CM-107 (Batch 50): salvage record. The Idol forensics run
            # (2026-08-19) errored mid-encode yet flushed 135,584 frames,
            # remuxed cleanly, and produced a playable 37-minute file --
            # and the UI still said only "Error". Record what actually
            # survived so callers (server -> UI, CLI) can report "PARTIAL:
            # encoded M of N" honestly instead of implying total loss.
            try:
                _real_enc = (encoder.underlying
                             if isinstance(encoder, AsyncEncoder) else encoder)
                _frames_enc = int(getattr(_real_enc, "_frames_encoded", 0) or 0)
                _needs_remux = bool(getattr(_real_enc, "_needs_remux", False))
                _remux_ok = bool(getattr(_real_enc, "_remux_ok", False))
                _out_p = Path(self.output_path)
                _out_ok = (_frames_enc > 0
                           and foi_capture is None
                           and _out_p.exists()
                           and _out_p.stat().st_size > 0
                           and ((not _needs_remux) or _remux_ok))
                self.last_run_salvage = {
                    "errored": bool(_inflight_exc),
                    "frames_encoded": _frames_enc,
                    "total_frames": int(total_frames),
                    "output_ok": bool(_out_ok),
                    "output_path": str(self.output_path),
                }
                if _inflight_exc and _out_ok:
                    _sec = (_frames_enc / fps) if fps > 0 else 0.0
                    print(f"[Pipeline] PARTIAL RESULT: the run errored, but "
                          f"the output contains the first {_frames_enc} "
                          f"frames (~{int(_sec // 60)}m{int(_sec % 60):02d}s "
                          f"of video) and is playable: {self.output_path}")
            except Exception:
                self.last_run_salvage = None

            # CM-111 (Batch 50): explicitly release the frame store. On a
            # clean run the final drain already emptied it; after a run
            # error the drain is skipped/fails and the full store would
            # otherwise stay resident -- the Idol post-mortem telemetry
            # showed ~8 GB (a full 4K MCL-300 host store) held by the
            # process for 10 hours after the failure, so a retry without
            # an app restart started 8 GB in the hole. Free it here,
            # unconditionally, before the exception propagates.
            try:
                _held = len(store.frames_bgr_u8)
                store.frames_bgr_u8.clear()
                store.frame_pts.clear()
                if _held:
                    print(f"[FrameStore] released {_held} undrained frames "
                          f"at cleanup (CM-111)")
                import gc as _gc
                _gc.collect()
                if self.device.type in ("cuda", "xpu"):
                    from chitramaya.device import empty_cache as _cm111_empty
                    _cm111_empty(self.device)
            except Exception:
                pass

            metrics.wall_end = _dt.datetime.now()

            sum_parts = metrics.sum_parts()
            overhead = t_total_no_mux - sum_parts
            t_total_with_mux = t_total_no_mux + metrics.t_mux

            print(
                f"[Pipeline] Processed {metrics.processed_frames} frames: "
                f"t_decode={metrics.t_decode:.2f}s t_det={metrics.t_det:.2f}s "
                f"t_track={metrics.t_track:.2f}s t_restore={metrics.t_restore:.2f}s "
                f"t_encode={metrics.t_encode:.2f}s"
            )
            print(
                f"[Pipeline] Prefetch stats: "
                f"t_queue_wait={metrics.t_queue_wait:.2f}s t_prepare={metrics.t_prepare:.2f}s "
                f"t_upload={metrics.t_upload:.2f}s t_csc={metrics.t_csc:.2f}s"
            )
            # [CHANGE 2] Print backpressure stats
            if metrics.backpressure_waits > 0:
                print(f"[Pipeline] Backpressure waits: {metrics.backpressure_waits} (store peaked at max_frames={store.max_frames})")
            # CM-191: the re-decode time is spent INSIDE t_encode (the drain
            # pulls the frame right before encoding it); say how much.
            # CM-196 VRAM ledger, last line: processing done (or died), before finalize.
            try:
                _snapE = vram_snapshot(self.device)
                if _snapE:
                    print(format_vram("end of processing", _snapE))
                    if self._report is not None:
                        self._report.vram("end_of_processing", _snapE, frame=int(metrics.processed_frames))
            except Exception:
                pass
            if metrics.redecode_frames > 0:
                print(f"[Pipeline] Re-decode (CM-191): {metrics.redecode_frames} frames, "
                      f"t_redecode={metrics.t_redecode:.2f}s (inside t_encode); "
                      f"alignment skipped={metrics.redecode_skipped} "
                      f"missing={metrics.redecode_missing}; "
                      f"peak pending patches {metrics.redecode_peak_patch_mb:.0f} MB "
                      f"in {'pinned RAM' if getattr(store, 'patch_home', 'host') == 'host' else 'VRAM'}")
            # CM-186: black-output guard hits (should be zero)
            if metrics.guard_black_frames:
                _gb = metrics.guard_black_frames
                print(f"[Pipeline] WARNING: CM-186 guard refused {len(_gb)} all-black "
                      f"restorer result(s) (frames {min(_gb)}..{max(_gb)}); those "
                      f"frames kept their source pixels.")
            print(
                f"[Pipeline] Processing time (no mux) = {t_total_no_mux:.2f}s "
                f"Overhead = {overhead:.2f}s (sum_parts={sum_parts:.2f}s)"
            )
            print(f"[Pipeline] Total time (with mux) = {t_total_with_mux:.2f}s (mux={metrics.t_mux:.2f}s)")
            print(f"[Pipeline] DONE: Processed  &  Remuxed {metrics.processed_frames} frames")
            print(f"[Pipeline] early_passthrough_frames={metrics.early_passthrough_frames}")

            fw, ft, tb, avg_area, pct = metrics.det_stats.summary()
            print(
                f"[DetStats] frames_with_det={fw}/{ft} total_boxes={tb} "
                f"avg_roi_area_px={avg_area:.2f} ({pct:.4f}% of frame)"
            )

            # Restoration coverage diagnostics.
            #
            # Four mutually exclusive buckets for every frame in the video:
            #
            #   1. restored          = pixels were modified by composite_clip_into_store
            #                          (includes detections directly AND TTL gap-fill)
            #   2. legit_passthrough = no detection in this frame AND no active scene
            #                          (warmup or genuine clean stretch). NOT a miss.
            #   3. restoration_miss  = detected but never made it to compositor.
            #                          Should be 0; non-zero is a bug.
            #   4. visible_miss      = everything else. These are the frames where the
            #                          user can SEE mosaic that should have been restored.
            #
            # The actionable list for tuning is visible_miss_frames. Use mask-viz mode
            # (--mode pseudo) to confirm them visually.
            all_frames = set(range(ft))
            det_set = metrics.det_stats.frames_with_det_set
            rest_set = metrics.frames_restored
            legit_set = metrics.frames_legit_passthrough
            miss_set = all_frames - rest_set - legit_set

            fr = len(rest_set)
            legit = len(legit_set)
            detected_and_restored = len(det_set & rest_set)
            restoration_miss_set = det_set - rest_set    # detected but never restored — BUG
            restoration_miss = len(restoration_miss_set)
            gap_fill_set = rest_set - det_set            # restored without a direct det (TTL bridge)
            gap_fill_bonus = len(gap_fill_set)
            visible_miss = len(miss_set)

            # Restorer workload — clips x clip-length drives restore cost far
            # better than box count. total_clip_frames = frames actually fed to
            # the restorer (overlapping clips counted), which is why two configs
            # with similar box counts can differ a lot in t_restore.
            clip_lens = sorted(int(x) for x in metrics.clip_lengths)
            n_clips = len(clip_lens)
            total_clip_frames = int(sum(clip_lens))
            if n_clips:
                clip_min = clip_lens[0]
                clip_max = clip_lens[-1]
                clip_med = clip_lens[n_clips // 2]
                clip_mean = round(total_clip_frames / n_clips, 1)
            else:
                clip_min = clip_max = clip_med = 0
                clip_mean = 0.0
            # CM-076: how many clips actually HIT the Max Clip Length cap?
            # Observed lengths run to cap+~3 (TTL grace), so count len>=cap.
            # This is the quality-vs-VRAM dial's instrument: capped=0 means a
            # bigger MCL buys nothing on this content; a high percentage
            # means scenes are being chopped and a larger MCL (or the jumpy-
            # boundary risk of a smaller one) is worth thinking about.
            _mcl = int(self.rest_max_clip_length)
            capped = sum(1 for x in clip_lens if _mcl > 0 and x >= _mcl)
            capped_pct = round(100.0 * capped / n_clips, 1) if n_clips else 0.0

            print(
                f"[RestStats] restored={fr}/{ft} "
                f"detected_and_restored={detected_and_restored} "
                f"gap_fill_bonus={gap_fill_bonus} "
                f"legit_passthrough={legit} "
                f"visible_miss={visible_miss} "
                f"restoration_miss={restoration_miss}"
            )
            print(
                f"[ClipStats] clips={n_clips} total_clip_frames={total_clip_frames} "
                f"len_min={clip_min} len_med={clip_med} len_max={clip_max} len_mean={clip_mean} "
                f"capped={capped} ({capped_pct}% at MCL={_mcl})"
            )

            # CM-077b: did the secondary (RTX Super-Res) actually fire, and
            # where? The gate opens only for crops whose original region
            # exceeds the clip size (256px longest side), so on some content
            # it legitimately never engages -- largest_crop_px tells you
            # whether that is this content or a bug. Frames with an upscale
            # are listed in the misses JSON (secondary_upscaled_frames) for
            # seek-and-inspect A/B at exactly those frames.
            _sec_stats = getattr(self._secondary, "stats", None) \
                if self._secondary is not None else None
            if _sec_stats is None and _region_stats.crops:
                _top = ", ".join(f"{d['frame']}({d['crop_px']}px)" for d in _region_stats.top_frames(5))
                print(f"[RegionStats] crops={_region_stats.crops} largest_crop_px={_region_stats.largest_px} "
                      f"biggest-crop frames: {_top} -- top 20 in the run report (counts.regions)")
            if _sec_stats is not None:
                _sec_pct = round(
                    100.0 * _sec_stats.crops_upscaled / _sec_stats.crops_seen, 1
                ) if _sec_stats.crops_seen else 0.0
                print(
                    f"[SecStats] mode={self.secondary_restoration} "
                    f"crops_seen={_sec_stats.crops_seen} "
                    f"upscaled={_sec_stats.crops_upscaled} ({_sec_pct}%) "
                    f"skipped_small={_sec_stats.skipped_small} "
                    f"skipped_geom={_sec_stats.skipped_geom} "
                    f"largest_crop_px={_sec_stats.largest_px} "
                    f"(gate opens above {self._secondary.min_apply_size}px) "
                    f"frames_with_upscale={len(_sec_stats.applied_frames)}"
                )
                # CM-077c: WHERE is that largest crop? Print the top frames
                # ranked by crop size -- the seek list for inspecting the
                # scaler where its contribution is biggest. Full top-20 (with
                # per-frame px) lands in the misses JSON.
                _sec_top = _sec_stats.top_frames(20)
                if _sec_top:
                    _preview = ", ".join(
                        f"{t['frame']}({t['crop_px']}px)" for t in _sec_top[:5])
                    print(f"[SecStats] biggest-crop frames: {_preview} "
                          f"-- top 20 in the misses JSON "
                          f"(secondary_biggest_frames)")
                if _sec_stats.crops_upscaled == 0:
                    print(
                        f"[SecStats] Secondary never engaged: no crop exceeded "
                        f"{self._secondary.min_apply_size}px on its longest side "
                        f"(largest seen: {_sec_stats.largest_px}px). This content "
                        f"pastes back at or below the restorer's native size, so "
                        f"there is nothing for the scaler to add."
                    )

            # CM-180: the run report (replaces the misses JSON). Everything
            # the old file carried is here in a defined shape, plus the
            # panel, the effective values, structured timing and events;
            # per-frame lists are run-length ranges (kilobytes, not
            # megabytes). Off with runReport = off.
            self.last_report_path = None
            if self._report is not None:
                try:
                    _rr = self._report
                    # events from state we can read here
                    if metrics.guard_black_frames:
                        _gb = sorted(metrics.guard_black_frames)
                        _rr.event("black_output_guard", _gb[0],
                                  f"restorer returned all-black for {len(_gb)} clip frame(s); source pixels kept (CM-186)",
                                  count=len(_gb))
                    if metrics.redecode_skipped or metrics.redecode_missing:
                        _rr.event("redecode_alignment", None,
                                  f"skipped={metrics.redecode_skipped} missing={metrics.redecode_missing} (CM-191)",
                                  count=int(metrics.redecode_skipped + metrics.redecode_missing))
                    _fl = getattr(store, "gap_filler", None)
                    if _fl is not None and getattr(_fl, "total_filled", 0):
                        _rr.event("gap_fill", None,
                                  f"{_fl.total_filled} duplicate frame(s) across {len(_fl.gaps)} gap(s) (CM-120)",
                                  count=int(_fl.total_filled))
                    if _fl is not None and getattr(_fl, "skipped_gaps", None):
                        _rr.event("timeline_jump_unfilled", None,
                                  f"{len(_fl.skipped_gaps)} jump(s) too large to fill", count=len(_fl.skipped_gaps))
                    _flaps = int(getattr(_watchdog, "flap_count", 0) or 0)
                    if _flaps:
                        _rr.event("pcie_flap", None, f"{_flaps} down-train/recover cycle(s)", count=_flaps)
                    _re_obj = locals().get("_real_encoder")
                    _head = float(getattr(_re_obj, "_head_skip_seconds", 0.0) or 0.0) if _re_obj is not None else 0.0
                    if _head > 0:
                        _rr.event("head_skip", 0, f"decoder discarded ~{_head:.3f}s at the stream head; remux delays the video")
                    if cancel_flag is not None and cancel_flag.is_set():
                        _rr.event("cancelled", int(metrics.processed_frames), "run cancelled by the user")
                    if _inflight_exc:
                        _rr.event("run_error", int(metrics.processed_frames),
                                  "the run raised; the output is a PARTIAL result"
                                  + (f" -- {_inflight_exc_text}" if _inflight_exc_text else " (see the console log)"))

                    _run = {
                        "total_frames": int(ft),
                        "processed_frames": int(metrics.processed_frames),
                        "fps": float(fps),
                        "resolution": [int(w), int(h)],
                        "source_container": str(getattr(decoder, "_container_format", "") or ""),
                        "partial": bool(_inflight_exc or (cancel_flag is not None and cancel_flag.is_set())),
                        "mode": self.mode,
                    }
                    _timing = {
                        "wall_s": round(float(t_total_with_mux), 2),
                        "processing_s": round(float(t_total_no_mux), 2),
                        "decode_s": round(metrics.t_decode, 2),
                        "redecode_s": round(metrics.t_redecode, 2),
                        "detect_s": round(metrics.t_det, 2),
                        "track_s": round(metrics.t_track, 2),
                        "restore_s": round(metrics.t_restore, 2),
                        "encode_s": round(metrics.t_encode, 2),
                        "queue_wait_s": round(metrics.t_queue_wait, 2),
                        "prepare_s": round(metrics.t_prepare, 2),
                        "remux_s": round(metrics.t_mux, 2),
                        "avg_fps_completed": round(float(metrics.processed_frames) / t_total_no_mux, 2) if t_total_no_mux > 0 else 0.0,
                        "redecode_frames": int(metrics.redecode_frames),
                        "redecode_peak_patch_mb": round(float(metrics.redecode_peak_patch_mb), 1),
                        "backpressure_waits": int(metrics.backpressure_waits),
                    }
                    _counts = {
                        "restored": int(fr),
                        "detected": int(len(det_set)),
                        "detected_and_restored": int(detected_and_restored),
                        "gap_fill_bonus": int(gap_fill_bonus),
                        "legit_passthrough": int(legit),
                        "visible_miss": int(visible_miss),
                        "restoration_miss": int(restoration_miss),
                        "early_passthrough_count": int(metrics.early_passthrough_frames),
                        "total_boxes": int(getattr(metrics.det_stats, "total_boxes", 0) or 0),
                        "clips": {
                            "count": int(n_clips),
                            "total_clip_frames": int(total_clip_frames),
                            "len_min": int(clip_min),
                            "len_median": int(clip_med),
                            "len_max": int(clip_max),
                            "len_mean": float(clip_mean),
                            "capped": int(capped),
                            "capped_pct": float(capped_pct),
                        },
                        "regions": {
                            "crops": int(_region_stats.crops),
                            "largest_crop_px": int(_region_stats.largest_px),
                            "biggest_frames": _region_stats.top_frames(20),
                        },
                        "secondary": None if _sec_stats is None else {
                            "mode": str(self.secondary_restoration),
                            "crops_seen": int(_sec_stats.crops_seen),
                            "crops_upscaled": int(_sec_stats.crops_upscaled),
                            "skipped_small": int(_sec_stats.skipped_small),
                            "skipped_geom": int(_sec_stats.skipped_geom),
                            "largest_crop_px": int(_sec_stats.largest_px),
                            "gate_threshold_px": int(self._secondary.min_apply_size),
                            "frames_with_upscale": len(_sec_stats.applied_frames),
                            "biggest_frames": _sec_stats.top_frames(20),
                        },
                    }
                    _frames = {
                        "visible_miss": sorted(miss_set),
                        "restoration_miss": sorted(restoration_miss_set),
                        "gap_fill": sorted(gap_fill_set),
                        "legit_passthrough": sorted(legit_set),
                        "secondary_upscaled": sorted(_sec_stats.applied_frames) if _sec_stats is not None else [],
                        "black_output_guard": sorted(metrics.guard_black_frames),
                    }
                    _debug = None
                    if metrics.det_stats.rois:
                        _debug = {
                            "detection_rois_format": "frame_num -> [[top,left,bottom,right], ...] (pixel coords, inclusive)",
                            "detection_rois": {str(k): v for k, v in sorted(metrics.det_stats.rois.items())},
                        }
                    _rep = _rr.build(run=_run, timing=_timing, counts=_counts, frames=_frames,
                                     debug=_debug, device=self.device)
                    _rp = _rr.write(_rep)
                    _lp = _rr.write_log()
                    if _rp:
                        self.last_report_path = _rp
                        print(f"[Pipeline] Run report: {_rp}" + (f"  (console log: {Path(_lp).name})" if _lp else ""))
                except Exception as e:
                    print(f"[Pipeline] WARNING: failed to write the run report: {e}")

            if metrics.wall_start and metrics.wall_end:
                elapsed = metrics.wall_end - metrics.wall_start
                print(f"[Pipeline] Wall clock: start={metrics.wall_start} end={metrics.wall_end} elapsed={elapsed}")
            print(f"[Pipeline] perf_counter elapsed = {t_total_with_mux:.2f}s")

            # [CHANGE 4] Cleanup timecodes file (kept only if VFR; CM-202:
            # never kept when the run-files switch is off)
            if tc_path and (not is_vfr or str(getattr(self, "run_report_mode", "beside")).lower() == "off"):
                try:
                    Path(tc_path).unlink(missing_ok=True)
                except Exception:
                    pass

            try:
                pbar.close()
            except Exception:
                pass

            # If the producer failed, surface it now — AFTER all cleanup, so
            # the encoder/NVENC session, async worker, and remux were never
            # stranded by a decode error. Guard on sys.exc_info(): if the
            # try-body is already propagating its own exception through this
            # finally, do not mask it with the producer's.
            if "e" in prod_exc and _sys.exc_info()[0] is None:
                raise prod_exc["e"]

        # FOI: if the target frame was a passthrough (no clip touched it), its
        # composited form equals the original. Also flag whether we saw it.
        if foi_capture is not None:
            foi_capture["found"] = ("original" in foi_capture)
            if foi_capture.get("composited") is None and "original" in foi_capture:
                foi_capture["composited"] = foi_capture["original"]

        # Additive: expose results to programmatic callers (the UI bridge).
        # The CLI path ignores this return value, so it is non-breaking.
        return metrics


# ===========================================================================
# UI BRIDGE (RM-030)
# ===========================================================================
# Adapter that lets the web UI (ChitraMaya/server.py) drive the proven one-shot
# `Pipeline` above. The server imports `MosaicPipeline` and `MosaicPipelineConfig`
# from this module; without them the mosaic UI raised ImportError on click.
#
# Design:
#   - MosaicPipelineConfig: flat config the server's MosaicConfig.to_pipeline_config()
#     constructs (17 fields). Several are INERT (the pipeline has no temporal
#     crossfade/color-match implementation) but are accepted so the server's
#     call signature is satisfied; they are documented as such.
#   - MosaicPipeline: holds ONE warm `Pipeline` instance (built from the config)
#     plus its detector/restorer, built ONCE so repeated UI previews don't
#     reload models. Each process_file() updates the input/output on the warm
#     Pipeline and calls run() with the warm models + progress/cancel injected.
#   - MosaicResult: the small result object the server reads counts off.
#
# NOTE (split-readiness): this bridge is self-contained mosaic code and imports
# nothing from the swap side. Keep it that way — it lifts out cleanly when the
# repos split.


@dataclass
class MosaicResult:
    """Result of a MosaicPipeline.process_file() call (what server.py reads)."""
    frames: int = 0
    detections: int = 0
    restorations: int = 0
    diag_path: Optional[str] = None


@dataclass
class MosaicPipelineConfig:
    """Flat runtime config consumed by MosaicPipeline.

    Mirrors exactly the keyword arguments built by
    ChitraMaya.models.MosaicConfig.to_pipeline_config(). Fields marked INERT are
    accepted for signature compatibility but have no effect: the proven
    pipeline implements neither temporal crossfade nor color matching (the
    compositor does spatial mask feathering only). They are surfaced here so a
    future implementation has a defined home, and so the server's mutation of
    _discard_margin / _blend_frames remains harmless.
    """
    detection_model: str = ""
    restoration_model: str = ""
    detection_score: float = 0.30
    detection_batch_size: int = 8
    max_clip_size: int = 60
    temporal_overlap: int = 0        # INERT: no temporal-overlap implementation
    crossfade: bool = False          # INERT: no crossfade implementation
    blend_frames: int = 0            # INERT: no crossfade implementation
    mask_preview: bool = False
    mask_color: tuple = (255, 0, 255)
    mask_opacity: float = 0.70
    # Auto-censor: pixelate detected regions instead of restoring them.
    censor: bool = False
    censor_block: int = 16
    detection_fp16: bool = True
    restoration_fp16: bool = True
    use_trt: bool = True
    color_match: bool = False        # INERT: no color-match implementation
    codec: str = "hevc"
    preset: str = "P5"
    qp: int = 18
    async_encoder: bool = False      # opt-in: overlap NVENC on a worker thread
    write_diagnostics: bool = True
    # Optional pass-throughs the UI may set later; safe defaults preserve
    # current behavior. SBS is exposed because the CLI supports it and the
    # lean-UI pass will add controls for it.
    sbs_enabled: bool = False
    sbs_layout: str = "lr"
    sbs_det_split: bool = False
    # CM-045: "none" | "fisheye" — per-eye hequirect->fisheye analysis warp
    # for studios that apply mosaic in viewing space (requires sbs_enabled).
    vr_projection: str = "none"
    # CM-077: "none" | "rtx-2x" | "rtx-4x" — RTX Super-Res upscale of restored
    # crops before paste-back (real restoration mode only; needs nvidia-vfx).
    secondary_restoration: str = "none"
    # CM-146 (Batch 70): Maxine DENOISE pass chained after the RTX upscale
    # ("none"|"low"|"medium"|"high"|"ultra"); ignored by esrgan-4x.
    secondary_denoise: str = "none"
    # CM-078: 0 = off, 1..3 = vs_temporalfix strength — temporal stabilization
    # of restored crops (7-frame window; real restoration mode only).
    temporal_stability: int = 0
    store_max_frames: int = 0
    # CM-084: FrameStore backend -- "auto" | "device" | "host".
    store_backend: str = "auto"
    # CM-202: the panel's run-files switch ("beside" | "temp" | "off");
    # "" = not set by the panel -> the flat runReport key decides.
    run_report: str = ""
    det_imgsz: int = 640
    det_iou: float = 0.70
    roi_dilate: int = 0
    use_seg_masks: bool = True
    feather_radius: int = 0
    blendmask: str = "none"


# ---------------------------------------------------------------------------
# Wiring guardrail (see ChitraMaya-WiringAudit)
# ---------------------------------------------------------------------------
# Every MosaicPipelineConfig field must be classified as CONSUMED (it flows to
# the pipeline/server and has an effect) or INERT (accepted for
# signature/UX compatibility but deliberately not implemented). Adding a field
# without classifying it — or renaming one and leaving a stale entry — raises
# at import. This is the check that would have caught the Detection-Batch /
# Feather / Blend-Mask "wired every layer except the last hop" bugs before they
# shipped. Pair it with tools/verify_wiring.py, which asserts the values
# actually survive the to_pipeline_config -> _build_base_config -> Pipeline
# round trip.
_MPC_CONSUMED_FIELDS = frozenset({
    "detection_model", "restoration_model", "detection_score",
    "detection_batch_size", "max_clip_size", "mask_preview", "mask_color",
    "mask_opacity", "censor", "censor_block", "detection_fp16", "restoration_fp16", "use_trt",
    "codec", "preset", "qp", "async_encoder", "write_diagnostics", "run_report",
    "sbs_enabled", "sbs_layout",
    "sbs_det_split", "vr_projection", "secondary_restoration",
    "secondary_denoise",
    "temporal_stability",
    "store_max_frames", "store_backend", "det_imgsz", "det_iou", "roi_dilate",
    "use_seg_masks", "feather_radius", "blendmask",
})
_MPC_INERT_FIELDS = frozenset({
    "temporal_overlap",   # no temporal-overlap implementation
    "crossfade",          # no crossfade implementation
    "blend_frames",       # no crossfade implementation
    "color_match",        # no color-match implementation
})


def _audit_mosaic_pipeline_config_fields() -> None:
    import dataclasses
    allf = {f.name for f in dataclasses.fields(MosaicPipelineConfig)}
    classified = _MPC_CONSUMED_FIELDS | _MPC_INERT_FIELDS
    unclassified = allf - classified
    stale = classified - allf
    if unclassified:
        raise RuntimeError(
            "MosaicPipelineConfig field(s) not classified as consumed/inert: "
            f"{sorted(unclassified)} — wire them (and add to "
            "_MPC_CONSUMED_FIELDS) or add to _MPC_INERT_FIELDS. See "
            "ChitraMaya-WiringAudit."
        )
    if stale:
        raise RuntimeError(
            "consumed/inert sets reference nonexistent MosaicPipelineConfig "
            f"field(s): {sorted(stale)} (rename left a stale entry)."
        )


_audit_mosaic_pipeline_config_fields()


class MosaicPipeline:
    """Warm, reusable mosaic pipeline for the UI.

    Builds the detector + restorer ONCE from `cfg` and reuses them across
    process_file() calls (UI previews + full runs), so models are not reloaded
    on every click. Wraps the proven one-shot `Pipeline` and drives it via the
    additive run() hooks (warm-model injection, progress_cb, cancel_flag).
    """

    def __init__(self, cfg: MosaicPipelineConfig, gpu_id: int = 0):
        self.config = cfg
        self.gpu_id = int(gpu_id)

        # Server mutates these in place for cached pipelines (inert knobs).
        self._discard_margin: int = max(0, int(getattr(cfg, "temporal_overlap", 0)))
        self._blend_frames: int = max(0, int(getattr(cfg, "blend_frames", 0)))

        # Build a warm host Pipeline from the config (no I/O happens until
        # run()). Borrow its proven _build_detector/_build_restorer so the
        # warm models are byte-identical to what the CLI builds.
        self._host = Pipeline(self._build_base_config())
        # Delegate to the host builders. mode drives everything: "real" ->
        # BasicVSR++, "pseudo" -> PseudoClipRestorer (flat fill), "none" ->
        # no restorer. The detector is always built (needed by real + pseudo).
        # This replaces the old detect_only guards, which wrongly nulled the
        # detector and let run() rebuild a restorer anyway.
        # CM-196 T10f: ledger the warm build -- these two are the largest
        # non-torch residents (TensorRT engines) and were invisible until now.
        self._host._vram_stage("before models (CUDA context)")
        self._detector = self._host._build_detector()
        self._host._vram_stage("detector (warm build)")
        self._restorer = self._host._build_restorer()
        self._host._vram_stage("restorer (warm build)")
        self._host._vram_ledger_warm = list(self._host._vram_ledger)

    # -- config construction -------------------------------------------------

    def _build_base_config(self, input_path: str = "", output_path: str = "") -> Config:
        """Translate the flat MosaicPipelineConfig into the nested Config dict
        the proven Pipeline.__post_init__ reads. Mirrors cli_config keys."""
        c = self.config
        data: dict = {
            "input": str(input_path),
            "output": str(output_path),
            "mode": ("mosaic" if bool(getattr(c, "censor", False))
                     else "pseudo" if bool(getattr(c, "mask_preview", False))
                     else "real"),
            "visualization": {
                "fill_color": list(getattr(c, "mask_color", (255, 0, 255))),
                "fill_opacity": float(getattr(c, "mask_opacity", 0.70)),
                "block": int(getattr(c, "censor_block", 16)),
            },
            "store_max_frames": int(getattr(c, "store_max_frames", 0)),
            "store_backend": str(getattr(c, "store_backend", "auto") or "auto"),
            "sbs_enabled": bool(getattr(c, "sbs_enabled", False)),
            "sbs_layout": str(getattr(c, "sbs_layout", "lr")),
            "sbs_det_split": bool(getattr(c, "sbs_det_split", False)),
            "vr_projection": str(getattr(c, "vr_projection", "none") or "none"),
            "secondary_restoration": str(getattr(c, "secondary_restoration", "none") or "none"),
            # Batch 74 (CM-146 fix): Batch 70 wired secondary_denoise through
            # models.py -> to_pipeline_config -> MosaicPipelineConfig -> the
            # consumed-keys allowlist, and missed THIS hop -- the one translation
            # the pipeline actually reads. cfg.get("secondary_denoise") therefore
            # always returned its default ("none") in every environment. Field
            # case 2026-09-01: four denoise A/B runs on the 3060 Ti all ran
            # denoise=none with Ultra selected in the UI.
            "secondary_denoise": str(getattr(c, "secondary_denoise", "none") or "none"),
            "temporal_stability": int(getattr(c, "temporal_stability", 0) or 0),
            "roi_dilate": int(getattr(c, "roi_dilate", 0)),
            "use_seg_masks": bool(getattr(c, "use_seg_masks", True)),
            "detection": {
                "model_path": c.detection_model,
                "conf_threshold": float(c.detection_score),
                "iou_threshold": float(getattr(c, "det_iou", 0.70)),
                "imgsz": int(getattr(c, "det_imgsz", 640)),
                "fp16": bool(getattr(c, "detection_fp16", True)),
                "batch_size": int(c.detection_batch_size),
            },
            "restoration": {
                "rest_model_path": c.restoration_model,
                "fp16": bool(getattr(c, "restoration_fp16", True)),
                "max_clip_length": int(c.max_clip_size),
                "backend": ("trt" if c.use_trt else "pytorch"),
                "feather_radius": int(getattr(c, "feather_radius", 0)),
                "blendmask": str(getattr(c, "blendmask", "none")),
            },
            "encoder": {
                "codec": str(c.codec),
                "preset": str(c.preset),
                "qp": int(c.qp),
                "async_encoder": bool(getattr(c, "async_encoder", False)),
                "gpu_id": self.gpu_id,
            },
            "decoder": {"gpu_id": self.gpu_id},
        }
        # CM-093 X5c: monitoring section. The GUI path never populated one,
        # so run()'s read of monitoring.watchdog_stall_seconds ALWAYS fell
        # back to its 120s default -- only the headless CLI flag
        # (--watchdog-stall-seconds) ever tuned the watchdog. Read the flat
        # ChitraMaya-config.json (the UI-state file) for an optional
        # "watchdogStallSeconds" number so a one-line config entry works for
        # GUI runs too. Anchored the same way the server anchors that file:
        # the exe's own dir when frozen, cwd when running from source.
        # <=0 disables the watchdog (existing StallWatchdog contract).
        # Batch 36r2: load the flat UI-state file ONCE; both the watchdog
        # threshold and the CM-084 FrameStore knobs below read from it.
        _flat: dict = {}
        try:
            import json as _json
            _base = (Path(_sys.executable).parent
                     if getattr(_sys, "frozen", False) else Path.cwd())
            _cfg_file = _base / "ChitraMaya-config.json"
            if _cfg_file.exists():
                _loaded = _json.loads(_cfg_file.read_text(encoding="utf-8"))
                if isinstance(_loaded, dict):
                    _flat = _loaded
        except Exception:
            _flat = {}

        _wd_stall = 120.0
        try:
            if "watchdogStallSeconds" in _flat:
                _wd_stall = float(_flat["watchdogStallSeconds"])
                print(f"[Pipeline] watchdog stall threshold: "
                      f"{_wd_stall:.0f}s (from ChitraMaya-config.json)")
        except Exception:
            _wd_stall = 120.0
        data["monitoring"] = {"watchdog_stall_seconds": _wd_stall}
        # CM-180: run report switch (flat file, hand-edit channel) + the
        # panel this call carries.
        try:
            from chitramaya.run_report import normalize_mode as _rr_mode
            data["runReport"] = _rr_mode(_flat.get("runReport", "beside"))
            if str(_flat.get("runReport", "beside")).lower() != data["runReport"]:
                print(f"[Pipeline] WARNING: ignoring invalid runReport {_flat.get('runReport')!r} "
                      f"in ChitraMaya-config.json (use beside, temp, or off).")
            # CM-202: the panel's "Run files" dropdown wins whenever it says
            # something explicit (same precedence as the store backend).
            _pr = str(getattr(self.config, "run_report", "") or "").strip().lower()
            if _pr in ("beside", "temp", "off"):
                data["runReport"] = _pr
        except Exception:
            data["runReport"] = "beside"
        data["run_panel"] = getattr(self, "_pending_panel", None)
        # Segment previews / Test Frame set write_diagnostics=False: no report
        # for temp outputs (the flag existed before but nothing read it).
        if not bool(getattr(self.config, "write_diagnostics", True)):
            data["runReport"] = "off"
        _td = str(getattr(self.config, "temp_dir", "") or "")
        if _td:
            data["temp_dir"] = _td

        # CM-084 (Batch 36r2): FrameStore knobs from the SAME flat file.
        # to_pipeline_config() (the UI path) carries no store_* fields, so
        # without this a GUI user could never reach them -- the exact gap
        # X5c fixed for watchdogStallSeconds, repeated. File value wins
        # over the dataclass default; the CLI still wins over the file
        # because headless runs build their Config from CLI args directly.
        try:
            # Precedence (Batch 38): the LIVE UI dropdown (arriving via the
            # dataclass, already in data) wins whenever it says something
            # explicit; the flat-file "storeBackend" key applies only when
            # the UI value is "auto" -- it remains the hand-edit channel
            # (the dropdown persists under its own control id, so these
            # two never collide).
            _sb = _flat.get("storeBackend")
            if (_sb is not None
                    and str(data.get("store_backend", "auto")).lower() == "auto"):
                _sb = str(_sb).strip().lower()
                if _sb in ("auto", "device", "host", "redecode"):
                    data["store_backend"] = _sb
                    print(f"[FrameStore] backend request: {_sb} "
                          f"(from ChitraMaya-config.json)")
                else:
                    print(f"[FrameStore] WARNING: ignoring invalid "
                          f"storeBackend {_sb!r} in ChitraMaya-config.json "
                          f"(use auto, redecode, device, or host).")
        except Exception:
            pass
        # CM-196 T10f: opt-in torch cache release after drains.
        try:
            _vcr = _flat.get("vramCacheRelease")
            if _vcr is not None:
                data["vram_cache_release"] = bool(_vcr)
                print(f"[Pipeline] VRAM cache release after drains: {bool(_vcr)} "
                      f"(from ChitraMaya-config.json)")
        except Exception:
            pass
        # CM-196: where the re-decode path parks pending patches (host RAM
        # by default; "device" = VRAM, the T10c behaviour, for A/B). Hand-edit
        # key only; the run report's `effective.redecode_patches` says what ran.
        try:
            _rp = _flat.get("redecodePatches")
            if _rp is not None:
                _rp = str(_rp).strip().lower()
                if _rp in ("host", "device"):
                    data["redecode_patches"] = _rp
                    print(f"[FrameStore] redecode patches: {_rp} "
                          f"(from ChitraMaya-config.json)")
                else:
                    print(f"[FrameStore] WARNING: ignoring invalid "
                          f"redecodePatches {_rp!r} in ChitraMaya-config.json "
                          f"(use host or device).")
        except Exception:
            pass
        try:
            _smf = _flat.get("storeMaxFrames")
            if _smf is not None:
                data["store_max_frames"] = int(_smf)
                print(f"[FrameStore] max frames: {int(_smf)} "
                      f"(from ChitraMaya-config.json)")
        except Exception:
            pass
        # Batch 42: PyTorch temporal-window cap from the same flat file
        # (hand-edit channel, same pattern as watchdogStallSeconds).
        # 0 = whole clip (default, lada semantics); N>0 caps the window
        # (low-VRAM safety valve; 32 = pre-Batch-42 behavior).
        try:
            _rcf = _flat.get("restoreChunkFrames")
            if _rcf is not None:
                data["restoration"]["chunk_frames"] = int(_rcf)
                print(f"[Restorer] chunk frames: {int(_rcf)} "
                      f"(from ChitraMaya-config.json)")
        except Exception:
            pass
        # Batch 44: opt-in per-frame ROI dump for tools/ab_eval.py.
        try:
            _dr = _flat.get("detDumpRois")
            if _dr is not None:
                data["detection"]["dump_rois"] = bool(_dr)
                print(f"[Detector] ROI dump: {bool(_dr)} "
                      f"(from ChitraMaya-config.json)")
        except Exception:
            pass
        # Mask Preview maps to mode="pseudo": the host builds the detector +
        # PseudoClipRestorer (flat fill), so detection + compositing run but
        # BasicVSR++ does not. mode="real" restores normally.
        return Config(data=data)

    # -- live config update (server applies non-load-affecting deltas) -------

    def apply_runtime_config(self, cfg: MosaicPipelineConfig) -> None:
        """Apply a non-model-reloading config delta to the warm pipeline.

        The server calls this implicitly by reassigning .config then mutating
        _discard_margin/_blend_frames/_detector. We also push the live
        detection score down to the attribute the detector actually reads.
        """
        self.config = cfg
        self._discard_margin = max(0, int(getattr(cfg, "temporal_overlap", 0)))
        self._blend_frames = max(0, int(getattr(cfg, "blend_frames", 0)))
        self._set_detection_score(float(cfg.detection_score))

    def _set_detection_score(self, score: float) -> None:
        """Set the live confidence threshold on the warm detector.

        The detector reads its threshold from `detector.model.conf` at detect
        time (ChitraMaya/mosaic/detector/yolo.py: self.conf, read as conf_thres).
        The server historically wrote `_detector.score_threshold`, which the
        detector never reads — so live score changes were silently ignored.
        This sets the attribute that is actually read.
        """
        det = self._detector
        if det is None:
            return
        model = getattr(det, "model", None)
        if model is not None and hasattr(model, "conf"):
            try:
                model.conf = float(score)
            except Exception:
                pass

    # -- run -----------------------------------------------------------------

    def process_file(
        self,
        input_path: str,
        output_path: str,
        *,
        progress_cb=None,
        use_tqdm: bool = False,
        cancel_flag=None,
        foi_capture=None,
        panel=None,
    ) -> MosaicResult:
        """Process one file using the warm models. Returns a MosaicResult.

        foi_capture: optional dict with 'target_frame'; when given, run() fills
        it with that frame's boxes + pre/post-composite tensors (FOI preview).
        panel (CM-180): the control panel as the UI submitted it (preset
        schema); lands verbatim in the run report.
        """
        # Rebuild the host's config for this input/output, re-running
        # __post_init__ so paths/knobs are picked up, then drive run() with the
        # warm models injected. Reusing the same Pipeline instance keeps the
        # builders/device consistent; only cfg-derived fields change.
        self._pending_panel = panel if isinstance(panel, dict) else None
        self._host.cfg = self._build_base_config(input_path, output_path)
        self._host.__post_init__()

        # Push current live detection score before running.
        self._set_detection_score(float(self.config.detection_score))

        metrics = self._host.run(
            detector_override=self._detector,
            restorer_override=self._restorer,
            progress_cb=progress_cb,
            cancel_flag=cancel_flag,
            foi_capture=foi_capture,
        )

        # CM-180: the run report path (None when runReport = off)
        diag_path = getattr(self._host, "last_report_path", None)

        if metrics is None:
            return MosaicResult()
        return MosaicResult(
            frames=int(metrics.processed_frames),
            detections=int(metrics.det_stats.frames_with_det),
            restorations=int(len(metrics.frames_restored)),
            diag_path=diag_path,
        )

    def partial_result(self) -> Optional[dict]:
        """CM-107 (Batch 50): what the last run salvaged, or None.

        When run() raises, the pipeline may still have flushed and remuxed
        a playable partial output (field case 2026-08-19: 135,584 frames /
        37 minutes survived an NVENC error-8 death). The server calls this
        from its except path to report "PARTIAL: encoded M of N" instead
        of a bare error. Keys: errored, frames_encoded, total_frames,
        output_ok, output_path."""
        host = getattr(self, "_host", None)
        rec = getattr(host, "last_run_salvage", None)
        return dict(rec) if isinstance(rec, dict) else None

    def close(self) -> None:
        """Release warm models/host and reclaim their GPU memory. Idempotent.

        Dropping the Python refs alone does NOT free VRAM: the detector and
        restorer hold TensorRT engines + execution contexts whose GPU
        allocations are released only when the objects are destroyed. TRT
        wrappers commonly form reference cycles, so those destructors run on a
        cyclic GC pass, not on a refcount drop. Without forcing it, each
        config-change rebuild leaks ~2 GB (detection context + restoration
        sub-engines), degrading restore speed run-over-run and eventually
        OOM-hanging on an 8 GB card. So: drop refs, force a GC to run the
        destructors, then return freed blocks to the driver (the same pattern
        used between batch engine compiles).
        """
        self._detector = None
        self._restorer = None
        self._host = None
        try:
            import gc
            gc.collect()                 # run TRT/torch destructors (breaks cycles)
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.empty_cache()      # CM-093 (no xpu ipc_collect)
        except Exception as e:
            print(f"[mosaic] WARNING: GPU cleanup during pipeline close failed: {e}")
