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
from chitramaya.mosaic.utils.config_util import Config
from chitramaya.video.decoder import Decoder
from chitramaya.video.encoder import Encoder

from .pipeline_utils import (
    Box,
    FrameStore,
    bgr_u8_to_bgra_u8,
    clip_box_to_bounds,
    cfg_first,
    cfg_path,
    compute_pts_fps,
    drain_store_to_encoder,
    drain_store_to_async_encoder,
    AsyncEncoder,
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

    def add(self, boxes, w: int, h: int, frame_num: int = -1) -> None:
        self.frames_total += 1
        if self.frame_area_px == 0:
            self.frame_area_px = int(w) * int(h)
        if not boxes:
            return

        self.frames_with_det += 1
        if frame_num >= 0:
            self.frames_with_det_set.add(int(frame_num))
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

    def _build_detector(self):
        if self.mode == "none":
            return None
        if not self.det_model:
            raise FileNotFoundError("Detector model path is empty (check config.json or --det-model)")

        det_type = str(self.cfg.get("detection", "det_type", default="yolo") or "yolo").lower()
        print(
            f"[Detector] type={det_type} imgsz={self.det_imgsz} "
            f"conf={self.det_conf} iou={self.det_iou} fp16={self.det_fp16}"
        )

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

        if not self.rest_model:
            raise FileNotFoundError("Restoration model path is empty (check config.json or --rest-model)")

        # Prefer TensorRT sub-engines if they exist for this checkpoint +
        # precision + max_clip_size. Falls back to PyTorch on any failure
        # so a partial/incompatible engine set never blocks restoration.
        if str(self.rest_backend).lower() != "pytorch":
            try:
                from chitramaya.mosaic.models.basicvsrpp.engine_paths import (
                    _basicvsrpp_sub_engine_dir,
                    all_basicvsrpp_sub_engines_exist,
                    list_basicvsrpp_compiled_clip_sizes,
                    pick_engine_clip_size,
                )

                # Engine filenames bind (precision, max_clip_size) at compile
                # time, but the compiled sets are DYNAMIC-batch (valid for
                # clips 1..N), so the compiled size only decides which FILES
                # to load — the user's request stays the RUNTIME clip
                # ceiling whenever a compiled set covers it. See
                # pick_engine_clip_size(). Without this resolution, a model
                # compiled at e.g. b90 hard-failed — or, worse, silently
                # fell back to PyTorch under backend=auto — just because
                # --rest-max-clip-length was omitted or different.
                req_mcl = int(self.rest_max_clip_length)
                engine_mcl = req_mcl
                if not all_basicvsrpp_sub_engines_exist(
                    self.rest_model, fp16=self.rest_fp16, max_clip_size=req_mcl,
                ):
                    avail = list_basicvsrpp_compiled_clip_sizes(
                        self.rest_model, self.rest_fp16,
                    )
                    pick = pick_engine_clip_size(avail, req_mcl)
                    if pick is not None:
                        engine_mcl, runtime_mcl = pick
                        if runtime_mcl == req_mcl:
                            print(
                                f"[Restorer] max_clip_length={req_mcl} has no "
                                f"exact engine set; loading the b{engine_mcl} "
                                f"set (dynamic, covers 1..{engine_mcl}) and "
                                f"running with max_clip_length={req_mcl} "
                                f"(compiled sizes: {avail})"
                            )
                        else:
                            # Only smaller sets are compiled — clips longer
                            # than the engine's ceiling cannot run, so the
                            # runtime ceiling caps. This happens BEFORE run()
                            # builds the tracker/store, so clip formation and
                            # the engines stay consistent.
                            print(
                                f"[Restorer] No TRT sub-engines compiled for "
                                f"max_clip_length>={req_mcl} "
                                f"(fp16={self.rest_fp16}); capping "
                                f"max_clip_length to {runtime_mcl} "
                                f"(compiled sizes: {avail})"
                            )
                            self.rest_max_clip_length = int(runtime_mcl)

                # VRAM-hint note, surfaced alongside run()'s VRAM warning: a
                # larger engine set reserves dramatically more activation
                # memory (observed: b180 + detection context left ~335 MB free
                # on an 8 GB card at 4K). If smaller compiled sets exist,
                # tell the user exactly which Max Clip selects them.
                self._rest_engine_note = ""
                try:
                    _avail_all = list_basicvsrpp_compiled_clip_sizes(
                        self.rest_model, self.rest_fp16,
                    )
                    _smaller = [n for n in _avail_all if n < int(engine_mcl)]
                    if _smaller:
                        self._rest_engine_note = (
                            f" Note: restoration engine set b{int(engine_mcl)} is in "
                            f"use; smaller compiled set(s) {_smaller} exist — a max "
                            f"clip length <= {max(_smaller)} selects b{max(_smaller)}, "
                            f"which reserves substantially less VRAM."
                        )
                except Exception:
                    pass

                if all_basicvsrpp_sub_engines_exist(
                    self.rest_model,
                    fp16=self.rest_fp16,
                    max_clip_size=int(engine_mcl),
                ):
                    from chitramaya.mosaic.restorer.basicvsrpp_trt_clip_restorer import (
                        BasicVSRPPTRTClipRestorer,
                    )
                    print(
                        f"[Restorer] Using TensorRT sub-engines "
                        f"(engine_set=b{engine_mcl}, "
                        f"max_clip_length={self.rest_max_clip_length}, "
                        f"fp16={self.rest_fp16})"
                    )
                    return BasicVSRPPTRTClipRestorer(
                        model_path=self.rest_model,
                        device=self.device,
                        fp16=self.rest_fp16,
                        max_clip_size=int(engine_mcl),
                    )
                elif str(self.rest_backend).lower() == "trt":
                    # User explicitly asked for TRT but no engine set exists
                    # for this precision at ANY clip size. Say exactly what
                    # was looked for, where, and what IS on disk.
                    eng_dir = _basicvsrpp_sub_engine_dir(self.rest_model)
                    other = list_basicvsrpp_compiled_clip_sizes(
                        self.rest_model, not self.rest_fp16,
                    )
                    if other:
                        hint = (
                            f" Engines for fp16={not self.rest_fp16} DO exist "
                            f"(clip sizes {other}) — match the compiled "
                            f"precision with "
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
                        f"{self.rest_model} --rest-max-clip-length "
                        f"{req_mcl}` or pass "
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

        print("[Restorer] Using PyTorch BasicVSR++ (no sub-engines)")
        return BasicVSRPPClipRestorer(
            model_path=self.rest_model,
            device=self.device,
            fp16=self.rest_fp16,
            max_frames=32,
        )

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

        metrics = PipelineMetrics()
        metrics.wall_start = _dt.datetime.now()
        t0_all = time.perf_counter()

        decoder = Decoder(
            input_path=str(inp),
            gpu_id=self.dec_gpu_id,
            batch_size=self.batch_size,
            output_format=self.dec_output_format,
            ffmpeg_input_args=self.dec_ffmpeg_input_args,
            trim_negative_pts=False,
        )

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

        # ChitraMaya's Encoder has a slimmer signature than gRestorer's.
        # Convert mux_audio (str "auto/copy/aac/none") to ChitraMaya's bool flag.
        _mux_audio_bool = str(self.mux_audio).lower() != "none"
        if foi_capture is not None:
            # FOI preview: capture-only. Never encode or mux (see _DiscardEncoder).
            encoder = _DiscardEncoder()
        else:
            encoder = Encoder(
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

            # Wrap in AsyncEncoder if enabled — runs encode_frame() on a
            # background thread, overlapping NVENC work with the main thread's
            # decode/detect/restore/composite pass.
            if self.async_encoder:
                print(f"[Encoder] Async encoder thread enabled (queue_size={self.async_encoder_queue})")
                encoder = AsyncEncoder(encoder, device=self.device, queue_size=self.async_encoder_queue)

        # Warm-model injection (additive): the UI bridge passes pre-built
        # models so repeated previews don't reload. When not provided, build
        # exactly as before. analysis_use_synth_rois still forces no detector.
        if self.analysis_use_synth_rois:
            detector = None
        elif detector_override is not None:
            detector = detector_override
        else:
            detector = self._build_detector()
        restorer = restorer_override if restorer_override is not None else self._build_restorer()

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
        final_cap = requested_cap
        if requested_cap > 0:
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
                if store_is_auto and _reduced:
                    final_cap = _plan_frames
                    # Honest wording: the reduction may have bottomed out at the
                    # one-clip floor, which does NOT necessarily fit free VRAM —
                    # the warning below states the real situation.
                    print(
                        f"[FrameStore] VRAM-aware: reduced auto max_frames "
                        f"{requested_cap}->{final_cap} (free VRAM after models: "
                        f"~{_free_b // (1024*1024)} MB; backpressure can still bump "
                        f"if a scene needs it)"
                    )
                if _warn is not None:
                    print(f"[Pipeline] WARNING: {_warn}"
                          f"{getattr(self, '_rest_engine_note', '')}")
        store = FrameStore(max_frames=final_cap)

        if store.max_frames > 0:
            est_mb = (store.max_frames * w * h * 3) / (1024.0 * 1024.0)
            print(f"[FrameStore] max_frames={store.max_frames} (~{est_mb:.0f} MB VRAM budget)")
        else:
            print("[FrameStore] max_frames=unlimited")

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

        def consume_batch(batch: List[object], batch_pts: Optional[List[Optional[int]]] = None) -> None:
            nonlocal frame_num

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
            batch_rgb: List[torch.Tensor] = []

            for item in batch:
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

                    batch_rgb.append(rgb.contiguous())
                    continue

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

                batch_rgb.append(rgb.contiguous())

            metrics.t_prepare += (time.perf_counter() - t0_prep)

            # Convert RGB -> BGR uint8 once per frame (LADA parity + reuse everywhere)
            batch_bgr_u8: List[torch.Tensor] = [rgb_hwc_to_bgr_hwc_u8(rgb) for rgb in batch_rgb]

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
                    for bgr in batch_bgr_u8:
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
                    detections = detector.detect_batch(batch_bgr_u8)
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

                t0 = time.perf_counter()
                step = tracker.step_frame(frame_num, bgr_u8, boxes, masks_list)
                metrics.t_track += (time.perf_counter() - t0)

                if step.new_clips and restorer is not None:
                    for clip in step.new_clips:
                        t0 = time.perf_counter()
                        restored = restorer.restore_clip(clip)
                        composite_clip_into_store(
                            clip=clip,
                            restored_frames_u8=restored,
                            store_bgr_u8=store.frames_bgr_u8,
                            model_dtype=restorer.model_dtype,
                            blendmask=self.rest_blendmask,
                            feather_radius=self.feather_radius,
                        )
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

            # Drain once per batch (sync once per drain happens inside drain_store_to_encoder)
            t0 = time.perf_counter()
            drain_store_to_encoder(
                store=store,
                safe_before=int(safe_before_batch),
                encoder=encoder,
                device=self.device,
                sync_before_encode=self.enc_sync_before_encode,
                pts_log=pts_log,
            )
            metrics.t_encode += (time.perf_counter() - t0)

            # Progress emit (additive): once per batch, only if a callback was
            # provided. Reads existing metrics/state; does not affect behavior.
            if progress_cb is not None:
                _now = time.perf_counter()
                _pf = int(metrics.processed_frames)
                # windowed fps: frames since last emit / wall since last emit
                _prev_f = getattr(self, "_pcb_prev_frames", 0)
                _prev_t = getattr(self, "_pcb_prev_time", t0_all)
                _dt_win = _now - _prev_t
                _fps_win = ((_pf - _prev_f) / _dt_win) if _dt_win > 1e-6 else 0.0
                _dt_all = _now - t0_all
                _fps_avg = (_pf / _dt_all) if _dt_all > 1e-6 else 0.0
                self._pcb_prev_frames = _pf
                self._pcb_prev_time = _now
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
                    )
                except Exception:
                    # A misbehaving UI callback must never crash the pipeline.
                    pass

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
                    while store.is_full():
                        metrics.backpressure_waits += 1

                        # Drain frames that are guaranteed not to be touched by any active clip.
                        min_start = tracker.min_active_start() if tracker is not None else None
                        tracker_sb = int(min_start) if min_start is not None else int(frame_num + 1)
                        sb = tracker_sb

                        drain_store_to_encoder(
                            store=store,
                            safe_before=sb,
                            encoder=encoder,
                            device=self.device,
                            sync_before_encode=self.enc_sync_before_encode,
                            pts_log=pts_log,
                        )

                        if not store.is_full():
                            break

                        # If we cannot drain because the oldest stored frame is still within an
                        # active clip, bump the cap up to an emergency ceiling.
                        if store.max_frames > 0 and min_start is not None and len(store.frames_bgr_u8) > 0:
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
                    while store.is_full():
                        metrics.backpressure_waits += 1

                        # Drain frames that are guaranteed not to be touched by any active clip.
                        min_start = tracker.min_active_start() if tracker is not None else None
                        tracker_sb = int(min_start) if min_start is not None else int(frame_num + 1)
                        sb = tracker_sb

                        drain_store_to_encoder(
                            store=store,
                            safe_before=sb,
                            encoder=encoder,
                            device=self.device,
                            sync_before_encode=self.enc_sync_before_encode,
                            pts_log=pts_log,
                        )

                        if not store.is_full():
                            break

                        # If we cannot drain because the oldest stored frame is still within an
                        # active clip, bump the cap up to an emergency ceiling.
                        if store.max_frames > 0 and min_start is not None and len(store.frames_bgr_u8) > 0:
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
                        restored = restorer.restore_clip(clip)
                        composite_clip_into_store(
                            clip=clip,
                            restored_frames_u8=restored,
                            store_bgr_u8=store.frames_bgr_u8,
                            model_dtype=restorer.model_dtype,
                            blendmask=self.rest_blendmask,
                            feather_radius=self.feather_radius,
                        )
                        metrics.frames_restored.update(int(fn) for fn in clip.frame_nums)
                        metrics.clip_lengths.append(int(len(clip.frame_nums)))
                        if (_foi_target is not None
                                and int(_foi_target) in clip.frame_nums
                                and int(_foi_target) in store.frames_bgr_u8):
                            foi_capture["composited"] = (
                                store.frames_bgr_u8[int(_foi_target)].detach().clone()
                            )

                drain_store_to_encoder(
                    store=store,
                    safe_before=10**18,
                    encoder=encoder,
                    device=self.device,
                    sync_before_encode=self.enc_sync_before_encode,
                    pts_log=pts_log,
                )
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

            t_total_no_mux = time.perf_counter() - t0_all

            # [CHANGE 4] Write timecodes file and compute PTS-derived fps
            tc_path: Optional[str] = None
            pts_fps: float = fps
            is_vfr: bool = False
            if pts_log and foi_capture is None:
                pts_fps, is_vfr = compute_pts_fps(pts_log, fallback_fps=fps)
                tc_path = write_timecodes_v2(pts_log, self.output_path, fps=fps)

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
                    encoder.close()
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
                f"len_min={clip_min} len_med={clip_med} len_max={clip_max} len_mean={clip_mean}"
            )

            # Write misses JSON next to the output. Lists are sorted for
            # readability + reproducibility. The actionable list is
            # `visible_miss_frames` — feed those into a viewer or rerun
            # under `--mode pseudo` to inspect them visually.
            try:
                import json
                miss_path = Path(self.output_path).with_suffix(".misses.json")
                report = {
                    "video": str(self.input_path),
                    "output": str(self.output_path),
                    "total_frames": int(ft),
                    "settings": {
                        "mode": self.mode,
                        "det_model": str(self.det_model),
                        "det_imgsz": int(self.det_imgsz),
                        "det_conf": float(self.det_conf),
                        "det_iou": float(self.det_iou),
                        "det_fp16": bool(self.det_fp16),
                        "rest_model": str(self.rest_model),
                        "rest_max_clip_length": int(self.rest_max_clip_length),
                        "rest_fp16": bool(self.rest_fp16),
                        "trk_ttl_after_end": int(self.trk_ttl_after_end),
                        "trk_crop_quant_px": int(self.trk_crop_quant_px),
                        "trk_crop_sticky": bool(self.trk_crop_sticky),
                        "trk_match_pad_px": int(self.trk_match_pad_px),
                    },
                    "summary": {
                        "restored": int(fr),
                        "detected": int(len(det_set)),
                        "detected_and_restored": int(detected_and_restored),
                        "gap_fill_bonus": int(gap_fill_bonus),
                        "legit_passthrough": int(legit),
                        "visible_miss": int(visible_miss),
                        "restoration_miss": int(restoration_miss),
                        "early_passthrough_count": int(metrics.early_passthrough_frames),
                    },
                    "clips": {
                        "count": int(n_clips),
                        "total_clip_frames": int(total_clip_frames),
                        "len_min": int(clip_min),
                        "len_median": int(clip_med),
                        "len_max": int(clip_max),
                        "len_mean": float(clip_mean),
                    },
                    "visible_miss_frames": sorted(miss_set),
                    "restoration_miss_frames": sorted(restoration_miss_set),
                    "gap_fill_frames": sorted(gap_fill_set),
                    "legit_passthrough_frames": sorted(legit_set),
                    # NOTE: detected_frames and restored_frames lists were
                    # intentionally dropped — they're huge on long videos
                    # (~270k entries each for a 2.5hr clip) and redundant
                    # with the count summary above. Use the four lists
                    # above for any inspection / tooling needs.
                }
                with miss_path.open("w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2)
                print(f"[Pipeline] Misses report: {miss_path}")
            except Exception as e:
                # Don't fail the whole run if JSON write hits an issue.
                print(f"[Pipeline] WARNING: failed to write misses JSON: {e}")

            if metrics.wall_start and metrics.wall_end:
                elapsed = metrics.wall_end - metrics.wall_start
                print(f"[Pipeline] Wall clock: start={metrics.wall_start} end={metrics.wall_end} elapsed={elapsed}")
            print(f"[Pipeline] perf_counter elapsed = {t_total_with_mux:.2f}s")

            # [CHANGE 4] Cleanup timecodes file (kept only if VFR)
            if tc_path and not is_vfr:
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
    store_max_frames: int = 0
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
    "mask_opacity", "detection_fp16", "restoration_fp16", "use_trt",
    "codec", "preset", "qp", "async_encoder", "write_diagnostics",
    "sbs_enabled", "sbs_layout",
    "sbs_det_split", "store_max_frames", "det_imgsz", "det_iou", "roi_dilate",
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
        self._detector = self._host._build_detector()
        self._restorer = self._host._build_restorer()

    # -- config construction -------------------------------------------------

    def _build_base_config(self, input_path: str = "", output_path: str = "") -> Config:
        """Translate the flat MosaicPipelineConfig into the nested Config dict
        the proven Pipeline.__post_init__ reads. Mirrors cli_config keys."""
        c = self.config
        data: dict = {
            "input": str(input_path),
            "output": str(output_path),
            "mode": ("pseudo" if bool(getattr(c, "mask_preview", False)) else "real"),
            "visualization": {
                "fill_color": list(getattr(c, "mask_color", (255, 0, 255))),
                "fill_opacity": float(getattr(c, "mask_opacity", 0.70)),
            },
            "store_max_frames": int(getattr(c, "store_max_frames", 0)),
            "sbs_enabled": bool(getattr(c, "sbs_enabled", False)),
            "sbs_layout": str(getattr(c, "sbs_layout", "lr")),
            "sbs_det_split": bool(getattr(c, "sbs_det_split", False)),
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
    ) -> MosaicResult:
        """Process one file using the warm models. Returns a MosaicResult.

        foi_capture: optional dict with 'target_frame'; when given, run() fills
        it with that frame's boxes + pre/post-composite tensors (FOI preview).
        """
        # Rebuild the host's config for this input/output, re-running
        # __post_init__ so paths/knobs are picked up, then drive run() with the
        # warm models injected. Reusing the same Pipeline instance keeps the
        # builders/device consistent; only cfg-derived fields change.
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

        diag_path = None
        try:
            diag_path = str(Path(output_path).with_suffix(".misses.json"))
        except Exception:
            diag_path = None

        if metrics is None:
            return MosaicResult()
        return MosaicResult(
            frames=int(metrics.processed_frames),
            detections=int(metrics.det_stats.frames_with_det),
            restorations=int(len(metrics.frames_restored)),
            diag_path=diag_path,
        )

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
        except Exception as e:
            print(f"[mosaic] WARNING: GPU cleanup during pipeline close failed: {e}")
