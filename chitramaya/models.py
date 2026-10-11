"""Core data models for the mosaic restoration pipeline.

All models are frozen dataclasses:
  - Frozen dataclasses as single source of truth for defaults
  - JSON round-trip support via to_dict/from_dict
  - Config file merges on top of dataclass defaults

Domain objects:
  - MosaicConfig:    User-facing mosaic restoration configuration
  - EncoderConfig:   NVENC encoding parameters
  - StreamInfo:      Probed video stream metadata
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional


# ── Enumerations ──────────────────────────────────────────────────────────

class InferenceBackend(Enum):
    """Model inference runtime."""
    TENSORRT = auto()      # Pure TensorRT engine
    ORT_TRT = auto()       # ONNX Runtime with TensorRT EP
    ORT_CUDA = auto()      # ONNX Runtime with CUDA EP (fallback)


# ── Stream information ────────────────────────────────────────────────────

@dataclass(frozen=True)
class StreamInfo:
    """Metadata extracted from a video stream."""
    path: Path
    width: int
    height: int
    fps: float
    num_frames: int
    duration: float
    codec: str = ""
    bitrate: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["path"] = str(self.path)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StreamInfo:
        data = dict(data)
        data["path"] = Path(data["path"])
        return cls(**data)


# ── Detection configuration ──────────────────────────────────────────────

# ── Swap parameters ──────────────────────────────────────────────────────

# ── Mask configuration ────────────────────────────────────────────────────

# ── Restorer configuration ────────────────────────────────────────────────

# ── Encoder configuration ────────────────────────────────────────────────

@dataclass(frozen=True)
class EncoderConfig:
    """NVENC encoding parameters."""
    codec: str = "hevc"
    preset: str = "P5"
    profile: str = ""
    qp: int = 18
    gpu_id: int = 0
    mux_audio: bool = True


# ── Top-level pipeline configuration ──────────────────────────────────────

# ── Mosaic restoration configuration ──────────────────────────────────────


def _hex_to_rgb(hexstr: str, default=(255, 0, 255)) -> list:
    """Convert '#RRGGBB' (or '#RGB') to an [r, g, b] list for the pipeline's
    RGB visualization.fill_color. Falls back to magenta on bad input."""
    try:
        s = str(hexstr).strip().lstrip("#")
        if len(s) == 3:
            s = "".join(c * 2 for c in s)
        return [int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)]
    except Exception:
        return list(default)


@dataclass(frozen=True)
class MosaicConfig:
    """User-facing configuration for the mosaic restoration pipeline.

    Flat fields for JSON round-trip, ``from_dict``
    filters unknown keys, ``to_pipeline_config`` produces the runtime
    ``MosaicPipelineConfig`` consumed by ``ChitraMaya.mosaic.pipeline``.
    """
    # Paths
    detection_model: str = ""           # e.g. ./models/lada_mosaic_detection_model_v4_accurate.pt
    restoration_model: str = ""         # e.g. ./models/lada_mosaic_restoration_model_generic_v1.2.pth

    # Detection
    # T20b: 0.20 and batch 8 = the recommended set baked in (Gman's field practice; 8 is
    # what the standard-set button compiles for). A run clamps the batch to its engine.
    mosaic_detection_score: float = 0.20
    mosaic_detection_batch_size: int = 8
    mosaic_det_imgsz: int = 640         # detector input size (multiple of 32); 960 for dense VR
    mosaic_iou: float = 0.70            # detection NMS IoU threshold
    mosaic_detection_fp16: bool = True  # detection engine precision
    mosaic_detection_trt: bool = True   # use TRT .engine for detection if available (else .pt)

    # SBS / stereo (side-by-side): one switch. When on, the frame is split at
    # the midline, each eye detected separately then merged, and restorations
    # are kept from crossing the seam. Layout (lr/rl) is irrelevant — the
    # split/merge is position-preserving, so we always split at w//2.
    mosaic_sbs_split: bool = False

    # CM-045 VR projection: "none" | "fisheye". With "fisheye", each eye is
    # warped hequirect->fisheye for detection/tracking/restoration (studios
    # that mosaic in viewing space), and restored regions are inverse-warped
    # back onto the original frames. Requires mosaic_sbs_split.
    mosaic_vr_projection: str = "none"

    # Restoration
    mosaic_max_clip_size: int = 90
    mosaic_temporal_overlap: int = 8
    mosaic_crossfade: bool = True
    mosaic_blend_frames: int = -1       # -1 = auto (temporal_overlap // 3 if crossfade else 0)
    mosaic_restoration_fp16: bool = True  # restoration engine precision
    mosaic_roi_dilate: int = 0          # grow detected ROI by N px before restoring (0 = off)
    mosaic_crop_cap: int = 0            # CM-227: open-scene crop storage cap, px on the long side (0 = off)
    mosaic_feather_radius: int = 0      # Face Swap feather px; 0 = auto (CM-214: 2.8 % of frame height)
    mosaic_blend_mask: str = "facefusion"   # compositor blend-mask mode: none | facefusion (CM-214: default since 1.80)
    mosaic_use_seg_masks: bool = True   # per-pixel seg masks vs bounding boxes
    # CM-205: after a real run, the detector scans the OUTPUT's key frames and
    # reports where mosaic is still visible (console + run report).
    mosaic_residual_scan: bool = True

    # CM-077 secondary restoration: "none" | "rtx-2x" | "rtx-4x". Upscales
    # restored crops with NVIDIA Maxine RTX Super-Res before paste-back, so
    # regions larger than the 256 clip size are downscaled into place instead
    # of stretched. Real restoration mode only; needs an RTX GPU + nvidia-vfx.
    # T20b: ON by default -- the field verdict ('RTXSR4 is still the king') sat behind
    # an Off default for every newcomer. Off-CUDA the server's default-config route
    # hands out esrgan-4x instead (CM-184 also maps rtx-* -> esrgan-4x at run time).
    mosaic_secondary: str = "rtx-4x"
    # Batch 70 (CM-146): Maxine denoise pass chained after the RTX
    # secondary's upscale (none|low|medium|high|ultra). RTX modes only;
    # ignored by the Real-ESRGAN secondary.
    mosaic_secondary_denoise: str = "none"
    # CM-217: Maxine Super-Res quality level for the RTX secondary
    # (low|medium|high|ultra; high = the pre-1.80 hardcoded value).
    mosaic_secondary_quality: str = "high"
    # CM-219 / CM-228: whole-frame upscale before encode -- a factor (2x |
    # 4x) or a target (1080p | 1440p | 2160p). Real restoration and the
    # upscale-only mode.
    mosaic_output_upscale: str = "none"
    # CM-228: which scaler does the whole frame: esrgan (Real-ESRGAN, all
    # GPUs) | rtx (Maxine VideoSuperRes, NVIDIA; falls back to esrgan).
    mosaic_upscale_scaler: str = "esrgan"
    # CM-228: Upscale only -- decode -> scale -> encode, no detector, no
    # restorer, no tracker (pipeline mode "none"). For SFW content you just
    # want bigger.
    mosaic_upscale_only: bool = False

    # CM-078: temporal stabilization of restored crops (vs_temporalfix).
    # 0 = off, 1..3 = strength (higher = more aggressive smoothing).
    # T20b: 1 by default (Gman: 'either 1 or 3, two was most expensive'; 3 may look
    # better and takes longer -- the tooltip says so).
    mosaic_temporal_stability: int = 1

    # CM-202 (GitHub #10): the suffix appended to the output file name in
    # restoration mode ("-restored" by default; "-censored"/"-mask" for the
    # other modes stay fixed) and the per-run files switch: "beside" (the
    # .run.json / .log / .timecodes.txt next to the video), "temp" (in the
    # Temp folder) or "off" (none written). "" = follow ChitraMaya-config.json.
    mosaic_output_suffix: str = "-restored"
    mosaic_run_report: str = ""

    # CM-084 (Batch 38): FrameStore backend -- "auto" | "device" | "host".
    # auto keeps frames in VRAM when they fit, offloads to system RAM when
    # the projected store would not (the long-Max-Clip enabler).
    mosaic_store_backend: str = "auto"

    # T20d (CM-234): Hardware decode for the ffmpeg lane (AMD / Intel; NVIDIA
    # only when NVDEC declines a stream) -- "auto" (probe qsv -> d3d11va ->
    # CPU) | "off" (CPU decode). The chroma guard repairs the d3d11va
    # download fault either way; Off is the way to avoid it outright.
    mosaic_hw_decode: str = "auto"

    # Spatial denoising on restored crops. Currently a no-op placeholder
    # (UI control removed; pipeline support added in a tuning session).
    mosaic_denoise: str = "none"        # none | low | medium | high

    # Modes / precision
    mosaic_restoration_trt: bool = True     # whether TRT restoration path is used; surfaces as restoration "Use Tensor"

    # Mask Preview (pseudo mode): flat-fill detected regions, skip restoration.
    # Fast coverage check — any leftover mosaic is a detection miss.
    mosaic_mask_preview: bool = False
    mosaic_mask_color: str = "#FF00FF"  # hex; converted to RGB for visualization.fill_color
    mosaic_mask_opacity: float = 0.70

    # Auto-censor (Auto Mosaic): pixelate detected regions instead of restoring.
    mosaic_censor: bool = False
    mosaic_censor_block: int = 16

    # Post
    mosaic_color_match: bool = False

    def to_pipeline_config(self, *, encoder: dict[str, Any] | None = None):
        """Build the runtime config the ``MosaicPipeline`` consumes.

        ``encoder`` is the shared encoder params dict (codec/preset/qp) used
        by face-swap too — there's a single source of truth in the UI's
        Encoder section. If omitted, sensible defaults apply.
        """
        from chitramaya.mosaic.pipeline import MosaicPipelineConfig
        enc = encoder or {}
        return MosaicPipelineConfig(
            detection_model=self.detection_model,
            restoration_model=self.restoration_model,
            detection_score=float(self.mosaic_detection_score),
            detection_batch_size=int(self.mosaic_detection_batch_size),
            det_imgsz=int(self.mosaic_det_imgsz),
            max_clip_size=int(self.mosaic_max_clip_size),
            temporal_overlap=int(self.mosaic_temporal_overlap),
            crossfade=bool(self.mosaic_crossfade),
            blend_frames=int(self.mosaic_blend_frames),
            mask_preview=bool(self.mosaic_mask_preview),
            mask_color=_hex_to_rgb(self.mosaic_mask_color),
            mask_opacity=float(self.mosaic_mask_opacity),
            censor=bool(self.mosaic_censor),
            censor_block=int(self.mosaic_censor_block),
            detection_fp16=bool(self.mosaic_detection_fp16),
            restoration_fp16=bool(self.mosaic_restoration_fp16),
            use_trt=bool(self.mosaic_restoration_trt),
            color_match=bool(self.mosaic_color_match),
            det_iou=float(self.mosaic_iou),
            roi_dilate=int(self.mosaic_roi_dilate),
            crop_cap_px=int(self.mosaic_crop_cap),          # CM-227
            use_seg_masks=bool(self.mosaic_use_seg_masks),
            residual_scan=bool(self.mosaic_residual_scan),
            feather_radius=int(self.mosaic_feather_radius),
            blendmask=str(self.mosaic_blend_mask),
            sbs_enabled=bool(self.mosaic_sbs_split),
            sbs_layout="lr",
            sbs_det_split=bool(self.mosaic_sbs_split),
            vr_projection=str(self.mosaic_vr_projection or "none").lower(),
            secondary_restoration=(str(self.mosaic_secondary) if str(self.mosaic_secondary or "").startswith("custom:")
                                   else str(self.mosaic_secondary or "none").lower()),   # CM-229: a file name keeps its case
            secondary_denoise=str(self.mosaic_secondary_denoise or "none").lower(),
            secondary_quality=str(self.mosaic_secondary_quality or "high").lower(),   # CM-217
            output_upscale=str(self.mosaic_output_upscale or "none").lower(),         # CM-219
            upscale_scaler=(str(self.mosaic_upscale_scaler) if str(self.mosaic_upscale_scaler or "").startswith("custom:")
                            else str(self.mosaic_upscale_scaler or "esrgan").lower()),   # CM-228 / CM-229
            upscale_only=bool(self.mosaic_upscale_only),                                # CM-228
            temporal_stability=int(self.mosaic_temporal_stability or 0),
            store_backend=str(self.mosaic_store_backend or "auto").lower(),
            hw_decode=str(self.mosaic_hw_decode or "auto").lower(),          # T20d
            run_report=str(self.mosaic_run_report or "").strip().lower(),   # CM-202
            codec=str(enc.get("codec", "hevc")),
            preset=str(enc.get("preset", "P5")),
            qp=int(enc.get("qp", 18)),
            async_encoder=bool(enc.get("async_encoder", False)),
            write_diagnostics=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MosaicConfig:
        import dataclasses
        valid = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)


# ── Source face ───────────────────────────────────────────────────────────

# ── Model manifest ───────────────────────────────────────────────────────

# ── Mosaic Pipeline Config ────────────────────────────────────────────────

@dataclass(frozen=True)
class MosaicDetectionConfig:
    """Configuration for YOLO mosaic detection."""
    model_name: str = "yolo-v4-accurate"
    score_threshold: float = 0.25
    iou_threshold: float = 0.7
    batch_size: int = 4
    imgsz: int = 640
    fp16: bool = True

@dataclass(frozen=True)
class MosaicRestorationConfig:
    """Configuration for BasicVSR++ mosaic restoration."""
    max_clip_size: int = 90
    temporal_overlap: int = 8
    crossfade: bool = True
    denoise: str = "none"  # none, low, medium, high
    fp16: bool = True
    restoration_size: int = 256
