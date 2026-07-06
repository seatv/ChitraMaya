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
    mosaic_detection_score: float = 0.25
    mosaic_detection_batch_size: int = 4
    mosaic_iou: float = 0.70            # detection NMS IoU threshold
    mosaic_detection_fp16: bool = True  # detection engine precision
    mosaic_detection_trt: bool = True   # use TRT .engine for detection if available (else .pt)

    # Restoration
    mosaic_max_clip_size: int = 90
    mosaic_temporal_overlap: int = 8
    mosaic_crossfade: bool = True
    mosaic_blend_frames: int = -1       # -1 = auto (temporal_overlap // 3 if crossfade else 0)
    mosaic_restoration_fp16: bool = True  # restoration engine precision
    mosaic_roi_dilate: int = 0          # grow detected ROI by N px before restoring (0 = off)
    mosaic_feather_radius: int = 0      # soften restored-region mask edge by N px (0 = hard)
    mosaic_blend_mask: str = "none"     # compositor blend-mask mode: none | facefusion
    mosaic_use_seg_masks: bool = True   # per-pixel seg masks vs bounding boxes

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
            max_clip_size=int(self.mosaic_max_clip_size),
            temporal_overlap=int(self.mosaic_temporal_overlap),
            crossfade=bool(self.mosaic_crossfade),
            blend_frames=int(self.mosaic_blend_frames),
            mask_preview=bool(self.mosaic_mask_preview),
            mask_color=_hex_to_rgb(self.mosaic_mask_color),
            mask_opacity=float(self.mosaic_mask_opacity),
            detection_fp16=bool(self.mosaic_detection_fp16),
            restoration_fp16=bool(self.mosaic_restoration_fp16),
            use_trt=bool(self.mosaic_restoration_trt),
            color_match=bool(self.mosaic_color_match),
            det_iou=float(self.mosaic_iou),
            roi_dilate=int(self.mosaic_roi_dilate),
            use_seg_masks=bool(self.mosaic_use_seg_masks),
            feather_radius=int(self.mosaic_feather_radius),
            blendmask=str(self.mosaic_blend_mask),
            codec=str(enc.get("codec", "hevc")),
            preset=str(enc.get("preset", "P5")),
            qp=int(enc.get("qp", 18)),
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
