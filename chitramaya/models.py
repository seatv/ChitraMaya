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

    # Restoration
    mosaic_max_clip_size: int = 90
    mosaic_temporal_overlap: int = 8
    mosaic_crossfade: bool = True
    mosaic_blend_frames: int = -1       # -1 = auto (temporal_overlap // 3 if crossfade else 0)

    # Spatial denoising on restored crops. Currently a no-op placeholder
    # — UI control exists, pipeline support added in a tuning session.
    mosaic_denoise: str = "none"        # none | low | medium | high

    # Modes / precision
    mosaic_fp16: bool = True
    mosaic_compile_trt: bool = True     # whether TRT path is used; surfaces as "Compile TRT" toggle
    mosaic_detect_only: bool = False    # Show Mask Only — overlay + skip restoration

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
            detect_only=bool(self.mosaic_detect_only),
            fp16=bool(self.mosaic_fp16),
            use_trt=bool(self.mosaic_compile_trt),
            color_match=bool(self.mosaic_color_match),
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
