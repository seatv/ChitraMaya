# ChitraMaya

A TensorRT-accelerated mosaic restoration studio with a real-time visual editor. Load a video, preview the restoration on your actual frames, and decide what to commit before spending time on a full encode.

## Why ChitraMaya?

Most restoration tools are batch processors: you set parameters, run a full pass, look at the result, and repeat until it's right. ChitraMaya is built the other way around — as an **interactive editor**.

- **Live segment preview.** Mark a segment, preview just that range, and decide whether to commit to a full run and save — before committing to a complete encode.
- **Visual feedback.** Detection and restoration controls update against your actual frames, so you can *see* what each adjustment does instead of inferring it from a number.
- **Hardware-accelerated throughout.** NVDEC decode, TensorRT-accelerated BasicVSR++ restoration, and NVENC encode keep frames resident on the GPU end to end — the preview is fast enough to actually iterate in.
- **VRAM-only pipeline.** No host round-trips during processing; everything stays on the GPU from decode through encode.

## Architecture

```
Decode (NVDEC) → Detect (YOLO) → Track scenes → Restore (BasicVSR++ / TensorRT) → Composite → Encode (NVENC)
```

ChitraMaya restores mosaic-obscured regions in video. A YOLO detector locates regions per frame, a scene tracker groups them into temporally coherent clips, and a BasicVSR++ restoration model (run through TensorRT sub-engines, or a PyTorch fallback) reconstructs each clip with temporal consistency. Restored regions are composited back with crossfade blending at clip boundaries.

## Prerequisites

- **GPU:** NVIDIA GPU with Turing architecture or newer (RTX 20xx+)
- **OS:** Windows 10/11 or Linux
- **Python:** 3.11 or 3.12
- **CUDA:** 12.x with matching cuDNN
- **TensorRT:** 10.x
- **ffmpeg:** on the system PATH (for audio remux)

## Installation

```bash
# Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux
source venv/bin/activate

# Install dependencies and the package
pip install -r requirements.txt
pip install -e .
```

Models are downloaded separately and placed in `models/` (see below) — they are not shipped with the repository.

## Models

Place detection and restoration model files in the `models/` directory. TensorRT engines are compiled per GPU architecture and cached in `models/engines/`.

| Type | Purpose |
|---|---|
| Mosaic detection (YOLO `.pt`) | Locate mosaic regions per frame |
| Mosaic restoration (BasicVSR++ `.pth`) | Reconstruct detected regions with temporal consistency |

Compile the TensorRT engines after placing the models:

```bash
# Build the BasicVSR++ restoration sub-engines
chitramaya -compile-rest --rest-model models/<restoration_model>.pth --rest-max-clip-length 60

# Build the YOLO detection engine
chitramaya -compile-det --det-model models/<detection_model>.pt --det-imgsz 640
```

## Usage

### Interactive UI (default)

```bash
chitramaya
```

Launches the desktop/web UI. Load a video, select your detection and restoration models, preview a segment, and run the restoration.

### Command line

```bash
chitramaya -restore \
    --input video.mp4 \
    --output restored.mp4 \
    --det-model models/<detection_model>.pt \
    --rest-model models/<restoration_model>.pth \
    --rest-max-clip-length 60 \
    --rest-backend trt
```

Useful flags:

| Flag | Description |
|---|---|
| `--rest-backend` | `trt` (TensorRT sub-engines) or `pytorch` (fallback, no precompiled engines needed) |
| `--rest-max-clip-length` | Frames per restoration clip (must match the compiled engine) |
| `--det-conf` | Detection confidence threshold |
| `--det-imgsz` | Detector input size |
| `--enc-codec` | Output codec: `hevc` or `h264` |
| `--enc-qp` | Encoder quantization parameter (lower = higher quality) |
| `--max-frames` | Process at most N frames (debug) |

Run `chitramaya -restore --help` for the full option list.

## License

See [LICENSE](LICENSE) for details.
