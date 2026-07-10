# ChitraMaya

A TensorRT-accelerated mosaic restoration studio with a real-time visual editor. Load a video, preview the restoration on your *actual* frames, and decide what to commit before spending time on a full encode.

![ChitraMaya — the mosaic input and the restored result, side by side](docs/InAction.png)

## Why ChitraMaya?

Some restoration tools are batch processors: set parameters, run a full pass, look at the result, repeat. ChitraMaya is built the other way around — as an **interactive editor**.

- **Test a single frame instantly.** Park the playhead on any frame and *Test Frame* restores a short window around it, showing each detected region as **Mosaic → Restored** side by side. Dial a setting, test again, watch it change — the loop is seconds, not a full encode.
- **Live segment preview.** Mark a segment, preview just that range, and decide whether to commit to a full run before encoding the whole video.
- **Hardware-accelerated throughout.** NVDEC decode, TensorRT-accelerated BasicVSR++ restoration, and NVENC encode keep frames on the GPU end to end.
- **Compiles for your GPU.** No models are shipped. You download the model checkpoints and compile TensorRT engines *for your specific card* — all from inside the app.

![Test Frame — every detected region shown as Mosaic then Restored, without a full encode](docs/InAction-FramePreview.png)

---

## Two ways to run it

- **[For Users](#for-users)** — download the installer, get models, compile, run. No Python, no build tools.
- **[For Developers](#for-developers)** — clone the repo, set up a venv, run from source, or build the installer.

---

## For Users

### 1. Requirements

- **GPU:** NVIDIA RTX card. Native TensorRT builders ship for **RTX 50-series (Blackwell), 40-series (Ada), and 30-series (Ampere)**. Other cards still work via a slower PTX fallback for the first compile.
- **OS:** Windows 10/11 with an up-to-date NVIDIA driver.
- Nothing else — CUDA, TensorRT, ffmpeg, and Python are all bundled in the installer.

### 2. Download and extract

Grab the latest release from the **[Releases](https://github.com/seatv/ChitraMaya/releases)** page. The installer is split into three parts to fit the download limit:

- `ChitraMaya-install.exe`
- `ChitraMaya-install.7z`
- `ChitraMaya-install.7z.002`

Put **ALL THREE** parts in the same folder and run `ChitraMaya-install.exe` — it reassembles and extracts automatically. You'll get a `ChitraMaya` folder containing `ChitraMaya.exe`, a `models\` folder, and `Compile-All-Engines.ps1`.

### 3. Get the models

No models ship with the app — you add them once. Two ways, both from **Manage Models** in the app (or just drop files into `models\`):

**In-app download (easiest):** launch ChitraMaya, click **Manage Models**, pick a source from the dropdown (the primary **lada** and VR-focused **zelefans** repositories are pre-loaded), click **Fetch**, select the detection (`.pt`) and restoration (`.pth`) files you want, and **Download**. They land in `models\` automatically. You can add your own Hugging Face repo URLs with **+ Add**.

![Manage Models — download checkpoints, then compile TensorRT engines for your GPU](docs/ModelManagement.png)

**Manual:** drop any detection `.pt` and restoration `.pth` files straight into the `models\` folder.

### 4. Compile engines for your GPU

TensorRT engines are hardware-specific, so they're built on your machine (once per model):

1. Open **Manage Models**. Downloaded models show as **Not compiled**.
2. Click **Select all not-compiled** (or pick individual rows).
3. Set **Image Size** (detection; 640 is the tested default, 960 helps dense VR content) and **Max Clip Length** (restoration).
4. Click **Compile** and watch the log. This takes a few minutes per model and pins the GPU — that's normal.

When it finishes, the badges flip to **Compiled** and the models are ready to use.

> On a 6 GB card, if a restoration compile runs out of memory, that's the one thing to watch — compiling is the most VRAM-hungry step.

### 5. Restore a video

1. Load a video (drag it in or use the file picker).
2. Pick your detection and restoration models in the Control Panel. Turn on **Use Tensor** to use the compiled engines. (Every control has a tooltip — hover to learn what it does.)
3. Park the playhead on a mosaic frame and click **Test Frame** to preview the result on that frame. Adjust settings and test again until it looks right.
4. Use **Restore** / **Restore & Save** to process a segment or the whole video.

For side-by-side (SBS) VR video, enable **Split SBS** in Detection so each eye is detected at full resolution.

![Restore & Save — the finished, restored output](docs/InAction-RestoreAndSave.png)

![Playing the restored result back in the built-in player](docs/InAction-RestoreAndSavePlaying.png)

---

## For Developers

### Prerequisites

- **GPU:** NVIDIA, Turing (RTX 20xx) or newer
- **OS:** Windows 10/11 or Linux
- **Python:** 3.11 or 3.12
- **CUDA:** 12.x with matching cuDNN
- **TensorRT:** 10.x
- **ffmpeg / ffprobe:** on the system PATH (used for audio remux and CPU-decode fallback)

### Install from source

```bash
git clone https://github.com/seatv/ChitraMaya.git
cd ChitraMaya

python -m venv venv
# Windows
venv\Scripts\activate
# Linux
source venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

Models are not shipped; place detection `.pt` and restoration `.pth` files in `models/` (compiled engines are cached in `models/engines/`).

### Compile engines

Compile everything found in `models/` in one shot:

```powershell
# Windows (defaults: DetImgsz 640, DetMaxBatch 8, workspace 2 GB, fp16 on)
powershell -ExecutionPolicy Bypass -File .\Compile-All-Engines.ps1
# Low-VRAM cards, if a compile OOMs:
powershell -ExecutionPolicy Bypass -File .\Compile-All-Engines.ps1 -RestWorkspace 1
```

Or compile individually:

```bash
chitramaya -compile-det  --det-model  models/<detection_model>.pt    --det-imgsz 640
chitramaya -compile-rest --rest-model models/<restoration_model>.pth --rest-max-clip-length 60
```

### Run

```bash
# Interactive UI (default)
chitramaya

# Headless CLI
chitramaya -restore \
    --input video.mp4 \
    --output restored.mp4 \
    --det-model  models/<detection_model>.pt \
    --rest-model models/<restoration_model>.pth \
    --rest-max-clip-length 60 \
    --rest-backend trt
```

Useful CLI flags:

| Flag | Description |
|---|---|
| `--rest-backend` | `trt` (TensorRT sub-engines) or `pytorch` (fallback, no precompiled engines needed) |
| `--rest-max-clip-length` | Frames per restoration clip (must match the compiled engine) |
| `--det-conf` | Detection confidence threshold |
| `--det-imgsz` | Detector input size (multiple of 32) |
| `--enc-codec` | Output codec: `hevc` or `h264` |
| `--enc-qp` | Encoder quantization parameter (lower = higher quality) |
| `--max-frames` | Process at most N frames (debug) |

Run `chitramaya -restore --help` for the full list.

### Build the installer

```powershell
# From the repo root, in your release venv:
powershell -ExecutionPolicy Bypass .\packaging\windows\chitramaya-packager.ps1
```

This produces a split, self-extracting installer under `dist\`. TensorRT builder resources are trimmed to the shipped consumer architectures (plus PTX) to keep the size down; edit `_DROP_TRT_BUILDER_ARCHS` in `packaging\windows\chitramaya.spec` to change which GPUs are supported natively.

---

## Architecture

```
Decode (NVDEC) -> Detect (YOLO) -> Track scenes -> Restore (BasicVSR++ / TensorRT) -> Composite -> Encode (NVENC)
```

A YOLO detector locates mosaic regions per frame, a scene tracker groups them into temporally coherent clips, and a BasicVSR++ restoration model (run through TensorRT sub-engines, or a PyTorch fallback) reconstructs each clip with temporal consistency. Restored regions are composited back into the frame — with an optional **FaceFusion** blend that follows the mosaic's actual shape for a softer edge.

Full restores stream the whole file through NVDEC for throughput; *Test Frame* and segment previews read just the frames they need.

## License

See [LICENSE](LICENSE) for details.
