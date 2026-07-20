# ChitraMaya

A TensorRT-accelerated mosaic restoration studio with a real-time visual editor. Load a video, preview the restoration on your *actual* frames, and decide what to commit before spending time on a full encode.

![ChitraMaya — the mosaic input and the restored result, side by side](docs/InAction.png)

## Why ChitraMaya?

Some restoration tools are batch processors: set parameters, run a full pass, look at the result, repeat. ChitraMaya is built the other way around — as an **interactive editor**.

- **Test a single frame instantly.** Park the playhead on any frame and *Test Frame* restores a short window around it, showing each detected region as **Mosaic → Restored** side by side. Dial a setting, test again, watch it change — the loop is seconds, not a full encode.
- **Live segment preview.** Mark a segment, preview just that range, and decide whether to commit to a full run before encoding the whole video.
- **Hardware-accelerated throughout.** NVDEC decode, TensorRT-accelerated BasicVSR++ restoration, and NVENC encode — to **HEVC, H.264, or AV1** — keep frames on the GPU end to end.
- **Compiles for your GPU.** No models are shipped. You download the model checkpoints and compile TensorRT engines *for your specific card* — all from inside the app.
- **Made for VR/SBS content.** Per-eye detection for side-by-side video, a runtime **Image Size** dial for dense high-resolution frames, **VR Projection** for studios whose mosaic arrives warped in the raw frame, and **SBS View**: a projected look-around preview (like a headset, on your desktop) with a draggable wipe to compare original vs restored inside the projection.
- **Add Mosaic.** The inverse operation — pixelate regions to produce shareable SFW clips. Draw rectangles by hand (precise, reliable), or let the app auto-detect regions with a detection model (**experimental** — see the warning below).

![Test Frame — every detected region shown as Mosaic then Restored, without a full encode](docs/InAction-FramePreview.png)

## Terms & Conditions

By downloading or using this software, in whole or in part, you agree to use it only for purposes that are lawful in your jurisdiction.

You are solely responsible for what you create with it and for complying with all applicable local, regional, and international laws — including, without limitation, those governing privacy, consent, publicity, defamation, and intellectual property. The authors and contributors of this software accept no responsibility and shall not be held liable for any use of the software or for anything produced with it. If you are unsure whether a use is lawful where you are, consult a legal professional before proceeding.

> [!CAUTION]
> **Automatic NSFW detection for mosaic addition is experimental and must not be relied on to censor content.** The "Auto-detect" / censor mode uses a third-party NSFW detection model that does **not** reliably find all explicit content — it will miss regions and whole frames. Do not use it to make content safe for publication, distribution, or any purpose where missed content has consequences. For censoring you can trust, use the **manual draw-rectangles** Add Mosaic and **review every frame of the output yourself** before sharing. See [Known Issues](#known-issues--not-yet-implemented).

---

## Two ways to run it

- **[For Users](#for-users)** — download the installer, get models, compile, run. No Python, no build tools.
- **[For Developers](#for-developers)** — clone the repo, set up a venv, run from source, or build the installer.

---

## For Users

### 1. Requirements

- **GPU:** NVIDIA RTX card. Native TensorRT builders ship for **RTX 50-series (Blackwell), 40-series (Ada), and 30-series (Ampere)**. Other cards still work via a slower PTX fallback for the first compile. **AV1 output** additionally requires an RTX 40-series or newer (older cards can decode AV1, but only Ada/Blackwell NVENC can encode it — the app checks and tells you).
- **OS:** Windows 10/11 with an up-to-date NVIDIA driver.
- Nothing else — CUDA, TensorRT, ffmpeg, and Python are all bundled in the installer.

### 2. Download and extract

Grab the latest release from the **[Releases](https://github.com/seatv/ChitraMaya/releases)** page.

> [!CAUTION]
> ### ⚠️ The installer is $${\color{red}THREE}$$ files — you need $${\color{red}ALL\ THREE}$$.
>
> The `.exe` **by itself is not the program**; it is only the unpacker for the
> other two parts. Download **all three** into the **same folder**:
>
> - [ ] `ChitraMaya-install.7z.001`
> - [ ] `ChitraMaya-install.7z.002`
> - [ ] `ChitraMaya-install.exe`

Run `ChitraMaya-install.exe` — it reassembles the parts and extracts automatically. You'll get a `ChitraMaya` folder containing `ChitraMaya.exe`, a `models\` folder, and `Compile-All-Engines.ps1`. If the install fails immediately, check that all three downloads completed and are in one folder.

### 3. Get the models

No models ship with the app — you add them once. Two ways, both from **Manage Models** in the app (or just drop files into `models\`):

**In-app download (easiest):** launch ChitraMaya, click **Manage Models**, pick a source from the dropdown (the primary **lada** and VR-focused **zelefans** repositories are pre-loaded), click **Fetch**, select the detection (`.pt`) and restoration (`.pth`) files you want, and **Download**. They land in `models\` automatically. You can add your own Hugging Face repo URLs with **+ Add**.

> Hugging Face throttles anonymous downloads (~1,000 requests/hour per IP) and answers with a 403 once you cross it — easy to hit on a heavy day of testing across machines behind one home IP. If downloads start failing, drop a free "read" token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) into a one-line `hf-token.txt` next to the app (or set the `HF_TOKEN` environment variable) and retry. Failed downloads now name the cause and the fix in the log rather than showing a bare error.

![Manage Models — download checkpoints, then compile TensorRT engines for your GPU](docs/ModelManagement.png)

**Manual:** drop any detection `.pt` and restoration `.pth` files straight into the `models\` folder.

### 4. Compile engines for your GPU

TensorRT engines are hardware-specific, so they're built on your machine (once per model):

1. Open **Manage Models**. Downloaded models show as **Not compiled**.
2. Click **Select all not-compiled** (or pick individual rows).
3. Set **Image Size** (detection; 640 is the tested default — **compile at 800 if you mostly restore 4K+ VR/SBS content**, see the tuning note below) and **Max Clip Length** (restoration).
4. Click **Compile** and watch the log. This takes a few minutes per model and pins the GPU — that's normal.

When it finishes, the badges flip to **Compiled** and the models are ready to use.

> On a 6 GB card, if a restoration compile runs out of memory, that's the one thing to watch — compiling is the most VRAM-hungry step.

### 5. Restore a video

1. Load a video (drag it in or use the file picker). The top bar shows the **full path of the loaded file**, so a long batch session never leaves you guessing which file is on screen.
2. Pick your detection and restoration models in the Control Panel. Turn on **Use Tensor** to use the compiled engines. (Every control has a tooltip — hover to learn what it does.)
3. Park the playhead on a mosaic frame and click **Test Frame** to preview the result on that frame. Adjust settings and test again until it looks right.
4. Use **Restore** / **Restore & Save** to process a segment or the whole video.

A few things worth knowing before a full run:

- **Max Clip Length is flexible.** A restoration engine set compiled at N handles any clip length up to N, so you can set Max Clip to any value up to your largest compiled size — the app loads the smallest set that covers your request and runs at the ceiling you asked for. Longer clips give better temporal stability but cost more VRAM.
- **VRAM pre-flight warning.** Before processing starts, ChitraMaya checks free GPU memory against what the run needs and warns you up front — from "headroom is thin" through "VRAM tight, may page" up to "this configuration does not fit this GPU" — and names the levers that would help (lower Max Clip, a smaller compiled engine set, PyTorch detection). Heed it, especially on 8 GB and smaller cards.
- **Async Encode is off by default.** Synchronous encoding is the dependable default and saves ~500–600 MB of VRAM at 4K. If your card has headroom, tick **Async Encode** in the Encoder panel to overlap encode with restoration for a faster run.
- **Output codec.** The Encoder panel offers **HEVC** (default), **H.264**, and **AV1** (RTX 40-series or newer). Your **QP** setting always uses the familiar HEVC 0–51 scale; for AV1 the app maps it internally to AV1's finer quantizer scale (the log shows the mapping), so QP 18 means the same quality intent regardless of codec.

![Restore & Save — the finished, restored output](docs/InAction-RestoreAndSave.png)

![Playing the restored result back in the built-in player](docs/InAction-RestoreAndSavePlaying.png)

### 6. VR / SBS content — the dials that matter

For side-by-side (SBS) VR video, enable **Split SBS** in Detection so each eye is detected at full resolution. Then two more dials decide how good the result gets:

**Image Size (Detection).** The detector's input resolution, now adjustable at runtime (640–960, steps of 32). On 4K-and-up VR frames the default 640 downscales too aggressively and can miss small or faint mosaics; **800 is the field-tested sweet spot for ≥4K VR/SBS** (640 missed frames in our tests; 800 and 960 caught everything, and 960 fragments clips without adding catches). With **Use Tensor** on, the compiled engine must match this size — if it doesn't, the run automatically falls back to PyTorch at your requested size and the log tells you to recompile the engine at that size in Manage Models. For regular flat content, 640 remains the tested default.

**VR Projection (Detection).** Some VR studios apply the mosaic to the raw video frame; others apply it in *viewing* space, so it looks square in a headset but arrives warped and trapezoidal in the raw frame — a pattern the detection and restoration models were never trained on and handle poorly. The **VR Projection** dropdown (requires Split SBS) fixes the second kind: with **Fisheye** selected, each eye is warped so those blocks become square again, detection and restoration run in that space, and only the restored regions are warped back onto the untouched original frames — background pixels are never resampled.

*Which setting for which video?* Open the video in any flat player (PotPlayer, VLC — no VR mode) and look at the mosaic:

- Blocks form a clean, even grid of **squares** → leave VR Projection **Off**. (Warping this content would hurt.)
- Blocks look **warped** — trapezoidal cells, rows that bow or fan out, especially away from the center → set VR Projection to **Fisheye**.

> [!IMPORTANT]
> On warped-mosaic content with projection Off, the run statistics can look perfect (every frame "detected and restored") while the output still shows mosaic — the models latch onto the warped blocks but cannot actually reconstruct them. **Judge quality with your eyes, not the stats:** use **Test Frame** on a mosaic-heavy frame with projection Off vs Fisheye and compare.

### 7. Compare in 3D — SBS View

For side-by-side VR content, the flat player shows two distorted fisheye-looking halves. **SBS View** (the button next to the volume control) projects the video the way a headset would — a natural look-around view — and lets you compare the original against your restored output side by side *inside* that projection.

It helps to understand the two independent choices in the top bar, because they answer different questions:

- **Eye (L / R)** — *which eye's image am I inspecting?* SBS video carries two pictures; this picks one. Your choice applies to everything on screen at once.
- **View (Original / Restored / Wipe)** — *which video fills the screen?* **Original** is the loaded video, **Restored** is your most recent output (run a restore, or Add Mosaic, first — until then only Original is available), and **Wipe** shows both at once, split by a draggable divider: **original on the left of the divider, restored on the right**. The divider splits the two *videos*, not the two eyes.

So "Eye = L, View = Wipe" means: show the left eye's picture, original left of the divider, restored right of it. Drag the divider across a restored region and watch it flip between mosaic and clean — that's the money shot.

Everything else in one place:

- **Look around:** drag with the mouse. **Zoom:** mouse wheel (FOV 30–110). **Reset view** or `0` recenters.
- **Playback:** both videos play together, frame-locked. Space = play/pause; `«  ‹  ›  »` buttons (or ←/→ arrows) skip by your configured skip amounts; `,` / `.` step a single frame; `m` unmutes (audio comes from the restored side). The `fN` counter in the time display is the current frame number.
- **Speed** (0.1×–1×): slow motion for close inspection — and if a very large original struggles to keep up with the restored side, a slower speed lets it stay in sync.
- **Offset:** aligns the clocks when the restored side is a *segment* preview (its 0:00 is the segment start). Auto-filled; you rarely need to touch it.
- **Esc** closes and frees the viewer's decoders.

> SBS View is a desktop editing aid, not a headset mode — for viewing in VR, open the output file in your usual VR player. It currently assumes equirect-180 side-by-side (left|right) content. (The new VR Projection option affects *restoration*; a matching projection selector for this viewer is planned.)

### 8. Add Mosaic — make SFW clips

The inverse of restoration: pixelate regions and save, for producing shareable, safe-for-work clips (this project's own demo material is made with it). There are two ways to place the mosaic — a **reliable manual** way and an **experimental automatic** way.

**Manual (reliable) — draw the rectangles.**

1. Load a video; optionally mark a segment to limit the scope.
2. Click the **Add Mosaic** button. The player pauses and a crosshair appears — **drag rectangles directly on the video** (up to three). Each shows its size and a ✕ to remove it. You can still scrub the timeline to a better frame while drawing.
3. For SBS video (with **Split SBS** on), draw on *either* eye — a dashed ghost mirrors the rectangle to the other eye at the same per-eye position, and both eyes get mosaiced.
4. **Done** opens a dialog with the exact pixel coordinates for fine-tuning (**Draw again** goes back to the video with your rectangles intact). Set **Block** for the mosaic cell size (16 is the classic look).
5. **Add & Save** encodes to `<name>-mosaic.mp4` (or `-mosaic-seg.mp4` for a segment) in your output folder. The result becomes the preview — open **SBS View** and wipe-compare original vs censored to verify placement in both eyes.

**Automatic (experimental) — let a model find the regions.**

> [!CAUTION]
> **This is experimental and will miss content. Do not use it to censor anything you intend to share.** It depends on a third-party NSFW detection model whose accuracy is limited — it misses regions and whole frames, and it is *not* production-grade. Treat any automatic result as a rough draft that **you must review frame by frame** (the playback-speed control and SBS View wipe help), and fix by hand. When it matters, use the manual method above.

Auto mode reuses the detection pipeline: pick a detection model (e.g. an NSFW detector) as the **Detection** model, then under **Alternate Execution Modes** in the Control Panel tick **Add Mosaic** and set **Block**. Now **Test Frame** shows each detected region as **Original → Censored** (the fast way to see what the model catches and misses on a given frame), **Restore** previews a censored segment, and **Restore & Save** writes `<name>-censored.mp4`. A detection model is all it needs — no restoration model. The one-shot **Auto-detect** button inside the Add Mosaic dialog does the same thing for a single run.

To review coverage: play the output back with the **playback-speed control** (left of the SBS button — slow down to catch a one-frame gap, speed up to skim), and use **SBS View → Wipe** to compare against the original in both eyes.

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
| `--rest-backend` | `auto` (default — loads a covering engine set, or falls back to PyTorch), `trt` (force TensorRT sub-engines), or `pytorch` (force fallback, no precompiled engines needed) |
| `--rest-max-clip-length` | Frames per restoration clip. Any value up to your largest compiled set — the app loads the smallest set that covers it and runs at this ceiling. Defaults to 30 if omitted. |
| `--det-conf` | Detection confidence threshold |
| `--det-imgsz` | Detector input size (multiple of 32). Also adjustable in the UI (Detection → Image Size). |
| `--sbs` / `--sbs-det-split` | Side-by-side handling / per-eye detection |
| `--vr-projection` | `none` (default) or `fisheye` — per-eye analysis projection for viewing-space mosaics (requires `--sbs`; see the VR section above) |
| `--async-encoder` | Overlap encode with restoration (opt-in; synchronous is the default). Faster on cards with VRAM headroom. |
| `--enc-codec` | Output codec: `hevc`, `h264`, or `av1` (AV1 needs RTX 40-series or newer NVENC) |
| `--enc-qp` | Encoder quantization parameter, HEVC 0–51 scale (lower = higher quality; mapped internally for AV1) |
| `--mp4-fast-start` / `--no-mp4-fast-start` | Move the MP4 index to the front for streaming (on by default) |
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
Decode (NVDEC) -> [VR Projection] -> Detect (YOLO) -> Track scenes -> Restore (BasicVSR++ / TensorRT) -> Composite -> Encode (NVENC)
```

A YOLO detector locates mosaic regions per frame, a scene tracker groups them into temporally coherent clips, and a BasicVSR++ restoration model (run through TensorRT sub-engines, or a PyTorch fallback) reconstructs each clip with temporal consistency. Restored regions are composited back into the frame — with an optional **Face Swap** blend that follows the mosaic's actual shape for a softer edge. With **VR Projection** enabled, detection through restoration run in a per-eye fisheye space and only the restored regions are warped back, leaving every other pixel of the original untouched.

Full restores stream the whole file through NVDEC for throughput; *Test Frame* and segment previews read just the frames they need.

## Known Issues / Not Yet Implemented

A few things are intentionally incomplete or have known limitations in this release:

**Not yet implemented (flags parse but have no effect):**

- **Batch / folder processing** — `--batch-video-extensions` and `--batch-skip-existing` are placeholders; ChitraMaya currently processes one input at a time. Batch-over-a-folder isn't wired yet.
- **Detection debug dumps** — `--debug-save-detection-frames` and `--debug-save-detection-json` don't write anything yet.

**Known limitations:**

- **Automatic mosaic detection is experimental — do not rely on it to censor.** The Auto-detect / censor mode depends on a third-party NSFW detection model that does not reliably detect all explicit content; it misses regions and whole frames and is not suitable for production censoring. Use the manual draw-rectangles Add Mosaic for anything you intend to share, and review every frame of any output yourself. Evaluating stronger detectors is on the roadmap.
- **On warped-mosaic VR content, run statistics cannot detect a quality failure.** With VR Projection Off on such content, the stats can report full coverage while the output still shows mosaic (the models "restore" blocks they cannot parse). Use **Test Frame** to judge — see the VR section.
- **VR Projection assumes FOV-180 content and requires Split SBS.** Fisheye-native sources with wider lenses (190/200) are handled with the same transform, which has been sufficient in testing; a per-title projection variant is a planned refinement if a title needs it.
- **Detection FP16 applies only to the PyTorch path.** For a compiled TensorRT detection engine, precision is baked in at compile time, so the runtime **Detection FP16** toggle has no effect — the app grays it out when a compiled engine is selected. It still applies to `.pt` PyTorch detection runs.
- **Test Frame preview rows can accumulate.** Running **Test Frame** repeatedly on the same frame may stack preview rows in the strip until you press **New** or the next result replaces them. Cosmetic; a fix is planned.
- **SBS View assumes equirect-180 side-by-side (left|right)** content; fisheye layouts aren't projected correctly yet (a projection selector is planned). Playback in the viewer uses the app's embedded browser decoder, not NVDEC — a very large (8K) HEVC master may not play there even though it restores fine; a downscaled copy will.
- **Add Mosaic rectangles are per-eye for SBS** and are clamped to the eye you drew them in — a rectangle can't span the eye seam. Both eyes receive the mosaic at the same per-eye position (no parallax offset), so pad rectangles generously on close subjects.

Found something else? Please open an issue — **without** attaching any explicit content (see the issue template).

## License

See [LICENSE](LICENSE) for details.
