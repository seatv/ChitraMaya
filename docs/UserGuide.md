# ChitraMaya User Guide

*Version 1.80.00 — for people who want a video fixed, not for people who want to build the program. Developers: see the README.*

ChitraMaya finds mosaic-censored regions in a video, restores them with a temporal model, and writes a new video file. It runs on Windows with an NVIDIA GeForce RTX, AMD Radeon or Intel Arc GPU. This guide covers installing it, getting the models, the everyday workflow, the settings that actually matter, what the output files are, what the console is telling you, what to expect from your graphics card, and what to do when something goes wrong.

This copy of the guide is installed next to `ChitraMaya.exe` as `UserGuide.html` (the **Help** button in the app opens it) and published at <https://get.chitramaya.org/userguide.html>.

---

## 1. What it does, and what it does not

**It does:** detect mosaic regions frame by frame with a detection model, track them across frames, restore each tracked region with a video restoration model (BasicVSR++, trained by the lada project), upscale the restored region with a second model so close-ups do not go soft, smooth frame-to-frame shimmer, paste the result back into the untouched original frame, and encode the whole thing on the GPU. It can also preview a single frame in seconds, run a whole folder overnight, upscale whole frames to 4K, add mosaic (the reverse operation, for making shareable clips), and compare original against restored inside a VR projection.

**It does not:** invent what was never in the picture — a mosaic that destroyed the content leaves a restoration that is plausible rather than true; find every mosaic — a thin sliver at the edge of the frame, a region whose neighbour is still mosaic, or a mosaic applied in an unusual projection can be missed, and the run report tells you where it suspects leftovers; or work on content the models were not trained on (text, logos, faces are not what the restoration model knows).

**Terms.** By using ChitraMaya you agree to use it only for purposes lawful where you are. You are responsible for what you make with it. The authors accept no liability for its use. The automatic mosaic detection in **Add Mosaic** is experimental and must not be relied on to censor anything you intend to share — see section 8.

---

## 2. Installing

### Which edition

There is one edition per GPU vendor, and they are separate downloads. Pick the one for your card:

| Your GPU | Installer |
|---|---|
| **NVIDIA** GeForce RTX 30 / 40 / 50-series | `ChitraMaya-install.exe` **plus** `ChitraMaya-install.7z.001` **plus** `ChitraMaya-install.7z.002` — all three |
| **AMD** Radeon RX 7000 / 9000 (RDNA 3 / 4) | `ChitraMaya-rocm-install.exe` — one file |
| **Intel** Arc A/B-series, Core Ultra iGPU | `ChitraMaya-xpu-install.exe` — one file |

Downloads: <https://get.chitramaya.org>. Every file is listed there with its SHA-256 checksum; a mainland-China mirror line is on the same page. The source code and the issue tracker live at <https://gitlab.com/seatv/chitramaya>.

The wrong edition tells you so: a startup check probes the GPU before anything loads and, if this edition cannot run on it, shows a message naming the GPUs Windows reports and the edition that matches them. `ChitraMaya-cli.exe -self-check` gives the full verdict. The title bar always says which edition is running.

### The NVIDIA installer is three files — you need all three

The `.exe` by itself is not the program; it is only the unpacker for the other two parts. Download all three into the same folder, then run `ChitraMaya-install.exe`. It reassembles the parts and extracts a `ChitraMaya` folder. The AMD and Intel installers are a single `.exe` you double-click.

Pick a **new, empty folder** when asked. Do not extract a new version over an old one, and do not mix editions in one folder.

### What is in the folder

`ChitraMaya.exe` is the app — a window, no terminal. Its console output goes to the in-app Console panel and to `ChitraMaya-console.log` next to the exe (the previous run is kept as `ChitraMaya-console.prev.log`). `ChitraMaya-cli.exe` is the same app with a console window, for terminal workflows: headless runs, engine compiles, diagnostics. `ChitraMaya.cmd` launches the app on the AMD and Intel editions. `VERSION.txt` says which version this folder is. `UserGuide.html` is this guide. `models\` is where models and compiled engines go (section 3). `presets\` holds your saved configurations.

### Updating

From one version to the next you do not re-download the multi-gigabyte installer: each release also carries **patch zips** of under 100 MB. Pick the one whose "from" version matches your install (`VERSION.txt`): a 1.70.00 install takes `ChitraMaya-patch-1.70.00-to-1.80.00.zip`. Extract it anywhere and run

```
powershell -ExecutionPolicy Bypass -File .\Apply-Patch.ps1
```

It finds your install (or asks), checks that every file it would replace belongs to the version the patch was built from (a mismatch refuses cleanly and changes nothing), backs up what it replaces, applies, and runs the self-check. Your models, compiled engines and settings are never touched, and running a patch twice is safe.

After updating, **Manage Models** may turn amber: that means a compiled engine no longer matches the new version's expectations and should be rebuilt once (section 3). The old engine still works until you do.

### Requirements

NVIDIA: Windows 10/11, an up-to-date GeForce driver, an RTX card. Native TensorRT support ships for 30-series (Ampere), 40-series (Ada) and 50-series (Blackwell); older cards still work through a slower generic path. Nothing else — CUDA, TensorRT, ffmpeg and Python are all inside the installer.

AMD: Windows 10/11, **AMD Software Adrenalin 26.2.2 or newer** (earlier drivers lack the runtime the bundled PyTorch stack needs), an RDNA 3 or RDNA 4 card — ROCm on Windows does not support older Radeons.

Intel: Windows 10/11, a current Intel graphics driver. On discrete Arc cards **Resizable BAR must be on** (BIOS: CSM off, Above 4G Decoding on, ReBAR on) — without it the same card ran twenty times slower in our tests. Integrated Arc (Core Ultra laptops) needs no such setting.

All editions: a few tens of GB free on the drive you use for temporary files; 16 GB of system RAM is a workable minimum, 32 GB comfortable for 4K.

---

## 3. First run: models and engines

No models ship with the app; you add them once.

**Get Models.** On a fresh install the header button is green and reads **Get Models**. Click it, then **Get standard set** — the detection and restoration models the app is built around are downloaded from Hugging Face into `models\`. Manual alternative: drop any detection `.pt` and restoration `.pth` files into `models\`. (The Temporal Stability weights are the one exception to "no models ship": they are tiny and bundled, so that feature works from the first run.)

If a download answers with a **403**, you have hit Hugging Face's anonymous-download limit (about 1,000 requests an hour per address — easy to reach on a day of heavy testing). Wait an hour, or download the files in a browser and drop them into `models\`.

**Compile (NVIDIA only).** TensorRT engines are specific to your card, so they are built on your machine, once per model. When one is missing for your settings the header button turns amber and reads **Compile Models**. Open **Manage Models**: downloaded models show as *Not compiled*; click **Select all not-compiled**, set the detector's **Image Size** and **Detection Batch** (640 and 8 are the defaults; **800 if you mostly restore 4K or VR content**, see section 6), and press **Compile**. The log is selectable and has a **Copy** button for bug reports. This takes a few minutes per model and pins the GPU — normal. When it finishes the badges read *Compiled* and the header button goes grey.

On a 6 GB card the restoration compile is the one step that can run out of memory; close other GPU programs first.

AMD and Intel have no compile step: TensorRT does not exist there, and the models run directly in PyTorch. The first run after installing into a new folder is slow once on AMD, while the ROCm math library compiles its kernels and caches them in `miopen-cache` next to the exe — the first clip can sit for a few minutes with no visible progress (the console says so at start), and the stall watchdog may print a stack dump while it waits; that is a diagnosis, not a crash. Later runs start immediately. Copying `miopen-cache` from an old install into a new one skips the wait.

**Custom super-resolution models** (section 7) go into `models\upscalers\`. On NVIDIA they too can be compiled, from a terminal — once per machine:

```
ChitraMaya-cli.exe -compile-sr 4xNomos8k_span_otf_strong.pth --secondary --bench
```

That builds the whole-frame engine and the region engine beside the weights and prints a benchmark of the model with and without the engine. The next run reports `TensorRT … ACTIVE` for the model at start. Engines are specific to the GPU they were built on; a run on a different card falls back to the un-compiled path and says why. (A Compile button for custom models in Manage Models is planned.)

---

## 4. The everyday workflow

The app is an editor that can also batch. The loop you will use most is: load a video, park on a mosaic frame, **Test Frame**, adjust, test again, and only then commit to a full run.

1. **Load a video** — drag it into the window or use the file picker. The top bar shows the full path of the loaded file, so a long session never leaves you guessing what is on screen. While a run is in progress the app refuses to load another file (the run keeps the file it started with).
2. **Pick the models** in the Control Panel: a detection model and a restoration model. On NVIDIA, **Use Tensor** uses the compiled engines. Every control has a tooltip; hover to learn what it does.
3. **Test Frame.** Park the playhead on a mosaic frame and click it. A short window of frames around that one is restored through the full pipeline and each detected region is shown **Mosaic → Restored** side by side. The enlarged view keeps your last five attempts on that frame — flip between them with the ‹ › arrows or the numbered slots (hover a number to see which settings produced it). This is the way to dial in a title: change one thing, test, compare.
4. **Restore** previews a marked segment without saving; **Save Segment** encodes just the marked range to a file (the way to test a setting on a minute of video before committing hours); **Restore & Save** runs the whole video. Before a full run, the VRAM pre-flight dialog tells you whether the run fits your card (section 10).

The built-in player plays the result afterwards; **SBS View** (section 6) compares original and restored inside a VR projection.

### The defaults are the recommended settings

A fresh install starts with Secondary = **RTX Super-Res 4×** (NVIDIA; **Real-ESRGAN 4×** on AMD and Intel), Detection Score 0.20, Detection Batch 8, Temporal Stability 1, Overlap 8 with Crossfade at Max Clip cuts, the Face Swap blend with Feather on auto, Max Clip Length 180, Crop Cap off, Async Encode off, HEVC at preset P5 and QP 18. Most titles need nothing changed.

### The dials that matter

**Secondary** (Restoration panel). The restorer works on a fixed-size crop; regions larger than that used to be stretched at paste-back and went soft — worst on close-ups and VR. With a Secondary on, the restored crop is upscaled first so the final resize shrinks instead of stretching; small regions automatically keep the untouched path. **RTX Super-Res** (NVIDIA) gives a smoother look, **Real-ESRGAN 4×** keeps the restorer's texture; neither is more correct — pick by eye with Test Frame. Any community super-resolution model placed in `models\upscalers\` appears in the dropdown too; `4xNomos8k_span_otf_strong` is a good one to try (section 7). If RTX Super-Res is selected on a machine without the Maxine runtime, the run falls back to Real-ESRGAN and says so.

**Temporal Stability** (0–3). Removes the frame-to-frame shimmer restoration models produce by smoothing each restored region across a seven-frame window, only where the content agrees — real motion passes through, and only restored pixels are touched. 1 is the default; 3 looks better on some titles and costs about 30 % more restore time. The stabilizer checks its own output; a window it cannot trust keeps its restored pixels unsmoothed and the run ends *with warnings* saying so.

**Max Clip Length.** The restorer looks at a run of consecutive frames of one region as a single sequence, and this is the longest such sequence. Longer windows let a frame borrow detail from its neighbours and mean fewer joins between consecutive sequences; but the useful reach is about a second of video, and every frame held open costs memory. The rule we measured: **about three times the frame rate** — 90 at 30 fps, 180 at 60 fps. Shorter saves memory (two gigabytes between 180 and 90 on a 4K60 title) at about 5 % more run time and more joins; longer buys almost nothing and, once the card is full, slows the whole run down. On an 8 GB card at 4K, lower this before reaching for Crop Cap.

**Crop Cap** (Restoration panel). The size above which a region's frames are stored downscaled while the run is in progress — the lever for 4K titles on 8 GB cards. Off by default; the pre-flight dialog proposes the value that fits. It costs a little fine texture on the very largest regions and nothing elsewhere.

**Feather and Region Margin** (Restoration panel, Face Swap blend). Region Margin grows the detected region by a number of pixels before restoration; Feather is the width of the soft edge where the restored patch blends into the original frame — 0 means auto, which scales with the frame (about 60 px at 2160p), and is the right setting for 4K, where a fixed small feather shows as a hard line.

**Detection Score and IoU** (Detection panel). Score is the confidence below which a detection is ignored; 0.20 is the default, lower finds more and risks false regions. IoU controls how aggressively overlapping candidate boxes are merged — it is not a sensitivity dial.

**Image Size** (Detection panel). The detector's input resolution, 640–960 in steps of 32. 640 is right for 1080p; **800 for 4K and VR/SBS**, where 640 downscales too far and misses small or faint mosaics. With Use Tensor on, the compiled engine must match this size; if it does not, a dialog at submit time offers to use the engine's size, continue on PyTorch, or open Manage Models.

### The Encoder panel

**Codec**: HEVC (default), H.264, or AV1 (RTX 40-series or newer; RDNA 3 or newer; Intel Arc). **QP** always uses the HEVC 0–51 scale; for AV1 the app maps it to AV1's finer scale, so QP 18 means the same quality intent on every codec. **Preset** runs from P1 (fastest) to P7 (best); on 8 GB cards stay at P5 or below. **Async Encode** overlaps encoding with restoration for a faster run at the cost of 500–600 MB of VRAM at 4K; it is off by default and the pre-flight's levers turn it off first when memory is short. **Output Upscale** brings the finished frames to 1440p or 2160p — a clip-or-overnight feature; see section 7.

### Saved configurations

The ⚙️ menu's **Save / manage configurations…** stores the complete current settings — every control plus the output and temp folders — under a name; one click loads one and makes it active, and it survives a restart. Files live in `presets\` next to the exe and can be copied between machines. A built-in **VR 4K** configuration sets Image Size 800, Max Clip 180 and Split SBS in one click.

---

## 5. A whole folder

**Process Folder** (Alternate Execution Modes) runs your current settings over every video in a folder. Pick the folder (paths pasted from Explorer work, quotes and all), optionally recursive, and the output location and suffix. The queue dialog lists every file and says which will be skipped and why — usually "output already exists", which is also the resume mechanism after an interruption. Models load once and stay warm for the batch; each file runs isolated, so one failure (even a GPU hang, which the watchdog catches) is logged and the batch moves on. The console log has a per-file summary at the end.

The pre-flight runs for the largest file in the plan, and a batch that does not fit runs on regardless of the warning — read it before leaving for the night. If the largest file is a 4K title and the card is an 8 GB one, set Crop Cap or a lower Max Clip for the batch.

Outputs never overwrite: if the name already exists the new file gets a `-2`, `-3`, … suffix and the log says so.

---

## 6. VR and side-by-side video

For side-by-side (SBS) VR video, enable **Split SBS** in Detection so each eye is detected at full resolution, and set Image Size to 800. Then one more decision:

**VR Projection.** Some studios apply the mosaic to the raw video frame; others apply it in viewing space, so it looks square in a headset but arrives warped and trapezoidal in the raw frame — a pattern the models were never trained on. Open the video in a flat player (PotPlayer, VLC — no VR mode) and look at the mosaic. Blocks in a clean, even grid of squares → leave VR Projection **Off**. Blocks warped — trapezoidal cells, rows that bow or fan out, especially away from the centre → set it to **Fisheye**: each eye is warped so the blocks become square, detection and restoration run in that space, and only the restored regions are warped back onto the untouched frame.

On warped-mosaic content with the projection off, the run statistics can look perfect while the output still shows mosaic. Judge with your eyes: Test Frame with projection Off and with Fisheye on a mosaic-heavy frame and compare.

**8K VR on cards before the RTX 50-series.** H.264 sources wider than 4096 px exceed the hardware decoder's limit on earlier cards; the app falls back to CPU decode for those files and says so at startup. It works; it is much slower, and that is a hardware limit.

**SBS View** (the button next to the volume control) projects the video the way a headset would and lets you compare original against restored inside that projection. **Eye (L / R)** picks which eye's picture you inspect. **View** picks what fills the screen: Original, Restored (your most recent output), or **Wipe** — both at once, split by a draggable divider, original on the left and restored on the right. Drag the divider across a restored region and watch it flip between mosaic and clean. Drag to look around, wheel to zoom, Space to play, ←/→ to skip, `,` and `.` to step a frame, `m` to unmute, Esc to close. It assumes equirect-180 side-by-side content and is a desktop aid, not a headset mode.

---

## 7. Upscaling and custom models

Two different things in the app scale pictures, and the [Scaling Guide](ScalingGuide.md) has the measurements for both.

The **Secondary** scales each restored region before paste-back; it costs a few percent of a run and is on by default (section 4).

**Output Upscale** (Encoder panel) scales the whole frame after restoration — 720p or 1080p to 1440p or 2160p — or, in **Upscale** mode under Alternate Execution Modes, does nothing else: decode, scale, encode, no models. It is a clip feature or an overnight one. Measured on an RTX 3060 Ti, a 66-minute 1080p title restored and upscaled to 2160p took 3h42m with RTX Super-Res and 5h04m with a compiled SPAN model; Real-ESRGAN on the same card is hours slower still. On AMD and Intel it is a start-before-bed job for a short title. The console's ETA line after the first 500 frames tells you what you are in for.

In Upscale mode the folder planner lists your earlier restored outputs as inputs (that is what you upscale), and the output gets `-upscaled` added to the name. Split SBS and VR Projection do nothing in this mode and are ignored.

**Custom models.** Drop any super-resolution model file into `models\upscalers\` (or **Manage Models → Custom scalers → Add**) and press **Rescan**; it appears in both scaler dropdowns with its architecture and factor. Some forty architectures are recognised from the file itself. Models the app cannot identify are listed with a reason. Where to find them: OpenModelDB and Philip Hofmann's models repository; check each model's licence, they differ. As a Secondary, none of the community models we measured clearly beat the two built-ins, and `4xNomos8k_span_otf_strong` came closest — slightly better on skin, and faster on AMD and Intel. As a whole-frame scaler the field is more open. On NVIDIA, compile the model once (section 3) and it runs about twice as fast.

---

## 8. Add Mosaic — making shareable clips

The reverse operation: pixelate regions and save, for producing safe-for-work clips. There is a reliable manual way and an experimental automatic way.

**Manual.** Load a video; optionally mark a segment. Click **Add Mosaic**: the player pauses and a crosshair appears — drag rectangles directly on the video (up to three); each shows its size and a ✕ to remove it, and you can still scrub to a better frame while drawing. For SBS video with Split SBS on, draw on either eye and a dashed ghost mirrors it to the other. **Done** opens a dialog with the exact pixel coordinates for fine-tuning; set **Block** for the cell size (16 is the classic look). **Add & Save** encodes to `<name>-mosaic.mp4` in your output folder; open SBS View and wipe-compare to verify placement.

**Automatic — experimental, and it will miss content.** Pick a detection model (an NSFW detector, for instance) as the Detection model, tick **Add Mosaic** under Alternate Execution Modes, set Block, and Test Frame shows each detected region Original → Censored; Restore & Save writes `<name>-censored.mp4`. Two techniques from the field: run detection with **Score 0.05 and IoU 1.0** (every candidate region is kept — over-coverage is safe, a miss is not), and afterwards run the same detector **over the censored output**: any frame where it still finds something is a leak candidate to patch by hand. None of this is a guarantee. Review every frame yourself before sharing anything made this way.

---

## 9. What the output files are

Beside every finished video you will find, depending on the **Run Files** switch (beside the video, in the Temp folder, or off):

- `<name><suffix>.mp4` — the result. The suffix is the **🏷️ Suffix** field in the header; `-restored` is the default. Upscale mode adds `-upscaled`; Add Mosaic writes `-mosaic` or `-censored`.
- `<name><suffix>.run.json` — the **run report**: what was asked versus what actually ran (which decoder, encoder, scaler, precision, and why), timings per stage, the VRAM ledger, detection and restoration counts, how many regions hit the Max Clip cap, the frames the detector bridged, the frames the guards touched, and the frames the **residual scan** still found mosaic on after the run. When something looks wrong, this is the file to send.
- `<name><suffix>.mp4.timecodes.txt` — the frame timing of the output, for frame-exact comparisons.
- `ChitraMaya-console.log` next to the exe — the console of the latest launch (`.prev.log` is the one before). The CLI exe keeps the newest twenty.

During a run you may see temporary files: `<name>.cm120-tsremux.mp4` next to a stream-capture source (section 13 explains), the raw bitstream and a `*-RECOVER.ps1` script beside the output, and an audio sidecar `.audio.mka`. These are the run's salvage: if the machine loses power in the last minute of a five-hour run, the recovery script turns the bitstream into a playable file without re-running anything. They are deleted once the output has been read back and verified. If a file ever reads back wrong, both the output and the salvage are kept and the console says so.

A run that stops with a hard error does not lose everything: what was encoded is flushed and packaged, the app says **PARTIAL: encoded M of N frames**, and the file plays up to that point.

---

## 10. Memory: the pre-flight dialog and the three levers

Video restoration is memory-bound before it is compute-bound, and 4K on an 8 GB card is where it shows. When you press **Restore & Save** (or start a folder), ChitraMaya estimates the run's working set from the frame size and your settings against the VRAM that will be free once the models are loaded — using what this card measured on its last run of the same shape, kept in `vram-ledger.json` next to the app, or an edition table before that first run; the dialog says which. If it does not fit you get three choices. **Apply levers and run** sets the panel controls that fit, in plain view, and starts. **Run anyway** keeps your settings. **Cancel**.

The levers, in the order the app tries them: **Crop Cap** to the value that fits; **Async Encode** off; a smaller **Detection Batch**; a shorter **Max Clip Length**; the Secondary off. The biggest single term in a 4K estimate is the scene buffers — the frames of every region currently being restored, held at native size — and they scale with Max Clip Length and with the size of the regions. A 4K60 title with Crop Cap off needs about 5.5 GB for those alone at Max Clip 180.

What happens if you run past the warning: the card fills, the driver starts paging to system memory over PCIe, and everything slows — not just the restorer; in one measured case detection time doubled. The console prints `VRAM PAGING` lines and the run report counts the seconds. On an external GPU enclosure the effect is several times worse, because the link is slower. Heed the dialog.

Older guidance about an "8 GB VRAM rule" and compile-time clip sizes no longer applies: compiled engines are independent of Max Clip Length, and the dial is a runtime setting from 30 to 720.

---

## 11. Reading the console

A few lines worth recognising:

`[VRAM] preflight …` — the same estimate as the dialog, printed after the models load, with the levers it would pull. `[Pipeline] frame N/M  avg X fps  elapsed  ETA` every 500 frames — the true average since the start (the progress bar shows the instantaneous rate between flushes). `[FrameStore] backend: REDECODE` — the normal path on NVIDIA: frames are decoded twice rather than held in memory. `[Decoder] MPEG-TS source detected … remuxing` — a stream capture being made safe to decode (section 13). `[Source] ffprobe auto-detection reports 'png_pipe' … forcing -f mpegts` — a disguised file being handled (section 13). `[Encoder] Audio sidecar: extracting the source audio once` — normal; `source has no audio stream` means the output will be silent on purpose. `[Scalers] … TensorRT … ACTIVE` — a compiled custom model in use. `DONE WITH WARNINGS` — the run finished but something was repaired or refused along the way; the Complete panel and the run report list the reasons. `Complete!` — a clean run. `PARTIAL: encoded M of N frames` — the run failed partway and the file plays to that point.

The stall watchdog monitors long runs and calls out a stalled pipeline instead of letting it sit; on external GPUs it also warns if the PCIe link is running below its trained speed under load. The default threshold is 120 seconds; add `"watchdogStallSeconds": 300` to `ChitraMaya-config.json` next to the exe if it is too eager for your hardware.

The machine stays awake while a run is in progress (the display may still turn off) and is released afterwards.

---

## 12. What to expect from your card

All figures are measured runs with the full default quality stack unless noted. Your titles will differ; the ETA line after 500 frames is the number to trust.

| Card | Job | Time |
|---|---|---|
| RTX 5060 Ti 8 GB | 2h41m 1080p60 feature, restore | 2h31m (TensorRT) |
| RTX 3060 12 GB | the same feature | 3h11m |
| RTX 3060 12 GB | 10-minute 4K60 cut, restore, Max Clip 180 | 46 min (13 fps) |
| RTX 3060 Ti 8 GB | 66-minute 1080p30 title, restore + upscale to 2160p, RTX Super-Res | 3h42m |
| RTX 3060 Ti 8 GB | the same, with a compiled SPAN model | 5h04m |
| RTX 3060 Laptop 6 GB | 4K60 title with Crop Cap 512 | about 23 fps |
| RX 9060 XT 16 GB | 2h41m 1080p60 feature, restore (PyTorch) | 6h51m–7h06m |
| RX 9060 XT 16 GB | 2h25m 720p title, restore + upscale to 2160p (SPAN) | 10h00m |
| Arc 140V laptop iGPU | 2h41m 1080p feature, restore (PyTorch) | 10h08m |

In words: the NVIDIA edition's TensorRT path is two to three times faster than any PyTorch path; a Radeon 9060 XT in PyTorch runs like an RTX 4060 in PyTorch; an Intel laptop is a start-it-before-bed machine. Whole-frame upscaling is hours per hour of video on every card except NVIDIA's RTX Super-Res path, and even there it dominates the run. The [Scaling Guide](ScalingGuide.md) has the full tables.

---

## 13. When something goes wrong

**The app says this is the wrong edition.** It is. Download the edition the message names.

**Get Models answers with 403.** Hugging Face's anonymous limit; wait an hour or download in a browser and drop the files into `models\`.

**The header button is amber after an update.** An engine needs rebuilding: Manage Models → Select all not-compiled → Compile.

**Out of memory in the middle of a run** (`CUDA out of memory`, usually at the title's biggest close-up). The pre-flight warned, or the title's largest region was wider than the estimate assumed. The run flushes what it has as a PARTIAL file. Re-run with Crop Cap set (or a lower Max Clip Length) — the dialog proposes the value.

**The run is very slow and the console shows VRAM PAGING.** The card is full; see section 10. On an external GPU enclosure, expect this at lower settings than on a desktop.

**The output has no sound, or the sound drifts.** Three known causes. A source that is a stream capture (MPEG-TS, often named `.mkv`): the app remuxes it before decoding since 1.61 — an output made by an older version from such a source cannot be repaired, re-run it. A source whose timeline is "30 fps plus a dropped frame every 33 s": fixed in 1.80 (the console line `gap-fill: … output rate` shows the correction); older outputs cannot be repaired, re-run. A source wearing a disguise — a small image glued in front of the video so that casual tools see a picture: since 1.80 the app notices (`[Source] … forcing -f mpegts`) and keeps the audio. If a source really has no audio stream the console says so and the output is silent on purpose.

**A single black, green or magenta frame in an AMD output.** The hardware-decode path occasionally hands over an unfinished frame; the app repairs most of them from the frame before and ends the run with warnings, but not all. The sure fix is **Hardware Decode = Off** (Restoration panel, under Frame Store), about 9 % slower.

**Everything is twenty times slower than the table on an Intel Arc card.** Resizable BAR is off. Turn it on in the BIOS. On an Intel laptop, `could not make an engine with allocator` in the self-check means the graphics driver is too old for the bundled stack; update it and reboot.

**The machine reboots or powers off mid-run, with no error in the log.** A log that simply stops, with no error line, is the signature of a power cut, not of the software: a failing power supply under hours of GPU load is the usual cause (check Event Viewer for Kernel-Power 41 without a bugcheck). Software failures leave a trace. A box that runs batches overnight deserves a reputable power supply with headroom.

**A watchdog stack dump on the first AMD run.** The kernel cache being built; wait.

**The output still shows mosaic in places.** Look at `frames.residual` in the run report: the residual scan lists the frames where the detector still finds mosaic on the finished output, with scores. Some are regions the restorer could not fully clean; some are regions the detector never found in the source — a thin sliver at the frame edge whose neighbour was still mosaic is the classic case. Test Frame on one of those frames shows which. A second pass over the flagged ranges is on the roadmap; today, Save Segment over the range with a lower Detection Score is the manual version.

**Where to report.** The issue tracker at <https://gitlab.com/seatv/chitramaya/-/issues>. Attach `ChitraMaya-console.log` and the `.run.json` beside the output, and state your GPU, driver version and (AMD/Intel) Resizable BAR state; `ChitraMaya-cli.exe -self-check` output helps. **No explicit content in issues** — frame numbers and the report are enough.

---

## 14. AMD and Intel in brief

The product is the same on every edition — editor, Test Frame, Process Folder, Add Mosaic, SBS View, watchdog, keep-awake, crash hardening. What differs:

AMD: no compile step; the first run in a new folder is slow once (kernel cache); detection runs in FP32 regardless of the toggle (FP16 detection was measured to stop finding mosaic after the first frames); restoration FP16 is fine and about 15 % faster; Temporal Stability runs in fp32 and checks each window; decode is D3D11VA with CPU as the fallback and encode is AMF (HEVC, H.264, AV1 on RDNA 3+); the Secondary is Real-ESRGAN by default; RTX Super-Res and the PCIe monitor are NVIDIA-only. Supported cards: RDNA 3 and 4.

Intel: no compile step; Quick Sync decode and encode (AV1, HEVC, H.264) with D3D11VA and CPU fallbacks; Resizable BAR effectively required on discrete cards; 8 GB Arc cards handle 4K flat video, 5K/VR is not recommended on 8 GB; on 16 GB laptops the host frame store may not fit 4K and the RAM guard slows the run rather than letting the machine run out. The `CM_HW_DECODE` environment variable (`auto`, `qsv`, `d3d11va`, `d3d12va`, `off`) overrides the Hardware Decode panel setting for A/B tests from a command window.

---

## 15. Languages, settings and files

The interface is available in English, Simplified Chinese, Traditional Chinese and Thai (the ⚙️ menu). Settings live in `ChitraMaya-config.json` next to the exe — Save Settings / Load Settings / Reset Defaults act on it, and a few advanced keys that have no panel control go in there by hand (`watchdogStallSeconds`, `restoreChunkFrames`). Saved configurations are in `presets\`. Two copies of the app can run at once: the second picks the next port and writes its own log.

---

## 16. Licence and credits

ChitraMaya is free software under the GNU AGPL-3.0: you may use, study, share and modify it, and anything built from it must be shared under the same terms. The models it uses carry their own licences — the lada detection and restoration models, the Real-ESRGAN weights (BSD-3-Clause), the vs_temporalfix weights (Apache-2.0) bundled with the app, and any community model you add. The README's acknowledgements name the people whose work this builds on.
