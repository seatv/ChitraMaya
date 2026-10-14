# Changelog

All notable changes to ChitraMaya. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the long-form release notes for each version are on the Releases page. CM-numbers are the project's change tickets.

## [1.80.00] — 2026-10-12

A feature release for every edition. Whole-frame upscaling as its own mode, custom super-resolution models (with TensorRT engines on NVIDIA), one-button model setup, the recommended settings as defaults, the 8 GB / 4K story closed on NVIDIA, Temporal Stability on AMD, a VRAM pre-flight that asks before the run, a User Guide, and the interface in 简体中文, 繁體中文 and Thai.

### Added
- Upscale mode and Output Upscale: whole-frame upscaling to 1080p/1440p/2160p with Real-ESRGAN (every edition) or RTX Super-Res (NVIDIA), inside a restoration run or on its own; Save Segment writes a marked range as a file (CM-219, CM-228).
- Custom super-resolution models in `models\upscalers\` appear in both scaler dropdowns; some forty architectures identified from the file; SPAN layers fused once at load (CM-229).
- TensorRT engines for custom scalers: `ChitraMaya-cli.exe -compile-sr <model> --secondary` builds the whole-frame and region engines beside the weights; measured 2.1× on the upscale (CM-242, CM-247).
- Manage Models does the whole first setup: Get Models fetches the standard set, Compile Models builds engines; the header button says which step is next; custom-scaler section (CM-232).
- VRAM pre-flight dialog before Restore & Save and folder batches, with "apply levers and run" (Crop Cap, Async off, Detection Batch, Max Clip, Secondary) from a per-card ledger (CM-221, CM-236).
- Re-decode frame store: no frames held between detection and paste-back on NVIDIA; Max Clip Length costs no frame memory (CM-191, CM-196).
- Right-sized detection engines: ~0.4 GB instead of 1.6–3.1 GB; engine flavours per Image Size and batch (CM-201, CM-218).
- Crop Cap: open-scene crops above N px stored downscaled, the lever for 4K close-ups on small cards; aligned resampling so capped crops keep native detail (CM-227, T26).
- Overlap + Crossfade at Max Clip cuts; outward-feathered Face Swap blend with a frame-scaled auto width; Scaler Quality; Region Margin (CM-213, CM-086, CM-214, CM-217, CM-208).
- Residual scan: the detector checks the finished output and reports frames where mosaic remains (CM-205).
- Run report `<name>.run.json` beside every output: asked vs ran, timings, VRAM ledger, counts, per-frame lists; output suffix field; Run Files switch (CM-180, CM-202).
- Watchdog GPU sampler with paging detection (CM-209); `DONE WITH WARNINGS` for runs whose guards acted (T20c).
- One-pass finalize with the index reserved up front; output verified by read-back; salvage (raw bitstream, RECOVER script, audio sidecar) kept until the last step (CM-187, CM-237).
- Audio sidecar extracted once at run start (CM-187); CM-246: sources with a picture header glued in front of the stream (ffprobe reports `png_pipe`) are handled by forcing the container from the extension, so their audio and the TS remux are kept.
- Upscale mode lists restored outputs as inputs and names outputs `-upscaled`; a run keeps the source it started with and load-video is refused while a run is in progress (CM-245).
- Hardware Decode control (AMD/Intel) and a decode guard that completes unfinished frames from the D3D11VA path (CM-234); Temporal Stability in fp32 on ROCm, each window checked.
- Interface in Simplified Chinese, Traditional Chinese (community) and Thai (machine draft); `hf-mirror.com` fallback for mainland China.
- Saved configurations with a built-in VR 4K; honest `avg fps / ETA` console checkpoints; keep-awake through finalize (CM-179).
- User Guide (`docs/UserGuide.md`, shipped as `UserGuide.html` with a Help button); this changelog; the README reduced to a front door (CM-248).

### Changed
- Defaults: Secondary on (RTX Super-Res 4× / Real-ESRGAN 4×), Detection Score 0.20, Detection Batch 8, Temporal Stability 1, Overlap 8 with Crossfade, Feather auto, Max Clip Length 180.
- Downloads move to <https://get.chitramaya.org>; source and issues to <https://gitlab.com/seatv/chitramaya>; the GitHub repositories become mirrors.
- The 8 GB advice: Max Clip Length about three times the frame rate (measured: 90/180/240 differ by a few percent of run time; 300 on a full card slows the whole run).

### Fixed
- 8 GB cards at 4K (1.70 known issue): a 2h20m 4K60 title completes on an RTX 3060 Ti (CM-191, CM-201).
- Face Swap edge irregularities (CM-086); detection batch above the engine profile (CM-203); RTX Super-Res without the Maxine runtime now falls back to Real-ESRGAN and says so (CM-184).
- Audio drift on "30 fps with a dropped frame every 33 s" sources: the container rate follows the encoded-vs-source counts (CM-235).
- A run that errored printed a false audio-sync warning; folder batches printed no checkpoints after the first file; lada `_full.pth` checkpoints load (#9).
- MPEG-TS temporary remux is video-only; audio comes from the original (CM-190); an all-black restorer result is not pasted (CM-186).
- Custom-scaler run report said `torch` for an engine run (read after the engines were released) (T25c).
- Detection FP16 toggle cannot stop detection on ROCm (fp32 by design, CM-169).

### Known issues
- AMD: unfinished frames from the D3D11VA decode path are guarded, not prevented; Hardware Decode = Off avoids them.
- The RTX Super-Res whole-frame scaler takes Quality and Denoise from the Secondary's controls.
- A run that delivered zero frames self-grades as complete (CM-129); HEVC sources may show a black in-app player without the HEVC Video Extensions (CM-130).
- A thin mosaic sliver at the frame edge whose neighbour is still mosaic can be missed by the detector; the residual scan reports it (a second pass is planned).

## [1.70.00] — 2026-09-10

AMD users: every previous AMD edition encoded on the CPU because of a probe bug, and the default blend mask compiled a kernel per region; both fixed and measured on an RX 9060 XT — a 2h41m feature in about seven hours instead of twenty-two (CM-167, CM-172). Detection FP16 off on ROCm by design (CM-169). All editions: the Real-ESRGAN secondary, named saved configurations, a startup check that names the right edition for the GPU, honest speed/ETA readouts, ffmpeg 9 compatibility, long-filename fixes. A 1.70.99-Beta preceded the 1.80.00 work.

## [1.61.00] — 2026-08

Audio desync on MPEG-TS stream captures fixed in three places: TS sources are losslessly remuxed to a temporary MP4 before decode, the timestamp safety net restored, the A/V start-offset probe fixed (CM-120, CM-121). The encoder cannot fail silently; raw bitstream plus a RECOVER script survive a failed finalize; finalize supervised by disk progress. 8K VR on pre-Blackwell cards falls back to CPU decode and works. Failed runs report what survived.

## [1.60.00] — 2026-08

TensorRT restoration engines are clip-size independent (Jasna's fixed-batch sub-engines): one compile per model, Max Clip Length a runtime dial from 30 to 720, ~1.6 GB of VRAM freed on 8 GB cards (CM-104). Host frame store with a RAM plan and a RAM guard for long clips. Dual-GPU groundwork (CM-083). 1.60.01 followed with a fix.

## [1.50.00] — 2026-08

Full-clip restoration on the PyTorch path (the Max Clip dial is the temporal window; AMD and Intel's biggest quality change). AMD Radeon (ROCm) edition first released; Intel Arc installer as an `.exe`. Colour tags propagated from source to output. Quality measurement tooling.

## [1.41.00] — 2026-08

Follow-ups to 1.40: fixes from the field, including the 5060 Ti hang forensics and AV1 seek-header repair for files made by earlier versions. 1.40.01 and 1.40.02 were the hotfixes in between.

## [1.40.00] — 2026-07-30

Process Folder batch mode with a queue, per-file isolation and a stall watchdog (CM-079, CM-080); RTX Super-Res secondary (CM-077); Temporal Stability 0–3 with bundled weights (CM-078); the windowed app plus `ChitraMaya-cli.exe` and the console drawer (CM-064); Test Frame history, five deep (CM-082); AV1 seek fix (CM-091); the full P1–P7 preset ladder; the final scene of a video now gets the secondary and projection like every other.

## [1.30.00] — 2026-07

VR Projection (fisheye) for mosaics applied in viewing space; AV1 output on RTX 40-series and newer with a QP mapping; Image Size as a runtime detection dial (800 for 4K/VR); the full path of the loaded file in the top bar; a detection post-processing time limit that silently skipped frames in dense scenes raised; AV1 remux fixed; settings changes always rebuild the pipeline.

## [1.20.00] — 2026-07

SBS View: a projected look-around comparison of original and restored for side-by-side VR content, with a wipe; Manage Models dialog fixes (#3); single-source version.

## [1.10.00] — 2026-07

Add Mosaic (manual rectangles; experimental auto-detect), per-eye handling for SBS, the first Intel Arc edition.

## [1.01.00] — 2026-06

Stability pass over 1.0: hangs on long restores surfaced as errors with the partial output saved; a VRAM pre-flight; Max Clip flexible across compiled engine sets; FaceFusion blending live; Detection Batch honoured; models resolve relative to the install folder; Hugging Face 403 fixed; async encoding opt-in.

## [1.00.00] — 2026-06

First release: the interactive editor with Test Frame, segment preview and Restore & Save; YOLO detection, BasicVSR++ restoration through TensorRT sub-engines, NVDEC/NVENC end to end; Manage Models with downloads and compiles; the split installer.
