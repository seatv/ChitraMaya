# tools/probe_rtx_vsr.py
r"""Probe for NVIDIA Maxine VideoSuperRes (nvidia-vfx) — ChitraMaya Batch 20 pre-check.

Works on a SINGLE IMAGE or a FOLDER OF FRAMES (e.g. from saveframes.ps1).
Folder mode applies the SAME crop window to every frame and writes numbered
output frames, ready to reassemble with ffmpeg — the exact commands are
printed at the end (including a side-by-side naive|vsr comparison video).
A moving sequence is the flicker test: if the vsr video shimmers where the
naive one is steady, that's the CM-078 (temporalfix) decision data.

Setup (in the ChitraMaya venv):
    pip install nvidia-vfx

Usage:
    python probe_rtx_vsr.py <image>                          # single image
    python probe_rtx_vsr.py <folder>                         # all png/jpg inside, center crop
    python probe_rtx_vsr.py <folder> --crop 1200,900,500,500 # fixed crop x,y,w,h for all frames
    python probe_rtx_vsr.py <folder> --scale 4 --fps 60 --denoise

The crop is resized to 256x256 (our restored-clip size), then upscaled.
Outputs (folder mode) land in <folder>\probe_out\:
    in256\NNNNN.png      the 256 input fed to SR
    naive_<S>x\NNNNN.png plain bilinear upscale (ChitraMaya's paste-back today)
    vsr_<S>x\NNNNN.png   Maxine SR upscale (the Batch 20 candidate)
    vsr_dn_<S>x\...      SR + denoise-medium (only with --denoise)
Single-image mode writes *_probe_*.png next to the input, as before.
ASCII-only output; exits nonzero with a named cause on failure.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time


def fail(msg: str, hint: str = "") -> None:
    print(f"[FAIL] {msg}")
    if hint:
        print(f"       HINT: {hint}")
    sys.exit(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="image file OR folder of frames (png/jpg)")
    ap.add_argument("--crop", default=None,
                    help="fixed crop window 'x,y,w,h' applied to every frame "
                         "(default: centered square). Get coords from Paint on one frame.")
    ap.add_argument("--scale", type=int, choices=[2, 4], default=4)
    ap.add_argument("--quality", default="high", choices=["low", "medium", "high", "ultra"])
    ap.add_argument("--denoise", action="store_true",
                    help="also produce an SR + denoise-medium output set")
    ap.add_argument("--fps", type=int, default=60, help="fps for the printed ffmpeg commands")
    args = ap.parse_args()

    # -- imports, each with a named failure -------------------------------
    try:
        import torch
    except ImportError:
        fail("PyTorch not importable", "run inside the ChitraMaya venv")
    if not torch.cuda.is_available():
        fail("CUDA not available to PyTorch", "NVIDIA GPU + driver required")
    try:
        import cv2
    except ImportError:
        fail("OpenCV not importable", "pip install opencv-python")

    # Preload pip TensorRT libs BEFORE nvvfx (Maxine bundles its own TRT and
    # loads it globally; ours must win symbol resolution — the jasna lesson).
    try:
        import ctypes
        import tensorrt_libs  # noqa: F401
        libs_dir = os.path.dirname(tensorrt_libs.__file__)
        if sys.platform == "win32":
            for name in ("nvinfer_10.dll", "nvinfer_plugin_10.dll"):
                p = os.path.join(libs_dir, name)
                if os.path.isfile(p):
                    ctypes.WinDLL(p)
        print("[OK]   pip TensorRT runtime preloaded (guards BasicVSR++ engines)")
    except ImportError:
        print("[note] tensorrt_libs not present; preload skipped (fine for this probe)")

    try:
        from nvvfx import VideoSuperRes
    except ImportError as e:
        fail(f"nvvfx not importable: {e}",
             "pip install nvidia-vfx  (needs an RTX GPU + recent driver)")

    # -- gather inputs ----------------------------------------------------
    if os.path.isdir(args.input):
        files = sorted(
            glob.glob(os.path.join(args.input, "*.png"))
            + glob.glob(os.path.join(args.input, "*.jpg"))
            + glob.glob(os.path.join(args.input, "*.jpeg"))
        )
        if not files:
            fail(f"no png/jpg frames found in folder: {args.input}")
        folder_mode = True
        out_root = os.path.join(args.input, "probe_out")
        print(f"[OK]   folder mode: {len(files)} frames from {args.input}")
    elif os.path.isfile(args.input):
        files = [args.input]
        folder_mode = False
        out_root = None
    else:
        fail(f"input not found: {args.input}")

    crop = None
    if args.crop:
        try:
            cx, cy, cw, ch = (int(v) for v in args.crop.split(","))
            if cw <= 0 or ch <= 0:
                raise ValueError
            crop = (cx, cy, cw, ch)
            print(f"[OK]   fixed crop window: x={cx} y={cy} w={cw} h={ch}")
        except ValueError:
            fail(f"bad --crop value: {args.crop!r}", "format is x,y,w,h in pixels, e.g. 1200,900,500,500")

    def crop256(img):
        h, w = img.shape[:2]
        if crop is not None:
            cx, cy, cw, ch = crop
            cx2, cy2 = min(w, cx + cw), min(h, cy + ch)
            cx0, cy0 = max(0, cx), max(0, cy)
            region = img[cy0:cy2, cx0:cx2]
            if region.size == 0:
                fail(f"crop window {crop} lies outside a {w}x{h} frame")
            side = min(region.shape[0], region.shape[1])
            region = region[:side, :side]
        else:
            side = min(h, w)
            y0, x0 = (h - side) // 2, (w - side) // 2
            region = img[y0:y0 + side, x0:x0 + side]
        return cv2.resize(region, (256, 256), interpolation=cv2.INTER_AREA)

    # -- load effects ONCE (reused across all frames) ---------------------
    device = torch.device("cuda:0")
    stream_ptr = torch.cuda.current_stream(device).cuda_stream
    out_size = 256 * args.scale
    qmap = {"low": VideoSuperRes.QualityLevel.LOW,
            "medium": VideoSuperRes.QualityLevel.MEDIUM,
            "high": VideoSuperRes.QualityLevel.HIGH,
            "ultra": VideoSuperRes.QualityLevel.ULTRA}

    try:
        sr = VideoSuperRes(device=0, quality=qmap[args.quality])
        sr.output_width = out_size
        sr.output_height = out_size
        sr.load()
        print(f"[OK]   VideoSuperRes loaded: {args.scale}x quality={args.quality} (256 -> {out_size})")
    except Exception as e:
        fail(f"VideoSuperRes load failed: {e}",
             "update the NVIDIA driver; Maxine needs a recent one. If it names a missing "
             "model/DLL, the nvidia-vfx wheel may not support this GPU/OS combo.")

    dn = None
    if args.denoise:
        try:
            dn = VideoSuperRes(device=0, quality=VideoSuperRes.QualityLevel.DENOISE_MEDIUM)
            dn.output_width = out_size
            dn.output_height = out_size
            dn.load()
            print("[OK]   denoise-medium pass loaded")
        except Exception as e:
            print(f"[note] denoise pass unavailable ({e}) — continuing with SR only")
            dn = None

    _layout = {"value": None}  # determined on first frame, reused after

    def run_effect(effect, frame_hwc):
        if _layout["value"] == "HWC":
            return torch.from_dlpack(effect.run(frame_hwc, stream_ptr=stream_ptr).image).clone()
        if _layout["value"] == "CHW":
            t = frame_hwc.permute(2, 0, 1).contiguous()
            return torch.from_dlpack(effect.run(t, stream_ptr=stream_ptr).image).clone()
        last = None
        for layout, tensor in (("HWC", frame_hwc),
                               ("CHW", frame_hwc.permute(2, 0, 1).contiguous())):
            try:
                out = torch.from_dlpack(effect.run(tensor, stream_ptr=stream_ptr).image).clone()
                _layout["value"] = layout
                print(f"[OK]   input layout resolved: {layout}")
                return out
            except Exception as e:
                last = e
        raise RuntimeError(f"both HWC and CHW layouts rejected: {last}")

    def to_bgr_u8(t: "torch.Tensor"):
        x = t.detach().float().cpu()
        if x.ndim == 3 and x.shape[0] in (1, 3):     # CHW -> HWC
            x = x.permute(1, 2, 0)
        if x.ndim == 3 and x.shape[-1] == 1:
            x = x.repeat(1, 1, 3)
        arr = (x.clamp(0, 1) * 255.0).round().byte().numpy()
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    # -- output dirs ------------------------------------------------------
    if folder_mode:
        dirs = {
            "in256": os.path.join(out_root, "in256"),
            "naive": os.path.join(out_root, f"naive_{args.scale}x"),
            "vsr": os.path.join(out_root, f"vsr_{args.scale}x"),
        }
        if dn is not None:
            dirs["vsr_dn"] = os.path.join(out_root, f"vsr_dn_{args.scale}x")
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)

    # -- process ----------------------------------------------------------
    t0 = time.perf_counter()
    for idx, path in enumerate(files, start=1):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[note] unreadable, skipped: {path}")
            continue
        img256 = crop256(img)
        naive = cv2.resize(img256, (out_size, out_size), interpolation=cv2.INTER_LINEAR)

        rgb = cv2.cvtColor(img256, cv2.COLOR_BGR2RGB)
        t_hwc = torch.from_numpy(rgb).to(device=device, dtype=torch.float32).div_(255.0).contiguous()

        try:
            out = run_effect(sr, t_hwc)
        except Exception as e:
            fail(f"SR inference failed on {os.path.basename(path)}: {e}")

        if folder_mode:
            n = f"{idx:05d}.png"
            cv2.imwrite(os.path.join(dirs["in256"], n), img256)
            cv2.imwrite(os.path.join(dirs["naive"], n), naive)
            cv2.imwrite(os.path.join(dirs["vsr"], n), to_bgr_u8(out))
            if dn is not None:
                hwc = out if (out.ndim == 3 and out.shape[-1] == 3) else out.permute(1, 2, 0).contiguous()
                out2 = run_effect(dn, hwc.float().clamp(0, 1))
                cv2.imwrite(os.path.join(dirs["vsr_dn"], n), to_bgr_u8(out2))
            if idx % 50 == 0 or idx == len(files):
                el = time.perf_counter() - t0
                print(f"[..]   {idx}/{len(files)} frames  ({idx / el:.1f} fps)")
        else:
            base, _ = os.path.splitext(path)
            cv2.imwrite(f"{base}_probe_256.png", img256)
            cv2.imwrite(f"{base}_probe_naive_{args.scale}x.png", naive)
            cv2.imwrite(f"{base}_probe_vsr_{args.scale}x.png", to_bgr_u8(out))
            if dn is not None:
                hwc = out if (out.ndim == 3 and out.shape[-1] == 3) else out.permute(1, 2, 0).contiguous()
                out2 = run_effect(dn, hwc.float().clamp(0, 1))
                cv2.imwrite(f"{base}_probe_vsr_dn_{args.scale}x.png", to_bgr_u8(out2))
            print(f"[OK]   wrote {base}_probe_*.png")

    sr.close()
    if dn is not None:
        dn.close()

    # -- assembly instructions -------------------------------------------
    print()
    if folder_mode:
        el = time.perf_counter() - t0
        print(f"RESULT: {len(files)} frames processed in {el:.1f}s ({len(files) / el:.1f} fps).")
        print()
        print("Assemble the videos (run from anywhere):")
        naive_d = dirs["naive"]
        vsr_d = dirs["vsr"]
        print(f'  ffmpeg -framerate {args.fps} -i "{naive_d}\\%05d.png" -c:v hevc_nvenc -preset p7 -cq 18 -pix_fmt yuv420p "{out_root}\\naive_{args.scale}x.mp4"')
        print(f'  ffmpeg -framerate {args.fps} -i "{vsr_d}\\%05d.png" -c:v hevc_nvenc -preset p7 -cq 18 -pix_fmt yuv420p "{out_root}\\vsr_{args.scale}x.mp4"')
        if dn is not None:
            dn_d = dirs["vsr_dn"]
            print(f'  ffmpeg -framerate {args.fps} -i "{dn_d}\\%05d.png" -c:v hevc_nvenc -preset p7 -cq 18 -pix_fmt yuv420p "{out_root}\\vsr_dn_{args.scale}x.mp4"')
        print()
        print("Side-by-side comparison (naive left | vsr right):")
        print(f'  ffmpeg -framerate {args.fps} -i "{naive_d}\\%05d.png" -framerate {args.fps} -i "{vsr_d}\\%05d.png" '
              f'-filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" '
              f'-c:v hevc_nvenc -preset p7 -cq 18 -pix_fmt yuv420p "{out_root}\\compare_{args.scale}x.mp4"')
        print()
        print("Judge two things: sharpness (any frame, 100% zoom) and FLICKER in motion")
        print("(vsr video shimmering where naive is steady = the CM-078 temporalfix signal).")
    else:
        print("RESULT: probe complete. Compare *_naive_* vs *_vsr_* at 100% zoom.")
    return 0


if __name__ == "__main__":
    sys.exit(main())