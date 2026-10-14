"""T25 (CM-242): compile a custom super-resolution model to TensorRT engine(s).

    ChitraMaya-cli.exe -compile-sr 4xNomos8k_span_otf_strong.pth --secondary --bench   (installed build)
    python -m tools.compile_custom_sr 4xNomos8k_span_otf_strong.pth                     (from source)
    python -m tools.compile_custom_sr 4xNomos8k_span_otf_strong.pth --secondary
    python -m tools.compile_custom_sr models/upscalers/foo.pth --tile 704 --tile 256 --bench
    python -m tools.compile_custom_sr foo.pth --bench --no-compile   # bench existing engines only

The model is a file name inside models\\upscalers (or a path). Default tile
704 = the whole-frame tiler's 640-px tile + 32-px halo each side; --secondary
adds a 256-px engine for the Secondary (crop) scaler. Engines and their .json
sidecars are written beside the weights (see custom_sr_trt.py). --bench times
one tile through torch and through the engine and projects a 1080p frame.

NVIDIA edition only. Run from the repo root with the app's venv active.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _resolve(name: str, base_dir: str | None) -> Path:
    p = Path(name)
    if p.is_file():
        return p
    from chitramaya.mosaic.restorer.custom_sr import find_custom_model, CUSTOM_PREFIX, is_custom
    val = name if is_custom(name) else CUSTOM_PREFIX + Path(name).name
    found = find_custom_model(val, base_dir)
    if found is None:
        raise SystemExit(f"not found: {name} (looked in models\\upscalers)")
    return Path(found)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="file name in models\\upscalers, or a path")
    ap.add_argument("--tile", type=int, action="append", help="engine tile (repeatable; default 704)")
    ap.add_argument("--secondary", action="store_true", help="also build the 256-px Secondary engine")
    ap.add_argument("--fp32", action="store_true", help="compile fp32 (default fp16 when the model allows)")
    ap.add_argument("--workspace-mb", type=int, default=2048)
    ap.add_argument("--opt-level", type=int, default=3, help="TensorRT builder optimization level 0..5")
    ap.add_argument("--bench", action="store_true", help="time torch vs engine per tile and per 1080p frame")
    ap.add_argument("--no-compile", action="store_true", help="skip compiling (bench existing engines)")
    ap.add_argument("--frame", default="1920x1080", help="frame size for the --bench projection, WxH")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--base-dir", default=None,
                    help="app base dir holding models\\upscalers (default: the folder next to the exe in an "
                         "installed build, else the current directory)")
    a = ap.parse_args(argv)
    # CM-247: from ChitraMaya-cli.exe the user can be in any folder; the models
    # live next to the exe, so that is where a bare file name is looked up.
    if a.base_dir is None and getattr(sys, "frozen", False):
        a.base_dir = str(Path(sys.executable).resolve().parent)

    import torch
    if not torch.cuda.is_available():
        print("CUDA is not available; TensorRT engines need an NVIDIA GPU.")
        return 2
    device = torch.device(a.device)
    model_path = _resolve(a.model, a.base_dir)
    tiles = list(a.tile or [])
    if not tiles:
        tiles = [704]
    if a.secondary and 256 not in tiles:
        tiles.append(256)
    fp16 = not a.fp32

    from chitramaya.mosaic.restorer import custom_sr_trt as trt
    print(f"[compile_custom_sr] {model_path}  tiles {tiles}  {'fp32' if a.fp32 else 'fp16'}  "
          f"device {torch.cuda.get_device_name(device)}")
    rc = 0
    if not a.no_compile:
        for t in tiles:
            try:
                meta = trt.compile_engine(model_path, device, tile=int(t), fp16=fp16,
                                          workspace_mb=int(a.workspace_mb), optimization_level=int(a.opt_level))
                print(f"[compile_custom_sr] OK  tile {t}: {Path(meta['engine']).name}  "
                      f"PSNR vs torch {meta['parity']['texture_psnr_db']} dB")
            except Exception as e:
                rc = 1
                print(f"[compile_custom_sr] FAILED tile {t}: {type(e).__name__}: {e}")
    if a.bench:
        try:
            w, h = (int(v) for v in a.frame.lower().split("x"))
        except Exception:
            w, h = 1920, 1080
        for t in tiles:
            try:
                trt.bench(model_path, device, tile=int(t), fp16=fp16, frame_hw=(h, w))
            except Exception as e:
                rc = 1
                print(f"[compile_custom_sr] bench FAILED tile {t}: {type(e).__name__}: {e}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
