# tools/compile_yolo.py
"""Compile a YOLO .pt checkpoint to a TensorRT .engine via ultralytics' export.

Ultralytics' AutoBackend (used by ``LadaYoloDetector``) auto-detects the file
extension, so after compilation you can swap ``--det-model X.pt`` for
``--det-model X.engine`` to take the TRT path. The .engine file lands next
to the .pt with the same stem.

Usage from the ChitraMaya unified CLI:

    ChitraMaya -compile-det --det-model PATH/TO/YOLO.pt --det-imgsz 640

Or directly:

    python -m tools.compile_yolo --det-model PATH/TO/YOLO.pt --det-imgsz 640
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile a YOLO .pt to a TensorRT .engine via ultralytics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--det-model", required=True,
        help="Path to YOLO .pt checkpoint",
    )
    parser.add_argument(
        "--det-imgsz", type=int, default=640,
        help="OPT image size (default: 640). With --dynamic, the engine still "
             "accepts any size from 32 to workspace*imgsz, but is optimized for "
             "this size. Pick the size you run at most often.",
    )
    parser.add_argument(
        "--fp16", action=argparse.BooleanOptionalAction, default=True,
        help="Build fp16 engine (default: True). Use --no-fp16 for fp32. "
             "Note: ultralytics' fp16 + static-shape export causes a cuTensor "
             "crash at runtime warmup (FP32 input vs FP16 weights mismatch). "
             "--dynamic=True is the workaround we use here.",
    )
    parser.add_argument(
        "--dynamic", action=argparse.BooleanOptionalAction, default=True,
        help="Build a dynamic-shape engine (default: True). One engine handles "
             "any imgsz from 32 to workspace*imgsz and any batch from 1 to "
             "--max-batch. With dynamic=False, the engine is locked to "
             "(--max-batch, --det-imgsz) at compile time.",
    )
    parser.add_argument(
        "--max-batch", type=int, default=16,
        help="Maximum batch size the engine should support (default: 16). "
             "Ultralytics requires this to be >1 when --dynamic=True. "
             "Increase if your pipeline ever exceeds 16 frames per detection batch.",
    )
    parser.add_argument(
        "--workspace", type=int, default=4,
        help="TRT workspace in GB (default: 4). With --dynamic, this also caps "
             "the engine's max accepted imgsz at workspace*imgsz "
             "(e.g. 4*640=2560).",
    )
    parser.add_argument(
        "--gpu-id", type=int, default=0,
        help="CUDA GPU index (default: 0)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recompile even if a matching .engine already exists",
    )
    args = parser.parse_args()

    model_path = Path(args.det_model)
    if not model_path.is_file():
        print(f"[!] Model not found: {model_path}", file=sys.stderr)
        return 1

    if model_path.suffix.lower() != ".pt":
        print(
            f"[!] Expected a .pt file, got: {model_path}",
            file=sys.stderr,
        )
        return 1

    # Engines live in <models>/engines/<stem>.engine, mirroring the convention
    # used for face-swap engines (convert_models.py). Ultralytics hardcodes its
    # output to <pt_parent>/<stem>.engine, so we let it write there and move
    # the file after — same logic for the intermediate .onnx ultralytics leaves
    # behind.
    engine_dir = model_path.parent / "engines"
    engine_path = engine_dir / f"{model_path.stem}.engine"
    ultra_engine_path = model_path.with_suffix(".engine")    # where ultralytics writes
    ultra_onnx_path = model_path.with_suffix(".onnx")        # intermediate ultralytics leaves

    print(f"[compile-yolo] checkpoint:    {model_path}")
    print(f"[compile-yolo] engine target: {engine_path}")
    print(f"[compile-yolo] dynamic:       {bool(args.dynamic)}")
    print(f"[compile-yolo] opt imgsz:     {args.det_imgsz}")
    print(f"[compile-yolo] max batch:     {args.max_batch}")
    print(f"[compile-yolo] precision:     {'fp16' if args.fp16 else 'fp32'}")
    print(f"[compile-yolo] workspace:     {args.workspace} GB")
    if args.dynamic:
        max_dim = args.workspace * args.det_imgsz
        print(f"[compile-yolo] dynamic shape range: batch 1..{args.max_batch}, "
              f"H,W 32..{max_dim} (opt at {args.det_imgsz})")
    print(f"[compile-yolo] device:        cuda:{args.gpu_id}")
    print()

    if engine_path.is_file() and not args.force:
        size_mb = engine_path.stat().st_size / (1024 * 1024)
        print(
            f"[compile-yolo] Engine already exists ({size_mb:.1f} MB). "
            f"Pass --force to rebuild."
        )
        return 0

    # Ensure engines/ exists
    engine_dir.mkdir(parents=True, exist_ok=True)

    # Force-clean: remove the engine at the final location AND any stale files
    # left in the ultralytics default location (from prior or failed runs).
    if args.force:
        for p in (engine_path, ultra_engine_path):
            try:
                p.unlink()
            except FileNotFoundError:
                pass

    try:
        import torch
        if not torch.cuda.is_available():
            print(
                "[!] CUDA not available; cannot compile TensorRT engines.",
                file=sys.stderr,
            )
            return 1
    except ImportError as e:
        print(f"[!] Failed to import torch: {e}", file=sys.stderr)
        return 1

    try:
        from ultralytics import YOLO
    except ImportError as e:
        print(f"[!] Failed to import ultralytics: {e}", file=sys.stderr)
        return 1

    print("[compile-yolo] Loading YOLO model ...")
    t0 = time.perf_counter()
    try:
        model = YOLO(str(model_path))
    except Exception as e:
        print(f"[!] YOLO load failed: {e}", file=sys.stderr)
        return 1
    print(f"[compile-yolo] Loaded ({time.perf_counter() - t0:.1f}s)")
    print()

    print("[compile-yolo] Exporting to TensorRT engine (this can take several minutes) ...")
    t0 = time.perf_counter()
    try:
        # Ultralytics writes the engine to model_path.with_suffix(".engine").
        # When dynamic=True, the engine accepts:
        #   - any batch size from 1 to args.max_batch
        #   - any imgsz from 32 to args.workspace * args.det_imgsz
        # opt-tuned for (args.max_batch, args.det_imgsz).
        # When dynamic=False, shape is locked to (args.max_batch, args.det_imgsz).
        # Ultralytics asserts batch>1 when dynamic=True; we honor that here.
        exported_path = model.export(
            format="engine",
            imgsz=int(args.det_imgsz),
            half=bool(args.fp16),
            dynamic=bool(args.dynamic),
            batch=int(args.max_batch),
            simplify=True,
            workspace=int(args.workspace),
            device=int(args.gpu_id),
            verbose=False,
        )
    except Exception as e:
        print(f"[!] YOLO export failed: {e}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - t0

    # Some ultralytics versions return PosixPath, some return str
    exported_path = Path(str(exported_path)) if exported_path else ultra_engine_path
    if not exported_path.is_file():
        # Ultralytics may also have written to the convention-based path
        exported_path = ultra_engine_path

    if not exported_path.is_file():
        print(
            f"[!] Export reported success but no engine file at {ultra_engine_path}",
            file=sys.stderr,
        )
        return 1

    # Move the engine from ultralytics' default location to <models>/engines/.
    # On Windows, shutil.move handles cross-directory atomicity better than
    # rename(), though here both paths are on the same volume.
    import shutil
    if exported_path.resolve() != engine_path.resolve():
        # Remove any leftover at the destination (already handled above for --force,
        # but be defensive in case engine_path was created by something else).
        if engine_path.is_file():
            engine_path.unlink()
        shutil.move(str(exported_path), str(engine_path))

    # Clean up the intermediate .onnx ultralytics leaves next to the .pt.
    # It's only useful during compile; recompile regenerates it.
    if ultra_onnx_path.is_file():
        try:
            ultra_onnx_path.unlink()
        except OSError as e:
            print(f"[compile-yolo] warning: could not remove intermediate "
                  f"{ultra_onnx_path}: {e}")

    size_mb = engine_path.stat().st_size / (1024 * 1024)
    print()
    print(f"[compile-yolo] Done in {elapsed:.1f}s.")
    print(f"[compile-yolo] Engine: {engine_path} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
