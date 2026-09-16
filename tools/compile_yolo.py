# tools/compile_yolo.py
"""Compile YOLO .pt checkpoint(s) to TensorRT .engine files via ultralytics' export.

CM-201 (2026-09-15): the default is now a FIXED-IMGSZ profile built by this
tool from ultralytics' ONNX export: batch 1..--max-batch, H,W 32..imgsz
(opt = --max-batch x imgsz x imgsz). ultralytics' own engine export ties the
maximum shape to the workspace number (max H,W = max(2, workspace) x imgsz),
so the field engines carried a batch-8 x 1600^2 profile while runs use
4 x 800^2 -- TensorRT sizes the execution context for the largest profile
shape, and the ledger measured that at 3.1 GB on a 3060 Ti (vs 1.1 GB for
the whole restorer). The engine header is ultralytics' (4-byte length +
JSON metadata + engine), so AutoBackend loads it unchanged; a
<engine>.json sidecar records the profile. --profile legacy keeps the old
ultralytics export path.

Ultralytics' AutoBackend (used by ``LadaYoloDetector``) auto-detects the file
extension, so after compilation you can swap ``--det-model X.pt`` for
``--det-model X.engine`` to take the TRT path. The .engine file lands in
``<models>/engines/<stem>.engine``.

``--det-model`` accepts EITHER a single .pt file OR a directory. When given a
directory, every ``*.pt`` file directly inside it is compiled in turn.

Usage from the ChitraMaya unified CLI:

    ChitraMaya -compile-det --det-model PATH/TO/YOLO.pt --det-imgsz 640
    ChitraMaya -compile-det --det-model PATH/TO/models   --det-imgsz 640   # all *.pt

Or directly:

    python -m tools.compile_yolo --det-model PATH/TO/YOLO.pt --det-imgsz 640
    python -m tools.compile_yolo --det-model PATH/TO/models   --det-imgsz 640
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path


def _free_cuda(model=None):
    """Release PyTorch's CUDA cache so TensorRT / the next model can use the VRAM.

    ultralytics' export loads the model onto CUDA (for the ONNX trace) and
    PyTorch's caching allocator does NOT return that memory to the driver on
    its own. On an 8 GB card this can leave TensorRT's builder — or the next
    model in a directory batch — without enough VRAM. Dropping the model
    reference + emptying the cache + gc gives it back.
    """
    try:
        if model is not None:
            del model
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _onnx_metadata(onnx_path: Path) -> dict:
    """ultralytics writes its export metadata into the ONNX metadata_props as
    str(value); recover the original types (dicts, lists, ints) so the engine
    header we write matches what ultralytics' own export would carry."""
    import ast
    import onnx
    m = onnx.load(str(onnx_path), load_external_data=False)
    out = {}
    for prop in m.metadata_props:
        v = prop.value
        try:
            v = ast.literal_eval(v)
        except Exception:
            pass
        out[prop.key] = v
    return out


def _build_fixed_profile_engine(onnx_path: Path, engine_path: Path, *, imgsz: int,
                                max_batch: int, fp16: bool, workspace_gb: int,
                                metadata: dict) -> None:
    """CM-201: build the TensorRT engine with the profile ChitraMaya runs:
    min (1,3,32,32) -- the rect letterbox is always <= imgsz on both sides --
    opt/max (max_batch, 3, imgsz, imgsz). Writes the ultralytics header."""
    import json
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    if workspace_gb and workspace_gb > 0:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb) * (1 << 30))
    try:
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    except Exception:
        flag = 0
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed: {errs}")
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        shp = tuple(int(d) for d in inp.shape)
        mn = tuple(d if d != -1 else lo for d, lo in zip(shp, (1, 3, 32, 32)))
        op = tuple(d if d != -1 else v for d, v in zip(shp, (int(max_batch), 3, int(imgsz), int(imgsz))))
        profile.set_shape(inp.name, min=mn, opt=op, max=op)
        print(f"[compile-yolo] profile {inp.name}: min {mn} opt {op} max {op}")
    config.add_optimization_profile(profile)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("TensorRT engine build failed (see the TensorRT log above)")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as t:
        meta = json.dumps(metadata)
        t.write(len(meta).to_bytes(4, byteorder="little", signed=True))
        t.write(meta.encode())
        t.write(bytes(engine))


def _write_profile_sidecar(engine_path: Path, *, imgsz: int, max_batch: int, fp16: bool,
                           profile: str, gpu_id: int, workspace_gb: int) -> None:
    """<engine>.json: what this engine accepts and what it was built with --
    read at model load (CM-201 line) and by the CM-151 manifest later."""
    import json
    import datetime as _dt
    info = {"profile": profile, "imgsz": int(imgsz), "max_batch": int(max_batch),
            "fp16": bool(fp16), "workspace_gb": int(workspace_gb),
            "built": _dt.datetime.now().isoformat(timespec="seconds")}
    if profile == "fixed":
        info["shape_min"] = [1, 3, 32, 32]
        info["shape_max"] = [int(max_batch), 3, int(imgsz), int(imgsz)]
    else:
        info["shape_min"] = [1, 3, 32, 32]
        info["shape_max"] = [int(max_batch), 3, 2 * int(imgsz), 2 * int(imgsz)]
        info["note"] = "ultralytics dynamic export: max H,W = max(2, workspace) x imgsz"
    try:
        import torch
        info["gpu"] = torch.cuda.get_device_name(int(gpu_id))
        import tensorrt as trt
        info["tensorrt"] = str(trt.__version__)
    except Exception:
        pass
    try:
        Path(str(engine_path) + ".json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[compile-yolo] warning: could not write the profile sidecar: {e}")


def compile_one(model_path: Path, args) -> int:
    """Compile a single YOLO .pt to a TensorRT engine.

    Returns 0 on success (or already-exists), non-zero on failure.
    """
    if model_path.suffix.lower() != ".pt":
        print(f"[!] Expected a .pt file, got: {model_path}", file=sys.stderr)
        return 1

    # Engines live in <models>/engines/<stem>.engine, mirroring the convention
    # used for the detection engines. Ultralytics hardcodes its output to
    # <pt_parent>/<stem>.engine, so we let it write there and move the file
    # after — same logic for the intermediate .onnx ultralytics leaves behind.
    engine_dir = model_path.parent / "engines"
    engine_path = engine_dir / f"{model_path.stem}.engine"
    ultra_engine_path = model_path.with_suffix(".engine")    # where ultralytics writes
    ultra_onnx_path = model_path.with_suffix(".onnx")        # intermediate ultralytics leaves

    print(f"[compile-yolo] checkpoint:    {model_path}")
    print(f"[compile-yolo] engine target: {engine_path}")
    print(f"[compile-yolo] profile:       {args.profile} "
          f"({'fixed imgsz, batch 1..N -- CM-201' if args.profile == 'fixed' else 'ultralytics dynamic export'})")
    print(f"[compile-yolo] dynamic:       {bool(args.dynamic)}")
    print(f"[compile-yolo] opt imgsz:     {args.det_imgsz}")
    print(f"[compile-yolo] max batch:     {args.max_batch}")
    print(f"[compile-yolo] precision:     {'fp16' if args.fp16 else 'fp32'}")
    ws = int(args.workspace)
    if ws > 0:
        print(f"[compile-yolo] workspace:     {ws} GB")
    else:
        print(f"[compile-yolo] workspace:     no cap (TensorRT default: "
              f"full device VRAM)")
    if args.profile == "fixed":
        print(f"[compile-yolo] shape range:   batch 1..{args.max_batch}, "
              f"H,W 32..{args.det_imgsz} (opt {args.max_batch} x {args.det_imgsz}^2)")
    elif args.dynamic:
        # Shape ceiling mirrors ultralytics' hard-coded formula
        # (ultralytics/utils/export/engine.py, onnx2engine):
        #     max H,W = max(2, workspace or 2) * imgsz
        # i.e. the SAME --workspace number is both the tactic memory pool
        # AND the shape-ceiling multiplier. There is no export() kwarg to
        # decouple them. workspace<=0 leaves the pool uncapped while the
        # ceiling falls back to 2*imgsz -- the only way to give the builder
        # more tactic memory WITHOUT inflating worst-case shapes.
        max_dim = max(2, ws if ws > 0 else 2) * args.det_imgsz
        print(f"[compile-yolo] dynamic shape range: batch 1..{args.max_batch}, "
              f"H,W 32..{max_dim} (opt at {args.det_imgsz})")
        if ws > 2:
            print(f"[compile-yolo] CAUTION: workspace {ws} raises the max "
                  f"accepted imgsz to {max_dim} (ultralytics couples the two). "
                  f"Worst-case activations at {max_dim}x{max_dim} can exceed "
                  f"VRAM and fail the build with 'could not find any "
                  f"implementation'. Field event 2026-08-16: workspace 4 at "
                  f"imgsz 800 -> 3200^2 -> 10GB tactic requests on an 8GB "
                  f"card. Use --workspace 0 to enlarge the memory pool "
                  f"without raising the ceiling.")
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

    if args.profile == "fixed":
        # CM-201: ONNX via ultralytics (dynamic axes on batch/H/W), engine by us.
        _trt_major = 0
        try:
            import tensorrt as _trt
            _trt_major = int(str(_trt.__version__).split(".")[0])
        except Exception:
            pass
        if _trt_major >= 11:
            print("[compile-yolo] TensorRT 11+ detected (strongly typed builder): the fixed "
                  "profile path needs an update; falling back to the ultralytics export.")
            args.profile = "legacy"
    if args.profile == "fixed":
        print("[compile-yolo] Exporting ONNX (ultralytics), then building the engine with the "
              "fixed profile (this can take several minutes) ...")
        t0 = time.perf_counter()
        try:
            onnx_out = model.export(
                format="onnx",
                imgsz=int(args.det_imgsz),
                half=False,
                dynamic=True,
                batch=int(args.max_batch),
                simplify=True,
                device=int(args.gpu_id),
                verbose=False,
            )
            onnx_path = Path(str(onnx_out)) if onnx_out else ultra_onnx_path
            if not onnx_path.is_file():
                onnx_path = ultra_onnx_path
            meta = _onnx_metadata(onnx_path)
            meta["batch"] = int(args.max_batch)
            meta["imgsz"] = [int(args.det_imgsz), int(args.det_imgsz)]
            meta["chitramaya_profile"] = {"kind": "fixed", "imgsz": int(args.det_imgsz),
                                          "max_batch": int(args.max_batch)}
            _free_cuda(None)
            _build_fixed_profile_engine(
                onnx_path, engine_path, imgsz=int(args.det_imgsz), max_batch=int(args.max_batch),
                fp16=bool(args.fp16), workspace_gb=int(args.workspace), metadata=meta)
        except Exception as e:
            print(f"[!] fixed-profile build failed: {e}", file=sys.stderr)
            return 1
        elapsed = time.perf_counter() - t0
        if ultra_onnx_path.is_file():
            try:
                ultra_onnx_path.unlink()
            except OSError:
                pass
        _write_profile_sidecar(engine_path, imgsz=int(args.det_imgsz), max_batch=int(args.max_batch),
                               fp16=bool(args.fp16), profile="fixed", gpu_id=int(args.gpu_id),
                               workspace_gb=int(args.workspace))
        size_mb = engine_path.stat().st_size / (1024 * 1024)
        print()
        print(f"[compile-yolo] Done in {elapsed:.1f}s.")
        print(f"[compile-yolo] Engine: {engine_path} ({size_mb:.1f} MB) -- profile batch 1..{args.max_batch}, "
              f"H,W 32..{args.det_imgsz}")
        _free_cuda(model)
        return 0

    print("[compile-yolo] Exporting to TensorRT engine (this can take several minutes) ...")
    t0 = time.perf_counter()
    try:
        # Ultralytics writes the engine to model_path.with_suffix(".engine").
        # When dynamic=True, the engine accepts:
        #   - any batch size from 1 to args.max_batch
        #   - any imgsz from 32 to max(2, args.workspace or 2) * args.det_imgsz
        # workspace<=0 -> pool limit never set (TRT 10 default: full device
        # VRAM), ceiling falls back to 2*imgsz.
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

    _write_profile_sidecar(engine_path, imgsz=int(args.det_imgsz), max_batch=int(args.max_batch),
                           fp16=bool(args.fp16), profile="legacy", gpu_id=int(args.gpu_id),
                           workspace_gb=int(args.workspace))
    size_mb = engine_path.stat().st_size / (1024 * 1024)
    print()
    print(f"[compile-yolo] Done in {elapsed:.1f}s.")
    print(f"[compile-yolo] Engine: {engine_path} ({size_mb:.1f} MB)")

    # Release PyTorch's CUDA cache so a directory batch doesn't accumulate
    # VRAM across models.
    _free_cuda(model)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile YOLO .pt checkpoint(s) to TensorRT engine(s) via ultralytics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--det-model", required=True,
        help="Path to a YOLO .pt checkpoint, OR a directory containing .pt files "
             "(every *.pt directly inside is compiled).",
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
        "--profile", choices=["fixed", "legacy"], default="fixed",
        help="fixed (default, CM-201): batch 1..--max-batch, H,W 32..--det-imgsz -- the "
             "shapes ChitraMaya actually runs, so TensorRT's execution context is sized "
             "for them (3.1 GB -> a few hundred MB on the v2 model at 800). legacy: "
             "ultralytics' own dynamic export (max H,W = max(2, workspace) x imgsz).",
    )
    parser.add_argument(
        "--max-batch", type=int, default=4,
        help="Maximum batch size the engine should support (default: 4 = the panel's "
             "Detection Batch default; the UI passes the panel value). "
             "Ultralytics requires this to be >1 when --dynamic=True. "
             "NOTE: larger max-batch raises the TRT builder's scratch-memory "
             "request; on an 8GB card, batch 16 on the 22M-param (YOLO11m) "
             "models requests ~2.1GB scratch and OVERFLOWS a 2GB --workspace, "
             "causing 'could not find any implementation' build failures. "
             "Batch 8 requests ~1.32GB and builds cleanly. Only raise this if "
             "your pipeline truly needs >8 frames per detection batch AND you "
             "also enlarge the pool -- prefer --workspace 0 (uncapped) over "
             "raising the workspace number, which would also raise the shape "
             "ceiling.",
    )
    parser.add_argument(
        "--workspace", type=int, default=2,
        help="TRT builder memory pool in GB (default: 2, matching the server's "
             "compile path). COUPLING (hard-coded in ultralytics): with "
             "--dynamic this same number also multiplies the engine's max "
             "accepted imgsz -- max H,W = max(2, workspace)*imgsz. Raising it "
             "past 2 therefore inflates worst-case shapes and can FAIL the "
             "build on 8GB cards (workspace 4 at imgsz 800 -> 3200^2 -> 10GB "
             "tactic requests, field event 2026-08-16). Use 0 to leave the "
             "pool UNCAPPED (TensorRT default: full device VRAM) while the "
             "shape ceiling stays at 2*imgsz -- the clean way to offer the "
             "builder more tactic memory.",
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

    target = Path(args.det_model)
    if not target.exists():
        print(f"[!] Path not found: {target}", file=sys.stderr)
        return 1

    # CUDA is required regardless of file/dir; check once up front.
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

    # Resolve the list of .pt files to compile: single file or all in a dir.
    if target.is_dir():
        models = sorted(target.glob("*.pt"))
        if not models:
            print(f"[!] No *.pt files found in directory: {target}", file=sys.stderr)
            return 1
        print(f"[compile-yolo] Directory mode: {len(models)} .pt file(s) in {target}")
        for m in models:
            print(f"               - {m.name}")
        print()
    else:
        models = [target]

    return _compile_all(models, args)


def _compile_all(models, args) -> int:
    # Compile each; keep going on failure but remember if any failed.
    failures = []
    for i, model_path in enumerate(models, 1):
        if len(models) > 1:
            print(f"===== [{i}/{len(models)}] {model_path.name} "
                  f"=======================================")
        # Batch 39 r2 (Gman: generic names are useless in a folder of
        # five logs): each MODEL gets its own log, named after the model,
        # written next to it -- our prints, ultralytics, and TensorRT's
        # native builder lines (the tactic/OOM warnings) all captured.
        from chitramaya.compile_log import (
            compile_log_path, tee_compile_output, write_log_header,
        )
        _log_path = compile_log_path(model_path.parent, model_path.stem)
        with tee_compile_output(_log_path):
            write_log_header(argv=sys.argv)
            print(f"[compile-yolo] model: {model_path}")
            rc = compile_one(model_path, args)
        if rc != 0:
            failures.append(model_path.name)
        # Always release CUDA between models so a failed or successful build
        # doesn't leave PyTorch squatting on VRAM for the next one.
        _free_cuda()
        if len(models) > 1:
            print()

    if failures:
        print(f"[compile-yolo] Completed with {len(failures)} failure(s): "
              f"{', '.join(failures)}", file=sys.stderr)
        return 1

    if len(models) > 1:
        print(f"[compile-yolo] All {len(models)} engine(s) compiled successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
