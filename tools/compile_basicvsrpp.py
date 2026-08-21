# tools/compile_basicvsrpp.py
"""Compile BasicVSR++ TensorRT sub-engines from a .pth checkpoint.

Builds the 6 sub-engines (4 loop-body + preprocess + upsample) for a given
checkpoint and precision (fp16/fp32). Engines are saved next to the .pth
file in a ``<stem>_sub_engines/`` directory.

v1.60 (CM-104): engines are CLIP-SIZE INDEPENDENT (fixed-batch preprocess
b60 / upsample b30, ported from Jasna v0.9.1). Compile ONCE per (model,
precision) — one set serves every Max Clip Length setting; the dial now
costs activation memory only. Old clip-sized bN sets remain loadable but
reserve far more VRAM; recompiling replaces them for practical purposes
(the old files can be deleted).

Usage from the chitramaya unified CLI:

    chitramaya -compile-rest --rest-model PATH/TO/MODEL.pth

Or directly:

    python -m tools.compile_basicvsrpp --rest-model PATH/TO/MODEL.pth \
        --fp16

Compile takes 5-15 minutes depending on optimization_level and the GPU.
The output engines are platform-specific (Windows engines won't load on
Linux and vice versa) — recompile per target machine.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

from chitramaya.compile_log import (
    compile_log_path,
    tee_compile_output,
    write_log_header,
)
from chitramaya.mosaic.models.basicvsrpp.engine_paths import (
    _basicvsrpp_sub_engine_dir,
    all_basicvsrpp_sub_engines_exist,
    get_basicvsrpp_sub_engine_paths,
)
from chitramaya.mosaic.models.basicvsrpp.inference import load_model
from chitramaya.mosaic.models.basicvsrpp.sub_engines import (
    compile_basicvsrpp_sub_engines,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compile BasicVSR++ TensorRT sub-engines from a .pth checkpoint."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--rest-model", required=True,
        help="Path to BasicVSR++ checkpoint (.pth)",
    )
    parser.add_argument(
        "--rest-max-clip-length", type=int, default=None,
        help="DEPRECATED (v1.60, CM-104): engines are clip-size independent; "
             "this flag is accepted for backward compatibility and ignored. "
             "One compile serves every Max Clip setting.",
    )
    parser.add_argument(
        "--fp16", action=argparse.BooleanOptionalAction, default=True,
        help="Build fp16 engines (default: True). Use --no-fp16 for fp32.",
    )
    parser.add_argument(
        "--optimization-level", type=int, default=5,
        help="TRT optimization level 1-5 (default: 5; higher=longer compile, "
             "more thorough kernel search)",
    )
    parser.add_argument(
        "--workspace", type=int, default=2,
        help="TRT build workspace in GB (default: 2). Bounds the scratch the "
             "builder may use, which caps the workspace the engine reserves "
             "RESIDENT at runtime. Smaller = more VRAM left for the frame store "
             "(chitramaya keeps everything resident, no host offload). Raise if a "
             "build reports it needs more scratch; lower for more headroom. "
             "Pass 0 to use the legacy 95%%-of-free behavior (unbounded).",
    )
    parser.add_argument(
        "--gpu-id", type=int, default=0,
        help="CUDA GPU index (default: 0)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recompile even if engines already exist on disk",
    )
    args = parser.parse_args()

    model_path = Path(args.rest_model)
    if not model_path.is_file():
        print(f"[!] Model not found: {model_path}", file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        print(
            "[!] CUDA not available; cannot compile TensorRT engines.",
            file=sys.stderr,
        )
        return 1

    device = torch.device(f"cuda:{int(args.gpu_id)}")
    fp16 = bool(args.fp16)

    if args.rest_max_clip_length is not None:
        print(
            f"[compile] NOTE: --rest-max-clip-length "
            f"({args.rest_max_clip_length}) no longer affects engine "
            f"compilation (v1.60, CM-104): engines are clip-size "
            f"independent. One compile serves every Max Clip setting."
        )

    engine_dir = _basicvsrpp_sub_engine_dir(str(model_path))

    # v1.60 (CM-104): engines are clip-size independent — the canonical set
    # is fixed-batch (preprocess b60 / upsample b30) and one compile serves
    # every Max Clip setting. --rest-max-clip-length is accepted for
    # backward compatibility and no longer affects what gets built.
    if (not args.force) and all_basicvsrpp_sub_engines_exist(
        str(model_path), fp16=fp16,
    ):
        print(
            f"[compile] All engines already present for this configuration. "
            f"Pass --force to rebuild."
        )
        for k, p in get_basicvsrpp_sub_engine_paths(
            str(model_path), fp16=fp16,
        ).items():
            size_mb = os.path.getsize(p) / (1024 * 1024) if os.path.isfile(p) else 0
            print(f"           {k:30s} ({size_mb:6.1f} MB)  {p}")
        return 0

    # Batch 39 (GenSRT port): everything from here on -- our prints,
    # PyTorch, and TensorRT's native builder output (the tactic/OOM
    # warnings that motivated this) -- is teed into a timestamped log
    # next to the engines. A failed or suspicious compile becomes a
    # shareable file instead of a scrolled-away memory. The no-op path
    # above exits BEFORE the tee so trivial re-runs create no log files.
    # Batch 39 r2: named after the MODEL, not a generic tag.
    _log_path = compile_log_path(engine_dir, Path(model_path).stem)
    with tee_compile_output(_log_path):
        write_log_header(argv=sys.argv, gpu_id=int(args.gpu_id))
        return _compile_body(args, model_path, device, fp16, engine_dir)


def _compile_body(args, model_path, device, fp16, engine_dir) -> int:
    print(f"[compile] checkpoint:      {model_path}")
    print(f"[compile] engine dir:      {engine_dir}")
    print(f"[compile] precision:       {'fp16' if fp16 else 'fp32'}")
    print(f"[compile] engine batches:  preprocess b60 / upsample b30 "
          f"(clip-size independent, v1.60)")
    print(f"[compile] device:          {device}")
    print(f"[compile] optimization:    {args.optimization_level}")
    print(f"[compile] workspace:       {args.workspace} GB" if args.workspace > 0
          else "[compile] workspace:       95% of free (unbounded)")
    print()

    if args.force:
        # Remove existing engines so compile doesn't skip them
        for p in get_basicvsrpp_sub_engine_paths(
            str(model_path), fp16=fp16,
        ).values():
            try:
                os.remove(p)
            except FileNotFoundError:
                pass

    # Load PyTorch model
    print("[compile] Loading PyTorch model ...")
    t0 = time.perf_counter()
    pt_model = load_model(
        config=None,
        checkpoint_path=str(model_path),
        device=device,
        fp16=fp16,
    )
    print(f"[compile] Model loaded ({time.perf_counter() - t0:.1f}s)")
    print()

    # Compile the 6 sub-engines. The function prints "Compiling sub-engine N/6 ..."
    # lines as it goes and skips engines that already exist.
    t0 = time.perf_counter()
    paths = compile_basicvsrpp_sub_engines(
        model=pt_model,
        device=device,
        fp16=fp16,
        model_weights_path=str(model_path),
        optimization_level=int(args.optimization_level),
        workspace_gb=int(args.workspace),
    )
    elapsed = time.perf_counter() - t0
    print()
    print(f"[compile] Done in {elapsed:.1f}s. Engines written to {engine_dir}:")
    for k, p in paths.items():
        size_mb = os.path.getsize(p) / (1024 * 1024) if os.path.isfile(p) else 0
        print(f"           {k:30s} ({size_mb:6.1f} MB)  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
