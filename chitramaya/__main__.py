# ChitraMaya/__main__.py
"""
ChitraMaya unified entry.

Usage:
    ChitraMaya                    Launch the web/desktop UI server (default)
    ChitraMaya -restore [opts]    Run the mosaic-restoration CLI
    ChitraMaya -h | --help        Show this help

`-restore` forwards all remaining arguments to the restoration CLI's own
argument parser, so anything you'd pass to `tools/process_mosaic.py` works
here too.

Invocation paths:
  - From source:    python -m ChitraMaya [args]
  - From package:   ChitraMaya.exe [args]    (after PyInstaller build via
                                            packager.ps1 — extract the
                                            dist/ChitraMaya directory and
                                            add it to PATH)
"""
from __future__ import annotations

import sys


USAGE = """\
Usage:
  ChitraMaya                       Launch the UI server
  ChitraMaya -restore     [opts]   Run the mosaic-restoration CLI
  ChitraMaya -compile-rest [opts]  Build/rebuild BasicVSR++ TensorRT sub-engines
  ChitraMaya -compile-det  [opts]  Build/rebuild the YOLO detection engine
  ChitraMaya -self-check           Verify this install (imports, GPU, ffmpeg)
  ChitraMaya -h | --help           Show this help

Forward all remaining arguments to the chosen CLI. For example:
  ChitraMaya -restore --input video.mp4 --output out.mp4 \\
      --det-model models/det.pt --rest-model models/rest.pth \\
      --det-conf 0.01 --det-imgsz 640

For the full CLI option list, run:
  ChitraMaya -restore      --help
  ChitraMaya -compile-rest --help
  ChitraMaya -compile-det  --help
"""


def _print_usage() -> None:
    print(USAGE)


def _apply_cuda_alloc_conf() -> None:
    """Batch 78 (CM-149): "cudaExpandableSegments": true in
    ChitraMaya-config.json sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments
    before torch makes its first CUDA allocation.

    Field basis: ProblemChild 4K60 on a 6GB card OOM'd at frame 197,368
    from allocator fragmentation; with expandable_segments it reached
    281,512 (+43%). Users should not have to remember a per-machine env
    var -- user knobs live in the config file (env-var doctrine). An env
    var the user DID set explicitly is respected and never clobbered.

    Must run before any subcommand touches torch/CUDA -- call first in
    main(). Config anchoring mirrors the server: exe dir when frozen,
    cwd when running from source."""
    import json
    import os
    from pathlib import Path
    try:
        base = (Path(sys.executable).parent
                if getattr(sys, "frozen", False) else Path.cwd())
        cfg_file = base / "ChitraMaya-config.json"
        if not cfg_file.exists():
            return
        flat = json.loads(cfg_file.read_text(encoding="utf-8"))
        if not isinstance(flat, dict) or not flat.get("cudaExpandableSegments"):
            return
        cur = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if "expandable_segments" in cur:
            print("[Alloc] PYTORCH_CUDA_ALLOC_CONF already configures "
                  "expandable_segments; leaving the environment as-is.")
            return
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
            (cur + "," if cur else "") + "expandable_segments:True")
        print("[Alloc] CUDA expandable_segments ENABLED "
              "(cudaExpandableSegments in ChitraMaya-config.json) -- "
              "reduces allocator fragmentation on long runs.")
    except Exception:
        # A malformed config must never block launch; self-check reports
        # JSON validity separately.
        pass


def main() -> int:
    args = sys.argv[1:]

    # Batch 78 (CM-149): allocator config BEFORE any torch/CUDA touch --
    # applies to the UI server, -restore CLI, and both compile paths.
    _apply_cuda_alloc_conf()

    # Subcommand dispatch first — help, restore, compile.
    if args and args[0] in ("-h", "--help", "help"):
        _print_usage()
        return 0

    # CM-110 (Batch 50): harden the CLI's console streams BEFORE any
    # subcommand runs. Field event 2026-08-18: -compile-rest died with
    # OSError(22, 'Incorrect function') / "lost sys.stderr" when the
    # shared Windows console handle went bad mid-run (GUI open at the
    # time) -- the compile itself was healthy. A dead console must never
    # kill a run; see chitramaya/safe_console.py. The GUI path is already
    # covered (console_buffer's tee swallows real-stream write errors).
    if args:
        try:
            from chitramaya.safe_console import install as _safe_console
            _safe_console()
        except Exception:
            pass

    if args and args[0] in ("-restore", "--restore", "restore"):
        sys.argv = ["ChitraMaya -restore"] + args[1:]
        from tools.process_mosaic import main as restore_main
        return int(restore_main() or 0)

    if args and args[0] in ("-compile-rest", "--compile-rest"):
        sys.argv = ["ChitraMaya -compile-rest"] + args[1:]
        from tools.compile_basicvsrpp import main as compile_rest_main
        return int(compile_rest_main() or 0)

    if args and args[0] in ("-compile-det", "--compile-det"):
        sys.argv = ["ChitraMaya -compile-det"] + args[1:]
        from tools.compile_yolo import main as compile_det_main
        return int(compile_det_main() or 0)

    if args and args[0] == "-self-check-devprobe":
        # Internal (Batch 34 r2): child-process half of the self-check's
        # GPU probe. Not documented in USAGE on purpose -- the self-check
        # spawns it so a native abort in the GPU runtime (ROCm without a
        # driver, field event 2026-08-15) cannot kill the whole check.
        from chitramaya.self_check import devprobe_main
        return int(devprobe_main() or 0)

    if args and args[0] in ("-self-check", "--self-check", "self-check"):
        # Batch 34: install verifier. Exit 0 = sound (warnings allowed),
        # 1 = broken. Consumed by humans, remote testers, and the CM-097
        # patch applier.
        from chitramaya.self_check import main as self_check_main
        return int(self_check_main() or 0)

    # Default: launch UI server. Forward UI-only flags (mirrors the
    # argparse that used to live in ChitraMaya/server.py's __main__).
    import argparse
    p = argparse.ArgumentParser(prog="ChitraMaya", description="ChitraMaya UI server")
    p.add_argument("--models-dir", type=str, default="./models")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--debug", action="store_true", help="Flask debug mode")
    p.add_argument("--console", action="store_true", help="Open WebView2 DevTools console")
    parsed = p.parse_args(args)

    from chitramaya.server import run
    run(
        models_dir=parsed.models_dir,
        gpu_id=parsed.gpu,
        debug=parsed.debug,
        console=parsed.console,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())