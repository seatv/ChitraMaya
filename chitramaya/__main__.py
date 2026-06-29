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


def main() -> int:
    args = sys.argv[1:]

    # Subcommand dispatch first — help, restore, compile.
    if args and args[0] in ("-h", "--help", "help"):
        _print_usage()
        return 0

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