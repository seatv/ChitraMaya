"""
ChitraMaya mosaic restoration CLI.

Usage:
    python -m tools.process_mosaic --input video.mp4 --output restored.mp4 \\
        --rest-model models/restoration.pth --det-model models/detection.pt

For the full argument list, see:
    python -m tools.process_mosaic --help

Architecture: ports gRestorer's single-hot-loop pipeline. Decode -> detect ->
track -> restore -> composite -> encode runs on the main thread, one batch at
a time. NVDEC and NVENC use their own hardware so they don't compete with
restoration kernels on the SMs. An optional decode prefetch thread is enabled
automatically for ffmpeg-CPU decoding only (not NVDEC).
"""
from __future__ import annotations

import sys


def main() -> int:
    try:
        from chitramaya.mosaic.cli_config import parse_args
        from chitramaya.mosaic.pipeline import Pipeline

        cfg = parse_args()
        pipeline = Pipeline(cfg)
        pipeline.run()
        return 0

    except KeyboardInterrupt:
        print("\n\n[!] Interrupted by user", file=sys.stderr)
        return 130

    except Exception as e:
        print(f"\n[!] Fatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
