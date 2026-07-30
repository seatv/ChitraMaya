"""
ChitraMaya mosaic restoration CLI.

Usage (single file):
    python -m tools.process_mosaic --input video.mp4 --output restored.mp4 \\
        --rest-model models/restoration.pth --det-model models/detection.pt

Usage (folder batch, CM-079):
    python -m tools.process_mosaic --input D:/videos --output D:/out \\
        --rest-model ... --det-model ...
    # When --input is a FOLDER, every video in it is processed with the same
    # settings, models load ONCE, and --output is treated as the output
    # DIRECTORY (default: the input folder). Per-file failures are isolated;
    # --batch-skip-existing resumes without redoing finished outputs;
    # --batch-video-extensions limits the extensions considered.

For the full argument list, see:
    python -m tools.process_mosaic --help

Architecture: ports gRestorer's single-hot-loop pipeline. Decode -> detect ->
track -> restore -> composite -> encode runs on the main thread, one batch at
a time. NVDEC and NVENC use their own hardware so they don't compete with
restoration kernels on the SMs. An optional decode prefetch thread is enabled
automatically for ffmpeg-CPU decoding only (not NVDEC).
"""
from __future__ import annotations

import os
import sys


def _suffix_for_mode(mode: str) -> str:
    m = (mode or "real").lower()
    if m == "mosaic":
        return "-censored"
    if m == "pseudo":
        return "-mask"
    return "-restored"


def _run_folder(cfg) -> int:
    """Batch-process a folder, reusing models across files (CM-079)."""
    from pathlib import Path

    from chitramaya.mosaic import batch as batchmod
    from chitramaya.mosaic.pipeline import Pipeline

    folder = str(cfg.get("input"))
    mode = str(cfg.get("mode", default="real"))
    suffix = _suffix_for_mode(mode)

    # --output is the output DIRECTORY for folder mode (default: input folder).
    out_dir = str(cfg.get("output", default="") or "").strip() or folder
    os.makedirs(out_dir, exist_ok=True)

    exts = cfg.get("batch_processing", "video_extensions", default=None)
    skip_existing = bool(cfg.get("batch_processing", "skip_existing", default=True))

    inputs = batchmod.enumerate_videos(folder, extensions=exts, recursive=False)
    if not inputs:
        print(f"[Batch] No videos found in {folder}")
        return 0
    plan = batchmod.plan_batch(inputs, out_dir, suffix, skip_existing=skip_existing)
    n_todo = sum(1 for it in plan if not it.skip)
    print(f"[Batch] {len(plan)} file(s) in {folder}; {n_todo} to process, "
          f"{len(plan) - n_todo} skipped (existing). Output -> {out_dir}")

    # Build the host + models ONCE, reuse across files (the warm pattern).
    host = Pipeline(cfg)
    detector = host._build_detector()
    restorer = host._build_restorer()

    def process_one(input_path: str, output_path: str):
        # Fresh output name if a prior output exists (collision-safe).
        final_out = output_path
        n = 2
        while Path(final_out).exists():
            p = Path(output_path)
            final_out = str(p.with_name(f"{p.stem}-{n}{p.suffix}"))
            n += 1
        host.cfg.set("input", value=input_path)
        host.cfg.set("output", value=final_out)
        host.__post_init__()
        return host.run(detector_override=detector, restorer_override=restorer)

    summary = batchmod.run_batch(
        plan, process_one,
        on_file_start=lambda i, it: print(
            f"[Batch] [{i + 1}/{len(plan)}] {Path(it.input_path).name}"),
        on_file_done=lambda i, r: (
            print(f"[Batch]   -> {r.status}"
                  + (f": {r.error}" if r.error and r.status == 'error' else ""))
            if r.status != "skipped" else None),
    )
    print(f"[Batch] Done: {summary.line()}")
    if summary.errors:
        print("[Batch] Files that FAILED (batch continued past them):")
        for r in summary.results:
            if r.status == "error":
                print(f"    {Path(r.input_path).name}: {r.error}")
    # Nonzero exit only if EVERYTHING failed; partial success is success.
    return 1 if (summary.errors and summary.done == 0) else 0


def main() -> int:
    try:
        from chitramaya.mosaic.cli_config import parse_args
        from chitramaya.mosaic.pipeline import Pipeline

        cfg = parse_args()

        # CM-079: folder input -> batch mode (models load once, per-file
        # errors isolated, --output is the output directory).
        if os.path.isdir(str(cfg.get("input"))):
            return _run_folder(cfg)

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
