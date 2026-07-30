# chitramaya/mosaic/batch.py
"""Folder batch processing for the mosaic pipeline (CM-079, Batch 22).

Process every video in a folder with the current settings, reusing the warm
models across all files (they load once, not per file). This is the headless
front end for large collections -- and stage 1 of the CM-080 dataset factory.

Design goals baked in from field experience:
  - EXCLUDE ChitraMaya's own outputs from the input list. A folder often holds
    both sources and prior `-restored`/`-censored` outputs; re-processing an
    already-censored file is the exact "double-censor" trap Gman hit. The
    enumerator filters our output suffixes and raw-bitstream sidecars.
  - Skip-existing: if a file's output already exists, skip it (resume a batch
    that was interrupted -- e.g. by the 5060 falling off the bus -- without
    redoing finished work).
  - Per-file error isolation: one file failing (including a GPU hang that
    kills a run) must NOT abort the whole batch. Log it, record it, continue.
    Combined with the per-file recovery sidecar (Batch 21b), a batch survives
    a mid-collection GPU incident and tells you exactly which file died.

This module is pure orchestration: no models, no GPU, no UI. The caller
supplies a `process_one(input_path, output_path) -> ProcessOutcome` callable
(the server wires the warm MosaicPipeline; the CLI wires its pipeline). Fully
unit-testable without CUDA.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence

# Container extensions we treat as processable video inputs.
DEFAULT_VIDEO_EXTS: tuple[str, ...] = (
    ".mp4", ".mkv", ".mov", ".m4v", ".avi", ".ts", ".webm", ".wmv", ".flv", ".mpg", ".mpeg",
)

# Output-name markers ChitraMaya itself produces. Any input whose stem ends
# with one of these is one of our OWN outputs -- never a source to reprocess.
_OUTPUT_STEM_SUFFIXES: tuple[str, ...] = (
    "-restored", "-censored", "-mask", "-mosaic",
    "-censored-seg", "-mosaic-seg",
    "-FIXED", "-PARTIAL",          # Batch 21b recovery products
)
# Sidecar / intermediate extensions to always ignore.
_IGNORE_EXTS: tuple[str, ...] = (".hevc", ".av1", ".h264", ".ps1", ".json", ".txt")
# Intermediate container fragments (...-restored.mp4.vtmp.mp4 etc.).
_IGNORE_SUBSTR: tuple[str, ...] = (".vtmp.",)


@dataclass
class BatchItem:
    """One planned unit of work."""
    input_path: str
    output_path: str
    skip: bool = False
    skip_reason: str = ""


@dataclass
class BatchFileResult:
    """Outcome of processing one file."""
    input_path: str
    output_path: str
    status: str                    # "done" | "skipped" | "error" | "cancelled"
    seconds: float = 0.0
    frames: int = 0
    detections: int = 0
    error: str = ""


@dataclass
class BatchSummary:
    total: int = 0
    done: int = 0
    skipped: int = 0
    errors: int = 0
    cancelled: int = 0
    seconds: float = 0.0
    results: List[BatchFileResult] = field(default_factory=list)

    def line(self) -> str:
        return (f"{self.done} done, {self.skipped} skipped, "
                f"{self.errors} error(s)"
                + (f", {self.cancelled} cancelled" if self.cancelled else "")
                + f" of {self.total} in {self.seconds:.0f}s")


def _is_own_output(stem: str) -> bool:
    low = stem.lower()
    return any(low.endswith(suf.lower()) for suf in _OUTPUT_STEM_SUFFIXES)


def enumerate_videos(
    folder: str | Path,
    extensions: Optional[Sequence[str]] = None,
    recursive: bool = False,
) -> List[Path]:
    """Return sorted source videos in `folder`, excluding ChitraMaya's own
    outputs and intermediates. Raises FileNotFoundError if the folder is bad."""
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"Not a folder: {folder}")
    exts = {e.lower() if e.startswith(".") else "." + e.lower()
            for e in (extensions or DEFAULT_VIDEO_EXTS)}
    it = root.rglob("*") if recursive else root.glob("*")
    out: List[Path] = []
    for p in it:
        if not p.is_file():
            continue
        name_low = p.name.lower()
        if p.suffix.lower() in _IGNORE_EXTS:
            continue
        if any(sub in name_low for sub in _IGNORE_SUBSTR):
            continue
        if p.suffix.lower() not in exts:
            continue
        if _is_own_output(p.stem):
            continue
        out.append(p)
    return sorted(out, key=lambda q: str(q).lower())


def output_path_for(input_path: str | Path, out_dir: str | Path, suffix: str) -> str:
    """Mirror the server's single-file output naming: <out_dir>/<stem><suffix>.mp4."""
    stem = Path(input_path).stem
    return str(Path(out_dir) / f"{stem}{suffix}.mp4")


def plan_batch(
    inputs: Sequence[Path],
    out_dir: str | Path,
    suffix: str,
    skip_existing: bool,
) -> List[BatchItem]:
    """Compute output paths and mark skips. Does not touch the GPU."""
    items: List[BatchItem] = []
    for src in inputs:
        outp = output_path_for(src, out_dir, suffix)
        skip = bool(skip_existing) and Path(outp).exists()
        items.append(BatchItem(
            input_path=str(src),
            output_path=outp,
            skip=skip,
            skip_reason="output exists" if skip else "",
        ))
    return items


# Callable contract: process_one(input_path, output_path) -> object with
# .frames / .detections (a MosaicResult), or None. May raise -- run_batch
# isolates it.
ProcessOne = Callable[[str, str], object]


def run_batch(
    items: Sequence[BatchItem],
    process_one: ProcessOne,
    *,
    monotonic: Callable[[], float] = time.perf_counter,
    on_file_start: Optional[Callable[[int, BatchItem], None]] = None,
    on_file_done: Optional[Callable[[int, BatchFileResult], None]] = None,
    cancel_flag=None,
) -> BatchSummary:
    """Run every planned item through `process_one`, isolating per-file errors.

    `monotonic` is injectable so tests don't depend on the wall clock (the
    workflow-script constraint that Date.now/perf_counter vary). `cancel_flag`
    (anything with .is_set()) stops the batch between files -- an in-flight
    file finishes or is left to its own cancel handling.
    """
    summary = BatchSummary(total=len(items))
    t_all = monotonic()

    for idx, item in enumerate(items):
        if cancel_flag is not None and cancel_flag.is_set():
            for rest in items[idx:]:
                summary.results.append(BatchFileResult(
                    input_path=rest.input_path, output_path=rest.output_path,
                    status="cancelled"))
                summary.cancelled += 1
            break

        if item.skip:
            res = BatchFileResult(
                input_path=item.input_path, output_path=item.output_path,
                status="skipped", error=item.skip_reason)
            summary.skipped += 1
            summary.results.append(res)
            if on_file_done:
                on_file_done(idx, res)
            continue

        if on_file_start:
            on_file_start(idx, item)

        t0 = monotonic()
        try:
            out = process_one(item.input_path, item.output_path)
            res = BatchFileResult(
                input_path=item.input_path, output_path=item.output_path,
                status="done", seconds=monotonic() - t0,
                frames=int(getattr(out, "frames", 0) or 0),
                detections=int(getattr(out, "detections", 0) or 0),
            )
            summary.done += 1
        except Exception as e:  # per-file isolation -- never abort the batch
            res = BatchFileResult(
                input_path=item.input_path, output_path=item.output_path,
                status="error", seconds=monotonic() - t0, error=str(e))
            summary.errors += 1

        summary.results.append(res)
        if on_file_done:
            on_file_done(idx, res)

    summary.seconds = monotonic() - t_all
    return summary


__all__ = [
    "DEFAULT_VIDEO_EXTS",
    "BatchItem", "BatchFileResult", "BatchSummary",
    "enumerate_videos", "output_path_for", "plan_batch", "run_batch",
]
