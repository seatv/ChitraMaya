# ChitraMaya/mosaic/models/basicvsrpp/engine_paths.py
"""Engine-path conventions for BasicVSR++ TensorRT sub-engines.

Sub-engines for a checkpoint at ``models/<stem>.pth`` are stored beside the
checkpoint in a directory named ``<stem>_sub_engines/``. The file naming
scheme distinguishes precision (fp16/fp32) and OS (.win/.linux) so engines
compiled on one machine aren't accidentally loaded on another.

v1.60.00 (CM-104): engines are CLIP-SIZE INDEPENDENT. The preprocess and
upsample stages are per-frame math, so their engines are compiled at FIXED
batch sizes (``BASICVSRPP_PREPROCESS_BATCH`` / ``BASICVSRPP_UPSAMPLE_BATCH``)
and the runtime loops the clip through in batches. Changing Max Clip Length
no longer selects different engine files, and nothing in this layer takes a
clip size any more. This scheme is ported from Jasna v0.9.1's fixed-batch
engines (see the derivative-work notice below), where it was measured to
shrink the six-engine resident set from ~4.5 GB (clip-sized b180) to
~0.9 GB with a ~2% wall cost.

BACKWARD COMPATIBILITY: pre-v1.60 "ladder" sets (preprocess_b{N} /
upsample_dyn_b{N} compiled at the clip size) remain loadable —
``find_basicvsrpp_engine_files`` picks the best files present and reports
per-file batch capacities so the runtime can chunk within them. A legacy
set runs correctly through the new batched runtime; it just keeps its old
(larger) VRAM reservation until the user recompiles once.

This module has no torch / tensorrt imports — pure path helpers.

Ported from Jasna's engine_paths.py (BasicVSR++ subset; fixed-batch
constants and naming follow Jasna v0.9.1).

Copyright (c) the Jasna authors (Kruk2). Derivative work of Jasna's
implementation; both projects are AGPL-3.0 (see LICENSE). Credit also
appears in the README Acknowledgements.
"""
from __future__ import annotations

import os
import re
from pathlib import Path


def engine_system_suffix() -> str:
    """Per-OS engine file suffix.

    TRT engines are tied to the platform they were compiled on, so we
    encode the OS in the filename to avoid accidental cross-platform loads.
    """
    return ".win" if os.name == "nt" else ".linux"


def engine_precision_name(*, fp16: bool) -> str:
    """Precision tag for engine filenames."""
    return "fp16" if bool(fp16) else "fp32"


# Order matters: these four directions are the recurrent propagation passes
# in BasicVSR++. The "_1" suffix is the first iteration, "_2" the second.
BASICVSRPP_DIRECTIONS = ("backward_1", "forward_1", "backward_2", "forward_2")

# v1.60.00 fixed batch sizes (CM-104, Jasna v0.9.1 values). The preprocess
# engine consumes consecutive-frame windows (SPyNet pairs) so its batches
# overlap by one frame at runtime; the upsample engine is strictly
# per-frame. Values chosen by Jasna's measured sweep: b60/b30 keep wall
# clock within ~2% of whole-clip calls while cutting resident memory by
# ~4x. A happy filename accident: users who compiled the OLD b60 ladder
# set already own the canonical preprocess file, and an old b30 set's
# upsample file IS the canonical upsample — many existing installs need
# no recompile at all.
BASICVSRPP_PREPROCESS_BATCH = 60
BASICVSRPP_UPSAMPLE_BATCH = 30

# The preprocess engine's minimum profile batch (SPyNet needs >= 2 frames
# for one pair; the profile floor has always been 3, with runtime padding
# for shorter clips).
BASICVSRPP_PREPROCESS_MIN_BATCH = 3


def _basicvsrpp_sub_engine_dir(model_weights_path: str) -> str:
    """Directory beside the checkpoint where engines for that checkpoint live."""
    stem = os.path.splitext(os.path.basename(model_weights_path))[0]
    return os.path.join(os.path.dirname(model_weights_path), f"{stem}_sub_engines")


def get_basicvsrpp_sub_engine_paths(
    model_weights_path: str, fp16: bool, max_clip_size: int | None = None,
) -> dict[str, str]:
    """Compute the CANONICAL on-disk path for every sub-engine.

    v1.60: the canonical preprocess/upsample names carry the FIXED batch
    sizes, not a clip size. ``max_clip_size`` is accepted for backward
    compatibility with pre-v1.60 callers: when given, it names the legacy
    clip-sized files instead (used only to inspect old ladder sets).
    """
    engine_dir = _basicvsrpp_sub_engine_dir(model_weights_path)
    prec = engine_precision_name(fp16=fp16)
    suf = engine_system_suffix()
    pre_b = int(max_clip_size) if max_clip_size else BASICVSRPP_PREPROCESS_BATCH
    up_b = int(max_clip_size) if max_clip_size else BASICVSRPP_UPSAMPLE_BATCH
    paths: dict[str, str] = {}
    for d in BASICVSRPP_DIRECTIONS:
        paths[f"loop_body_{d}"] = os.path.join(
            engine_dir, f"loop_body_{d}.trt_{prec}{suf}.engine",
        )
    paths["preprocess"] = os.path.join(
        engine_dir, f"preprocess_b{pre_b}.trt_{prec}{suf}.engine",
    )
    paths["upsample"] = os.path.join(
        engine_dir, f"upsample_dyn_b{up_b}.trt_{prec}{suf}.engine",
    )
    return paths


def all_basicvsrpp_sub_engines_exist(
    model_weights_path: str, fp16: bool, max_clip_size: int | None = None,
) -> bool:
    """True iff every CANONICAL sub-engine file is present.

    v1.60 note: prefer ``basicvsrpp_engines_usable`` for "can TRT run at
    all?" questions — it also accepts legacy ladder sets.
    """
    return all(
        os.path.isfile(p)
        for p in get_basicvsrpp_sub_engine_paths(
            model_weights_path, fp16, max_clip_size,
        ).values()
    )


def _scan_batch_files(engine_dir: str, stem: str, prec: str, suf: str) -> dict[int, str]:
    """Map {batch_size: path} for files named ``<stem>_b<N>.trt_<prec><suf>.engine``."""
    out: dict[int, str] = {}
    if not os.path.isdir(engine_dir):
        return out
    pat = re.compile(
        r"^" + re.escape(stem) + r"_b(\d+)\.trt_" + re.escape(prec)
        + re.escape(suf) + r"\.engine$"
    )
    for name in os.listdir(engine_dir):
        m = pat.match(name)
        if m:
            out[int(m.group(1))] = os.path.join(engine_dir, name)
    return out


def _pick_batch_file(candidates: dict[int, str], target: int) -> tuple[int, str] | None:
    """Choose the best file for a runtime that chunks at ``target``.

    Preference: exact target; else the LARGEST size below it (smallest
    memory among still-chunkable files); else the SMALLEST size above it
    (a bigger legacy file still works — the runtime simply chunks at
    ``target`` and the file's larger profile is wasted reservation).
    """
    if not candidates:
        return None
    if target in candidates:
        return target, candidates[target]
    below = [n for n in candidates if n < target]
    if below:
        n = max(below)
        return n, candidates[n]
    n = min(candidates)
    return n, candidates[n]


def find_basicvsrpp_engine_files(
    model_weights_path: str, fp16: bool,
) -> dict | None:
    """Locate a loadable sub-engine set, canonical OR legacy (v1.60, CM-104).

    Returns None when no complete set exists. Otherwise a dict:
      paths           {engine_key: path} for the 6 engines to load
      preprocess_b    batch size baked into the chosen preprocess file
      upsample_b      batch size baked into the chosen upsample file
      preprocess_cap  runtime chunk cap = min(BASICVSRPP_PREPROCESS_BATCH, file)
      upsample_cap    runtime chunk cap = min(BASICVSRPP_UPSAMPLE_BATCH, file)
      fixed           True when both files are the canonical fixed-batch ones
                      (the memory-lean v1.60 set); False = legacy files in use
    """
    engine_dir = _basicvsrpp_sub_engine_dir(model_weights_path)
    prec = engine_precision_name(fp16=fp16)
    suf = engine_system_suffix()

    paths: dict[str, str] = {}
    for d in BASICVSRPP_DIRECTIONS:
        p = os.path.join(engine_dir, f"loop_body_{d}.trt_{prec}{suf}.engine")
        if not os.path.isfile(p):
            return None
        paths[f"loop_body_{d}"] = p

    pre = _pick_batch_file(
        _scan_batch_files(engine_dir, "preprocess", prec, suf),
        BASICVSRPP_PREPROCESS_BATCH,
    )
    up = _pick_batch_file(
        _scan_batch_files(engine_dir, "upsample_dyn", prec, suf),
        BASICVSRPP_UPSAMPLE_BATCH,
    )
    if pre is None or up is None:
        return None
    pre_b, pre_path = pre
    up_b, up_path = up
    paths["preprocess"] = pre_path
    paths["upsample"] = up_path
    return {
        "paths": paths,
        "preprocess_b": pre_b,
        "upsample_b": up_b,
        "preprocess_cap": min(BASICVSRPP_PREPROCESS_BATCH, pre_b),
        "upsample_cap": min(BASICVSRPP_UPSAMPLE_BATCH, up_b),
        "fixed": (pre_b == BASICVSRPP_PREPROCESS_BATCH
                  and up_b == BASICVSRPP_UPSAMPLE_BATCH),
    }


def basicvsrpp_engines_usable(model_weights_path: str, fp16: bool) -> bool:
    """True when SOME complete sub-engine set (canonical or legacy) exists."""
    return find_basicvsrpp_engine_files(model_weights_path, fp16) is not None


def pick_engine_clip_size(
    compiled_sizes: list[int], requested: int,
) -> tuple[int, int] | None:
    """LEGACY (pre-v1.60) resolver, kept for compatibility.

    v1.60 runtime no longer snaps Max Clip to compiled sizes — engines are
    clip-size independent and the runtime chunks within per-file caps
    (see ``find_basicvsrpp_engine_files``). Retained because old sets are
    still meaningful to describe, and external scripts may import this.
    """
    if not compiled_sizes:
        return None
    req = int(requested)
    sizes = sorted(int(n) for n in compiled_sizes)
    if req in sizes:
        return (req, req)
    ge = [n for n in sizes if n >= req]
    if ge:
        return (min(ge), req)
    return (sizes[-1], sizes[-1])


def list_basicvsrpp_compiled_clip_sizes(
    model_weights_path: str, fp16: bool,
) -> list[int]:
    """Batch sizes (bN) of the COMPLETE preprocess-keyed sets on disk.

    Pre-v1.60 this was "which clip sizes are compiled". Post-CM-104 the
    number no longer limits Max Clip; it is informational (which files
    exist / how big their reservations are). A canonical fixed-batch set
    reports as [BASICVSRPP_PREPROCESS_BATCH] via the b60 preprocess file
    when the b30 upsample exists (see ``all_basicvsrpp_sub_engines_exist``
    fallthrough below, which accepts the canonical upsample for any N).
    """
    engine_dir = _basicvsrpp_sub_engine_dir(model_weights_path)
    if not os.path.isdir(engine_dir):
        return []
    prec = engine_precision_name(fp16=fp16)
    suf = engine_system_suffix()
    pre_files = _scan_batch_files(engine_dir, "preprocess", prec, suf)
    up_files = _scan_batch_files(engine_dir, "upsample_dyn", prec, suf)
    loop_ok = all(
        os.path.isfile(os.path.join(
            engine_dir, f"loop_body_{d}.trt_{prec}{suf}.engine"))
        for d in BASICVSRPP_DIRECTIONS
    )
    if not loop_ok:
        return []
    sizes: set[int] = set()
    for n in pre_files:
        # A set is complete when a same-N upsample exists (legacy ladder)
        # OR the canonical fixed-batch upsample exists (v1.60 set).
        if n in up_files or BASICVSRPP_UPSAMPLE_BATCH in up_files:
            sizes.add(n)
    return sorted(sizes)
