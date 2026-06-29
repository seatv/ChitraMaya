# ChitraMaya/mosaic/models/basicvsrpp/engine_paths.py
"""Engine-path conventions for BasicVSR++ TensorRT sub-engines.

Sub-engines for a checkpoint at ``models/<stem>.pth`` are stored beside the
checkpoint in a directory named ``<stem>_sub_engines/``. The file naming
scheme distinguishes precision (fp16/fp32) and OS (.win/.linux) so engines
compiled on one machine aren't accidentally loaded on another.

This module has no torch / tensorrt imports — pure path helpers.

Faithful port of Jasna's engine_paths.py (BasicVSR++ subset only).
"""
from __future__ import annotations

import os
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


def _basicvsrpp_sub_engine_dir(model_weights_path: str) -> str:
    """Directory beside the checkpoint where engines for that checkpoint live."""
    stem = os.path.splitext(os.path.basename(model_weights_path))[0]
    return os.path.join(os.path.dirname(model_weights_path), f"{stem}_sub_engines")


def get_basicvsrpp_sub_engine_paths(
    model_weights_path: str, fp16: bool, max_clip_size: int = 60,
) -> dict[str, str]:
    """Compute the on-disk path for every sub-engine of a given checkpoint.

    Engines whose batch size is bound at compile time encode ``max_clip_size``
    in the filename (``preprocess``, ``upsample``). Per-direction loop-body
    engines use static batch=1 and don't.
    """
    engine_dir = _basicvsrpp_sub_engine_dir(model_weights_path)
    prec = engine_precision_name(fp16=fp16)
    suf = engine_system_suffix()
    paths: dict[str, str] = {}
    for d in BASICVSRPP_DIRECTIONS:
        paths[f"loop_body_{d}"] = os.path.join(
            engine_dir, f"loop_body_{d}.trt_{prec}{suf}.engine",
        )
    paths["preprocess"] = os.path.join(
        engine_dir, f"preprocess_b{max_clip_size}.trt_{prec}{suf}.engine",
    )
    paths["upsample"] = os.path.join(
        engine_dir, f"upsample_dyn_b{max_clip_size}.trt_{prec}{suf}.engine",
    )
    return paths


def all_basicvsrpp_sub_engines_exist(
    model_weights_path: str, fp16: bool, max_clip_size: int = 60,
) -> bool:
    """True iff every sub-engine file for this checkpoint+precision is present."""
    return all(
        os.path.isfile(p)
        for p in get_basicvsrpp_sub_engine_paths(
            model_weights_path, fp16, max_clip_size,
        ).values()
    )
