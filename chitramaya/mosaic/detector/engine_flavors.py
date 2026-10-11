# chitramaya/mosaic/detector/engine_flavors.py
"""CM-218: several compiled TensorRT engines ("flavors") per detection model.

A detection engine bakes in its input size (Image Size) and its batch
profile (1..max_batch) and precision at compile time. Until 1.80 there was
one engine per model, ``<models>/engines/<stem>.engine``: compiling at 800
destroyed the 640 engine, and a panel dial that disagreed with the file
dropped the whole run to PyTorch detection. Now every compile lands at

    <models>/engines/<stem>__i<imgsz>_b<max_batch>_<fp16|fp32>.engine

beside any others, each with its CM-201 sidecar (``<engine>.json``), and
the run picks the flavor that matches the panel. The pre-1.80 file
``<stem>.engine`` is still recognised as a "legacy" flavor whose size and
batch are read from the ultralytics header / sidecar, so nothing has to be
recompiled.

Picking rules (``pick_detection_engine``): Image Size must match exactly
(it is baked in); among those, the asked precision first, then the smallest
batch profile that covers the asked Detection Batch, else the largest
profile (the pipeline's CM-203 clamp then applies). No flavor at the asked
size -> None and a reason that names what IS compiled.
"""
from __future__ import annotations

import json
import os
import re
import struct
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

_FLAVOR_RE = re.compile(r"^(?P<stem>.+)__i(?P<imgsz>\d+)_b(?P<batch>\d+)_(?P<prec>fp16|fp32)$")


@dataclass
class EngineFlavor:
    path: str
    imgsz: Optional[int]
    max_batch: Optional[int]
    fp16: Optional[bool]
    legacy: bool = False

    def label(self) -> str:
        """`640/b8` (+ ` fp32` when not fp16; legacy files get a `*`)."""
        s = f"{self.imgsz if self.imgsz else '?'}/b{self.max_batch if self.max_batch else '?'}"
        if self.fp16 is False:
            s += " fp32"
        if self.legacy:
            s += "*"
        return s

    def as_dict(self) -> dict:
        d = asdict(self)
        d["label"] = self.label()
        return d


def engines_dir_for(pt_path: str) -> Path:
    return Path(pt_path).parent / "engines"


def flavor_engine_path(pt_path: str, imgsz: int, max_batch: int, fp16: bool) -> str:
    """Where ``compile_yolo`` writes the engine for this flavor."""
    p = Path(pt_path)
    return str(engines_dir_for(pt_path) / f"{p.stem}__i{int(imgsz)}_b{int(max_batch)}_{'fp16' if fp16 else 'fp32'}.engine")


def legacy_engine_path(pt_path: str) -> str:
    p = Path(pt_path)
    return str(engines_dir_for(pt_path) / f"{p.stem}.engine")


def engine_header_imgsz(engine_path: str) -> Optional[int]:
    """The imgsz from the ultralytics header (4-byte LE length + JSON)."""
    try:
        with open(engine_path, "rb") as f:
            meta_len = struct.unpack("<i", f.read(4))[0]
            if not (0 < meta_len < 100_000):
                return None
            meta = json.loads(f.read(meta_len).decode("utf-8", errors="replace"))
        imgsz = meta.get("imgsz")
        if isinstance(imgsz, (list, tuple)) and imgsz:
            return int(imgsz[0])
        if isinstance(imgsz, (int, float)):
            return int(imgsz)
    except Exception:
        pass
    return None


def engine_sidecar(engine_path: str) -> dict:
    try:
        sc = engine_path + ".json"
        if os.path.isfile(sc):
            with open(sc, "r", encoding="utf-8") as f:
                d = json.load(f)
            return d if isinstance(d, dict) else {}
    except Exception:
        pass
    return {}


def describe_engine(engine_path: str, *, legacy: bool) -> EngineFlavor:
    """Flavor facts for one engine file: the file name first, then the
    sidecar, then the ultralytics header. An engine without a sidecar was
    exported before CM-201 with a batch-8 profile (CM-203's rule)."""
    stem = Path(engine_path).stem
    m = _FLAVOR_RE.match(stem)
    sc = engine_sidecar(engine_path)
    imgsz = int(m.group("imgsz")) if m else None
    batch = int(m.group("batch")) if m else None
    fp16: Optional[bool] = (m.group("prec") == "fp16") if m else None
    if imgsz is None:
        imgsz = int(sc["imgsz"]) if sc.get("imgsz") else engine_header_imgsz(engine_path)
    if batch is None:
        batch = int(sc["max_batch"]) if sc.get("max_batch") else 8
    if fp16 is None:
        fp16 = bool(sc["fp16"]) if "fp16" in sc else None
    return EngineFlavor(path=str(engine_path).replace("\\", "/"), imgsz=imgsz, max_batch=batch,
                        fp16=fp16, legacy=bool(legacy))


def list_detection_flavors(pt_path: str) -> List[EngineFlavor]:
    """Every compiled engine for this .pt: flavored files plus the legacy one."""
    out: List[EngineFlavor] = []
    try:
        d = engines_dir_for(pt_path)
        stem = Path(pt_path).stem
        if not d.is_dir():
            return out
        for f in sorted(d.iterdir()):
            if not f.is_file() or f.suffix.lower() != ".engine":
                continue
            m = _FLAVOR_RE.match(f.stem)
            if m and m.group("stem") == stem:
                out.append(describe_engine(str(f), legacy=False))
            elif f.stem == stem:
                out.append(describe_engine(str(f), legacy=True))
    except Exception:
        pass
    out.sort(key=lambda e: ((e.imgsz or 0), (e.max_batch or 0), 0 if e.fp16 else 1, e.legacy))
    return out


def pick_detection_engine(pt_path: str, imgsz: int, batch: int, fp16: bool = True
                          ) -> Tuple[Optional[EngineFlavor], str]:
    """The flavor the run should load for the panel's (Image Size, Detection
    Batch, FP16), or (None, reason). Image Size must match exactly."""
    flavors = list_detection_flavors(pt_path)
    if not flavors:
        return None, "no compiled engine"
    same = [f for f in flavors if f.imgsz == int(imgsz)]
    if not same:
        have = ", ".join(f.label() for f in flavors)
        return None, f"no engine at Image Size {int(imgsz)} (compiled: {have})"
    cover = [f for f in same if (f.max_batch or 0) >= int(batch)]
    pool = cover if cover else same
    # precision first (fp16 vs fp32 is a 2x speed difference; a legacy file of
    # unknown precision counts as matching), then the smallest covering batch
    # profile (or the largest available), then a flavored file over a legacy one
    def _prec(f):
        return 0 if (f.fp16 is None or f.fp16 == bool(fp16)) else 1
    if cover:
        pool.sort(key=lambda f: (_prec(f), (f.max_batch or 0), f.legacy))
    else:
        pool.sort(key=lambda f: (_prec(f), -(f.max_batch or 0), f.legacy))
    best = pool[0]
    why = f"flavor {best.label()}"
    if not cover:
        why += f" (largest batch profile for this size; Detection Batch {int(batch)} clamps to {best.max_batch})"
    if best.fp16 is not None and best.fp16 != bool(fp16):
        why += f"; precision {'fp16' if best.fp16 else 'fp32'} is what is compiled"
    return best, why


__all__ = [
    "EngineFlavor", "flavor_engine_path", "legacy_engine_path", "list_detection_flavors",
    "pick_detection_engine", "describe_engine", "engine_header_imgsz", "engine_sidecar",
]
