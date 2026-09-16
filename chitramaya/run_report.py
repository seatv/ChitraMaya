# ChitraMaya/run_report.py
"""CM-180 -- the run report: one file per run that says what ran.

Replaces the ``.misses.json`` (a miss list that grew into our run record by
accretion). Written beside the output as ``<output stem>.run.json`` together
with ``<output stem>.log`` (the whole console for THIS run, not a 400-line
tail). Config key ``runReport``: ``beside`` (default) | ``temp`` | ``off``.

Top-level keys, in order: chitramaya, machine, run, panel, config,
effective, timing, counts, events, frames, debug. ``panel`` is the control
panel exactly as submitted (the preset schema -- copy it into
``<app base>/presets/<name>.json`` and it IS a preset). ``effective`` holds
only fields that can differ from what was asked, each as
``{asked, ran, reason}``. ``frames`` keeps the per-frame lists losslessly as
sorted inclusive run-length ranges.

Weights travel; content never does: nothing derived from frame content goes
in here, and the report never leaves the machine unless the user sends it.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPORT_MODES = ("beside", "temp", "off")


def ranges(values: Sequence[int]) -> List[List[int]]:
    """Sorted inclusive run-length ranges: [1,2,3,7,9,10] -> [[1,3],[7,7],[9,10]]."""
    vs = sorted(int(v) for v in values)
    out: List[List[int]] = []
    for v in vs:
        if out and v == out[-1][1] + 1:
            out[-1][1] = v
        elif out and v == out[-1][1]:
            continue
        else:
            out.append([v, v])
    return out


def expand_ranges(rs: Any) -> List[int]:
    """Inverse of ranges(); also accepts an old flat list."""
    out: List[int] = []
    for item in rs or []:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            a, b = int(item[0]), int(item[1])
            out.extend(range(a, b + 1))
        else:
            out.append(int(item))
    return out


def normalize_mode(value: Any) -> str:
    m = str(value or "beside").strip().lower()
    return m if m in REPORT_MODES else "beside"


def _machine_info(device=None) -> Dict[str, Any]:
    info: Dict[str, Any] = {"os": f"{platform.system()} {platform.release()} ({platform.version()})"}
    try:
        import torch
        info["torch"] = str(torch.__version__)
        cuda_v = getattr(torch.version, "cuda", None)
        hip_v = getattr(torch.version, "hip", None)
        if hip_v:
            info["runtime"] = f"ROCm {hip_v}"
        elif cuda_v:
            info["runtime"] = f"CUDA {cuda_v}"
        if device is not None and getattr(device, "type", "") == "cuda" and torch.cuda.is_available():
            idx = device.index if device.index is not None else 0
            info["gpu"] = torch.cuda.get_device_name(idx)
            try:
                _free, total = torch.cuda.mem_get_info(idx)
                info["vram_total_mb"] = int(total // (1024 * 1024))
            except Exception:
                pass
        elif device is not None and getattr(device, "type", "") == "xpu":
            try:
                info["gpu"] = torch.xpu.get_device_name(device.index or 0)
                info["runtime"] = "XPU"
            except Exception:
                pass
    except Exception:
        pass
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        drv = pynvml.nvmlSystemGetDriverVersion()
        info["driver"] = drv.decode() if isinstance(drv, bytes) else str(drv)
        pynvml.nvmlShutdown()
    except Exception:
        pass
    try:
        import psutil
        info["ram_total_mb"] = int(psutil.virtual_memory().total // (1024 * 1024))
    except Exception:
        pass
    return info


def vram_snapshot(device=None) -> Optional[Dict[str, int]]:
    """CM-196: one line of the VRAM ledger, in MB. ``used``/``free``/``total``
    come from the driver (everything on the card: this process's torch
    tensors, TensorRT engines, RTX SS, NVDEC/NVENC surfaces, other
    processes); ``torch_allocated``/``torch_reserved`` from torch's caching
    allocator. ``other = used - torch_reserved`` is what torch cannot see.
    None when the device is not CUDA/ROCm."""
    try:
        import torch
        if device is None or getattr(device, "type", "") != "cuda" or not torch.cuda.is_available():
            return None
        idx = device.index if device.index is not None else 0
        free_b, total_b = torch.cuda.mem_get_info(idx)
        alloc_b = int(torch.cuda.memory_allocated(idx))
        resv_b = int(torch.cuda.memory_reserved(idx))
        mb = 1024 * 1024
        used_b = int(total_b) - int(free_b)
        return {
            "used": used_b // mb, "free": int(free_b) // mb, "total": int(total_b) // mb,
            "torch_allocated": alloc_b // mb, "torch_reserved": resv_b // mb,
            "other": max(0, used_b - resv_b) // mb,
        }
    except Exception:
        return None


def format_vram(label: str, snap: Optional[Dict[str, int]]) -> str:
    if not snap:
        return f"[VRAM] {label}: n/a"
    return (f"[VRAM] {label}: used {snap['used']} / {snap['total']} MB "
            f"(free {snap['free']}; torch allocated {snap['torch_allocated']}, "
            f"reserved {snap['torch_reserved']}; other {snap['other']} = engines/RTX SS/NVDEC/NVENC/driver)")


class RunReport:
    """Collects the run record and writes it at the end (or not at all)."""

    def __init__(self, *, mode: str, output_path: str, input_path: str,
                 panel: Optional[dict] = None, config: Optional[dict] = None,
                 version: str = "", edition: str = "", temp_dir: str = "") -> None:
        self.mode = normalize_mode(mode)
        self.output_path = str(output_path)
        self.input_path = str(input_path)
        self.panel = panel if isinstance(panel, dict) else None
        self.config = config if isinstance(config, dict) else None
        self.version = str(version)
        self.edition = str(edition)
        self.started = _dt.datetime.now()
        self._t0 = time.perf_counter()
        self.effective: Dict[str, Dict[str, Any]] = {}
        self.events: Dict[str, Dict[str, Any]] = {}
        self.checkpoints: List[List[float]] = []
        self.vram_samples: List[Dict[str, Any]] = []     # CM-196 ledger
        self._vram_peak_used: int = 0
        self.report_path: Optional[str] = None
        self.log_path: Optional[str] = None
        self._log_offset: Optional[int] = None
        self._log_source: Optional[str] = None
        if self.mode != "off":
            stem = Path(self.output_path)
            if self.mode == "temp":
                base = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir())
                stem = base / stem.name
            self.report_path = str(stem.with_suffix(".run.json"))
            self.log_path = str(stem.with_suffix(".log"))
        # Remember where the process console log stands right now, so the
        # per-run log is exactly this run's lines.
        try:
            from chitramaya.console_buffer import get_buffer
            buf = get_buffer()
            if buf is not None:
                self._log_source = getattr(buf, "log_path", None)
                self._log_offset = buf.log_offset()
        except Exception:
            self._log_source = None
            self._log_offset = None

    # -- collection ---------------------------------------------------------
    def note(self, field: str, asked: Any, ran: Any, reason: str = "") -> None:
        """Record a field that CAN differ from the panel. Kept even when
        asked == ran, so the report answers 'what ran' in one place."""
        self.effective[str(field)] = {"asked": asked, "ran": ran, "reason": str(reason or "")}

    def event(self, kind: str, frame: Optional[int] = None, message: str = "", count: int = 1) -> None:
        e = self.events.setdefault(str(kind), {"count": 0, "first": None})
        e["count"] += int(count)
        if e["first"] is None:
            e["first"] = {"frame": (int(frame) if frame is not None else None),
                          "message": str(message)[:500],
                          "time": _dt.datetime.now().strftime("%H:%M:%S")}

    def checkpoint(self, frame: int, elapsed_s: float, max_points: int = 2000,
                   vram_used_mb: Optional[int] = None) -> None:
        """[frame, elapsed_s] -- or [frame, elapsed_s, vram_used_mb] when the
        CM-196 ledger has a driver reading (the third column lines up with a
        monitor CSV without guessing the run window)."""
        if len(self.checkpoints) < max_points:
            row: List[float] = [int(frame), round(float(elapsed_s), 1)]
            if vram_used_mb is not None:
                row.append(int(vram_used_mb))
                if int(vram_used_mb) > self._vram_peak_used:
                    self._vram_peak_used = int(vram_used_mb)
            self.checkpoints.append(row)

    def vram(self, label: str, snap: Optional[Dict[str, int]], frame: Optional[int] = None,
             delta_mb: Optional[int] = None) -> None:
        """CM-196: record one ledger line (see ``vram_snapshot``). ``delta_mb``
        (T10f) = what the step that just finished added on the card."""
        if not snap:
            return
        row: Dict[str, Any] = {"label": str(label),
                               "elapsed_s": round(time.perf_counter() - self._t0, 1)}
        if frame is not None:
            row["frame"] = int(frame)
        if delta_mb is not None:
            row["delta_mb"] = int(delta_mb)
        row.update(snap)
        self.vram_samples.append(row)
        if int(snap.get("used", 0)) > self._vram_peak_used:
            self._vram_peak_used = int(snap.get("used", 0))

    # -- output -------------------------------------------------------------
    def build(self, *, run: Dict[str, Any], timing: Dict[str, Any], counts: Dict[str, Any],
              frames: Dict[str, Any], debug: Optional[Dict[str, Any]] = None,
              device=None) -> Dict[str, Any]:
        finished = _dt.datetime.now()
        run = dict(run)
        run.setdefault("started", self.started.isoformat(timespec="seconds"))
        run.setdefault("finished", finished.isoformat(timespec="seconds"))
        run.setdefault("wall_seconds", round(time.perf_counter() - self._t0, 1))
        run.setdefault("input", self.input_path)
        run.setdefault("output", self.output_path)
        timing = dict(timing)
        if self.checkpoints:
            timing["checkpoints"] = self.checkpoints
        rep: Dict[str, Any] = {
            "chitramaya": {"version": self.version, "edition": self.edition, "report_format": 1},
            "machine": _machine_info(device),
            "run": run,
            "panel": self.panel,
            "config": self.config,
            "effective": self.effective,
            "timing": timing,
            "counts": counts,
            "events": self.events,
            "frames": {k: (ranges(v) if isinstance(v, (list, set, tuple)) else v)
                       for k, v in frames.items()},
        }
        if self.vram_samples or self._vram_peak_used:
            rep["vram"] = {"peak_used_mb": int(self._vram_peak_used),
                           "samples": self.vram_samples}
        if debug:
            rep["debug"] = debug
        return rep

    def write(self, report: Dict[str, Any]) -> Optional[str]:
        if self.mode == "off" or not self.report_path:
            return None
        p = Path(self.report_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        return str(p)

    def write_log(self) -> Optional[str]:
        """Copy this run's slice of the process console log beside the report."""
        if self.mode == "off" or not self.log_path:
            return None
        text = None
        try:
            from chitramaya.console_buffer import get_buffer
            buf = get_buffer()
            if buf is not None:
                text = buf.read_log_from(self._log_offset)
        except Exception:
            text = None
        if text is None:
            return None
        try:
            with open(self.log_path, "w", encoding="utf-8", errors="replace") as f:
                f.write(text)
            return self.log_path
        except Exception:
            return None


__all__ = ["RunReport", "ranges", "expand_ranges", "normalize_mode", "REPORT_MODES"]
