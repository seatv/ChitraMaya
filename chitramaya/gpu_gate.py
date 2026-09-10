# chitramaya/gpu_gate.py
"""CM-163: the startup GPU gate.

Field event 2026-09-06: a user (well, Gman playing one) installed the NVIDIA
edition on an AMD machine. The window opened and vanished with no message.
The self-check knew the answer (CM-132) but nobody runs the self-check
first. This module runs the SAME child-process GPU probe the self-check uses
(self_check._spawn_devprobe -- a child, because the ROCm runtime can abort
the process outright when the driver is missing) BEFORE the UI server
touches torch, and if the edition cannot run here it says so in plain
words, names the GPUs Windows reports and the edition that matches them,
writes the same lines to the console log, and exits with code 3 (which the
.cmd launcher turns into a paused window -- CM-165).

Cost: one extra torch import in a child process, a few seconds. To keep it
off the common path, a PASS is cached in <app base>/.gpu-gate-ok keyed by
build version + GPU inventory; the probe reruns only after an update or a
hardware change. "gpuGate": false in ChitraMaya-config.json disables it.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

_EDITION_LABEL = {"cuda": "NVIDIA", "rocm": "AMD (ROCm)", "xpu": "Intel Arc (XPU)", "cpu": "CPU-only"}
_EDITION_REPO = {
    "cuda": "github.com/seatv/ChitraMaya",
    "rocm": "github.com/seatv/ChitraMaya-AMD-ROCM",
    "xpu": "github.com/seatv/ChitraMaya-Intel-ARC",
}


def _build_edition() -> str:
    """Which edition this BUILD is -- from torch/lib file names, no import
    (mirrors __main__._looks_like_rocm_build; extended for xpu/cuda)."""
    try:
        import importlib.util
        spec = importlib.util.find_spec("torch")
        if spec is None or not spec.submodule_search_locations:
            return "cpu"
        lib = Path(list(spec.submodule_search_locations)[0]) / "lib"
        names = [p.name.lower() for p in lib.iterdir()] if lib.is_dir() else []
        if any(n.startswith("amdhip64") or n.startswith("miopen") for n in names):
            return "rocm"
        if any("sycl" in n or n.startswith("ur_") or "level_zero" in n for n in names):
            return "xpu"
        if any(n.startswith("cudart") or n.startswith("cudnn") or n.startswith("cublas") for n in names):
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _inventory() -> List[str]:
    try:
        from chitramaya.self_check import _windows_gpus
        return _windows_gpus()
    except Exception:
        return []


def _classify(names: List[str]) -> Tuple[bool, bool, bool, bool]:
    low = " ; ".join(n.lower() for n in names)
    has_nvidia = any(s in low for s in ("nvidia", "geforce", "quadro", "rtx", "gtx"))
    has_amd = ("amd" in low) or ("radeon" in low)
    has_arc = "arc" in low
    has_pre_arc_intel = ("intel" in low) and not has_arc
    return has_nvidia, has_amd, has_arc, has_pre_arc_intel


def _cache_path(base: Path) -> Path:
    return base / ".gpu-gate-ok"


def _cache_key(version: str, names: List[str]) -> str:
    return json.dumps({"v": version, "gpus": sorted(names)}, sort_keys=True)


def compose_message(edition: str, names: List[str], detail: str) -> str:
    """The gentle text. Says what we know, not what we guess."""
    label = _EDITION_LABEL.get(edition, edition)
    has_nvidia, has_amd, has_arc, has_pre = _classify(names)
    lines = [
        f"Please make sure you downloaded the ChitraMaya edition that matches your GPU.",
        "",
        f"This is the {label} edition, and it could not start on this machine's GPU.",
    ]
    if names:
        lines += ["", "GPUs Windows reports on this machine:"] + [f"    - {n}" for n in names]
    else:
        lines += ["", "Windows did not report any GPU (or the query timed out)."]
    lines.append("")
    matches = []
    if has_nvidia and edition != "cuda":
        matches.append(f"an NVIDIA GPU is present -- the NVIDIA edition matches: {_EDITION_REPO['cuda']}")
    if has_amd and edition != "rocm":
        matches.append(f"an AMD GPU is present -- the AMD edition may match (RDNA 3 / RDNA 4 cards): {_EDITION_REPO['rocm']}")
    if has_arc and edition != "xpu":
        matches.append(f"an Intel Arc GPU is present -- the Intel Arc edition matches: {_EDITION_REPO['xpu']}")
    same_vendor = ((edition == "cuda" and has_nvidia) or (edition == "rocm" and has_amd)
                   or (edition == "xpu" and has_arc))
    if matches:
        lines.append("What matches this machine:")
        lines += [f"    - {m}" for m in matches]
    elif same_vendor:
        lines.append("The right kind of GPU is present, so this is most likely a driver problem:")
        if edition == "cuda":
            lines.append("    - install the current NVIDIA GeForce/Studio driver and try again.")
        elif edition == "rocm":
            lines.append("    - install AMD Software Adrenalin 26.2.2 or newer from amd.com; a basic "
                         "display driver from Windows Update is not enough. ROCm supports RDNA 3 and "
                         "RDNA 4 cards only.")
        else:
            lines.append("    - install the current Intel graphics driver (run the installer at the "
                         "machine, not over a remote session) and reboot.")
    elif has_pre and edition == "xpu":
        lines.append("Only a pre-Arc Intel GPU (UHD / Iris / HD) is present. No driver makes it "
                     "run this edition; ChitraMaya needs an Intel Arc, NVIDIA RTX, or supported AMD Radeon GPU.")
    else:
        lines.append("No GPU that any ChitraMaya edition supports (NVIDIA RTX / Intel Arc / "
                     "AMD RDNA 3-4) was found.")
    lines += ["",
              "For the exact reason run:  ChitraMaya-cli.exe -self-check",
              "This text is also in ChitraMaya-console.log next to the app."]
    if detail:
        lines += ["", f"Probe result: {detail}"]
    return "\n".join(lines)


def _spawn_probe() -> Tuple[Optional[int], str]:
    """Run self_check.devprobe_main in a child. Frozen: prefer the console
    sibling (ChitraMaya-cli.exe) so the child HAS a stdout to report on --
    the windowed exe's print() is a no-op when nothing captures it; fall back
    to our own exe. Source: python -m chitramaya. No console window flashes
    (NOWINDOW)."""
    import subprocess
    try:
        from chitramaya.winproc import NOWINDOW
    except Exception:
        NOWINDOW = {}
    if getattr(sys, "frozen", False):
        exe = Path(sys.executable)
        cli = exe.with_name(exe.stem + "-cli" + exe.suffix)
        cmd = [str(cli if cli.exists() else exe), "-self-check-devprobe"]
    else:
        cmd = [sys.executable, "-m", "chitramaya", "-self-check-devprobe"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=300, **NOWINDOW)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except BaseException as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def _show_blocking(title: str, text: str) -> None:
    """Native message box on Windows (works even when the WebView cannot);
    stdout elsewhere."""
    print(text, flush=True)
    if os.name != "nt":
        return
    try:
        import ctypes
        MB_OK, MB_ICONWARNING, MB_TOPMOST = 0x0, 0x30, 0x40000
        ctypes.windll.user32.MessageBoxW(None, text, title, MB_OK | MB_ICONWARNING | MB_TOPMOST)
    except Exception:
        pass


def run_gate(base: Path, version: str, *, enabled: bool = True) -> Optional[int]:
    """Return None to continue startup, or an exit code to stop.

    Never raises. A probe that fails for reasons unrelated to the GPU (the
    child could not be spawned at all) lets startup continue -- the gate must
    never be the thing that blocks a working machine."""
    if not enabled:
        return None
    try:
        edition = _build_edition()
        if edition == "cpu":
            # A CPU-only torch (dev box without a GPU build) has nothing to
            # gate; the pipeline reports its own device choice later.
            return None
        names = _inventory()
        key = _cache_key(version, names)
        cp = _cache_path(base)
        try:
            if cp.exists() and cp.read_text(encoding="utf-8").strip() == key:
                return None  # same build, same GPUs, passed before
        except Exception:
            pass

        rc, out = _spawn_probe()
        line = ""
        for ln in (out or "").splitlines():
            if ln.startswith("DEVPROBE|"):
                line = ln.strip()
        parts = line.split("|") if line else []
        status = parts[1] if len(parts) > 1 else ""
        name = parts[2] if len(parts) > 2 else ""

        if status == "ok":
            print(f"[GPU gate] {_EDITION_LABEL.get(edition, edition)} edition on {name}: OK")
            try:
                cp.write_text(key, encoding="utf-8")
            except Exception:
                pass
            return None

        if not line and (rc is None or rc == 0):
            # Could not run the child (timeout / spawn failure), or it ran but
            # could not report (a windowed exe with no stdout). Never block on
            # that; the normal path will report whatever is wrong.
            print(f"[GPU gate] probe gave no result (rc={rc}, {out.strip()[:120]!r}); continuing.")
            return None

        if status == "none":
            detail = "no usable GPU device for this edition"
        elif status == "matmulfail":
            detail = f"device {name} enumerates but cannot run a kernel ({parts[3] if len(parts) > 3 else ''})"
        else:
            detail = f"the GPU probe process ended with code {rc} and no result" + \
                     (" -- the runtime aborted (a missing or wrong driver does this)" if rc not in (0, None) else "")
        text = compose_message(edition, names, detail)
        print("[GPU gate] STOP")
        _show_blocking("ChitraMaya - GPU check", text)
        return 3
    except Exception as e:  # noqa: BLE001
        print(f"[GPU gate] skipped ({type(e).__name__}: {e})")
        return None
