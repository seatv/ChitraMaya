# chitramaya/self_check.py
"""
ChitraMaya install self-check (Batch 34, CM-097 groundwork; Batch 66
CM-132: honest GPU verdict).

Answers ONE question: "is this install sound?" -- without starting the UI
or touching any user file. Three consumers, in order of who asked for it:

  1. Remote field testing (the ROCm tester): a pasted self-check output
     tells us the edition, the torch build, whether the GPU is visible,
     and what the bundled ffmpeg can actually do -- the whole first round
     of debugging questions, answered in one shot.
  2. The patch system (CM-097): Apply-Patch verifies files, then runs
     `ChitraMaya -self-check` and refuses to call the patch good if it
     fails. GenSRT's notes: this check is worth more than the patching.
  3. Release QA: run it on the build machine before packaging.

Design rules:
  * ASCII-only output (survives every Windows codepage).
  * Never raises: every probe is guarded; the summary line + exit code
    carry the verdict. Exit 0 = PASS (warnings allowed), 1 = FAIL.
  * A missing/unusable GPU is a WARNING, not a failure -- the install
    can be perfectly sound on a machine whose driver is missing/wrong,
    and the patch verifier must not be held hostage by the environment.
    Import failures and missing/broken bundled binaries are FAILURES.
  * CM-132 (field 2026-08-30): a WARNING must not be MISLEADING. On a
    ThinkStation whose only GPU is a pre-Xe UHD P630, the old message
    said "driver missing or too old" (implying fixable) and the verdict
    said plain PASS. Now: when the GPU probe fails, the check enumerates
    the machine's actual GPUs (WMI) and says which of two situations the
    user is in -- (a) right GPU present, update the driver, or (b) this
    hardware can NEVER run this edition (and which edition, if any,
    matches what IS present) -- and the verdict banner becomes
    "PASS (install intact) -- GPU NOT USABLE". Exit code stays 0 for
    that case: Apply-Patch runs this check and must stay able to verify
    an install on a GPU-less bench machine.
  * Edition-aware without hardcoding editions: a module that fails to
    import because it needs a stack this edition deliberately excludes
    (tensorrt on XPU/ROCm, PyNvVideoCodec on XPU/ROCm, ...) is reported
    SKIP, not FAIL. Anything else is a real breakage.
"""
from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# Explicit module ledger (project style: explicit beats discovered --
# pkgutil walking is unreliable under PyInstaller's FrozenImporter, and a
# static list means a module that VANISHES from the build is noticed).
# Keep in sync when batches add modules; the spec ledgers are the
# file-level twin of this list.
_MODULE_LEDGER = [
    "chitramaya.compile_log",
    "chitramaya.config",
    "chitramaya.console_buffer",
    "chitramaya.device",
    "chitramaya.keep_awake",
    "chitramaya.models",
    # chitramaya.pipeline deliberately NOT listed: it is a dead legacy
    # top-level module (nothing imports it; its own import of
    # "chitramaya.pipeline_utils" has been broken since the mosaic/
    # reorganisation -- found by this very self-check on first run).
    # Candidate for deletion from the repo; the live one is
    # chitramaya.mosaic.pipeline below.
    "chitramaya.server",
    "chitramaya.winproc",
    "chitramaya.gui",
    "chitramaya.video.decoder",
    "chitramaya.video.encoder",
    "chitramaya.mosaic.add_mosaic",
    "chitramaya.mosaic.batch",
    "chitramaya.mosaic.cli_config",
    "chitramaya.mosaic.pipeline",
    "chitramaya.mosaic.pipeline_utils",
    "chitramaya.mosaic.session",
    "chitramaya.mosaic.vr_projection",
    "chitramaya.mosaic.watchdog",
    "chitramaya.mosaic.core.clip",
    "chitramaya.mosaic.core.scene",
    "chitramaya.mosaic.core.scene_tracker",
    "chitramaya.mosaic.detector.core",
    "chitramaya.mosaic.detector.lada_yolo",
    "chitramaya.mosaic.detector.yolo",
    "chitramaya.mosaic.models.basicvsrpp.engine_paths",
    "chitramaya.mosaic.models.basicvsrpp.inference",
    "chitramaya.mosaic.models.basicvsrpp.sub_engines",
    "chitramaya.mosaic.models.basicvsrpp.trt_export",
    "chitramaya.mosaic.models.basicvsrpp.grestorer.basicvsr_plusplus_net",
    "chitramaya.mosaic.models.basicvsrpp.grestorer.deformconv",
    "chitramaya.mosaic.models.basicvsrpp.grestorer.flow_warp",
    "chitramaya.mosaic.models.basicvsrpp.grestorer.model_utils",
    "chitramaya.mosaic.restorer.base_restorer",
    "chitramaya.mosaic.restorer.basicvsrpp_clip_restorer",
    "chitramaya.mosaic.restorer.basicvsrpp_trt_clip_restorer",
    "chitramaya.mosaic.restorer.clip_restorer",
    "chitramaya.mosaic.restorer.compositor",
    "chitramaya.mosaic.restorer.mosaic_clip_restorer",
    "chitramaya.mosaic.restorer.none_restorer",
    "chitramaya.mosaic.restorer.pseudo_clip_restorer",
    "chitramaya.mosaic.restorer.pseudo_restorer",
    "chitramaya.mosaic.restorer.rtx_secondary",
    "chitramaya.mosaic.restorer.temporal_stabilizer",
    "chitramaya.mosaic.restorer.temporalfix_arch",
    "chitramaya.mosaic.utils.config_util",
    "chitramaya.mosaic.utils.image_utils",
    "chitramaya.mosaic.utils.mask_utils",
    "chitramaya.mosaic.utils.visualization",
    "tools.process_mosaic",
    "tools.compile_basicvsrpp",
    "tools.compile_yolo",
]

# Root packages whose ABSENCE is an edition choice, not a breakage. If a
# ledger module fails to import and the missing name's root is in this
# set, that is a SKIP. (mmengine: raw-training-checkpoint dependency --
# never shipped by any edition, see CM-095.)
_EDITION_OPTIONAL = {
    "tensorrt", "torch_tensorrt", "PyNvVideoCodec", "nvvfx", "pynvml",
    "cuda", "nvidia", "mmengine", "triton", "onnxruntime",
}

# ffmpeg encoder inventory: name -> which edition path cares.
_FFMPEG_ENCODERS = [
    ("hevc_qsv",  "Intel Arc hardware HEVC"),
    ("h264_qsv",  "Intel Arc hardware H.264"),
    ("av1_qsv",   "Intel Arc hardware AV1"),
    ("hevc_amf",  "AMD Radeon hardware HEVC"),
    ("h264_amf",  "AMD Radeon hardware H.264"),
    ("av1_amf",   "AMD Radeon hardware AV1 (RDNA4+)"),
    ("libx265",   "software HEVC fallback"),
    ("libx264",   "software H.264 fallback"),
    ("libsvtav1", "software AV1 fallback (full build)"),
]


class _Tally:
    def __init__(self) -> None:
        self.ok = 0
        self.skip = 0
        self.warn = 0
        self.fail = 0
        # CM-132: set when the GPU probe establishes that video
        # processing will NOT run on this machine as it stands (driver
        # problem or wrong/absent hardware). Does not affect the exit
        # code -- the install itself can still be intact and Apply-Patch
        # must remain able to verify it -- but the verdict banner and
        # trailer say so in plain words instead of a bare PASS.
        self.gpu_unusable = False

    def line(self, status: str, text: str) -> None:
        print(f"[SelfCheck] {status:5s} {text}")
        if status == "OK":
            self.ok += 1
        elif status == "SKIP":
            self.skip += 1
        elif status == "WARN":
            self.warn += 1
        elif status == "FAIL":
            self.fail += 1

    def info(self, text: str) -> None:
        print(f"[SelfCheck] {text}")


def _missing_root(exc: BaseException) -> Optional[str]:
    """Root package name a ModuleNotFoundError complains about, else None."""
    name = getattr(exc, "name", None)
    if not name and isinstance(exc, ImportError):
        # "No module named 'x.y'" -- parse defensively.
        msg = str(exc)
        if "No module named" in msg:
            name = msg.split("'")[1] if "'" in msg else None
    return name.split(".")[0] if name else None


def _check_modules(t: _Tally) -> None:
    t.info("-- module ledger --")
    for mod in _MODULE_LEDGER:
        try:
            importlib.import_module(mod)
            t.line("OK", mod)
        except BaseException as e:  # noqa: BLE001 -- report, never die
            root = _missing_root(e)
            if root and root in _EDITION_OPTIONAL:
                t.line("SKIP", f"{mod} (needs {root}; not part of this "
                               f"edition)")
            else:
                t.line("FAIL", f"{mod} ({type(e).__name__}: {e})")


def _torch_edition(torch) -> str:
    try:
        if getattr(getattr(torch, "version", None), "hip", None):
            return "rocm"
    except Exception:
        pass
    if "+xpu" in str(getattr(torch, "__version__", "")):
        return "xpu"
    try:
        if getattr(getattr(torch, "version", None), "cuda", None):
            return "cuda"
    except Exception:
        pass
    return "cpu"


def devprobe_main() -> int:
    """Child-process half of the GPU probe (Batch 34 r2). Touches the
    device runtime -- enumeration AND a tiny fp16 matmul -- and prints
    ONE machine-readable line. It runs in its own process because the
    ROCm runtime can natively ABORT (not raise) on enumeration when the
    AMD driver is absent; field event 2026-08-15: that abort killed the
    first blind build mid-spec. In a child, a crash costs the parent a
    warning line instead of the whole self-check."""
    import torch
    edition = _torch_edition(torch)
    dev = None
    name = ""
    if edition in ("cuda", "rocm"):
        if torch.cuda.is_available():
            dev, name = "cuda", torch.cuda.get_device_name(0)
    elif edition == "xpu":
        if getattr(torch, "xpu", None) and torch.xpu.is_available():
            dev, name = "xpu", torch.xpu.get_device_name(0)
    # v1.50.00: append VRAM size to the device name -- the 8GB-vs-16GB
    # question should never need a follow-up message (field event: the
    # ROCm tester's card variant stayed unknown through a full PASS).
    # Rides inside the name field, so the parent's parser is unchanged.
    if dev is not None:
        try:
            if dev == "cuda":
                _tot = torch.cuda.get_device_properties(0).total_memory
            else:
                _tot = torch.xpu.get_device_properties(0).total_memory
            name = f"{name} ({_tot / (1024**3):.1f} GB VRAM)"
        except Exception:
            pass
    if dev is None:
        print("DEVPROBE|none||")
        return 0
    try:
        a = torch.randn(1024, 1024, device=dev, dtype=torch.float16)
        _t0 = time.perf_counter()
        b = a @ a
        if dev == "cuda":
            torch.cuda.synchronize()
        else:
            torch.xpu.synchronize()
        _ms = (time.perf_counter() - _t0) * 1000.0
        del a, b
        print(f"DEVPROBE|ok|{name}|{_ms:.1f}")
    except BaseException as e:  # noqa: BLE001
        print(f"DEVPROBE|matmulfail|{name}|{type(e).__name__}: {e}")
    return 0


def _spawn_devprobe() -> Tuple[Optional[int], str]:
    """Run devprobe_main in a child process. Frozen: re-exec our own exe
    with the hidden flag; source: python -m chitramaya with the flag."""
    if getattr(sys, "frozen", False):
        cmd = [sys.executable, "-self-check-devprobe"]
    else:
        cmd = [sys.executable, "-m", "chitramaya", "-self-check-devprobe"]
    try:
        rc, out = _run(cmd, timeout=300)
        return rc, out
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except BaseException as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def _windows_gpus() -> List[str]:
    """CM-132: names of every display adapter Windows knows about, via
    WMI (Win32_VideoController). Empty list on non-Windows, on timeout,
    or on any failure -- callers fall back to generic advice."""
    if os.name != "nt":
        return []
    try:
        rc, out = _run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_VideoController | "
             "Select-Object -ExpandProperty Name"],
            timeout=25)
        if rc != 0:
            return []
        return [ln.strip() for ln in out.splitlines() if ln.strip()]
    except BaseException:  # noqa: BLE001 -- advisory only, never die
        return []


def _point_to_matching_edition(t: _Tally, has_nvidia: bool,
                               has_arc: bool, has_amd: bool) -> None:
    """CM-132: when THIS edition can never run here, say which edition
    (if any) matches the hardware that IS present."""
    if has_nvidia:
        t.info("      An NVIDIA GPU IS present -- the NVIDIA edition "
               "matches this machine: github.com/seatv/ChitraMaya")
    if has_arc:
        t.info("      An Intel Arc GPU IS present -- the Intel Arc "
               "edition matches this machine: "
               "github.com/seatv/ChitraMaya-Intel-ARC")
    if has_amd:
        t.info("      An AMD GPU IS present -- the AMD edition may match "
               "this machine (supported RDNA3/RDNA4 cards only): "
               "github.com/seatv/ChitraMaya-AMD-ROCM")
    if not (has_nvidia or has_arc or has_amd):
        t.info("      No GPU any ChitraMaya edition supports (NVIDIA / "
               "Intel Arc / supported AMD Radeon) is present on this "
               "machine.")


def _gpu_inventory_advice(t: _Tally, edition: str) -> None:
    """CM-132 (field 2026-08-30): after a failed GPU probe, look at what
    GPUs Windows actually reports and split two very different fates the
    old message conflated: (a) the right GPU is present and a DRIVER
    update fixes it, vs (b) this machine's hardware can NEVER run this
    edition and no driver will change that. Field event: a ThinkStation
    P340 whose only GPU is a pre-Xe UHD P630 was told 'driver missing or
    too old' by the XPU edition -- advice that cannot work."""
    names = _windows_gpus()
    if not names:
        hint = (" ROCm needs AMD Adrenalin 26.2.2+ and a supported "
                "RDNA3/RDNA4 card." if edition == "rocm" else "")
        t.info(f"      Could not enumerate this machine's GPUs. If the "
               f"right GPU is installed, update its driver and re-run "
               f"the self-check.{hint}")
        return
    t.info(f"      GPUs on this machine: {'; '.join(names)}")
    low = " ; ".join(n.lower() for n in names)
    has_nvidia = any(s in low for s in ("nvidia", "geforce", "quadro",
                                        "rtx", "gtx"))
    has_amd = ("amd" in low) or ("radeon" in low)
    has_arc = "arc" in low  # discrete A/B-series and Arc-based iGPUs
    has_pre_arc_intel = ("intel" in low) and not has_arc  # UHD/Iris/HD

    if edition == "xpu":
        if has_arc:
            t.info("      An Intel Arc GPU is present -- this is a DRIVER "
                   "problem, not a hardware problem. Install the current "
                   "Intel graphics driver and re-run the self-check "
                   "(old drivers enumerate the GPU but cannot launch "
                   "compute kernels).")
        else:
            if has_pre_arc_intel:
                t.info("      This Intel GPU is pre-Arc (UHD / Iris / HD "
                       "Graphics). The XPU edition requires an Intel Arc "
                       "GPU -- a discrete A- or B-series card, or an "
                       "Arc-based Core Ultra iGPU. NO driver update will "
                       "enable it on this GPU: this machine cannot run "
                       "the XPU edition.")
            else:
                t.info("      No Intel GPU is present: this machine "
                       "cannot run the XPU edition.")
            _point_to_matching_edition(t, has_nvidia, False, has_amd)
    elif edition == "rocm":
        if has_amd:
            t.info("      An AMD GPU is present. Two requirements, both "
                   "mandatory: AMD Software Adrenalin 26.2.2 or newer, "
                   "AND a card ROCm supports on Windows (RDNA3 / RDNA4 "
                   "-- e.g. RX 7700 XT and up, RX 9060 XT and up). On an "
                   "older or unsupported Radeon, NO driver version will "
                   "make this edition work.")
        else:
            t.info("      No AMD GPU is present: this machine cannot run "
                   "the ROCm edition.")
            _point_to_matching_edition(t, has_nvidia, has_arc, False)
    elif edition == "cuda":
        if has_nvidia:
            t.info("      An NVIDIA GPU is present -- this is a DRIVER "
                   "problem, not a hardware problem. Install the current "
                   "NVIDIA driver and re-run the self-check.")
        else:
            t.info("      No NVIDIA GPU is present: this machine cannot "
                   "run the NVIDIA edition.")
            _point_to_matching_edition(t, False, has_arc, has_amd)


def _check_torch(t: _Tally) -> str:
    t.info("-- torch / GPU --")
    try:
        import torch
    except BaseException as e:
        t.line("FAIL", f"import torch ({type(e).__name__}: {e})")
        return "unknown"
    edition = _torch_edition(torch)
    t.line("OK", f"torch {torch.__version__} (edition: {edition})")
    if edition == "rocm":
        t.info(f"      hip runtime: {torch.version.hip}")

    if edition == "cpu":
        t.line("OK", "CPU-only torch build (no device check applies)")
        return edition

    # Device enumeration + matmul run in a CHILD process (see
    # devprobe_main's docstring for why). Crash tolerance here is the
    # whole point: a dead child is a diagnosis, not a dead self-check.
    rc, out = _spawn_devprobe()
    line = ""
    for _ln in out.splitlines():
        if _ln.startswith("DEVPROBE|"):
            line = _ln.strip()
    if rc == 0 and line:
        _, status, name, extra = (line.split("|", 3) + ["", "", ""])[:4]
        if status == "ok":
            t.line("OK", f"device 0: {name}")
            t.line("OK", f"fp16 matmul 1024x1024 ({extra} ms)")
        elif status == "matmulfail":
            t.line("OK", f"device 0: {name}")
            # CM-132: WARN, not FAIL -- the install is sound; the driver
            # vintage is the problem (field: Lunar Lake factory driver).
            # A FAIL here made Apply-Patch refuse a good patch on a
            # machine whose only sin was an old driver.
            t.line("WARN", f"fp16 matmul on {name} ({extra}) -- the "
                           f"device enumerates but kernels do not "
                           f"launch. This is a driver-vintage problem, "
                           f"not a bad install: update the GPU driver "
                           f"and re-run the self-check.")
            t.gpu_unusable = True
        else:
            t.line("WARN", f"no {edition} device visible -- the install "
                           f"is intact but the pipeline cannot run here "
                           f"as things stand")
            t.gpu_unusable = True
            _gpu_inventory_advice(t, edition)
    elif rc is None:
        t.line("WARN", f"device probe did not finish ({out}) -- treat as "
                       f"no device visible")
        t.gpu_unusable = True
        _gpu_inventory_advice(t, edition)
    else:
        # CM-132: state the fact (the runtime aborted), then let the GPU
        # inventory say whether a driver update can fix it or the
        # hardware itself is the limit -- the old text asserted "driver
        # missing or too old" even on machines with no eligible GPU.
        t.line("WARN", f"device probe CRASHED (exit {rc}) -- the "
                       f"{edition} runtime aborted during GPU "
                       f"enumeration. The install itself is intact; "
                       f"whether this machine can run at all is "
                       f"assessed below.")
        t.gpu_unusable = True
        _gpu_inventory_advice(t, edition)
    return edition


def _resolve_ffmpeg() -> Tuple[Optional[str], Optional[str]]:
    ff = os.environ.get("CHITRAMAYA_FFMPEG") or shutil.which("ffmpeg")
    fp = os.environ.get("CHITRAMAYA_FFPROBE") or shutil.which("ffprobe")
    return ff, fp


def _run(cmd: List[str], timeout: int = 30) -> Tuple[int, str]:
    try:
        from chitramaya.winproc import NOWINDOW
    except Exception:
        NOWINDOW = {}
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace",
                       timeout=timeout, **NOWINDOW)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def _check_ffmpeg(t: _Tally) -> None:
    t.info("-- ffmpeg / ffprobe --")
    ff, fp = _resolve_ffmpeg()
    for label, exe in (("ffmpeg", ff), ("ffprobe", fp)):
        if not exe:
            t.line("FAIL", f"{label}: not found (bundled bin/ missing and "
                           f"not on PATH)")
            continue
        try:
            rc, out = _run([exe, "-version"], timeout=30)
            first = out.splitlines()[0].strip() if out.strip() else "(no output)"
            if rc == 0:
                t.line("OK", f"{label}: {first}")
                t.info(f"      path: {exe}")
            else:
                t.line("FAIL", f"{label}: exit {rc} running -version")
        except BaseException as e:
            t.line("FAIL", f"{label}: {type(e).__name__}: {e}")
    if not ff:
        return
    # Encoder inventory: presence only (a 2-frame init probe per encoder
    # runs in the app itself; here we report what this BUILD carries).
    try:
        rc, enc_out = _run([ff, "-hide_banner", "-encoders"], timeout=30)
        if rc == 0:
            for name, why in _FFMPEG_ENCODERS:
                have = name in enc_out
                t.info(f"      encoder {name:10s} "
                       f"{'present' if have else 'absent '}  ({why})")
        else:
            t.line("WARN", f"could not list encoders (exit {rc})")
    except BaseException as e:
        t.line("WARN", f"could not list encoders ({type(e).__name__}: {e})")
    try:
        rc, hw_out = _run([ff, "-hide_banner", "-hwaccels"], timeout=30)
        if rc == 0:
            accels = [ln.strip() for ln in hw_out.splitlines()[1:]
                      if ln.strip()]
            t.info(f"      hwaccels: {', '.join(accels) if accels else '(none)'}")
    except BaseException:
        pass


def _check_weights(t: _Tally) -> None:
    t.info("-- bundled weights --")
    try:
        import chitramaya
        wdir = Path(chitramaya.__file__).resolve().parent / \
            "mosaic" / "restorer" / "weights"
        pths = sorted(wdir.glob("*.pth")) if wdir.is_dir() else []
        if len(pths) >= 3:
            t.line("OK", f"temporal-stabilizer weights: {len(pths)} files "
                         f"in {wdir.name}/")
        elif wdir.is_dir():
            t.line("FAIL", f"weights dir present but only {len(pths)} .pth "
                           f"files (expected 3+) -- broken bundle")
        else:
            t.line("FAIL", f"weights dir missing: {wdir}")
    except BaseException as e:
        t.line("FAIL", f"weights check ({type(e).__name__}: {e})")


def _check_config(t: _Tally) -> None:
    t.info("-- config --")
    base = Path(os.environ.get("CHITRAMAYA_HOME") or Path.cwd())
    cfg = base / "ChitraMaya-config.json"
    if not cfg.is_file():
        t.line("OK", "no ChitraMaya-config.json yet (created on first run)")
        return
    try:
        json.loads(cfg.read_text(encoding="utf-8"))
        t.line("OK", f"ChitraMaya-config.json parses ({cfg})")
    except BaseException as e:
        t.line("FAIL", f"ChitraMaya-config.json is not valid JSON "
                       f"({type(e).__name__}: {e}) -- fix or delete it")


def _check_optional_stacks(t: _Tally, edition: str) -> None:
    t.info("-- optional stacks --")
    # (import name, friendly, editions where absence is a FAILURE)
    probes = [
        ("tensorrt", "TensorRT", ("cuda",)),
        ("PyNvVideoCodec", "PyNvVideoCodec (NVDEC/NVENC)", ("cuda",)),
        ("ultralytics", "ultralytics (YOLO)", ("cuda", "xpu", "rocm", "cpu")),
        ("cv2", "OpenCV", ("cuda", "xpu", "rocm", "cpu")),
        ("numpy", "NumPy", ("cuda", "xpu", "rocm", "cpu")),
        ("flask", "Flask", ("cuda", "xpu", "rocm", "cpu")),
    ]
    for mod, friendly, required_on in probes:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", None)
            if not ver:
                # v1.50.00: fall back to installed-distribution metadata --
                # after the PyNvVideoCodec 2.2 API break, the exact wheel
                # version in a paste is diagnostic gold, and some builds
                # do not expose __version__.
                try:
                    from importlib import metadata as _md
                    _dist = {"PyNvVideoCodec": "pynvvideocodec",
                             "cv2": "opencv-python",
                             "flask": "Flask"}.get(mod, mod)
                    ver = _md.version(_dist)
                except Exception:
                    ver = "?"
            t.line("OK", f"{friendly} {ver}")
        except BaseException as e:
            if edition in required_on:
                t.line("FAIL", f"{friendly} missing on the {edition} "
                               f"edition ({type(e).__name__})")
            else:
                t.line("SKIP", f"{friendly} (not part of this edition)")
    # GUI backend: absence only matters for the windowed exe; the CLI is
    # complete without it, so WARN not FAIL.
    try:
        importlib.import_module("webview")
        t.line("OK", "pywebview (UI window backend)")
    except BaseException:
        t.line("WARN", "pywebview missing -- UI window will not open "
                       "(CLI/-restore unaffected)")


def main() -> int:
    t = _Tally()
    try:
        from chitramaya import __version__ as _ver
    except BaseException:
        _ver = "?"
    frozen = bool(getattr(sys, "frozen", False))
    t.info(f"ChitraMaya v{_ver} self-check  "
           f"(python {sys.version.split()[0]}, "
           f"{'frozen' if frozen else 'source'})")
    t.info(f"home: {os.environ.get('CHITRAMAYA_HOME') or Path.cwd()}")

    edition = _check_torch(t)
    _check_modules(t)
    _check_optional_stacks(t, edition)
    _check_ffmpeg(t)
    _check_weights(t)
    _check_config(t)

    # CM-132: three verdicts, not two. A machine whose GPU cannot run
    # this edition used to get a bare PASS -- true for the INSTALL,
    # false for the user's actual question ("will it work?").
    if t.fail:
        verdict = "FAIL"
    elif t.gpu_unusable:
        verdict = "PASS (install intact) -- GPU NOT USABLE"
    else:
        verdict = "PASS"
    t.info(f"==== {verdict}  ({t.ok} ok, {t.skip} skipped, {t.warn} "
           f"warnings, {t.fail} failures) ====")
    if t.fail:
        t.info("A FAIL above means this install is broken -- reinstall or "
               "re-apply the patch. Warnings are environmental (driver/"
               "GPU) and do not indicate a bad install.")
    elif t.gpu_unusable:
        t.info("The install itself is sound (patch verification can "
               "trust it), but video processing will NOT run on this "
               "machine as it stands. See the '-- torch / GPU --' "
               "section above for whether a driver update fixes that "
               "or the hardware itself is the limit.")
    # Exit code stays 0 for the GPU-unusable case ON PURPOSE:
    # Apply-Patch treats non-zero as a failed patch, and a bench machine
    # without the target GPU must still be able to verify an install.
    return 1 if t.fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
