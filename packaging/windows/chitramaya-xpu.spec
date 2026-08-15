# -*- mode: python ; coding: utf-8 -*-
# ChitraMaya PyInstaller spec (Windows) -- XPU EDITION (Intel Arc).
#
# Sibling of chitramaya.spec (the NVIDIA/CUDA spec). Same skeleton --
# guardrail, module ledger, GUI stack, dual bootloaders -- with the NVIDIA
# freight removed (TensorRT, PyNvVideoCodec, nvvfx/Maxine, CUDA DLLs) and
# the Intel pieces added:
#   * torch +xpu build assertion (refuses to build from the wrong venv)
#   * Intel oneAPI runtime DLL collection (SYCL/UR/MKL constellation --
#     these are torch-xpu's equivalent of the CUDA DLL carriage)
#   * ffmpeg/ffprobe bundling is REQUIRED by default: on Arc, ffmpeg IS
#     the decoder (X2 qsv/d3d11va) and the encoder (X3 QSV); a build
#     without it is dead on arrival. The spec also probes the bundled
#     ffmpeg for QSV encoders so a capability gap is visible at BUILD
#     time, not on a tester's machine.
#
# Build (from repo root, inside the ARC venv):
#   python -m PyInstaller --noconfirm --clean packaging/windows/chitramaya-xpu.spec
#
# What the user must provide on the target machine: the Intel graphics
# driver (ships the Level Zero / compute runtime). Everything else rides
# in the onedir.

import argparse
import pathlib
import shutil
import subprocess
import sys

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)


def get_project_root() -> pathlib.Path:
    root = pathlib.Path(".").absolute()
    assert (root / "pyproject.toml").exists(), \
        "Run PyInstaller from the repo root (pyproject.toml must exist)."
    return root


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="ChitraMaya", help="Base exe/folder name.")
    p.add_argument("--skip-ffmpeg", action="store_true",
                   help="Do not bundle ffmpeg/ffprobe. NOT recommended for "
                        "the XPU build -- ffmpeg is the decode AND encode "
                        "path on Arc.")
    return p.parse_args()


args = parse_args()
NAME = args.name
project_root = get_project_root()

# ── Version guardrail (Batch 31; shared ledger with the NVIDIA spec) ───────
import re as _re

_init_text = (project_root / "chitramaya" / "__init__.py").read_text(encoding="utf-8")
_vm = _re.search(r'^__version__\s*=\s*"([^"]+)"', _init_text, _re.MULTILINE)
if not _vm:
    raise SystemExit("[spec] FATAL: __version__ not found in chitramaya/__init__.py")
APP_VERSION = _vm.group(1)

print("=" * 62)
print(f"[spec]   BUILDING  ChitraMaya  v{APP_VERSION}   (XPU / Intel Arc)")
print("=" * 62)

_released_file = project_root / "packaging" / "windows" / "released-versions.txt"
if _released_file.is_file():
    _released = {ln.strip() for ln in
                 _released_file.read_text(encoding="utf-8").splitlines()
                 if ln.strip() and not ln.strip().startswith("#")}
    if APP_VERSION in _released:
        raise SystemExit(
            f"[spec] FATAL: version {APP_VERSION} is already RELEASED (listed in "
            f"packaging/windows/released-versions.txt). Bump __version__ in "
            f"chitramaya/__init__.py before building."
        )
else:
    print("[spec] NOTE: released-versions.txt not found -- version-reuse "
          "check skipped.")

# ── Build-venv assertion: this MUST be the torch +xpu venv ────────────────
# The most likely packaging accident on a multi-venv dev box is freezing
# the CUDA torch into the "Arc" build (or vice versa). The wheel tags the
# version string, so check it here and refuse to continue on a mismatch.
# torch.xpu.is_available() is deliberately NOT required -- the build
# machine may not have an Arc plugged in; the +xpu wheel is what matters.
try:
    import torch as _torch
    _tv = str(_torch.__version__)
except Exception as _te:
    raise SystemExit(f"[spec] FATAL: cannot import torch in this venv ({_te})")
if "xpu" not in _tv:
    raise SystemExit(
        f"[spec] FATAL: torch {_tv} is not an +xpu build. Activate the Arc "
        f"venv (requirements-xpu.txt) before building the XPU edition."
    )
print(f"[spec] torch: {_tv} (+xpu wheel confirmed)")
try:
    if _torch.xpu.is_available():
        print(f"[spec] xpu device present on build machine: "
              f"{_torch.xpu.get_device_name(0)}")
    else:
        print("[spec] NOTE: no xpu device on the build machine (fine -- "
              "wheel check passed; runtime needs the Intel driver).")
except Exception:
    pass

# ── Bundled binaries: ffmpeg / ffprobe -> bin/ (REQUIRED on XPU) ───────────
binaries = []


def _which(exe):
    try:
        return shutil.which(exe)
    except Exception:
        return None


if args.skip_ffmpeg:
    print("[spec] WARNING: --skip-ffmpeg on the XPU build -- the packaged "
          "app will have NO decode or encode path unless the user installs "
          "ffmpeg themselves. You almost certainly do not want this.")
else:
    _ff = _which("ffmpeg.exe")
    _fp = _which("ffprobe.exe")
    if not (_ff and _fp):
        raise SystemExit(
            "[spec] FATAL: ffmpeg.exe/ffprobe.exe not found on PATH. On the "
            "XPU build ffmpeg IS the decoder (X2 qsv/d3d11va) and encoder "
            "(X3 QSV) -- put the chosen build (gyan.dev full or essentials) "
            "on PATH, or pass --skip-ffmpeg to build without it (not "
            "recommended)."
        )
    binaries += [(_ff, "bin"), (_fp, "bin")]
    # Capability probe of the EXACT binary being bundled: QSV encoders and
    # the software-AV1 ladder rungs. A gap here is a field bug tomorrow.
    try:
        _enc = subprocess.run([_ff, "-hide_banner", "-encoders"],
                              capture_output=True, text=True, timeout=30).stdout
        for _name, _why in (
            ("av1_qsv",  "AV1 hardware encode (Arc)"),
            ("hevc_qsv", "HEVC hardware encode (Arc)"),
            ("h264_qsv", "H.264 hardware encode (Arc)"),
            ("libsvtav1", "software AV1 fallback (full build only)"),
            ("libx265",  "software HEVC fallback"),
        ):
            _have = _name in _enc
            print(f"[spec] bundled ffmpeg: {_name:10s} "
                  f"{'OK' if _have else 'MISSING'}  ({_why})")
    except Exception as _fe:
        print(f"[spec] WARNING: could not probe bundled ffmpeg ({_fe})")
    print(f"[spec] bundling ffmpeg from: {_ff}")

# ── Data files (same set as the NVIDIA spec) ───────────────────────────────
datas = []
datas.append((str(project_root / "chitramaya" / "static"), "chitramaya/static"))
datas.append((str(project_root / "chitramaya" / "templates"), "chitramaya/templates"))
datas += collect_data_files("ultralytics", include_py_files=False)
datas.append((str(project_root / "chitramaya" / "mosaic" / "restorer" / "weights"),
              "chitramaya/mosaic/restorer/weights"))

# ── Hidden imports ─────────────────────────────────────────────────────────
hiddenimports = []
hiddenimports += collect_submodules("chitramaya")
hiddenimports += collect_submodules("tools")

# ── Module ledger (shared with the NVIDIA spec; keep in sync) ──────────────
_expected_modules = [
    ("chitramaya/device.py",          "CM-093 X1 accelerator abstraction"),
    ("chitramaya/keep_awake.py",      "Batch 32 sleep inhibit"),
    ("chitramaya/self_check.py",      "Batch 34 install self-check"),
    ("chitramaya/mosaic/batch.py",    "Batch 22 folder batch"),
    ("chitramaya/mosaic/watchdog.py", "Batch 23 stall watchdog"),
    ("chitramaya/console_buffer.py",  "Batch 23/24 console drawer + log"),
    ("chitramaya/winproc.py",         "Batch 24 no-window subprocesses"),
    ("chitramaya/mosaic/restorer/temporalfix_arch.py",
                                      "Batch 26 temporal stabilizer arch"),
    ("chitramaya/mosaic/restorer/temporal_stabilizer.py",
                                      "Batch 26 temporal stabilizer"),
    ("chitramaya/mosaic/restorer/weights/temporalfix_s1_v1.1.pth",
                                      "Batch 29 bundled stabilizer weights s1"),
    ("chitramaya/mosaic/restorer/weights/temporalfix_s2_v1.pth",
                                      "Batch 29 bundled stabilizer weights s2"),
    ("chitramaya/mosaic/restorer/weights/temporalfix_s3_v1.pth",
                                      "Batch 29 bundled stabilizer weights s3"),
    ("chitramaya/mosaic/restorer/weights/LICENSE-vs_temporalfix.txt",
                                      "Batch 29 upstream Apache-2.0 license"),
]
_missing_modules = [(p, why) for (p, why) in _expected_modules
                    if not (project_root / p).is_file()]
if _missing_modules:
    for _p, _why in _missing_modules:
        print(f"[spec] MISSING: {_p}  ({_why})")
    raise SystemExit(
        "[spec] FATAL: the module file(s) above are not in this repo -- "
        "integrate the batch zip(s) that add them, then re-run."
    )
hiddenimports += [
    "chitramaya.device", "chitramaya.keep_awake", "chitramaya.self_check",
    "chitramaya.mosaic.batch", "chitramaya.mosaic.watchdog",
    "chitramaya.console_buffer", "chitramaya.winproc",
    "chitramaya.mosaic.restorer.temporalfix_arch",
    "chitramaya.mosaic.restorer.temporal_stabilizer",
]
hiddenimports += collect_submodules("ultralytics")
hiddenimports += collect_submodules("cv2")
hiddenimports += collect_submodules("torch")
hiddenimports += ["flask", "flask.json", "werkzeug", "jinja2"]

# ── Native dynamic libs ────────────────────────────────────────────────────
binaries += collect_dynamic_libs("torch")
binaries += collect_dynamic_libs("cv2")

# ── Intel oneAPI runtime constellation ─────────────────────────────────────
# The +xpu wheels depend on Intel runtime wheels (SYCL runtime, oneMKL,
# Unified Runtime, UMF, ...). Depending on the wheel generation, their
# DLLs land in <venv>/Library/bin (wheel data scheme) and/or inside
# intel_*/mkl_* site-packages dirs -- torch adds those directories via
# os.add_dll_directory at import, which does not exist in a frozen app.
# Collect from BOTH locations into the bundle ROOT (the exe dir is always
# on the frozen DLL search path via the runtime hook). PyInstaller dedups
# by destination name. The sycl sanity check below fails the build if the
# constellation is missing everywhere -- a frozen app without it imports
# torch fine and then runs CPU-only, the worst kind of silent failure.
_intel_dll_count = 0
_venv = pathlib.Path(sys.prefix)
_lib_bin = _venv / "Library" / "bin"
if _lib_bin.is_dir():
    for _dll in sorted(_lib_bin.glob("*.dll")):
        binaries.append((str(_dll), "."))
        _intel_dll_count += 1
    print(f"[spec] Intel runtime: {_intel_dll_count} DLLs from {_lib_bin}")

_site = pathlib.Path(_torch.__file__).resolve().parent.parent  # site-packages
_INTEL_DIR_PREFIXES = ("intel", "umf", "tcm", "onemkl", "mkl", "dpcpp",
                       "pti", "oneccl", "icx")
_site_dll_count = 0
for _d in sorted(_site.iterdir()):
    if _d.is_dir() and _d.name.lower().startswith(_INTEL_DIR_PREFIXES):
        for _dll in sorted(_d.rglob("*.dll")):
            binaries.append((str(_dll), "."))
            _site_dll_count += 1
if _site_dll_count:
    print(f"[spec] Intel runtime: {_site_dll_count} DLLs from site-packages "
          f"intel*/mkl*/umf* dirs")

# Sanity: a SYCL runtime DLL must exist SOMEWHERE in what we collected
# (torch/lib in newer wheels, Library/bin or intel_* dirs in older ones).
_all_dll_names = [pathlib.Path(b[0]).name.lower() for b in binaries]
_sycl_present = any(("sycl" in n) or n.startswith("ur_") for n in _all_dll_names)
if not _sycl_present:
    raise SystemExit(
        "[spec] FATAL: no SYCL/UR runtime DLL found in torch/lib, "
        "<venv>/Library/bin, or intel_* site-packages dirs. The frozen app "
        "would import torch but run CPU-ONLY on the user's machine. "
        "Check the Arc venv's intel-* runtime wheels."
    )
print("[spec] SYCL runtime: present in collected binaries")

# ── GUI stack: pywebview + .NET backend ────────────────────────────────────
for pkg in ("webview", "pythonnet", "clr_loader"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

runtime_hooks = [
    str(project_root / "packaging" / "windows" / "pyinstaller_runtime_hook_chitramaya.py")
]

entry_script = str(project_root / "packaging" / "windows" / "chitramaya_entrypoint.py")

a = Analysis(
    [entry_script],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=runtime_hooks,
    # NVIDIA-only stacks excluded explicitly: absent from the Arc venv
    # anyway, but a shared/dev venv must never leak them into this build.
    # triton (torch.compile backend) excluded until the compile experiment
    # lands -- it is dead weight in eager mode.
    excludes=["onnxruntime", "tensorrt", "torch_tensorrt",
              "PyNvVideoCodec", "nvvfx", "triton"],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

# Dual bootloaders, one payload (Batch 24 pattern): windowed UI exe +
# console CLI exe for -restore / headless workflows.
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
)

exe_cli = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=f"{NAME}-cli",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
)

coll = COLLECT(
    exe,
    exe_cli,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name=NAME,
)
