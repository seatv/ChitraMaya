# -*- mode: python ; coding: utf-8 -*-
# ChitraMaya PyInstaller spec (Windows) — GUI (pywebview) + CUDA/torch + TensorRT
# + PyNvVideoCodec + ultralytics, with bundled ffmpeg/ffprobe.
#
# Modeled on the proven gRestorer CLI spec, plus the GUI collection the CLI
# didn't need. Build via packaging/windows/chitramaya-packager.ps1.

import argparse
import pathlib
import shutil

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
                   help="Do not bundle ffmpeg/ffprobe (expect them on PATH).")
    return p.parse_args()


args = parse_args()
NAME = args.name
project_root = get_project_root()

# ── Bundled binaries: ffmpeg / ffprobe → bin/ ──────────────────────────────
binaries = []


def _which(exe):
    try:
        return shutil.which(exe)
    except Exception:
        return None


if not args.skip_ffmpeg:
    for exe in ("ffmpeg.exe", "ffprobe.exe"):
        p = _which(exe)
        if p:
            binaries.append((p, "bin"))

# ── Data files ─────────────────────────────────────────────────────────────
datas = []
# GUI assets (Flask templates + static JS/CSS). Explicit is safer than relying
# on package-data collection for these.
datas.append((str(project_root / "chitramaya" / "static"), "chitramaya/static"))
datas.append((str(project_root / "chitramaya" / "templates"), "chitramaya/templates"))
# ultralytics ships yaml/config data files it loads at runtime.
datas += collect_data_files("ultralytics", include_py_files=False)

# ── Hidden imports ─────────────────────────────────────────────────────────
hiddenimports = []
hiddenimports += collect_submodules("chitramaya")
hiddenimports += collect_submodules("tools")          # -restore / -compile-* subcommands
hiddenimports += collect_submodules("ultralytics")
hiddenimports += collect_submodules("cv2")
hiddenimports += collect_submodules("torch")
hiddenimports += collect_submodules("PyNvVideoCodec")
hiddenimports += collect_submodules("tensorrt")       # runtime .engine loading
hiddenimports += collect_submodules("torch_tensorrt")  # BasicVSR++ restorer engine loader
hiddenimports += ["flask", "flask.json", "werkzeug", "jinja2"]

# ── Native dynamic libs (the CUDA/TRT/codec DLLs) ──────────────────────────
binaries += collect_dynamic_libs("torch")
binaries += collect_dynamic_libs("cv2")
binaries += collect_dynamic_libs("PyNvVideoCodec")
binaries += collect_dynamic_libs("tensorrt")
binaries += collect_dynamic_libs("torch_tensorrt")

# ── TensorRT builder-resource filter (applied AFTER Analysis) ──────────────
# tensorrt_libs bundles a per-architecture compile-time builder DLL
# (nvinfer_builder_resource_sm<ARCH>_10.dll), ~150-640 MB each. We ship only
# the consumer archs we've tested — plus the PTX builder, the JIT fallback so
# an untested/newer GPU still compiles (slower first time) instead of hard-
# failing. Trims ~1.3 GB with no loss for tested cards.
#
# These DLLs are added by PyInstaller's standard hook-tensorrt_libs during
# Analysis, so we must filter a.binaries below, not this pre-Analysis list.
#
# DENYLIST (drop these) — edit to widen/narrow per release. Denylisting the
# dropped archs (vs allow-listing kept ones) means a future TRT that adds a new
# consumer arch survives the filter automatically.
_DROP_TRT_BUILDER_ARCHS = ("sm90", "sm100", "sm80", "sm75")  # Hopper, DC-Blackwell, A100, Turing


def _keep_binary(dest_name: str) -> bool:
    low = dest_name.lower().replace("\\", "/").rsplit("/", 1)[-1]
    if "nvinfer_builder_resource_" not in low:
        return True  # not a builder resource — always keep
    # Per-arch builder resource; drop only denylisted archs. PTX
    # ("..._resource_ptx_10.dll") isn't in the denylist, so it's kept.
    return not any(f"_{arch}_" in low for arch in _DROP_TRT_BUILDER_ARCHS)

# ── GUI stack: pywebview + its .NET backend (WinForms/Chromium) ────────────
# collect_all returns (datas, binaries, hiddenimports); pywebview needs its
# bundled JS, and pythonnet/clr_loader are the WinForms host on Windows.
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

# Freeze via the bootstrap entrypoint (NOT chitramaya/__main__.py directly —
# that breaks package-relative imports in a frozen build).
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
    excludes=["onnxruntime"],  # not used; avoid multi-GB bloat
    noarchive=False,
    optimize=0,
)

# Drop the TensorRT builder resources for archs we don't ship (see above).
# a.binaries entries are (dest_name, src_path, typecode); filter on dest_name.
_before = len(a.binaries)
a.binaries = [b for b in a.binaries if _keep_binary(str(b[0]))]
print(f"[spec] TRT builder filter: {_before} -> {len(a.binaries)} binaries "
      f"(dropped archs: {', '.join(_DROP_TRT_BUILDER_ARCHS)})")

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX on CUDA DLLs is risky; keep off for reliability
    console=True,       # ChitraMaya logs heavily to console; keep it visible
    disable_windowed_traceback=False,
    argv_emulation=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name=NAME,
)
