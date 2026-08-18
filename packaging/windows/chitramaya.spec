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

# ── Version guardrail (Batch 31) ───────────────────────────────────────────
# v1.20 AND v1.30 both shipped with a stale __version__ (title bar said
# 1.20.00 for two releases) because the bump relied on release-day memory.
# Enforce it here instead: read the single-source constant, announce it
# loudly, and refuse to build a version that is already on the released
# list. Workflow: bump chitramaya/__init__.py -> build -> publish -> append
# the version to packaging/windows/released-versions.txt.
import re as _re

_init_text = (project_root / "chitramaya" / "__init__.py").read_text(encoding="utf-8")
_vm = _re.search(r'^__version__\s*=\s*"([^"]+)"', _init_text, _re.MULTILINE)
if not _vm:
    raise SystemExit("[spec] FATAL: __version__ not found in chitramaya/__init__.py")
APP_VERSION = _vm.group(1)

print("=" * 62)
print(f"[spec]   BUILDING  ChitraMaya  v{APP_VERSION}")
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
            f"chitramaya/__init__.py before building -- the title bar, log "
            f"header, and HF User-Agent all read it."
        )
else:
    print("[spec] NOTE: packaging/windows/released-versions.txt not found -- "
          "version-reuse check skipped. Create it (one version per line) to "
          "arm the guardrail.")

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
# Batch 29: bundled temporal-stabilizer weights (vs_temporalfix, Apache-2.0;
# upstream LICENSE ships in the same dir). Placed at the SAME package path
# as in the source tree so bundled_weights_dir() resolves identically in
# the frozen build. Without these, Temporal Stability silently degrades to
# a WARNING on every user machine -- the ledger below fails the build if
# they are absent.
datas.append((str(project_root / "chitramaya" / "mosaic" / "restorer" / "weights"),
              "chitramaya/mosaic/restorer/weights"))

# ── Hidden imports ─────────────────────────────────────────────────────────
hiddenimports = []
hiddenimports += collect_submodules("chitramaya")
hiddenimports += collect_submodules("tools")          # -restore / -compile-* subcommands

# ── New-module existence check (belt & braces) ─────────────────────────────
# collect_submodules walks the package AS PRESENT ON DISK at build time, so
# a new module that never landed in this repo (the WinMerge new-file blind
# spot -- same class as the missing installer-scripts incident) disappears
# from the build SILENTLY and surfaces as "No module named ..." on an end
# user's machine. Fail the build loudly instead, naming what to integrate.
_expected_modules = [
    ("chitramaya/device.py",          "CM-093 X1 accelerator abstraction"),
    ("chitramaya/keep_awake.py",      "Batch 32 sleep inhibit"),
    ("chitramaya/self_check.py",      "Batch 34 install self-check"),
    ("chitramaya/compile_log.py",     "Batch 39 compile-output capture"),
    ("chitramaya/mosaic/batch.py",    "Batch 22 folder batch"),
    ("chitramaya/mosaic/watchdog.py", "Batch 23 stall watchdog"),
    ("chitramaya/console_buffer.py",  "Batch 23/24 console drawer + log"),
    ("chitramaya/winproc.py",         "Batch 24 no-window subprocesses"),
    ("chitramaya/mosaic/restorer/temporalfix_arch.py",
                                      "Batch 26 temporal stabilizer arch"),
    ("chitramaya/mosaic/restorer/temporal_stabilizer.py",
                                      "Batch 26 temporal stabilizer"),
    # Batch 29: data files ride the same ledger -- a missing weights file
    # would not crash the build OR the app, just silently ship a Temporal
    # Stability dial that warns-and-disables on every user machine.
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
        "[spec] FATAL: the module file(s) above are not in this repo -- the "
        "build would package WITHOUT them and crash on users' machines with "
        "'No module named ...'. Integrate the batch zip(s) that add them, "
        "then re-run the packager."
    )
# Explicit entries too, in case a future refactor makes any of these
# reachable only via function-level or dynamic import.
hiddenimports += [
    "chitramaya.device", "chitramaya.keep_awake", "chitramaya.self_check", "chitramaya.compile_log",
    "chitramaya.mosaic.batch", "chitramaya.mosaic.watchdog",
    "chitramaya.console_buffer", "chitramaya.winproc",
    "chitramaya.mosaic.restorer.temporalfix_arch",
    "chitramaya.mosaic.restorer.temporal_stabilizer",
]
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

# ── CM-077 Secondary Restoration: NVIDIA Maxine binding (pip: nvidia-vfx) ──
# The importable package is `nvvfx`; its wheel carries the Python binding
# plus Maxine DLLs and model payloads that PyInstaller has no hook for, so
# collect_all is required. Optional at BUILD time to mirror the runtime
# graceful fallback: without nvidia-vfx in the release venv the build still
# succeeds and the packaged app prints its [Secondary] WARNING if the user
# selects RTX Super-Res. The prints below make the build's capability
# explicit so a missing wheel never ships silently by accident — check the
# build log for "BUNDLED" before a release that advertises Secondary.
# Clean-machine validation: run the packaged exe with Secondary = RTX 2x on
# a short clip; "[Secondary] RTX Super-Res ACTIVE" = packaging complete.
try:
    _nv_d, _nv_b, _nv_h = collect_all("nvvfx")
    datas += _nv_d
    binaries += _nv_b
    hiddenimports += _nv_h
    print(f"[spec] nvvfx (RTX Super-Res secondary): BUNDLED "
          f"({len(_nv_d)} data files, {len(_nv_b)} dynamic libs)")
except Exception as _nv_err:
    print(f"[spec] nvvfx (RTX Super-Res secondary): NOT bundled ({_nv_err}). "
          f"Packaged app will run with Secondary unavailable — "
          f"'pip install nvidia-vfx' in the release venv to include it.")

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

# Batch 24: two bootloaders, one shared onedir build.
#   ChitraMaya.exe      windowed (console=False) -- the UI. All console
#                       output is mirrored to the in-app Console drawer and
#                       to ChitraMaya-console.log next to the exe, so no
#                       terminal window opens or is needed.
#   ChitraMaya-cli.exe  console -- for the terminal workflows (-restore,
#                       -compile-rest, -compile-det) which need live stdout/
#                       tqdm in PowerShell. A windowed exe detaches from the
#                       calling console and would print NOTHING there.
# Both EXEs are tiny bootloaders sharing the same COLLECT payload, so the
# install size is unchanged (+~2 MB for the second bootloader).
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
    console=False,      # UI build: no terminal; Console drawer + log file
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
    console=True,       # CLI build: live stdout/tqdm in the terminal
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
