# -*- mode: python ; coding: utf-8 -*-
# ChitraMaya PyInstaller spec (Windows) -- ROCm EDITION (AMD Radeon).
#
# Third sibling: chitramaya.spec (NVIDIA/CUDA), chitramaya-xpu.spec
# (Intel Arc), and this one. Same skeleton -- version guardrail, module
# ledger, GUI stack, dual bootloaders -- with the AMD specifics:
#   * torch +rocm build assertion (refuses to build from the wrong venv;
#     the wheel version string is "2.9.1+rocm7.2.1"-shaped)
#   * ROCm SDK runtime collection. AMD ships the runtime as pip wheels
#     (rocm_sdk_core / rocm_sdk_libraries / devel, from repo.radeon.com,
#     requirements-rocm.txt) that install python packages plus native
#     DLLs and GPU kernel packs into site-packages. The exact layout is
#     wheel-generation-dependent, so this spec DISCOVERS rather than
#     assumes: collect_all() on the importable rocm packages, a
#     tree-copy of any rocm/hip-prefixed site-packages dir that is not
#     importable, and collect_dynamic_libs("torch") for anything living
#     in torch/lib. A build-time ASSERTION then requires the HIP runtime
#     DLL (amdhip64*) to be present SOMEWHERE in the collection --
#     without it the frozen app imports torch and silently runs
#     CPU-only, the exact failure mode the xpu spec's SYCL check was
#     built to prevent. If AMD's layout shifts, the build fails loudly
#     here, on OUR machine, not on a tester's.
#   * ffmpeg REQUIRED and capability-probed for the AMF encoders
#     (hevc_amf/h264_amf/av1_amf) -- on Radeon, ffmpeg is the decoder
#     (d3d11va ladder) and the encoder (Batch 34 AMF rung).
#
# Build (from repo root, inside the ROCm venv, python 3.12):
#   python -m PyInstaller --noconfirm --clean packaging/windows/chitramaya-rocm.spec
#
# What the user must provide on the target machine: the AMD Adrenalin
# graphics driver, 26.2.2 or newer (AMD's floor for the ROCm 7.2.1
# PyTorch-on-Windows runtime). Everything else rides in the onedir.

import argparse
import os
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

# r3 (field crash #2, 2026-08-15): hide HIP devices for the whole build.
# On a driverless machine the ROCm runtime ABORTS the process on any
# GPU touch (even import-time, see the torch walk below); an empty
# HIP_VISIBLE_DEVICES makes enumeration short-circuit cleanly on stacks
# that honor it. Build-time only -- this must NEVER be set in the
# runtime hook, where it would blind the user's real GPU. Belt and
# braces alongside the subprocess isolation + safe walk below.
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ.setdefault("ROCR_VISIBLE_DEVICES", "")


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
                        "the ROCm build -- ffmpeg is the decode AND encode "
                        "path on Radeon.")
    return p.parse_args()


args = parse_args()
NAME = args.name
project_root = get_project_root()

# ── Version guardrail (Batch 31; shared ledger with the other specs) ───────
import re as _re

_init_text = (project_root / "chitramaya" / "__init__.py").read_text(encoding="utf-8")
_vm = _re.search(r'^__version__\s*=\s*"([^"]+)"', _init_text, _re.MULTILINE)
if not _vm:
    raise SystemExit("[spec] FATAL: __version__ not found in chitramaya/__init__.py")
APP_VERSION = _vm.group(1)

print("=" * 62)
print(f"[spec]   BUILDING  ChitraMaya  v{APP_VERSION}   (ROCm / AMD Radeon)")
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

# ── Build-venv assertion: this MUST be the torch +rocm venv ───────────────
# Same multi-venv accident guard as the xpu spec: the wheel tags the
# version string ("2.9.1+rocm7.2.1"), so check it and refuse a mismatch.
# torch.cuda.is_available() is deliberately NOT required -- the build
# machine may have no AMD card (blind-build workflow, validated on the
# Arc edition); the +rocm wheel is what matters.
try:
    import torch as _torch
    _tv = str(_torch.__version__)
except Exception as _te:
    raise SystemExit(f"[spec] FATAL: cannot import torch in this venv ({_te})")
if "rocm" not in _tv:
    raise SystemExit(
        f"[spec] FATAL: torch {_tv} is not a +rocm build. Activate the ROCm "
        f"venv (requirements-rocm.txt) before building the ROCm edition."
    )
print(f"[spec] torch: {_tv} (+rocm wheel confirmed)")
_hip_ver = getattr(getattr(_torch, "version", None), "hip", None)
print(f"[spec] hip runtime (per torch): {_hip_ver}")
# Device-presence probe, SUBPROCESS-ISOLATED (r2, field crash 2026-08-15):
# on a build machine with no AMD driver, torch.cuda.is_available() on a
# +rocm build can ABORT the process natively -- not a catchable Python
# exception -- and it killed the first blind-build attempt mid-spec with
# no traceback. Run the enumeration in a child process so a crash there
# costs a NOTE line, not the build.
try:
    _dp = subprocess.run(
        [sys.executable, "-c",
         "import torch; a = torch.cuda.is_available(); "
         "print('DEV|' + (torch.cuda.get_device_name(0) if a else 'none'))"],
        capture_output=True, text=True, timeout=180)
    _dev = None
    for _ln in (_dp.stdout or "").splitlines():
        if _ln.startswith("DEV|"):
            _dev = _ln[4:].strip()
    if _dp.returncode == 0 and _dev and _dev != "none":
        print(f"[spec] hip device present on build machine: {_dev}")
    elif _dp.returncode == 0:
        print("[spec] NOTE: no AMD device on the build machine (fine -- "
              "wheel check passed; runtime needs the Adrenalin driver).")
    else:
        print(f"[spec] NOTE: device probe exited {_dp.returncode} -- the "
              f"ROCm runtime aborts enumeration without an AMD GPU/driver; "
              f"expected on a blind-build machine. Wheel check passed; "
              f"continuing.")
except Exception as _pe:
    print(f"[spec] NOTE: device probe skipped ({_pe})")

# ── Bundled binaries: ffmpeg / ffprobe -> bin/ (REQUIRED on ROCm) ──────────
binaries = []


def _which(exe):
    try:
        return shutil.which(exe)
    except Exception:
        return None


if args.skip_ffmpeg:
    print("[spec] WARNING: --skip-ffmpeg on the ROCm build -- the packaged "
          "app will have NO decode or encode path unless the user installs "
          "ffmpeg themselves. You almost certainly do not want this.")
else:
    _ff = _which("ffmpeg.exe")
    _fp = _which("ffprobe.exe")
    if not (_ff and _fp):
        raise SystemExit(
            "[spec] FATAL: ffmpeg.exe/ffprobe.exe not found on PATH. On the "
            "ROCm build ffmpeg IS the decoder (d3d11va ladder) and encoder "
            "(Batch 34 AMF rung) -- put the gyan.dev 'full' build on PATH, "
            "or pass --skip-ffmpeg to build without it (not recommended)."
        )
    binaries += [(_ff, "bin"), (_fp, "bin")]
    # Capability probe of the EXACT binary being bundled: the AMF rungs
    # this edition encodes with, plus the software ladder. A gap here is
    # a field bug tomorrow. (Presence check only -- AMF cannot INITIALIZE
    # without an AMD GPU, which the build machine may not have; the
    # 2-frame init probe runs in the app on the target machine.)
    try:
        _enc = subprocess.run([_ff, "-hide_banner", "-encoders"],
                              capture_output=True, text=True, timeout=30).stdout
        _missing_hw = []
        for _name, _why, _required in (
            ("hevc_amf",  "HEVC hardware encode (Radeon)", True),
            ("h264_amf",  "H.264 hardware encode (Radeon)", True),
            ("av1_amf",   "AV1 hardware encode (RDNA4+)", True),
            ("libsvtav1", "software AV1 fallback (full build only)", False),
            ("libx265",   "software HEVC fallback", False),
        ):
            _have = _name in _enc
            print(f"[spec] bundled ffmpeg: {_name:10s} "
                  f"{'OK' if _have else 'MISSING'}  ({_why})")
            if _required and not _have:
                _missing_hw.append(_name)
        if _missing_hw:
            raise SystemExit(
                f"[spec] FATAL: bundled ffmpeg lacks AMF encoders "
                f"({', '.join(_missing_hw)}). This build of ffmpeg cannot "
                f"hardware-encode on Radeon at all -- the tester would "
                f"silently run CPU x265. Use the gyan.dev 'full' build."
            )
    except SystemExit:
        raise
    except Exception as _fe:
        print(f"[spec] WARNING: could not probe bundled ffmpeg ({_fe})")
    print(f"[spec] bundling ffmpeg from: {_ff}")

# ── Data files (same set as the other specs) ───────────────────────────────
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

# ── Module ledger (shared with the other specs; keep in sync) ──────────────
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

# ── torch submodule walk, CRASH-SAFE (r3, field crash #2 2026-08-15) ──────
# collect_submodules("torch") enumerates by IMPORTING every torch
# subpackage in an isolated child -- and on a driverless machine,
# importing torch.utils.benchmark touches the HIP runtime and natively
# aborts the child (exit 0xC0000005), killing the build. The filter
# gates recursion BEFORE import (verified against PyInstaller 6.22
# source), so filtered subtrees are never touched. Start with the known
# crashers (plus tensorboard, which just warns about a missing package);
# if a NEW subtree crashes the child, parse its name out of the error,
# add it to the skip list, and retry -- self-healing, bounded, loud.
# Pre-seeded with every crasher found in the field (2026-08-15 build:
# the three _inductor entries were discovered by the self-healing loop).
# Seeding them saves ~3 walk retries per build.
_torch_skip = ["torch.utils.benchmark", "torch.utils.tensorboard",
               "torch._inductor.template_heuristics",
               "torch._inductor.kernel",
               "torch._inductor.codegen.cutedsl"]
_torch_mods = None
for _attempt in range(8):
    def _torch_filter(n, _skip=tuple(_torch_skip)):
        return not any(n == s or n.startswith(s + ".") for s in _skip)
    try:
        _torch_mods = collect_submodules("torch", filter=_torch_filter)
        break
    except Exception as _ce:
        _m = _re.search(r"args=\('([\w.]+)'", str(_ce))
        if _m and _m.group(1) not in _torch_skip:
            print(f"[spec] NOTE: torch submodule walk crashed importing "
                  f"{_m.group(1)} (native GPU-runtime abort, driverless "
                  f"machine) -- excluding that subtree and retrying")
            _torch_skip.append(_m.group(1))
        else:
            raise SystemExit(
                f"[spec] FATAL: torch submodule walk keeps dying and the "
                f"crashing module could not be identified ({_ce}). Add the "
                f"culprit to _torch_skip in this spec."
            )
if _torch_mods is None:
    raise SystemExit(
        f"[spec] FATAL: torch submodule walk did not converge after 8 "
        f"attempts; skip list so far: {_torch_skip}"
    )
hiddenimports += _torch_mods
print(f"[spec] torch submodules collected: {len(_torch_mods)} "
      f"(skipped subtrees: {', '.join(_torch_skip)})")
# Export the skip list for the hook override in pyinstaller_hooks_rocm/
# (the stock contrib hook-torch runs its own UNFILTERED walk during
# Analysis and would crash the same way; our override skips the walk and
# defers to the list collected here).
os.environ["CHITRAMAYA_TORCH_SKIP"] = ",".join(_torch_skip)

hiddenimports += ["flask", "flask.json", "werkzeug", "jinja2"]

# ── Native dynamic libs ────────────────────────────────────────────────────
binaries += collect_dynamic_libs("torch")
binaries += collect_dynamic_libs("cv2")

# ── ROCm SDK runtime collection ────────────────────────────────────────────
# Strategy 1: collect_all() on every importable rocm package. AMD's
# wheels install python packages (rocm_sdk and friends); collect_all
# preserves package-relative layout, which matters because HIP resolves
# device libraries / kernel packs relative to its install root.
_rocm_pkg_hits = []
for _pkg in ("rocm_sdk", "_rocm_sdk", "rocm", "rocm_sdk_core",
             "rocm_sdk_libraries", "hip"):
    try:
        _d, _b, _h = collect_all(_pkg)
        if _d or _b or _h:
            datas += _d
            binaries += _b
            hiddenimports += _h
            _rocm_pkg_hits.append(f"{_pkg} ({len(_b)} libs, {len(_d)} data)")
    except Exception:
        pass
if _rocm_pkg_hits:
    print(f"[spec] ROCm packages collected: {', '.join(_rocm_pkg_hits)}")

# Strategy 1b (defensive, r3): sweep <venv>/Library/bin if it exists --
# the wheel data scheme some Windows wheels use (the xpu edition's Intel
# runtime landed there; the stock torch hook collects MKL from there on
# mkl-dependent builds). Expected empty on today's ROCm venv; harmless.
_venv_lib_bin = pathlib.Path(sys.prefix) / "Library" / "bin"
if _venv_lib_bin.is_dir():
    _lb_count = 0
    for _dll in sorted(_venv_lib_bin.glob("*.dll")):
        binaries.append((str(_dll), "."))
        _lb_count += 1
    if _lb_count:
        print(f"[spec] Library/bin sweep: {_lb_count} DLLs")

# Strategy 2: tree-copy any rocm/hip/amd-prefixed site-packages directory
# that strategy 1 did not reach (not importable, or layout PyInstaller's
# hooks skip). Copied under its own name so relative lookups survive.
_site = pathlib.Path(_torch.__file__).resolve().parent.parent  # site-packages
_ROCM_DIR_PREFIXES = ("rocm", "_rocm", "hip", "hsa", "amd", "roc")
_seen_roots = {str(_dest).replace("\\", "/").split("/")[0]
               for _src, _dest in datas}
_tree_dll_count = 0
for _d in sorted(_site.iterdir()):
    if not (_d.is_dir() and _d.name.lower().startswith(_ROCM_DIR_PREFIXES)):
        continue
    if _d.name.endswith(".dist-info") or _d.name in _seen_roots:
        continue
    for _f in sorted(_d.rglob("*")):
        if _f.is_file():
            _rel = _f.relative_to(_site).parent
            if _f.suffix.lower() in (".dll", ".pyd"):
                binaries.append((str(_f), str(_rel)))
                _tree_dll_count += 1
            else:
                datas.append((str(_f), str(_rel)))
if _tree_dll_count:
    print(f"[spec] ROCm runtime: {_tree_dll_count} additional native libs "
          f"from site-packages rocm*/hip*/amd* trees")

# ── HIP-present-or-fatal assertion (the xpu spec's SYCL check, in AMD) ─────
# amdhip64*.dll is the HIP runtime torch loads on Windows. If it is not
# in the collection, the frozen app imports torch and runs CPU-ONLY on
# the tester's machine -- the worst kind of silent failure. Fail HERE.
_all_native_names = [pathlib.Path(b[0]).name.lower() for b in binaries] + \
                    [pathlib.Path(d[0]).name.lower() for d in datas]
_hip_present = any("amdhip" in n for n in _all_native_names)
if not _hip_present:
    raise SystemExit(
        "[spec] FATAL: no amdhip64* HIP runtime DLL found in torch/lib, "
        "collected rocm packages, or rocm*/hip*/amd* site-packages trees. "
        "The frozen app would import torch but run CPU-ONLY on the user's "
        "machine. Check that the rocm_sdk wheels (requirements-rocm.txt) "
        "are installed in THIS venv -- and if AMD moved the DLLs, extend "
        "the collection strategies above to the new location."
    )
print("[spec] HIP runtime (amdhip64*): present in collected files")

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

# NVIDIA-only stacks excluded explicitly (absent from the ROCm venv
# anyway, but a shared/dev venv must never leak them in), plus the
# same dead weight the xpu build drops.
_base_excludes = ["onnxruntime", "tensorrt", "torch_tensorrt",
                  "PyNvVideoCodec", "nvvfx", "triton"]

# ── Analysis, CRASH-SAFE (r5, field crash #3 2026-08-15) ──────────────────
# After Analysis, PyInstaller's binary-dependency stage IMPORTS every
# collected package in a child process to resolve DLLs -- and modules the
# safe walk skipped can still enter the bundle via STATIC analysis
# (torch.utils.benchmark did: some torch module references it), whereupon
# the bindepend child imports it and hits the same driverless HIP abort.
# Closure: everything on the crash skip-list is also EXCLUDED from the
# module graph (excluded modules are never collected, so bindepend never
# imports them -- all are torch.compile-only machinery this eager-mode
# edition does not ship), and the Analysis call itself self-heals the
# same way the walk does: parse the crashing package out of the error,
# exclude it, re-run. Each retry re-runs the full Analysis (~minutes),
# so crashers are pre-seeded in _torch_skip above as they are found.
a = None
for _attempt in range(6):
    try:
        a = Analysis(
            [entry_script],
            pathex=[str(project_root)],
            binaries=binaries,
            datas=datas,
            hiddenimports=hiddenimports,
            # r3: local hook dir FIRST -- overrides the contrib hook-torch
            # whose unfiltered submodule walk would repeat the driverless
            # HIP abort (see pyinstaller_hooks_rocm/hook-torch.py).
            hookspath=[str(project_root / "packaging" / "windows" /
                           "pyinstaller_hooks_rocm")],
            hooksconfig={},
            runtime_hooks=runtime_hooks,
            excludes=_base_excludes + _torch_skip,
            noarchive=False,
            optimize=0,
        )
        break
    except Exception as _ae:
        _m = _re.search(r"importing package '([\w.]+)'", str(_ae))
        if _m and _m.group(1) not in _torch_skip:
            print(f"[spec] NOTE: binary-dependency scan crashed importing "
                  f"{_m.group(1)} (native GPU-runtime abort, driverless "
                  f"machine) -- excluding it and RE-RUNNING Analysis "
                  f"(takes a few minutes; add it to _torch_skip in this "
                  f"spec to pre-seed future builds)")
            _torch_skip.append(_m.group(1))
        else:
            raise
if a is None:
    raise SystemExit(
        f"[spec] FATAL: Analysis did not converge after 6 attempts; "
        f"skip list so far: {_torch_skip}"
    )

pyz = PYZ(a.pure)

# Dual bootloaders, one payload (Batch 24 pattern): windowed UI exe +
# console CLI exe for -restore / -self-check / headless workflows.
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
