# packaging/windows/pyinstaller_hooks_rocm/hook-torch.py
#
# ChitraMaya ROCm-build override of the stock pyinstaller-hooks-contrib
# torch hook (r3, field crash #2 2026-08-15). The stock hook runs an
# UNFILTERED collect_submodules("torch"), which imports every torch
# subpackage in an isolated child process -- and on a driverless build
# machine, importing torch.utils.benchmark touches the HIP runtime and
# natively aborts the child (0xC0000005), killing the Analysis stage.
#
# This override keeps the stock hook's collection behavior (data files,
# dynamic libs, collection mode) but does NOT walk submodules here: the
# rocm spec performs the walk itself with a crash-safe, self-healing
# filter and passes the full module list into Analysis(hiddenimports=)
# directly. hookspath entries take precedence over contrib stdhooks, so
# this file wins for the ROCm build only -- the NVIDIA and XPU builds
# keep the stock hook (their wheels enumerate cleanly; field-proven).
#
# NOTE (mkl): the stock hook additionally collects MKL DLLs on Windows
# when torch's metadata depends on `mkl` (Intel builds). The AMD ROCm
# wheel has no mkl dependency; if AMD ever adds one, the spec's
# <venv>/Library/bin sweep is the place to extend.

from PyInstaller.utils.hooks import (
    PY_DYLIB_PATTERNS,
    collect_data_files,
    collect_dynamic_libs,
)

module_collection_mode = "pyz+py"
warn_on_missing_hiddenimports = False

datas = collect_data_files(
    "torch",
    excludes=[
        "**/*.h",
        "**/*.hpp",
        "**/*.cuh",
        "**/*.lib",
        "**/*.cpp",
        "**/*.pyi",
        "**/*.cmake",
    ],
)

# No submodule walk here (see header). The spec supplies hiddenimports.
hiddenimports = []

binaries = collect_dynamic_libs(
    "torch",
    search_patterns=PY_DYLIB_PATTERNS + ["*.so.*"],
)
