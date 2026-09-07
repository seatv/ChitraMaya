# packaging/windows/chitramaya_entrypoint.py
#
# Tiny bootstrap so PyInstaller freezes ChitraMaya as a PACKAGE import, not by
# pointing at chitramaya/__main__.py directly. __main__ and its imports
# (chitramaya.server, tools.*) use package-relative resolution; freezing the
# module itself as the entry script breaks that ("attempted relative import
# with no known parent package"). Importing chitramaya.__main__ here keeps the
# package context intact.
from __future__ import annotations

import multiprocessing


def main() -> int:
    from chitramaya.__main__ import main as _main
    return int(_main() or 0)


if __name__ == "__main__":
    # Batch 80 (CM-112 sprint): REQUIRED for ultralytics training in the
    # frozen app. Windows multiprocessing spawns dataloader workers by
    # re-executing this exe with "--multiprocessing-fork parent_pid=...";
    # without freeze_support() that child falls into the normal argv
    # dispatch and dies on the UI argparse (field: first frozen -train-det,
    # 2026-09-02). freeze_support() intercepts the handshake and turns the
    # child into a worker; it is a no-op in every other invocation.
    multiprocessing.freeze_support()
    raise SystemExit(main())
