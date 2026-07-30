# chitramaya/winproc.py
"""Windows subprocess helper (Batch 24).

When the packaged app runs WINDOWED (console=False in chitramaya.spec),
every console child process it spawns -- ffmpeg remuxes, ffprobe probes,
engine-compile subprocesses -- would otherwise make Windows pop a visible
black console window over the UI for the child's lifetime. Passing
CREATE_NO_WINDOW suppresses that.

Usage at every subprocess call site:

    from chitramaya.winproc import NOWINDOW
    subprocess.run(cmd, ..., **NOWINDOW)

Harmless in console/dev runs: ChitraMaya captures or discards child output
at every call site (capture_output / PIPE / DEVNULL), so the child never
needed the parent's console anyway. No-op on non-Windows platforms.
"""

from __future__ import annotations

import subprocess
import sys

if sys.platform == "win32":
    NOWINDOW: dict = {"creationflags": subprocess.CREATE_NO_WINDOW}
else:
    NOWINDOW = {}

__all__ = ["NOWINDOW"]
