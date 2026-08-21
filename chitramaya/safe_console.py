# chitramaya/safe_console.py
"""Console-write hardening for CLI runs (CM-110, Batch 50).

Field event 2026-08-18 (bedroom dev box, dual-GPU compile session): a
``ChitraMaya-cli -compile-rest`` run crashed with
``OSError(22, 'Incorrect function')`` / "lost sys.stderr" while the GUI
was open; closing the GUI made the identical command work, and so did
redirecting the CLI's output (``*> file.txt``). The trigger is a console
handle that stops accepting WriteConsoleW mid-run -- Windows console
handles are shared, stateful objects, and another process changing the
console's mode (or the handle simply going bad) turns every subsequent
Python print/tqdm write into an OSError that kills the run.

The compile itself was FINE in every such event. Losing a 40-minute
engine build because a *console write* failed inverts the priorities,
and chitramaya/compile_log.py already holds the doctrine for its own
scope: "a dead console must never kill a compile." This module extends
that doctrine to the whole CLI process:

  ``install()`` wraps ``sys.stdout`` and ``sys.stderr`` in write-through
  guards. The first time a write raises, that stream is marked dead and
  every later write is silently dropped -- the run continues, and the
  compile-log tee / console-log file (which write to their own file
  handles, not the console) keep the full record.

Notes:
  - Idempotent; safe to call before console_buffer.install() or the
    compile-log tee (both re-wrap whatever is in sys.stdout/sys.stderr
    at their install time, so the layers compose).
  - The guard deliberately reports success (returns len(s)) after a
    failure: callers like tqdm treat a write error as fatal ("lost
    sys.stderr"), which is precisely the behavior being removed.
  - fileno()/isatty()/encoding pass through so libraries that probe the
    stream (tqdm, colorama, logging) behave exactly as before while the
    console is healthy.

Pure stdlib; ASCII-only output, as everywhere in ChitraMaya.
"""
from __future__ import annotations

import io
import sys


class _SafeStream(io.TextIOBase):
    """Write-through wrapper that survives a dying console handle."""

    def __init__(self, real, label: str):
        self._real = real
        self._label = str(label)
        self._dead = False

    # -- writing -----------------------------------------------------------
    def write(self, s) -> int:
        if isinstance(s, (bytes, bytearray)):
            s = s.decode("utf-8", "replace")
        else:
            s = str(s)
        if self._real is None or self._dead:
            return len(s)
        try:
            self._real.write(s)
        except Exception:
            # Console handle went bad (field: OSError 22 'Incorrect
            # function' with the GUI open). Mark dead and carry on -- the
            # run matters, the console echo does not. No note is printed
            # here on purpose: the sibling stream may be equally dead, and
            # file-based logs (compile log tee, ChitraMaya-console.log)
            # still capture everything written from now on.
            self._dead = True
        return len(s)

    def flush(self) -> None:
        if self._real is None or self._dead:
            return
        try:
            self._real.flush()
        except Exception:
            self._dead = True

    # -- passthroughs some libraries probe -----------------------------------
    def fileno(self) -> int:
        if self._real is not None and hasattr(self._real, "fileno"):
            return self._real.fileno()
        raise io.UnsupportedOperation("fileno")

    def isatty(self) -> bool:
        try:
            return bool(self._real is not None and self._real.isatty())
        except Exception:
            return False

    @property
    def encoding(self):  # type: ignore[override]
        return getattr(self._real, "encoding", "utf-8") or "utf-8"

    @property
    def errors(self):  # type: ignore[override]
        return getattr(self._real, "errors", "replace") or "replace"

    def writable(self) -> bool:
        return True


def install() -> None:
    """Wrap sys.stdout/sys.stderr in write-guards. Idempotent."""
    if not isinstance(sys.stdout, _SafeStream):
        sys.stdout = _SafeStream(sys.stdout, "stdout")
    if not isinstance(sys.stderr, _SafeStream):
        sys.stderr = _SafeStream(sys.stderr, "stderr")


__all__ = ["install"]
