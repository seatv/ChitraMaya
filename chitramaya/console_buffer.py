# chitramaya/console_buffer.py
"""In-process console capture for the UI console drawer (Batch 23).

Everything ChitraMaya prints -- the batch plan roster, skip reasons, the
Batch 21b REMUX FAILED recovery banner, watchdog alarms -- currently goes to
stdout/stderr. In the packaged .exe there is no console window, so exe users
never see any of it. This module tees stdout/stderr into a bounded in-memory
ring buffer that the Flask server exposes at ``/api/console`` and the UI
renders in a collapsible drawer.

Design:
  - ``install()`` wraps ``sys.stdout`` and ``sys.stderr`` once (idempotent).
    The wrapper writes through to the real stream (console users lose
    nothing) AND feeds the ring buffer.
  - Lines are committed on ``\\n``. Carriage-return progress spinners (tqdm
    writes ``...\\r...\\r...`` on stderr) are collapsed to their final
    segment, so a progress bar lands in the buffer once -- as its last
    state -- instead of thousands of times.
  - The buffer holds the most recent ``max_lines`` lines, each stamped with
    a monotonically increasing sequence number. Clients poll with
    ``?since=<seq>`` and receive only new lines -- cheap enough for a 1 Hz
    poll from the drawer.
  - Frozen/windowed builds may have ``sys.stdout is None``; the tee then
    writes to the buffer only.

Pure stdlib, no Flask/GPU imports -- unit-testable anywhere.
"""

from __future__ import annotations

import io
import sys
import threading
from collections import deque
from typing import List, Optional, Tuple

_MAX_LINE_CHARS = 2000   # hard cap per line (a runaway line can't eat the buffer)


class ConsoleBuffer:
    """Bounded ring buffer of console lines with sequence numbers.

    Optionally mirrors every committed line to a log file (``log_path``),
    flushed per line. In the windowed .exe (console=False) this file is the
    only post-mortem when the app dies before the Console drawer was opened
    -- a startup crash in a windowed app is otherwise completely silent.
    The file is truncated at install time, so it always holds exactly the
    current session.
    """

    def __init__(self, max_lines: int = 2000, log_path: Optional[str] = None):
        self._lines: deque[Tuple[int, str]] = deque(maxlen=int(max_lines))
        self._seq = 0
        self._partial = ""
        self._lock = threading.Lock()
        self._fh = None
        if log_path:
            try:
                self._fh = open(log_path, "w", encoding="utf-8",
                                errors="replace")
            except Exception:
                self._fh = None  # unwritable location must never break prints

    def feed(self, text: str) -> None:
        """Feed raw stream text; commit completed lines on newline."""
        if not text:
            return
        with self._lock:
            self._partial += text
            if len(self._partial) > _MAX_LINE_CHARS * 4:
                # Pathological no-newline stream: force-commit to bound memory.
                self._commit_locked(self._partial)
                self._partial = ""
                return
            while "\n" in self._partial:
                line, self._partial = self._partial.split("\n", 1)
                self._commit_locked(line)

    def _commit_locked(self, line: str) -> None:
        # tqdm-style updates: keep only the segment after the last \r.
        if "\r" in line:
            line = line.rsplit("\r", 1)[-1]
        line = line.rstrip()
        if not line:
            return
        if len(line) > _MAX_LINE_CHARS:
            line = line[:_MAX_LINE_CHARS] + " ...[truncated]"
        self._seq += 1
        self._lines.append((self._seq, line))
        if self._fh is not None:
            try:
                self._fh.write(line + "\n")
                self._fh.flush()   # per-line: the log must survive a hard crash
            except Exception:
                self._fh = None    # disk full / handle gone -> stop mirroring

    def snapshot(self, since: int = 0) -> dict:
        """Lines with seq > ``since``. Returns {"next": <latest seq>, "lines": [...]}.

        ``next`` is what the client passes back as ``since`` on its next poll.
        If the client's cursor predates the oldest buffered line (buffer
        wrapped), it simply gets everything still buffered.
        """
        with self._lock:
            out: List[str] = [ln for (seq, ln) in self._lines if seq > since]
            return {"next": self._seq, "lines": out}


class _TeeStream(io.TextIOBase):
    """Write-through wrapper: real stream (may be None) + ConsoleBuffer."""

    def __init__(self, real, buf: ConsoleBuffer):
        self._real = real
        self._buf = buf

    # -- writing ----------------------------------------------------------
    def write(self, s) -> int:
        s = str(s)
        if self._real is not None:
            try:
                self._real.write(s)
            except Exception:
                pass  # a dead console must never kill a print()
        try:
            self._buf.feed(s)
        except Exception:
            pass
        return len(s)

    def flush(self) -> None:
        if self._real is not None:
            try:
                self._real.flush()
            except Exception:
                pass

    # -- passthroughs some libraries probe --------------------------------
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


_buffer: Optional[ConsoleBuffer] = None
_install_lock = threading.Lock()


def install(max_lines: int = 2000, log_path: str | None = None) -> ConsoleBuffer:
    """Tee sys.stdout/sys.stderr into a shared ConsoleBuffer. Idempotent.

    ``log_path``: optional file that mirrors every line (truncated now) --
    the windowed .exe's post-mortem. Ignored on repeat calls."""
    global _buffer
    with _install_lock:
        if _buffer is not None:
            return _buffer
        buf = ConsoleBuffer(max_lines=max_lines, log_path=log_path)
        sys.stdout = _TeeStream(sys.stdout, buf)
        sys.stderr = _TeeStream(sys.stderr, buf)
        _buffer = buf
        return buf


def get_buffer() -> Optional[ConsoleBuffer]:
    """The installed buffer, or None if install() was never called."""
    return _buffer


__all__ = ["ConsoleBuffer", "install", "get_buffer"]
