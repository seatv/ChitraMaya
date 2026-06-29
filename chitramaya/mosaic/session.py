"""Cross-process session indicator.

Writes a small JSON to a well-known location when a job is running, so a
second instance (UI opening while CLI batch is running, or vice versa)
can detect the situation and warn the user.

Design notes:
    - Stored in the system temp dir as ``ChitraMaya.session.json``.
    - Contents: ``{pid, mode, started_at, paths}``. ``mode`` is one of
      ``cli``, ``cli-batch``, ``ui``, ``ui-batch``.
    - Stale-lock cleanup: if the recorded PID is dead, treat the lock
      as absent (read_session returns None). This avoids the cost of a
      Windows file lock and the friction of stale-lock removal after a
      crash.
    - **This is advisory.** The lockfile does NOT prevent concurrent
      runs at the OS level — only one ``MosaicPipeline`` per GPU works
      well, but two GPUs (or future parallel pipelines) would be fine.
      Callers decide what to do with the information.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

SESSION_FILENAME = "ChitraMaya.session.json"


def _session_path() -> Path:
    """Lockfile location. Override with $CHITRAMAYA_SESSION_FILE if needed."""
    custom = os.environ.get("CHITRAMAYA_SESSION_FILE")
    if custom:
        return Path(custom)
    return Path(tempfile.gettempdir()) / SESSION_FILENAME


def _pid_alive(pid: int) -> bool:
    """Cross-platform PID liveness check."""
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            # Windows: opening the process succeeds iff it exists.
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            handle = kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid),
            )
            if not handle:
                return False
            # Check exit code — STILL_ACTIVE (259) = running.
            exit_code = ctypes.c_ulong()
            kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            kernel32.CloseHandle(handle)
            return exit_code.value == 259
        else:
            # POSIX: signal 0 doesn't deliver but errors if the process is gone.
            os.kill(pid, 0)
            return True
    except (PermissionError, ProcessLookupError, OSError):
        return False
    except Exception:
        # Be conservative — assume alive if we couldn't tell.
        return True


@dataclass
class SessionInfo:
    pid: int
    mode: str
    started_at: float
    paths: list[str]
    age_sec: float

    @property
    def is_us(self) -> bool:
        return self.pid == os.getpid()


def read_session() -> SessionInfo | None:
    """Return the live session if any, else None.

    Stale locks (PID dead) are auto-cleaned and reported as None.
    """
    path = _session_path()
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception:
        # Corrupt — best to wipe.
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        return None

    pid = int(data.get("pid", 0))
    if not _pid_alive(pid):
        # Stale — clean up and report as no-session.
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        return None

    started_at = float(data.get("started_at", 0.0))
    return SessionInfo(
        pid=pid,
        mode=str(data.get("mode", "unknown")),
        started_at=started_at,
        paths=list(data.get("paths") or []),
        age_sec=max(0.0, time.time() - started_at),
    )


def write_session(*, mode: str, paths: Iterable[str | None] | None = None) -> None:
    """Stamp a session lockfile. Overwrites any existing one."""
    payload = {
        "pid": os.getpid(),
        "mode": str(mode),
        "started_at": time.time(),
        "paths": [str(p) for p in (paths or []) if p],
    }
    path = _session_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8",
        )
    except Exception:
        logger.exception("Failed to write session lockfile: %s", path)


def clear_session() -> None:
    """Remove the lockfile. Idempotent."""
    path = _session_path()
    try:
        path.unlink(missing_ok=True)
    except Exception:
        logger.exception("Failed to clear session lockfile: %s", path)


class SessionLock:
    """Context manager: write on enter, clear on exit.

    Usage::

        with SessionLock(mode="cli", paths=[input_path]):
            run_job()
    """
    def __init__(self, *, mode: str, paths: Iterable[str | None] | None = None):
        self.mode = mode
        self.paths = list(paths or [])

    def __enter__(self) -> "SessionLock":
        write_session(mode=self.mode, paths=self.paths)
        return self

    def __exit__(self, *exc):
        clear_session()
        return False
