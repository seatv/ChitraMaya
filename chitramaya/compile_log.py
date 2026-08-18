# chitramaya/compile_log.py
"""
Compile-output capture (Batch 39, ported idea from GenSRT).

Engine compilation is the most failure-prone, hardware-dependent thing
ChitraMaya does, and until now its output was ephemeral: the Manage
Models panel streams it into an in-memory buffer that resets on the next
compile, and CLI runs scroll it into the void. Field event 2026-08-16:
an OOM and several "tactic not tried -- insufficient memory" warnings
rolled by during a recompile with no way to share them afterwards.

This module tees BOTH stdout and stderr at the OS file-descriptor level
into a per-compile log file while still passing everything through to
the console (or the server's pipe, for UI-driven compiles). The fd-level
tee matters: TensorRT's builder logs from C++ directly to the process
stderr, bypassing sys.stderr entirely -- a Python-level tee would keep
our [compile] prints and lose exactly the tactic/OOM lines that made
this feature necessary.

Usage (both compile tools):

    log_path = compile_log_path(engine_dir, "restore")
    with tee_compile_output(log_path):
        write_log_header(argv=sys.argv)
        ... the whole compile ...

Never breaks a compile: every failure path in here degrades to "no log"
(or "log without C++ lines" on the Python-level fallback) with a printed
note, and the compile itself proceeds.
"""
from __future__ import annotations

import datetime
import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Optional


def compile_log_path(anchor_dir: str | os.PathLike, tag: str) -> str:
    """<anchor_dir>/<tag>-compile-YYYYMMDD-HHMMSS.log (dir created)."""
    d = Path(anchor_dir)
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        d = Path(".")
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return str(d / f"{tag}-compile-{stamp}.log")


def write_log_header(argv=None) -> None:
    """Print (through the tee) a self-describing header: what ran, on what
    hardware, against which stack. This is the block that turns a pasted
    log into a remotely debuggable artifact."""
    print("=" * 70)
    print(f"[compile] ChitraMaya engine compile log")
    print(f"[compile] started: {datetime.datetime.now().isoformat(timespec='seconds')}")
    if argv:
        print(f"[compile] argv:    {' '.join(str(a) for a in argv)}")
    try:
        from chitramaya import __version__ as _v
        print(f"[compile] app:     v{_v}")
    except Exception:
        pass
    print(f"[compile] python:  {sys.version.split()[0]}")
    try:
        import torch
        print(f"[compile] torch:   {torch.__version__}")
        if torch.cuda.is_available():
            _free, _total = torch.cuda.mem_get_info()
            print(f"[compile] gpu:     {torch.cuda.get_device_name(0)} "
                  f"(free {_free // (1024*1024)} MB / "
                  f"total {_total // (1024*1024)} MB)")
    except Exception:
        pass
    try:
        import tensorrt as trt
        print(f"[compile] tensorrt: {trt.__version__}")
    except Exception:
        pass
    print("=" * 70)


class _PyTee:
    """Python-level fallback tee (captures our prints; C++ output passes
    through to the console uncaptured). Used only when fd duplication is
    unavailable (e.g. no valid console handles)."""

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, s):
        try:
            self._stream.write(s)
        except Exception:
            pass
        try:
            self._fh.write(s)
        except Exception:
            pass
        return len(s)

    def flush(self):
        for t in (self._stream, self._fh):
            try:
                t.flush()
            except Exception:
                pass

    def __getattr__(self, name):
        return getattr(self._stream, name)


class tee_compile_output:
    """Context manager: everything written to fd 1/2 (and therefore to
    sys.stdout/sys.stderr AND to any C/C++ code in-process) goes to both
    the original destination and `log_path`. Restores fds on exit and
    writes any in-flight exception's traceback into the log first."""

    def __init__(self, log_path: str):
        self.log_path = str(log_path)
        self._fh = None
        self._saved = {}      # fd -> saved dup
        self._pumps = []
        self._py_fallback = None
        self._lock = threading.Lock()

    # -- internals ---------------------------------------------------------

    def _pump(self, rfd: int, orig_fd: Optional[int]):
        while True:
            try:
                chunk = os.read(rfd, 65536)
            except OSError:
                break
            if not chunk:
                break
            if orig_fd is not None:
                try:
                    os.write(orig_fd, chunk)
                except OSError:
                    pass
            with self._lock:
                try:
                    self._fh.write(chunk.decode("utf-8", "replace"))
                    self._fh.flush()
                except Exception:
                    pass
        try:
            os.close(rfd)
        except OSError:
            pass

    # -- context protocol ----------------------------------------------------

    def __enter__(self):
        try:
            self._fh = open(self.log_path, "w", encoding="utf-8",
                            errors="replace", newline="")
        except Exception as e:
            print(f"[compile] NOTE: cannot create log file "
                  f"{self.log_path} ({e}); continuing without one.")
            return self

        hooked_any = False
        for fd in (1, 2):
            try:
                saved = os.dup(fd)
                r, w = os.pipe()
                os.dup2(w, fd)
                os.close(w)
            except OSError:
                continue
            t = threading.Thread(target=self._pump, args=(r, saved),
                                 daemon=True)
            t.start()
            self._saved[fd] = saved
            self._pumps.append(t)
            hooked_any = True
        # r3 (field finding, restore log 2026-08-15): with fd1 swapped to a
        # pipe, Python BLOCK-buffers stdout, so sparse [compile] prints sat
        # in the buffer until exit while stderr streamed through -- the log
        # read as "all warnings first, then the whole compile". Force line
        # buffering so every line lands near its real time.
        if hooked_any:
            for _st in (sys.stdout, sys.stderr):
                try:
                    _st.reconfigure(line_buffering=True)
                except Exception:
                    pass

        if not hooked_any:
            # No usable console fds (windowed frozen exe corner). Fall back
            # to a Python-level tee: our prints are captured, C++ output is
            # not -- still infinitely better than nothing.
            self._py_fallback = (sys.stdout, sys.stderr)
            sys.stdout = _PyTee(sys.stdout, self._fh)
            sys.stderr = _PyTee(sys.stderr, self._fh)
            print("[compile] NOTE: fd-level capture unavailable; log will "
                  "contain ChitraMaya's own output but may miss native "
                  "TensorRT lines.")
        print(f"[compile] full log: {self.log_path}")
        return self

    def __exit__(self, exc_type, exc, tb):
        # A dying compile writes its traceback THROUGH the tee first, so
        # the log always ends with the reason.
        if exc_type is not None:
            try:
                print(f"\n[compile] *** FAILED: {exc_type.__name__}: {exc} ***")
                traceback.print_exception(exc_type, exc, tb)
            except Exception:
                pass
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass

        if self._py_fallback is not None:
            sys.stdout, sys.stderr = self._py_fallback
            self._py_fallback = None

        for fd, saved in self._saved.items():
            try:
                os.dup2(saved, fd)      # also drops the pipe's last writer
                os.close(saved)
            except OSError:
                pass
        self._saved.clear()
        for t in self._pumps:
            t.join(timeout=5)
        self._pumps.clear()
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception:
                pass
            self._fh = None
        return False   # never swallow the exception
