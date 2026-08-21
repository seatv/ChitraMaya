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


def write_log_header(argv=None, gpu_id: int | None = None) -> None:
    """Print (through the tee) a self-describing header: what ran, on what
    hardware, against which stack. This is the block that turns a pasted
    log into a remotely debuggable artifact.

    ``gpu_id``: the CUDA device the compile will actually run on. v1.60
    fix (field event, dual-GPU box 2026-08-18): the header hardcoded
    device 0 and reported the WRONG card for ``--gpu-id 1`` compiles —
    on a Franken+12GB box every log claimed the 6GB laptop chip."""
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
            _dev = int(gpu_id) if gpu_id is not None else 0
            _free, _total = torch.cuda.mem_get_info(_dev)
            print(f"[compile] gpu:     cuda:{_dev} "
                  f"{torch.cuda.get_device_name(_dev)} "
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
        self._orig_std = None       # (sys.stdout, sys.stderr) before rebind
        self._new_std = None
        self._retargeted = []       # [(logging handler, original stream)]
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
        try:
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
            if hooked_any:
                # r4 (field crash 2026-08-16, interactive Windows console):
                # sys.stdout/sys.stderr there are _WindowsConsoleIO objects,
                # which write via WriteConsoleW -- legal ONLY on console
                # handles. After the dup2 swap, their fds are pipes, so the
                # very first print raised OSError(22, 'Incorrect function')
                # and killed the compile ("lost sys.stderr"). The server
                # path never hit it (subprocess stdout is a pipe from
                # birth), and Linux never hits it -- the exact
                # worked-in-every-test-on-Linux trap the GenSRT notes warn
                # about. Fix: rebind Python's stream objects to fresh
                # line-buffered wrappers over the (now piped) fds -- valid
                # everywhere -- and retarget logging handlers that captured
                # the old objects at import time (torch/ultralytics do).
                # Line buffering here also subsumes the r3 ordering fix.
                self._rebind_python_streams()
        except Exception:
            # Doctrine: the capture must NEVER break the compile. Unwind
            # any partial fd surgery and fall back to the Python-level tee.
            self._unwind_fds()
            hooked_any = False

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

    def _rebind_python_streams(self) -> None:
        import logging
        old_out, old_err = sys.stdout, sys.stderr
        new_out = open(1, "w", buffering=1, encoding="utf-8",
                       errors="replace", closefd=False)
        new_err = open(2, "w", buffering=1, encoding="utf-8",
                       errors="replace", closefd=False)
        sys.stdout, sys.stderr = new_out, new_err
        self._orig_std = (old_out, old_err)
        self._new_std = (new_out, new_err)
        try:
            handlers = list(logging.root.handlers)
            for lg in logging.Logger.manager.loggerDict.values():
                if isinstance(lg, logging.Logger):
                    handlers.extend(lg.handlers)
            for h in handlers:
                stream = getattr(h, "stream", None)
                if not isinstance(h, logging.StreamHandler):
                    continue
                if stream is old_out:
                    self._retargeted.append((h, old_out))
                    h.setStream(new_out)
                elif stream is old_err:
                    self._retargeted.append((h, old_err))
                    h.setStream(new_err)
        except Exception:
            pass

    def _restore_python_streams(self) -> None:
        for h, orig in self._retargeted:
            try:
                h.setStream(orig)
            except Exception:
                pass
        self._retargeted = []
        if self._new_std is not None:
            for st in self._new_std:
                try:
                    st.flush()
                except Exception:
                    pass
        if self._orig_std is not None:
            sys.stdout, sys.stderr = self._orig_std
            self._orig_std = None
        self._new_std = None

    def _unwind_fds(self) -> None:
        self._restore_python_streams()
        for fd, saved in list(self._saved.items()):
            try:
                os.dup2(saved, fd)
                os.close(saved)
            except OSError:
                pass
        self._saved.clear()
        for t in self._pumps:
            t.join(timeout=5)
        self._pumps = []

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

        # r4: put Python's original stream objects back BEFORE the fds
        # become consoles again (no prints may happen in between).
        self._restore_python_streams()

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
