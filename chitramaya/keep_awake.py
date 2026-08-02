# chitramaya/keep_awake.py
"""Batch 32: keep the system awake while a run is in flight.

Field story (2026-08-01/02, the Arc "soak crashes"): two long runs died
minutes after the user disconnected RDP for the night -- at 21:30 and
21:29 on consecutive evenings. The interactive/RDP session had been
resetting Windows' idle timer all day; once disconnected, the timer ran
out, the machine slept (event trail: IGCL device-enumeration failures,
overlay teardown, NIC link drop within 30 seconds), and on resume the
GPU compute context was gone -- surfacing as Level Zero
UR_RESULT_ERROR_UNKNOWN mid-op on xpu, and would surface as a CUDA
context error on the NVIDIA fleet. Background compute does NOT count as
user activity; an application doing hours of unattended work must say
so itself. This module is that statement. (Same bug class was
previously hit and fixed in other long-running video tools.)

Mechanics: SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
tells Windows the SYSTEM must stay awake while the calling thread holds
the claim. The display is deliberately allowed to sleep (no
ES_DISPLAY_REQUIRED) -- a dark monitor is fine and polite; only the
machine itself must keep running. The state is THREAD-scoped and
continuous: acquire() and release() must be called from the same thread
(Pipeline.run's worker thread satisfies this). Process exit clears the
state automatically, so a crash can never leave the machine insomniac.
No-op on non-Windows platforms and on any failure -- never raises.
"""
from __future__ import annotations

import os

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def acquire(label: str = "") -> bool:
    """Claim system-stays-awake for the CALLING thread.

    Returns True when the claim was made (Windows and the API accepted
    it); False on non-Windows or failure. Safe to call repeatedly."""
    if os.name != "nt":
        return False
    try:
        import ctypes
        prev = ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        ok = bool(prev)  # API returns the previous state; NULL on failure
        if ok:
            print("[Pipeline] sleep inhibit: ON -- system stays awake "
                  "during processing"
                  + (f" ({label})" if label else ""), flush=True)
        else:
            print("[Pipeline] WARNING: sleep inhibit failed; the machine "
                  "may sleep on idle timeout during long runs", flush=True)
        return ok
    except Exception:
        return False


def release() -> None:
    """Drop the claim (call from the same thread as acquire()).

    Best-effort; never raises. Leaves ES_CONTINUOUS alone-cleared so the
    normal power policy resumes immediately."""
    if os.name != "nt":
        return
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        print("[Pipeline] sleep inhibit: released", flush=True)
    except Exception:
        pass


__all__ = ["acquire", "release", "ES_CONTINUOUS", "ES_SYSTEM_REQUIRED"]
