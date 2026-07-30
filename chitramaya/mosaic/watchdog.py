# chitramaya/mosaic/watchdog.py
"""Stall watchdog + PCIe link canary (CM-081, Batch 23).

Two field incidents motivated this module, both on the 5060 Ti:

  1. A run hung with zero console output; nvidia-smi later reported
     "GPU has been lost". Nothing in the log said WHERE the pipeline was
     stuck, so diagnosis needed a full monitor-CSV forensic pass.
  2. That forensic pass showed the PCIe link down-training (Gen3 -> Gen1)
     a full ~18 seconds BEFORE bus transactions stopped and VRAM was
     dropped. The link generation is an early-warning canary: by the time
     the GPU "silently leaves", the down-train has already announced it.

The watchdog is a tiny daemon thread that runs alongside the pipeline loop:

  - STALL DETECTION: polls a caller-supplied progress counter (processed
    frames). If it stops advancing for ``stall_seconds``, it prints a loud
    banner plus a stack dump of every Python thread (via
    ``sys._current_frames``, so it works even while another thread is
    blocked inside a CUDA/TRT call -- the dump shows the last Python frame
    that entered it, which is exactly the "where is it stuck" answer).
    It re-alarms on a cooldown while the stall persists, and reminds the
    user that the Batch 21b ``-RECOVER.ps1`` sidecar can rebuild the
    partial output if the process must be killed.

  - PCIE CANARY (best-effort, needs NVML): samples the link generation and
    width at start; if the generation later drops below the baseline, it
    prints an immediate alarm -- on the 5060 incident this fires ~18s
    before the GPU is lost, early enough to see WHICH file was in flight
    and that the hardware (not the software) is failing.

The watchdog only ever prints (stdout is also mirrored to the UI console
drawer by chitramaya/console_buffer.py); it never touches the pipeline's
state and can never abort a run. Alarms are diagnostic, not corrective.

ASCII-only output, as everywhere in ChitraMaya.
"""

from __future__ import annotations

import sys
import threading
import time
import traceback
from typing import Callable, Optional


def _dump_all_thread_stacks() -> str:
    """Format the current Python stack of every live thread (ASCII)."""
    names = {t.ident: t.name for t in threading.enumerate()}
    parts = []
    frames = sys._current_frames()
    for ident, frame in frames.items():
        name = names.get(ident, "?")
        parts.append(f"--- Thread {name} (ident={ident}) ---")
        parts.append("".join(traceback.format_stack(frame)).rstrip())
    return "\n".join(parts)


class _PcieCanary:
    """Best-effort NVML watcher for PCIe link down-training."""

    def __init__(self, gpu_index: int):
        self._nvml = None
        self._handle = None
        self._base_gen = None
        self._base_width = None
        self._max_gen = None
        self._best_gen = None     # highest gen observed (links idle at Gen1)
        self._alarmed_gen = None  # lowest gen already alarmed on
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
            self._base_gen = int(pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle))
            self._base_width = int(pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle))
            try:
                self._max_gen = int(pynvml.nvmlDeviceGetMaxPcieLinkGeneration(handle))
            except Exception:
                self._max_gen = None
            self._nvml = pynvml
            self._handle = handle
        except Exception:
            self._nvml = None  # no NVML -> canary silently off

    @property
    def active(self) -> bool:
        return self._nvml is not None

    def describe(self) -> str:
        if not self.active:
            return "pcie-canary=off (NVML unavailable)"
        return f"pcie-canary=on (baseline Gen{self._base_gen} x{self._base_width})"

    def under_load_warning(self) -> Optional[str]:
        """Warn when the link has NOT trained up to full speed UNDER LOAD.

        Batch 25c/25d, field-learned twice over: (1) a run armed with
        'baseline Gen1 x4' on a Gen3-capable eGPU link and died of NVENC
        starvation minutes later -- the quiet baseline note was easy to
        miss; (2) BUT links legitimately drop to Gen1 at IDLE (ASPM power
        management) and train up under load, so judging at arm time can cry
        wolf. This check is therefore called by the watchdog a few polls
        INTO processing, when the GPU is demonstrably working: if the best
        generation observed so far is still below the device's max, the
        link genuinely failed to train up."""
        if not self.active or self._max_gen is None:
            return None
        best = self._best_gen if self._best_gen is not None else self._base_gen
        if best is not None and best < self._max_gen:
            return (f"PCIe link has NOT trained up under load: running "
                    f"Gen{best} but the device supports Gen{self._max_gen}. "
                    f"Bandwidth is cut ~{2 ** (self._max_gen - best)}x -- "
                    f"paging/oversubscribed runs that survive at full link "
                    f"speed can starve NVENC and fail at this speed. A "
                    f"reboot/power-cycle usually retrains the link; if it "
                    f"comes back degraded after a cold boot, suspect the "
                    f"riser/cable/slot.")
        return None

    def check(self) -> Optional[str]:
        """Return an alarm string on a NEW down-train, else None."""
        if not self.active:
            return None
        try:
            gen = int(self._nvml.nvmlDeviceGetCurrPcieLinkGeneration(self._handle))
            width = int(self._nvml.nvmlDeviceGetCurrPcieLinkWidth(self._handle))
        except Exception as e:
            # NVML dying mid-run is itself a symptom of the GPU going away.
            self._nvml = None
            return (f"PCIE CANARY: NVML query failed mid-run ({e}) -- "
                    f"this can itself mean the GPU is dropping off the bus.")
        # Track the BEST generation seen: links idle at Gen1 by design
        # (ASPM), so the meaningful reference is the fastest state the link
        # has demonstrated during this run, not the arm-time snapshot.
        if self._best_gen is None or gen > self._best_gen:
            self._best_gen = gen
        if self._best_gen is not None and gen < self._best_gen:
            if self._alarmed_gen is None or gen < self._alarmed_gen:
                self._alarmed_gen = gen
                return (f"PCIE LINK DOWN-TRAIN: Gen{self._best_gen} -> Gen{gen} "
                        f"(width x{width}). On a prior 5060 incident this "
                        f"preceded 'GPU has been lost' by ~18 seconds. If the "
                        f"run dies, the -RECOVER.ps1 sidecar next to the raw "
                        f"bitstream rebuilds the partial output.")
        return None

    def close(self) -> None:
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml = None


class StallWatchdog:
    """Daemon thread: frame-progress stall detector + PCIe canary.

    Usage:
        wd = StallWatchdog(lambda: metrics.processed_frames,
                           stall_seconds=120, gpu_index=0)
        wd.start()
        try:
            ... pipeline loop ...
        finally:
            wd.stop()

    ``stall_seconds <= 0`` disables the watchdog entirely (start() is a
    no-op), which is the CLI's ``--watchdog-stall-seconds 0``.
    """

    POLL_SECONDS = 5.0

    def __init__(
        self,
        get_progress: Callable[[], int],
        *,
        stall_seconds: float = 120.0,
        gpu_index: int = 0,
        label: str = "",
    ):
        self._get_progress = get_progress
        self._stall_seconds = float(stall_seconds)
        self._gpu_index = int(gpu_index)
        self._label = str(label)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._canary: Optional[_PcieCanary] = None

    @property
    def enabled(self) -> bool:
        return self._stall_seconds > 0

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._canary = _PcieCanary(self._gpu_index)
        print(f"[Watchdog] armed: stall-threshold={self._stall_seconds:.0f}s, "
              f"{self._canary.describe()}")
        self._thread = threading.Thread(
            target=self._loop, name="stall-watchdog", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            try:
                t.join(timeout=2.0)
            except Exception:
                pass
            self._thread = None
        if self._canary is not None:
            self._canary.close()
            self._canary = None

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        last_val = None
        last_change = time.monotonic()
        last_alarm = 0.0
        polls = 0
        while not self._stop.wait(self.POLL_SECONDS):
            now = time.monotonic()
            polls += 1

            # PCIe canary first: it fires BEFORE progress visibly stalls.
            if self._canary is not None:
                alarm = None
                try:
                    alarm = self._canary.check()
                except Exception:
                    pass
                if alarm:
                    print(f"[Watchdog] ALARM: {alarm}")
                # 25d: judge link speed UNDER LOAD, not at arm time (idle
                # links legitimately sit at Gen1 via ASPM). By the 3rd poll
                # (~15s into the main loop) the GPU is demonstrably working;
                # if the link still has not trained up, say so once.
                if polls == 3:
                    try:
                        lw = self._canary.under_load_warning()
                    except Exception:
                        lw = None
                    if lw:
                        print(f"[Watchdog] WARNING: {lw}")

            try:
                val = int(self._get_progress())
            except Exception:
                continue  # progress source briefly unavailable; not a stall

            if last_val is None or val != last_val:
                last_val = val
                last_change = now
                continue

            stalled_for = now - last_change
            if stalled_for >= self._stall_seconds and (now - last_alarm) >= self._stall_seconds:
                last_alarm = now
                where = f" ({self._label})" if self._label else ""
                print(
                    f"[Watchdog] STALL DETECTED{where}: no frame progress for "
                    f"{stalled_for:.0f}s (stuck at frame {val}). Dumping all "
                    f"thread stacks so the hang site is on record:"
                )
                try:
                    print(_dump_all_thread_stacks())
                except Exception as e:
                    print(f"[Watchdog] stack dump failed: {e}")
                print(
                    "[Watchdog] If this run must be killed, the encoder's "
                    "-RECOVER.ps1 sidecar (next to the raw bitstream in the "
                    "output folder) rebuilds a playable file from the frames "
                    "encoded so far. If nvidia-smi reports 'GPU has been "
                    "lost', the bus dropped the card -- reboot required; "
                    "check the PCIe canary lines above for the down-train."
                )


__all__ = ["StallWatchdog"]
