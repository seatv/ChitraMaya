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

import statistics
import sys
import threading
import time
import traceback
from collections import deque
from typing import Callable, Dict, List, Optional


class GpuSampler:
    """CM-209: NVML readings the watchdog takes every poll -- VRAM used,
    PCIe TX/RX throughput, encoder/decoder/GPU utilization -- kept for the
    run (medians, percentiles) and for the window since the last 500-frame
    checkpoint (the checkpoint line and the run report), plus a PAGING
    detector.

    Why: on 09-17 a 5060 Ti sat at 8,041 of 8,151 MB with 5 GB/s on the
    bus in both directions and the encoder at 0 % for a quarter of an hour
    before the card fell off the bus -- the same signature as the Dell's
    09-14 runs. The external monitor CSV showed it after the fact; the run
    itself said nothing. VRAM at the wall alone is NOT the signature (the
    torch cache sits at the wall on every long run and the Dell's A/B runs
    did that without paging); VRAM at the wall AND the bus busy, sustained,
    is: nothing in the re-decode pipeline moves frames over the bus, so
    sustained GB/s at a full card is the driver swapping GPU memory through
    host RAM.

    NVIDIA only (NVML). Elsewhere the sampler is silently off; the monitor
    CSV remains the instrument there (WDDM backend, nvGPUMonitor v0.10).
    Best-effort: every failure turns the sampler off, never the run.
    """

    # 60 s of 5 s polls decides "sustained"; a single busy poll is a drain.
    WINDOW = 12
    PAGING_VRAM_FRAC = 0.97
    PAGING_PCIE_GBS = 2.0
    PAGING_REPEAT_S = 300.0

    def __init__(self, gpu_index: int):
        self._nvml = None
        self._handle = None
        self.samples: List[Dict[str, float]] = []      # every poll, whole run
        self._ck: List[Dict[str, float]] = []           # since the last checkpoint
        self._win: deque = deque(maxlen=self.WINDOW)
        self.paging_seconds: float = 0.0
        self.paging_episodes: int = 0
        self._paging_now = False
        self._paging_last_warn = 0.0
        self._last_t: Optional[float] = None
        self._vram_total_mb: float = 0.0
        self.notes: List[str] = []
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
            mi = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            self._vram_total_mb = float(mi.total) / (1024 * 1024)
            self._nvml = pynvml
        except Exception:
            self._nvml = None

    @property
    def active(self) -> bool:
        return self._nvml is not None

    def sample(self) -> Optional[str]:
        """Take one reading. Returns a console line when the paging state
        changes (or repeats while it persists), else None."""
        if self._nvml is None:
            return None
        nv = self._nvml
        h = self._handle
        now = time.monotonic()
        try:
            mi = nv.nvmlDeviceGetMemoryInfo(h)
            used_mb = float(mi.used) / (1024 * 1024)
            tx = float(nv.nvmlDeviceGetPcieThroughput(h, nv.NVML_PCIE_UTIL_TX_BYTES)) / 1e6   # KB/s -> GB/s
            rx = float(nv.nvmlDeviceGetPcieThroughput(h, nv.NVML_PCIE_UTIL_RX_BYTES)) / 1e6
            try:
                enc = float(nv.nvmlDeviceGetEncoderUtilization(h)[0])
            except Exception:
                enc = -1.0
            try:
                dec = float(nv.nvmlDeviceGetDecoderUtilization(h)[0])
            except Exception:
                dec = -1.0
            try:
                ur = nv.nvmlDeviceGetUtilizationRates(h)
                gpu = float(ur.gpu)
            except Exception:
                gpu = -1.0
        except Exception as e:
            self.notes.append(f"sampler off: {e}")
            self._nvml = None
            return None
        s = {"t": now, "vram_mb": used_mb, "tx_gbs": tx, "rx_gbs": rx,
             "enc": enc, "dec": dec, "gpu": gpu}
        self.samples.append(s)
        self._ck.append(s)
        self._win.append(s)

        line = None
        if len(self._win) >= self.WINDOW and self._vram_total_mb > 0:
            v = statistics.median(x["vram_mb"] for x in self._win) / self._vram_total_mb
            bus = statistics.median(max(x["tx_gbs"], x["rx_gbs"]) for x in self._win)
            encm = statistics.median(x["enc"] for x in self._win)
            paging = (v >= self.PAGING_VRAM_FRAC and bus >= self.PAGING_PCIE_GBS)
            if paging:
                if self._last_t is not None:
                    self.paging_seconds += max(0.0, now - self._last_t)
                if not self._paging_now:
                    self._paging_now = True
                    self.paging_episodes += 1
                    self._paging_last_warn = now
                    line = self._paging_line(v, bus, encm, first=True)
                elif now - self._paging_last_warn >= self.PAGING_REPEAT_S:
                    self._paging_last_warn = now
                    line = self._paging_line(v, bus, encm, first=False)
            elif self._paging_now:
                self._paging_now = False
                line = (f"[Watchdog] VRAM paging cleared: bus back to {bus:.1f} GB/s "
                        f"(paged for {self.paging_seconds:.0f} s so far this run)")
        self._last_t = now
        return line

    def _paging_line(self, v: float, bus: float, encm: float, *, first: bool) -> str:
        head = "[Watchdog] WARNING: VRAM PAGING" if first else "[Watchdog] WARNING: still paging"
        idle = ""
        if 0 <= encm <= 5:
            idle = " NVENC is idle, so almost no frames are completing;"
        return (f"{head} -- VRAM at {100 * v:.0f} % of the card and {bus:.1f} GB/s on the PCIe bus "
                f"for the last 60 s: the driver is swapping GPU memory through system RAM.{idle} "
                f"the run is running far below its speed and an 8 GB card in this state has lost "
                f"NVENC before (CM-122). Levers, cheapest first: Frame Store = Re-decode; "
                f"vram_cache_release on (ChitraMaya-config.json); Scaler Denoise off; "
                f"Temporal Fix off; Max Clip Length 180 -> 120; recompile the detector "
                f"engine if the console said it was built before CM-201 (CM-209).")

    def checkpoint_summary(self) -> Optional[Dict[str, float]]:
        """Medians since the last checkpoint (and reset that window)."""
        if not self._ck:
            return None
        ck = self._ck
        self._ck = []
        def med(k):
            vals = [x[k] for x in ck if x[k] >= 0]
            return round(statistics.median(vals), 2) if vals else None
        return {"n": len(ck), "vram_mb": med("vram_mb"), "pcie_tx_gbs": med("tx_gbs"),
                "pcie_rx_gbs": med("rx_gbs"), "enc_util": med("enc"), "dec_util": med("dec"),
                "gpu_util": med("gpu"), "paging": bool(self._paging_now)}

    def run_summary(self) -> Optional[Dict[str, object]]:
        if not self.samples:
            return None
        def q(k, p):
            vals = sorted(x[k] for x in self.samples if x[k] >= 0)
            if not vals:
                return None
            i = min(len(vals) - 1, max(0, int(round(p * (len(vals) - 1)))))
            return round(vals[i], 2)
        return {
            "samples": len(self.samples),
            "poll_seconds": StallWatchdog.POLL_SECONDS,
            "vram_total_mb": int(self._vram_total_mb),
            "vram_used_mb": {"median": q("vram_mb", 0.5), "p90": q("vram_mb", 0.9), "max": q("vram_mb", 1.0)},
            "pcie_tx_gbs": {"median": q("tx_gbs", 0.5), "p90": q("tx_gbs", 0.9), "max": q("tx_gbs", 1.0)},
            "pcie_rx_gbs": {"median": q("rx_gbs", 0.5), "p90": q("rx_gbs", 0.9), "max": q("rx_gbs", 1.0)},
            "enc_util": {"median": q("enc", 0.5), "p90": q("enc", 0.9)},
            "dec_util": {"median": q("dec", 0.5), "p90": q("dec", 0.9)},
            "gpu_util": {"median": q("gpu", 0.5), "p90": q("gpu", 0.9)},
            "paging_seconds": round(self.paging_seconds, 0),
            "paging_episodes": int(self.paging_episodes),
            "notes": list(self.notes[-5:]),
        }

    def close(self) -> None:
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml = None


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
        self.last_gen = None      # T23e: the generation the last check() read
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
        # T23e (CM-241): links sit at Gen1 at idle (ASPM); the arm-time
        # reading is not the reference -- the best generation seen under
        # load is (check() tracks it). Say so, so a 'baseline Gen1' line on
        # a Gen4 card stops reading like a fault.
        idle = ""
        if self._max_gen is not None and self._base_gen is not None and self._base_gen < self._max_gen:
            idle = f", idle now; Gen{self._max_gen} link judged under load"
        return f"pcie-canary=on (baseline Gen{self._base_gen} x{self._base_width}{idle})"

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
        self.last_gen = gen
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

    def recovery_note(self) -> Optional[str]:
        """Return a one-time note when a previously alarmed link has
        re-trained back to its best generation.

        v1.50.00 (field, MCL-300 run 2026-08-16): the link dipped
        Gen3->Gen2 mid-run, ALARMed, then re-trained to Gen3 within
        seconds -- power-state churn, not the 5060 pre-failure signature
        (whose tell is a down-train WITHOUT recovery). Without this note
        a clean run ends with an unresolved alarm in the log, and the
        user is left worried about a link that healed itself. Resets the
        alarm latch so a later genuine down-train alarms again."""
        if not self.active or self._alarmed_gen is None:
            return None
        try:
            gen = int(self._nvml.nvmlDeviceGetCurrPcieLinkGeneration(self._handle))
        except Exception:
            return None
        if self._best_gen is not None and gen >= self._best_gen:
            self._alarmed_gen = None
            return (f"PCIe link RECOVERED: back at Gen{gen}. The earlier "
                    f"down-train was transient (power management), not the "
                    f"pre-failure signature -- that one does not recover.")
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

    # Batch 50: PCIe flap governor tuning. ESCALATE is how long a held
    # down-train may stay unrecovered before it alarms in full anyway --
    # the 5060 pre-failure signature killed the GPU ~18s after the
    # down-train, so 30s of no recovery is decisively "not a flap".
    # SUMMARY is the minimum spacing between flap-count one-liners.
    PCIE_FLAP_ESCALATE_S = 30.0
    PCIE_FLAP_SUMMARY_S = 60.0

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
        self.gpu: Optional[GpuSampler] = None          # CM-209: read by the pipeline

    @property
    def enabled(self) -> bool:
        return self._stall_seconds > 0

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._canary = _PcieCanary(self._gpu_index)
        self.gpu = GpuSampler(self._gpu_index)
        print(f"[Watchdog] armed: stall-threshold={self._stall_seconds:.0f}s, "
              f"{self._canary.describe()}, "
              f"{'gpu-sampler=on (VRAM, PCIe, NVENC every 5 s; paging detector)' if self.gpu.active else 'gpu-sampler=off (NVML unavailable)'}")
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
        if self.gpu is not None:
            self.gpu.close()      # keeps its samples; the pipeline reads them after stop()

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        last_val = None
        last_change = time.monotonic()
        last_alarm = 0.0
        polls = 0
        # Batch 50: PCIe flap governor state. Field 2026-08-19 (the Idol
        # forensics run AND the 4060 benchmark): during GPU-starved crawl
        # phases the link power-states down and re-trains every few
        # seconds -- ~30 verbatim ALARM/RECOVERED pairs buried the log's
        # real information. Policy:
        #   - the FIRST down-train and FIRST recovery print verbatim (the
        #     full educational text must appear once per run);
        #   - later down-trains are HELD for PCIE_FLAP_ESCALATE_S: if the
        #     link recovers inside the window the pair is counted silently
        #     and a one-line flap summary prints at most every
        #     PCIE_FLAP_SUMMARY_S;
        #   - a held down-train that does NOT recover inside the window
        #     escalates with the full alarm text plus a "has not
        #     recovered" note. The non-recovering down-train (the 5060
        #     'GPU has been lost' signature) is exactly the event this
        #     governor must never suppress.
        flap_first_alarm = False    # first ALARM printed verbatim?
        flap_first_rec = False      # first RECOVERED printed verbatim?
        flap_pending = None         # (held_since, alarm_text) or None
        flap_escalated = False      # an escalated alarm awaits its all-clear
        flap_count = 0
        self.flap_count = 0
        flap_last_summary = 0.0
        idle_note_printed = False   # T23e: the idle-link note, once per run
        while not self._stop.wait(self.POLL_SECONDS):
            now = time.monotonic()
            polls += 1

            # CM-209: GPU sampler -- one NVML reading per poll; prints only
            # when the paging state changes (or every 5 min while it lasts).
            # T23e: sampled BEFORE the canary so a down-train can be judged
            # against what the GPU was doing in the same poll.
            gpu_idle = False
            if self.gpu is not None and self.gpu.active:
                try:
                    _gl = self.gpu.sample()
                    if _gl:
                        print(_gl)
                    _last = self.gpu.samples[-1] if self.gpu.samples else None
                    if _last is not None:
                        gpu_idle = (0 <= _last["gpu"] < 15
                                    and _last["enc"] <= 5 and _last["dec"] <= 5)
                except Exception:
                    pass

            # PCIe canary: it fires BEFORE progress visibly stalls.
            if self._canary is not None:
                alarm = None
                try:
                    alarm = self._canary.check()
                except Exception:
                    pass
                if alarm and gpu_idle and "NVML query failed" not in alarm:
                    # T23e (CM-241): the GPU was IDLE when the link dropped
                    # -- the link power-managed because nothing was being
                    # sent (field 09-28, batch 1: an 8-minute decoder wait on
                    # a sleeping USB drive read as a PCIe alarm). Not the
                    # pre-failure signature, which arrives under load. Held
                    # like a flap: if it does not recover in 30 s it still
                    # escalates in full.
                    if not idle_note_printed:
                        idle_note_printed = True
                        print(f"[Watchdog] PCIe link at Gen{self._canary.last_gen} "
                              f"while the GPU is idle: the pipeline is waiting on something "
                              f"upstream of the GPU (source read, decoder, audio sidecar) and "
                              f"the link power-managed. Not an alarm; a down-train that does "
                              f"NOT recover within {int(self.PCIE_FLAP_ESCALATE_S)}s still "
                              f"alarms in full, and a stall prints its stack at "
                              f"{self._stall_seconds:.0f}s (CM-241).")
                    flap_pending = (now, alarm)
                    flap_count += 1
                    self.flap_count = flap_count
                elif alarm:
                    if not flap_first_alarm:
                        flap_first_alarm = True
                        print(f"[Watchdog] ALARM: {alarm}")
                    else:
                        flap_pending = (now, alarm)
                        flap_count += 1
                        self.flap_count = flap_count   # CM-180: read by the run report
                # v1.50.00: close the loop on transient down-trains -- a
                # recovered link gets an explicit all-clear so a completed
                # run never ends on an unresolved alarm.
                try:
                    _rec = self._canary.recovery_note()
                except Exception:
                    _rec = None
                if _rec:
                    if flap_pending is not None:
                        # Recovered inside the hold window: a flap. Count
                        # it; summarize on a cadence instead of printing
                        # the pair.
                        flap_pending = None
                        if (now - flap_last_summary) >= self.PCIE_FLAP_SUMMARY_S:
                            flap_last_summary = now
                            print(f"[Watchdog] PCIe link flapping: "
                                  f"{flap_count} down-train/recover "
                                  f"cycle(s) so far this run -- transient "
                                  f"power-state churn while the GPU is "
                                  f"starved, not the pre-failure "
                                  f"signature. Individual alarms are "
                                  f"suppressed; a down-train that does "
                                  f"NOT recover within "
                                  f"{int(self.PCIE_FLAP_ESCALATE_S)}s "
                                  f"still alarms in full.")
                    if (not flap_first_rec) or flap_escalated:
                        flap_first_rec = True
                        flap_escalated = False
                        print(f"[Watchdog] {_rec}")
                # Escalation: a held down-train that never recovered.
                if (flap_pending is not None
                        and (now - flap_pending[0]) >= self.PCIE_FLAP_ESCALATE_S):
                    print(f"[Watchdog] ALARM: {flap_pending[1]} NOTE: this "
                          f"down-train has NOT recovered for "
                          f"{int(self.PCIE_FLAP_ESCALATE_S)}s -- unlike "
                          f"the earlier transient flaps, this matches the "
                          f"pre-failure signature.")
                    flap_pending = None
                    flap_escalated = True
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


__all__ = ["StallWatchdog", "GpuSampler"]
