# chitramaya/video/finalize.py
"""CM-187 -- one-pass finalize helpers (v1.71).

Why this module exists (field, 2026-09-11): finishing a run made NINE
whole-file passes on the NVIDIA edition (source decode, raw write, raw
read, temp-container write, temp read, source re-read for the audio,
output write, then ffmpeg's ``+faststart`` read+rewrite of the finished
output) and seven on AMD/Intel -- ~150 GB of disk traffic to finish a
31 GB 4K output, which is why the disk-progress watchdog (CM-124) kept
killing 4K remuxes and every one of them was hand-fixed. This module cuts
the finalize to ONE pass over the output:

  * ``AudioSidecar`` -- the source's audio is extracted ONCE, at run start,
    with stream copy into ``<stem>.audio.mka`` (a few minutes at most,
    while the GPU is busy anyway). The finalize and the RECOVER script then
    never touch the source again -- which also removes the dependency on
    the source drive being readable at the end of a 5-hour run.
  * ``moov_size`` -- the mp4 muxer reserves room for the index at the FRONT
    of the file while writing, so the file is "faststart" in one pass
    instead of a second read+rewrite of the whole output. If the estimate
    is ever too small ffmpeg says so at the trailer and the caller retries
    once without the reservation (index at the end; plays everywhere).
  * A/V start offset -- measured empirically (ffmpeg 6.1 and 2026 master):
    for the P-only streams NVENC writes (bf=0, the shipped default) the
    offset is restored by an EDIT LIST on the audio: ``-itsoffset -X`` on
    the sidecar with ``-avoid_negative_ts disabled`` makes the muxer write
    a sample-accurate skip (verified: skip 4800 samples for X = 0.100 s at
    48 kHz). For container inputs (the AMD/Intel ``.venc.mp4``, or a raw
    stream that carries B-frames) the shipped shape is kept -- ``-itsoffset
    X`` on the VIDEO input -- because that is the form that measured
    correct for those inputs; only the source is replaced by the sidecar.

Every command built here is plain stream copy; no pixels are touched.
"""
from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Optional

from chitramaya.winproc import NOWINDOW

# Windows: run the background extraction below the encode's own priority
# so it never competes with the run for CPU; harmless elsewhere.
_LOWPRIO: dict = dict(NOWINDOW)
if os.name == "nt":
    try:
        _LOWPRIO["creationflags"] = (
            _LOWPRIO.get("creationflags", 0) | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    except AttributeError:
        pass


def sidecar_path_for(stem: str) -> str:
    """``<stem>.audio.mka`` -- Matroska holds any audio codec unchanged."""
    return f"{stem}.audio.mka"


def source_has_audio(ffprobe: str, src: str) -> Optional[bool]:
    """True/False, or None when ffprobe could not tell (treat as True)."""
    try:
        r = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", str(src)],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=60, **NOWINDOW)
        if r.returncode != 0:
            return None
        return bool((r.stdout or "").strip())
    except Exception:
        return None


class AudioSidecar:
    """Extract the source audio once, in the background, at run start.

    ``start()`` returns immediately; ``wait()`` blocks until the extraction
    ends (it is normally long done before the finalize). ``ready`` is True
    only when ffmpeg exited 0 and the file exists with bytes in it;
    ``none`` is True when the source has no audio at all (nothing to mux,
    the finalize skips the audio input). Anything else (extraction failed,
    was never started) -> the caller falls back to the source file exactly
    as before this change.
    """

    def __init__(self, ffmpeg: str, ffprobe: str, src: str, out_path: str):
        self.ffmpeg = str(ffmpeg)
        self.ffprobe = str(ffprobe)
        self.src = str(src)
        self.path = str(out_path)
        self.none = False
        self.rc: Optional[int] = None
        self.error = ""
        self.seconds = 0.0
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._cancelled = False
        self.method = ""

    # -- lifecycle --------------------------------------------------------
    def start(self) -> "AudioSidecar":
        if self._started:
            return self
        self._started = True
        has = source_has_audio(self.ffprobe, self.src)
        if has is False:
            self.none = True
            print("[Encoder] Audio sidecar: source has no audio stream; "
                  "the output will be video-only.")
            return self
        base = [self.ffmpeg, "-hide_banner", "-y", "-loglevel", "error",
                "-nostdin", "-i", self.src, "-vn", "-sn", "-dn", "-map", "0:a"]
        # Attempt ladder (field 2026-09-12: a TS capture's audio could not
        # be stream-copied into ANY container -- the shipped source-based
        # remux failed on the same packets): 1) stream copy; 2) re-encode
        # to AAC-LC 192 kb/s so the run still ends with sound. Audio
        # re-encoding is a small CPU job that runs in the background while
        # the GPU works.
        attempts = [
            ("stream copy", base + ["-c", "copy", "-f", "matroska", self.path]),
            ("re-encode to AAC 192 kb/s", base + ["-c:a", "aac", "-b:a", "192k",
                                              "-f", "matroska", self.path]),
        ]
        print(f"[Encoder] Audio sidecar: extracting the source audio once "
              f"(stream copy, background) -> {Path(self.path).name}")
        t0 = time.monotonic()

        def _reap() -> None:
            for i, (label, cmd) in enumerate(attempts):
                if self._cancelled:
                    return
                try:
                    self._proc = subprocess.Popen(
                        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                        **_LOWPRIO)
                    _, err = self._proc.communicate()
                    rc = self._proc.returncode
                    err_s = (err or b"").decode("utf-8", "replace").strip()
                except Exception as e:  # pragma: no cover
                    rc, err_s = -1, f"{type(e).__name__}: {e}"
                self.rc, self.error = rc, err_s
                self.seconds = time.monotonic() - t0
                if rc == 0 and self.ready:
                    sz = os.path.getsize(self.path) / 1e6
                    self.method = label
                    print(f"[Encoder] Audio sidecar ready: {sz:.1f} MB in "
                          f"{self.seconds:.0f}s ({label})")
                    return
                try:
                    os.unlink(self.path)
                except OSError:
                    pass
                tail = err_s.splitlines()[-1] if err_s else ""
                if i + 1 < len(attempts) and not self._cancelled:
                    print(f"[Encoder] Audio sidecar: {label} failed (rc={rc}) "
                          f"{tail} -- trying {attempts[i + 1][0]}.")
            if not self._cancelled:
                print(f"[Encoder] Audio sidecar FAILED (rc={self.rc}) "
                      f"-- the finalize will read the source instead.")

        self._thread = threading.Thread(target=_reap, name="cm-audio-sidecar",
                                        daemon=True)
        self._thread.start()
        return self

    def wait(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def cancel(self) -> None:
        """Kill a still-running extraction (run aborted early)."""
        self._cancelled = True
        p = self._proc
        if p is not None and p.poll() is None:
            try:
                p.kill()
            except Exception:
                pass
        self.wait(5)

    @property
    def ready(self) -> bool:
        if self.none or self.rc != 0:
            return False
        try:
            return os.path.getsize(self.path) > 0
        except OSError:
            return False

    @property
    def decided(self) -> bool:
        """True once we know whether the sidecar can be used (done or none)."""
        return self.none or self.rc is not None

    def cleanup(self) -> None:
        try:
            if os.path.isfile(self.path):
                os.unlink(self.path)
        except OSError:
            pass


# -- moov reservation ----------------------------------------------------
def estimate_moov_bytes(n_video_frames: int, duration_s: float,
                        audio_packets_per_s: float = 50.0) -> int:
    """Bytes to reserve at the front of the mp4 for the index.

    Measured: ~12 bytes per sample entry (stts/stsz/stco/ctts/stss across a
    601-frame + 939-packet test file = 18.7 KB). Budget 48 bytes per entry
    (64-bit chunk offsets on files > 4 GB, keyframe table, ctts on
    B-frame streams) plus 1 MB of headroom, floored at 2 MB. A 2h41m
    1080p60 title (578,704 frames, ~480k audio packets) reserves ~52 MB --
    nothing next to the 10-30 GB it fronts.
    """
    entries = max(0, int(n_video_frames)) + int(max(0.0, duration_s) * audio_packets_per_s)
    return max(2 * 1024 * 1024, 48 * entries + 1024 * 1024)


_MOOV_TOO_SMALL = re.compile(r"reserved_moov_size is too small", re.I)


def moov_too_small(stderr_text: str) -> bool:
    return bool(stderr_text and _MOOV_TOO_SMALL.search(stderr_text))


_AUDIO_MUX_FAIL = re.compile(
    r"aost#|codec not currently supported in container|"
    r"Could not find tag for codec (aac_latm|mp2|pcm_\w+|vorbis|\w+) in stream|"
    r"Error submitting a packet to the muxer", re.I)


def audio_mux_failed(stderr_text: str) -> bool:
    """True when ffmpeg's complaint points at the AUDIO stream copy (LATM AAC
    from a TS capture, damaged ADTS packets, a codec mp4 cannot hold...).
    Field 2026-09-12: 'Error submitting a packet to the muxer: Invalid data
    found when processing input' on aost#0:1 -- the video was fine."""
    return bool(stderr_text and _AUDIO_MUX_FAIL.search(stderr_text))


def with_audio_reencode(cmd: List[str], bitrate: str = "192k") -> List[str]:
    """Same finalize command with the audio re-encoded to AAC-LC instead of
    stream-copied (video stays copy). A small CPU job over the audio only."""
    out = list(cmd)
    for i in range(len(out) - 1):
        if out[i] == "-c:a" and out[i + 1] == "copy":
            out[i + 1] = "aac"
            out[i + 2:i + 2] = ["-b:a", bitrate]
            break
    return out


# -- command builders -----------------------------------------------------
def build_onepass_raw_cmd(
    ffmpeg: str, *, fps_str: str, input_fmt_args: List[str], raw_path: str,
    sidecar: Optional[str], audio_itsoffset: float,
    color_args: List[str], tag_args: List[str], timescale_args: List[str],
    extra_args: List[str], moov_bytes: int, out_path: str,
    duration_s: float = 0.0,
) -> List[str]:
    """Raw elementary stream (NVENC output) + audio sidecar -> mp4, one pass.

    ``duration_s`` (video frames / fps) clamps the output: the sidecar holds
    the WHOLE source audio, so a PARTIAL result (run errored mid-title)
    would otherwise carry an hour of sound over no picture (field
    2026-09-12: PotPlayer froze at 42 min and played audio to the end).

    ``audio_itsoffset`` = audio_delay - video_delay (seconds): negative
    trims the audio start by an edit list (video started later than the
    audio in the source), positive delays the audio. Zero = no option.
    """
    cmd = [ffmpeg, "-hide_banner", "-y", "-loglevel", "warning", "-nostdin",
           "-fflags", "+genpts", "-analyzeduration", "10M", "-probesize", "50M",
           "-r", fps_str, *input_fmt_args, "-i", raw_path]
    if sidecar:
        if abs(audio_itsoffset) > 1e-4:
            cmd += ["-itsoffset", f"{audio_itsoffset:.6f}"]
        cmd += ["-i", sidecar]
    cmd += ["-map", "0:v:0", "-c:v", "copy"] + list(color_args) + list(tag_args)
    if sidecar:
        cmd += ["-map", "1:a?", "-c:a", "copy"]
    # Negative audio start = the edit list that restores the source's A/V
    # start offset; the default (make_non_negative) would silently shift
    # it away. Only for that direction -- with a positive audio delay the
    # default handling is what measures correct on current ffmpeg.
    if sidecar and audio_itsoffset < -1e-4:
        cmd += ["-avoid_negative_ts", "disabled"]
    cmd += list(timescale_args)
    if moov_bytes > 0:
        cmd += ["-moov_size", str(int(moov_bytes))]
    if duration_s > 0:
        cmd += ["-t", f"{duration_s:.3f}"]
    cmd += list(extra_args) + [out_path]
    return cmd


def build_onepass_container_cmd(
    ffmpeg: str, *, video_path: str, video_itsoffset: float,
    sidecar: Optional[str], audio_itsoffset: float,
    tag_args: List[str], timescale_args: List[str], extra_args: List[str],
    duration_s: float, moov_bytes: int, out_path: str,
) -> List[str]:
    """Container video (fragmented .venc.mp4, or a wrapped temp) + audio
    sidecar -> mp4, one pass. Same offset shape the shipped code used
    (``-itsoffset`` on the video input; verified correct for container
    inputs), only the source is replaced by the sidecar."""
    cmd = [ffmpeg, "-hide_banner", "-y", "-loglevel", "warning", "-nostdin"]
    if sidecar:
        if audio_itsoffset > 1e-4:
            cmd += ["-itsoffset", f"{audio_itsoffset:.6f}"]
        cmd += ["-i", sidecar]
        if video_itsoffset > 1e-4:
            cmd += ["-itsoffset", f"{video_itsoffset:.6f}"]
        cmd += ["-i", video_path, "-map", "1:v:0", "-c:v", "copy"]
        cmd += list(tag_args) + ["-map", "0:a?", "-c:a", "copy"]
    else:
        if video_itsoffset > 1e-4:
            cmd += ["-itsoffset", f"{video_itsoffset:.6f}"]
        cmd += ["-i", video_path, "-map", "0:v:0", "-c:v", "copy"] + list(tag_args)
    cmd += list(timescale_args)
    if moov_bytes > 0:
        cmd += ["-moov_size", str(int(moov_bytes))]
    if duration_s > 0:
        cmd += ["-t", f"{duration_s:.3f}"]
    cmd += list(extra_args) + [out_path]
    return cmd


# -- post-finalize check ----------------------------------------------------
def probe_audio_desc(ffprobe: str, path: str) -> str:
    """'aac 48000 Hz stereo' for the finalize log line ('' on failure)."""
    try:
        r = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_name,profile,sample_rate,channels",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=60, **NOWINDOW)
        parts = [t for t in (r.stdout or "").strip().splitlines()[0].split(",") if t] \
            if (r.stdout or "").strip() else []
        return " ".join(parts[:2] + [f"{parts[2]} Hz" if len(parts) > 2 else "",
                                     f"{parts[3]} ch" if len(parts) > 3 else ""]).strip()
    except Exception:
        return ""


def probe_av_start(ffprobe: str, path: str) -> tuple[Optional[float], Optional[float], Optional[int]]:
    """(first video frame pts s, first audio packet pts s, audio skip samples)
    of a finished file -- what a player will present. Two tiny reads."""
    v = a = None
    skip = None
    try:
        r = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "frame=pts_time", "-of", "csv=p=0",
             "-read_intervals", "%+#4", str(path)],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=60, **NOWINDOW)
        vals = []
        for ln in (r.stdout or "").splitlines():
            tok = ln.split(",")[0].strip()
            try:
                vals.append(float(tok))
            except ValueError:
                pass
        if vals:
            v = min(vals)
    except Exception:
        pass
    try:
        r = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "a:0",
             "-show_entries",
             "packet=pts_time:packet_side_data=side_data_type,skip_samples",
             "-of", "csv=p=0", "-read_intervals", "%+#1", str(path)],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=60, **NOWINDOW)
        first = (r.stdout or "").strip().splitlines()
        if first:
            parts = first[0].split(",")
            try:
                a = float(parts[0])
            except ValueError:
                a = None
            m = re.search(r"Skip Samples,(\d+)", first[0])
            if m is None:
                m = re.search(r"skip_samples=(\d+)", first[0])
            if m:
                skip = int(m.group(1))
    except Exception:
        pass
    return v, a, skip


def first_presented_time(ffmpeg: str, path: str, kind: str = "v") -> Optional[float]:
    """Presentation time of the first decoded video frame ('v') or audio
    frame ('a') as a player sees it (edit lists applied). One frame is
    decoded; used for verification and the finalize log line."""
    if kind == "v":
        cmd = [ffmpeg, "-v", "info", "-nostdin", "-i", str(path), "-map", "0:v:0",
               "-vf", "showinfo", "-frames:v", "1", "-f", "null", "-"]
    else:
        cmd = [ffmpeg, "-v", "info", "-nostdin", "-i", str(path), "-map", "0:a:0",
               "-af", "ashowinfo", "-frames:a", "1", "-f", "null", "-"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=120, **NOWINDOW)
        m = re.search(r"pts_time:\s*(-?[0-9.]+)", r.stderr or "")
        return float(m.group(1)) if m else None
    except Exception:
        return None
