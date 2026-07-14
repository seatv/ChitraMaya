"""
ChitraMaya video encoder — NVENC via PyNvVideoCodec.

Simplified from gRestorer's encoder with focus on:
  - Clean timestamp handling (no VFR complexity)
  - Simple API: encode_frame(tensor) → bitstream
  - ffmpeg remux with audio at close()

Input format: BGRA HWC4 uint8 CUDA tensor (from rgbp_to_packed).
Output: H.264/HEVC elementary stream → ffmpeg remux to MP4/MKV with audio.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import PyNvVideoCodec as nvc


def _raw_ext(codec: str) -> str:
    return ".hevc" if codec in ("hevc", "h265") else ".h264"


def _create_nvenc_with_fallback(
    width: int,
    height: int,
    fmt: str,
    full_opts: Dict[str, str],
    base_opts: Dict[str, str],
):
    """
    Try to create an NVENC encoder with the full option set. If the SDK
    rejects an unknown key (different PyNvVideoCodec / NVENC SDK versions
    use slightly different key names), drop the offending key and retry.

    Returns (encoder, opts_actually_used). opts_actually_used contains only
    keys that NVENC accepted, so the caller can print exactly what's active.

    base_opts is the floor we won't drop below — if those fail, propagate
    the error.
    """
    base_keys = set(base_opts.keys())
    opts = dict(full_opts)
    last_err: Optional[Exception] = None

    # First try the full set; on failure, drop unrecognized keys one at a
    # time (skipping base_keys). Up to ~10 attempts is plenty since the
    # quality_opts set is small.
    for _ in range(12):
        try:
            return nvc.CreateEncoder(width, height, fmt, False, **opts), opts
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            # Identify a probably-unknown key from the error message and drop it.
            # PyNvVideoCodec errors typically mention the offending key name.
            dropped = None
            for k in list(opts.keys()):
                if k in base_keys:
                    continue
                if k.lower() in msg:
                    dropped = k
                    break
            if dropped is None:
                # Couldn't identify — drop ANY non-base key to make progress.
                for k in list(opts.keys()):
                    if k not in base_keys:
                        dropped = k
                        break
            if dropped is None:
                # Nothing left to drop and we're still failing → it's the base set.
                break
            opts.pop(dropped, None)

    # Final fallback: minimal base set.
    try:
        minimal = dict(base_opts)
        return nvc.CreateEncoder(width, height, fmt, False, **minimal), minimal
    except Exception as e:
        # Re-raise the last meaningful error.
        raise (last_err or e)


def _ffmpeg_input_fmt(codec: str) -> str:
    if codec in ("hevc", "h265"):
        return "hevc"
    if codec in ("h264", "avc"):
        return "h264"
    raise ValueError(f"Unsupported codec: {codec}")


def _fps_to_rational(fps: float) -> str:
    """Convert fps to rational string for ffmpeg."""
    frac = Fraction(fps).limit_denominator(10001)
    if frac.denominator == 1:
        return str(frac.numerator)
    return f"{frac.numerator}/{frac.denominator}"


def _derive_ffprobe(ffmpeg_path: str) -> str:
    """Best-effort ffprobe path derived from the configured ffmpeg path.

    "ffmpeg" -> "ffprobe"; "C:/x/ffmpeg.exe" -> "C:/x/ffprobe.exe".
    """
    p = str(ffmpeg_path)
    base = os.path.basename(p)
    if "ffmpeg" in base.lower():
        d = os.path.dirname(p)
        probe = base.lower().replace("ffmpeg", "ffprobe")
        return os.path.join(d, probe) if d else probe
    return "ffprobe"


def _probe_stream_start_seconds(ffprobe: str, path: str, stream: str) -> Optional[float]:
    """Return a stream's start_time in seconds via ffprobe.

    Returns None if the stream is absent, reports 'N/A', or the probe fails.
    """
    try:
        cmd = [
            ffprobe, "-v", "error", "-select_streams", stream,
            "-show_entries", "stream=start_time",
            "-of", "default=nw=1:nk=1", path,
        ]
        out = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=30,
        )
        s = (out.stdout or "").strip()
        return float(s) if s else None
    except Exception:
        return None


class Encoder:
    """NVENC video encoder with ffmpeg remux.

    Usage::

        enc = Encoder("output.mp4", width=1920, height=1080, fps=30.0,
                       input_path="input.mp4")  # for audio source
        for frame in frames:
            enc.encode_frame(bgra_hwc4_tensor)
        enc.close()  # flush + remux with audio
    """

    def __init__(
        self,
        output_path: str | Path,
        width: int,
        height: int,
        fps: float,
        codec: str = "hevc",
        preset: str = "P7",
        qp: int = 15,
        gpu_id: int = 0,
        input_path: str | Path | None = None,
        mux_audio: bool = True,
        mp4_faststart: bool = True,
        mux_extra_args: str = "",
        ffmpeg_path: str = "ffmpeg",
        # NVENC advanced options. Defaults below are "do not include" sentinels:
        # we pass NOTHING beyond codec/preset/QP and let the card pick the rest.
        # Empirically every additional knob (tune, lookahead, multipass, B-frames,
        # spatial AQ) adds NVENC workload, and on some cards full-load 4K
        # restoration saturates the hardware encoder enough to throw error 8
        # (nvEncLockBitstream INVALID_PARAM). A future free-form override
        # (`--enc-options`) will let users opt back in.
        tune: str = "",
        spatial_aq: bool = False,
        aq_strength: int = 0,
        bf: int = 0,                 # 0 = no B-frames (clean PTS=DTS so the
                                     #     start offset survives remux); -1 =
                                     #     don't set (NVENC default, may reorder)
        bref: str = "",
        rc_lookahead: int = 0,
        multipass: str = "disabled",
        temporal_aq: bool = False,
    ) -> None:
        self.output_path = str(output_path)
        self.input_path = str(input_path) if input_path else None
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.fps_str = _fps_to_rational(fps)
        self.codec = str(codec).lower()
        self.preset = preset
        self.qp = int(qp)
        self.gpu_id = int(gpu_id)
        self.mux_audio = bool(mux_audio)
        self.mp4_faststart = bool(mp4_faststart)
        self.mux_extra_args = str(mux_extra_args or "")
        self.ffmpeg_path = str(ffmpeg_path)
        self.tune = str(tune)
        self.spatial_aq = bool(spatial_aq)
        self.aq_strength = int(aq_strength)
        self.bf = int(bf)
        self.bref = str(bref)
        self.rc_lookahead = int(rc_lookahead)
        self.multipass = str(multipass)
        self.temporal_aq = bool(temporal_aq)

        # Determine if we need remux
        suffix = Path(self.output_path).suffix.lower()
        self._container = "mp4" if suffix in (".mp4", ".m4v", ".mov") else (
            "mkv" if suffix == ".mkv" else None
        )
        self._needs_remux = self._container is not None
        self._raw_path = self.output_path
        if self._needs_remux:
            self._raw_path = self.output_path + _raw_ext(self.codec)

        self._file = open(self._raw_path, "wb")
        self._frames_encoded = 0
        self._closed = False
        self._remux_ok = False

        # NVENC options
        gop = max(1, int(round(self.fps * 2.0)))

        # Minimum-viable base options. Codec / preset / QP only —
        # everything else (bf, aq, lookahead, multipass, tune) defaults
        # to whatever NVENC + the card pick. Keeps the encoder workload
        # as light as possible to avoid hardware saturation.
        base_opts = {
            "codec": self.codec,
            "preset": self.preset,
            "fps": self.fps_str,
            "gop": str(gop),
            "idrperiod": str(gop),
            "rc": "constqp",
            "constqp": str(self.qp),
        }

        # Optional/advanced quality options — only populated if a caller
        # explicitly set the corresponding kwarg. Defaults to all sentinels
        # = empty dict (NVENC picks its own defaults). Future free-form
        # ffmpeg-style override CLI will populate this via the kwargs.
        quality_opts: dict[str, str] = {}
        if self.tune:
            quality_opts["tuninginfo"] = self.tune
        if self.spatial_aq:
            quality_opts["aq"] = "1"
            if self.aq_strength > 0:
                quality_opts["aqstrength"] = str(self.aq_strength)
        if self.temporal_aq:
            quality_opts["temporalaq"] = "1"
        if self.bf >= 0:
            quality_opts["bf"] = str(self.bf)
        if self.bf > 0 and self.bref:
            quality_opts["bframerefmode"] = self.bref
        if self.rc_lookahead > 0:
            quality_opts["lookahead"] = str(self.rc_lookahead)
        if self.multipass and self.multipass.lower() != "disabled":
            quality_opts["multipass"] = self.multipass

        enc_opts = {**base_opts, **quality_opts}
        self._encoder, used_opts = _create_nvenc_with_fallback(
            self.width, self.height, "ARGB", enc_opts, base_opts
        )
        self._active_opts = used_opts

        # Pretty-print active quality settings so the user can see what's on.
        active_quality = {k: used_opts[k] for k in quality_opts if k in used_opts}
        banner_base = (
            f"[Encoder] {self.width}×{self.height} @ {self.fps:.2f}fps, "
            f"{self.codec}/{self.preset}/QP{self.qp}"
        )
        if active_quality:
            quality_str = " ".join(f"{k}={v}" for k, v in active_quality.items())
            print(f"{banner_base} {quality_str}")
        else:
            print(banner_base)
        if quality_opts and len(active_quality) < len(quality_opts):
            dropped = sorted(set(quality_opts.keys()) - set(active_quality.keys()))
            print(
                f"[Encoder] NVENC rejected (your PyNvVideoCodec build doesn't expose these keys): {dropped}"
            )

    def encode_frame(self, frame: torch.Tensor) -> None:
        """Encode a single BGRA HWC4 uint8 CUDA tensor."""
        if self._closed:
            return

        frame = frame.contiguous()
        if frame.device.type == "cuda":
            torch.cuda.synchronize(device=frame.device)

        bitstream = self._encoder.Encode(frame)
        if bitstream:
            self._file.write(bytearray(bitstream))
        self._frames_encoded += 1

    def flush(self) -> None:
        """Flush remaining frames from encoder."""
        try:
            tail = self._encoder.EndEncode()
            if tail:
                self._file.write(bytearray(tail))
        except Exception as e:
            print(f"[Encoder] Flush error: {e}")
        print(f"[Encoder] Flushed ({self._frames_encoded} frames)")

    def close(self) -> None:
        """Flush, close file, remux with audio if needed."""
        if self._closed:
            return
        self._closed = True

        self.flush()
        self._file.close()

        if self._needs_remux:
            self._remux_ok = self._remux()

        # Cleanup raw bitstream only if remux succeeded
        if self._needs_remux and self._remux_ok:
            raw = Path(self._raw_path)
            if raw.exists():
                try:
                    raw.unlink()
                except Exception:
                    pass
        elif self._needs_remux and not self._remux_ok:
            print(f"[Encoder] Raw bitstream kept for debugging: {self._raw_path}")

        print(f"[Encoder] Done: {self.output_path}")

    def _run_ffmpeg(self, cmd: List[str], label: str = "ffmpeg") -> bool:
        """Run one ffmpeg command; return True on rc==0. Shared by both remux
        passes. UTF-8 decoding avoids cp1252 crashes on non-ASCII filenames."""
        print(f"[Encoder] {label}: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=900,
            )
            if result.returncode != 0:
                print(f"[Encoder] {label} failed (rc={result.returncode})")
                if result.stderr:
                    for line in result.stderr.strip().split("\n")[-5:]:
                        print(f"  {line}")
                return False
            print(f"[Encoder] {label} OK")
            return True
        except Exception as e:
            print(f"[Encoder] {label} error: {e}")
            return False

    def _remux(self) -> bool:
        """Remux raw bitstream with audio from input using ffmpeg. Returns True on success."""
        raw_path = Path(self._raw_path)
        out_path = Path(self.output_path)

        if not raw_path.exists():
            print(f"[Encoder] Remux skipped: {raw_path} not found")
            return False

        input_fmt = _ffmpeg_input_fmt(self.codec)

        # Reintroduce the source's start-PTS offset between audio and video.
        # NVDEC numbers frames from 0 and NVENC emits a raw elementary stream
        # with no start offset, so the remux regenerates video PTS from t=0.
        # But source containers usually carry a small offset between their video
        # and audio streams (e.g. video start_time=0.033, audio start_time=0.000
        # => audio leads video by 33ms). Collapsing that to 0/0 shifts A/V by a
        # constant amount from the very first frame. Probe the source and delay
        # whichever stream started later, matching lada's -itsoffset behaviour.
        video_delay = 0.0
        audio_delay = 0.0
        if self.mux_audio and self.input_path and Path(self.input_path).exists():
            ffprobe = _derive_ffprobe(self.ffmpeg_path)
            v_start = _probe_stream_start_seconds(ffprobe, self.input_path, "v:0")
            a_start = _probe_stream_start_seconds(ffprobe, self.input_path, "a:0")
            if v_start is not None and a_start is not None:
                rel = v_start - a_start
                if rel > 1e-4:
                    video_delay = rel        # audio led video in the source
                elif rel < -1e-4:
                    audio_delay = -rel       # video led audio (rare)
                if video_delay or audio_delay:
                    print(
                        f"[Encoder] Restoring source A/V start offset: "
                        f"v_start={v_start:.6f}s a_start={a_start:.6f}s "
                        f"-> video_delay={video_delay:.6f}s audio_delay={audio_delay:.6f}s"
                    )

        has_audio_source = bool(
            self.mux_audio and self.input_path and Path(self.input_path).exists()
        )

        # Reusable arg groups.
        tag_args: List[str] = []
        if self._container == "mp4":
            if self.codec in ("hevc", "h265"):
                tag_args = ["-tag:v", "hvc1"]
            elif self.codec in ("h264", "avc"):
                tag_args = ["-tag:v", "avc1"]

        timescale_args = (
            ["-video_track_timescale", "90000"] if self._container == "mp4" else []
        )
        # faststart is mp4-only AND now honors the mp4_faststart flag (was
        # previously forced on with no way to disable it).
        faststart_args = (
            ["-movflags", "+faststart"]
            if (self._container == "mp4" and self.mp4_faststart) else []
        )
        # User-supplied extra remux args (must not include -i), appended just
        # before the output path in whichever final remux command runs.
        extra_args: List[str] = []
        if self.mux_extra_args.strip():
            extra_args = shlex.split(self.mux_extra_args)
            for _t in extra_args:
                if _t == "-i" or _t.startswith("-i"):
                    raise ValueError("mux_extra_args must not include -i")

        # Cap output near the (possibly delayed) video length so trailing audio
        # doesn't extend the file, without clipping the delayed video's tail.
        # (Avoids -shortest, which fails with raw bitstreams lacking timestamps.)
        dur_args: List[str] = []
        if self._frames_encoded > 0 and self.fps > 0:
            duration = video_delay + (self._frames_encoded / self.fps)
            dur_args = ["-t", f"{duration:.3f}"]

        ff = self.ffmpeg_path
        temp_video: Optional[Path] = None
        try:
            if video_delay > 0:
                # -itsoffset is silently DROPPED on a raw annexb input when
                # ffmpeg CFR-stamps it from frame 0 (confirmed: output video
                # start_time stayed 0.000). So first bounce the raw stream into
                # a temp CONTAINER (lossless copy) to give it real timestamps,
                # then apply -itsoffset on that container in the audio-mux pass.
                # -itsoffset on a container input is reliable (lada's pattern).
                temp_video = Path(str(out_path) + ".vtmp" + out_path.suffix)

                step1 = [ff, "-hide_banner", "-y", "-loglevel", "warning",
                         "-fflags", "+genpts",
                         "-analyzeduration", "10M", "-probesize", "50M",
                         "-r", self.fps_str,
                         "-f", input_fmt, "-i", str(raw_path),
                         "-map", "0:v:0", "-c:v", "copy"]
                step1 += tag_args + timescale_args + [str(temp_video)]
                if not self._run_ffmpeg(step1, "video-container"):
                    return False

                # Mux with lada's input ordering: the un-delayed AUDIO source is
                # input 0, the delayed restored VIDEO is input 1. ffmpeg baselines
                # output timestamps against input 0 (audio @ 0), so the video's
                # +offset survives as an edit list. With the delayed video as
                # input 0 instead, ffmpeg re-zeroed it and the offset was lost
                # (confirmed: output start_time stayed 0.000).
                if has_audio_source:
                    step2 = [ff, "-hide_banner", "-y", "-loglevel", "warning",
                             "-i", self.input_path,
                             "-itsoffset", f"{video_delay:.6f}", "-i", str(temp_video),
                             "-map", "1:v:0", "-c:v", "copy"] + tag_args
                    step2 += ["-map", "0:a?", "-c:a", "copy"]
                else:
                    step2 = [ff, "-hide_banner", "-y", "-loglevel", "warning",
                             "-itsoffset", f"{video_delay:.6f}", "-i", str(temp_video),
                             "-map", "0:v:0", "-c:v", "copy"] + tag_args
                step2 += faststart_args + timescale_args + dur_args + extra_args + [str(out_path)]
                return self._run_ffmpeg(step2, "remux")

            # No video delay: single pass raw -> final. audio_delay (rare: source
            # video led its audio) is applied on the audio CONTAINER input, which
            # is reliable.
            cmd = [ff, "-hide_banner", "-y", "-loglevel", "warning",
                   "-fflags", "+genpts",
                   "-analyzeduration", "10M", "-probesize", "50M",
                   "-r", self.fps_str,
                   "-f", input_fmt, "-i", str(raw_path)]
            if has_audio_source:
                if audio_delay > 0:
                    cmd += ["-itsoffset", f"{audio_delay:.6f}"]
                cmd += ["-i", self.input_path]
            cmd += ["-map", "0:v:0", "-c:v", "copy"] + tag_args
            if has_audio_source:
                cmd += ["-map", "1:a?", "-c:a", "copy"]
            cmd += faststart_args + timescale_args + dur_args + extra_args + [str(out_path)]
            return self._run_ffmpeg(cmd, "remux")
        finally:
            if temp_video is not None:
                try:
                    Path(temp_video).unlink(missing_ok=True)
                except Exception:
                    pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def rgbp_to_packed(
    rgbp_chw: torch.Tensor,
    out_hwc4: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert planar RGB CHW uint8 → packed BGRA HWC4 uint8 for NVENC.

    This is the format conversion between swap_core output and encoder input.
    Matches gRestorer's copy_stream_gpu.rgbp_to_packed() with pack="argb".
    """
    if rgbp_chw.ndim != 3 or rgbp_chw.shape[0] != 3:
        raise ValueError(f"Expected [3,H,W], got {tuple(rgbp_chw.shape)}")

    h, w = rgbp_chw.shape[1], rgbp_chw.shape[2]

    if out_hwc4 is None or out_hwc4.shape != (h, w, 4):
        out_hwc4 = torch.empty((h, w, 4), device=rgbp_chw.device, dtype=torch.uint8)

    # BGRA layout (ARGB word order on little-endian)
    out_hwc4[..., 0] = rgbp_chw[2]   # B
    out_hwc4[..., 1] = rgbp_chw[1]   # G
    out_hwc4[..., 2] = rgbp_chw[0]   # R
    out_hwc4[..., 3] = 255           # A

    return out_hwc4