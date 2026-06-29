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
        bf: int = -1,                # -1 = don't set; 0 explicitly disables B-frames
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

    def _remux(self) -> bool:
        """Remux raw bitstream with audio from input using ffmpeg. Returns True on success."""
        raw_path = Path(self._raw_path)
        out_path = Path(self.output_path)

        if not raw_path.exists():
            print(f"[Encoder] Remux skipped: {raw_path} not found")
            return False

        input_fmt = _ffmpeg_input_fmt(self.codec)

        # Build ffmpeg command
        cmd = [self.ffmpeg_path, "-hide_banner", "-y", "-loglevel", "warning"]

        # Input: raw video bitstream
        # -fflags +genpts: generate PTS for demuxed packets
        # -analyzeduration/-probesize: give ffmpeg time to find VPS/SPS/PPS in NVENC output
        cmd += ["-fflags", "+genpts",
                "-analyzeduration", "10M",
                "-probesize", "50M",
                "-r", self.fps_str,
                "-f", input_fmt,
                "-i", str(raw_path)]

        # Input: original file for audio (if available)
        has_audio_source = False
        if self.mux_audio and self.input_path and Path(self.input_path).exists():
            cmd += ["-i", self.input_path]
            has_audio_source = True

        # Video: copy (already encoded)
        cmd += ["-map", "0:v:0", "-c:v", "copy"]

        # Video tag for MP4 compatibility
        if self._container == "mp4":
            if self.codec in ("hevc", "h265"):
                cmd += ["-tag:v", "hvc1"]
            elif self.codec in ("h264", "avc"):
                cmd += ["-tag:v", "avc1"]

        # Audio: copy from source if available
        if has_audio_source:
            cmd += ["-map", "1:a?", "-c:a", "copy"]

        # Container options
        if self._container == "mp4":
            cmd += ["-movflags", "+faststart"]
            # Force the standard 90 kHz video timescale. NVENC's raw stream
            # carries a non-standard timebase (observed 1/15360); with -c:v copy
            # ffmpeg preserves it into the MP4 instead of normalizing. Some
            # players (notably VR/headset players) mis-reconcile a non-90 kHz
            # video clock against the 44.1 kHz audio clock, producing
            # progressive A/V drift over long files even when per-frame PTS is
            # evenly spaced. 90000 is evenly divisible for common rates
            # (90000/60 = 1500) and is what virtually all MP4s use, so players
            # are optimized for it. This matches gRestorer's remux.
            cmd += ["-video_track_timescale", "90000"]

        # Set output duration to match our video (avoid -shortest which
        # fails with raw bitstreams that lack proper timestamps)
        if self._frames_encoded > 0 and self.fps > 0:
            duration = self._frames_encoded / self.fps
            cmd += ["-t", f"{duration:.3f}"]

        cmd += [str(out_path)]

        print(f"[Encoder] Remuxing: {' '.join(cmd)}")
        try:
            # Force UTF-8 decoding of ffmpeg output. Without encoding=, Python
            # falls back to the system codepage (cp1252 on Windows), which
            # crashes when the input video filename contains non-ASCII
            # characters (e.g. CJK). errors='replace' is defensive against
            # any genuinely malformed bytes in ffmpeg's progress lines.
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=300,
            )
            if result.returncode != 0:
                print(f"[Encoder] Remux failed (rc={result.returncode})")
                if result.stderr:
                    lines = result.stderr.strip().split("\n")
                    for line in lines[-5:]:
                        print(f"  {line}")
                return False
            else:
                print(f"[Encoder] Remux OK: {out_path}")
                return True
        except Exception as e:
            print(f"[Encoder] Remux error: {e}")
            return False

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
