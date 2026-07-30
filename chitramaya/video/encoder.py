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
import struct
import subprocess
from chitramaya.winproc import NOWINDOW
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import PyNvVideoCodec as nvc


def _raw_ext(codec: str) -> str:
    if codec in ("hevc", "h265"):
        return ".hevc"
    if codec == "av1":
        return ".av1"    # raw low-overhead OBU stream (ffmpeg 'av1' demuxer)
    return ".h264"


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


def _av1_obu_spans(buf: bytes):
    """Yield (obu_type, start, end) spans over one temporal unit's bytes.

    Best-effort low-overhead OBU walk (header + leb128 size); stops rather
    than guesses on a malformed header. Types: 1=SEQUENCE_HEADER,
    2=TEMPORAL_DELIMITER, 6=FRAME."""
    i, n = 0, len(buf)
    while i < n:
        start = i
        b = buf[i]
        t = (b >> 3) & 0xF
        has_ext = (b >> 2) & 1
        has_size = (b >> 1) & 1
        i += 1 + has_ext
        if not has_size:
            yield (t, start, n)
            return
        sz, shift = 0, 0
        while i < n:
            c = buf[i]
            i += 1
            sz |= (c & 0x7F) << shift
            if not (c & 0x80):
                break
            shift += 7
        end = i + sz
        yield (t, start, min(end, n))
        i = end


def _av1_fix_seq_headers(raw_path: Path) -> "tuple[int, int] | None":
    """CM-091: make every AV1 temporal unit self-contained, in place.

    NVENC emits the sequence-header OBU exactly once, at stream start, and
    PyNvVideoCodec 2.1 silently ignores the repeatseqhdr request -- so
    mid-stream keyframes are not valid seek targets and dav1d-based players
    (ffmpeg/VLC/mpv) fail every seek with 'Error parsing OBU data' (root-
    caused on a 542,865-frame field file; the container index was perfect,
    the bytes it pointed at were not). This rewrites the IVF: capture the
    seq-header OBU from the first temporal unit that has one, then insert
    it (after the temporal delimiter) into every unit that lacks one.
    Spec-legal, ~15 bytes/frame, pixels untouched; validated end-to-end on
    a synthetic header-stripped stream and on the field file itself.

    Rewrites via a sibling temp file + os.replace so a failure at any point
    leaves the original raw stream intact. Returns (total_units, inserted)
    or None on any failure -- the caller proceeds with the original file
    (playable, seek-impaired) rather than risking a finished encode."""
    tmp_path = Path(str(raw_path) + ".seekfix")
    fin = fout = None
    try:
        fin = open(raw_path, "rb", buffering=1024 * 1024)
        header = fin.read(32)
        if len(header) < 32 or header[:4] != b"DKIF":
            fin.close()
            return None
        fout = open(tmp_path, "wb", buffering=1024 * 1024)
        fout.write(header)
        seq_bytes = None
        units = 0
        inserted = 0
        while True:
            fh = fin.read(12)
            if len(fh) < 12:
                break
            size, ts = struct.unpack("<IQ", fh)
            data = fin.read(size)
            if len(data) < size:
                # Truncated tail (should not happen post-flush): keep as-is.
                fout.write(fh)
                fout.write(data)
                break
            spans = list(_av1_obu_spans(data))
            types = [t for (t, _, _) in spans]
            if seq_bytes is None and 1 in types:
                for (t, s, e) in spans:
                    if t == 1:
                        seq_bytes = data[s:e]
                        break
            if seq_bytes is not None and 1 not in types:
                ins = 0
                if spans and spans[0][0] == 2:   # after temporal delimiter
                    ins = spans[0][2]
                data = data[:ins] + seq_bytes + data[ins:]
                inserted += 1
            fout.write(struct.pack("<IQ", len(data), ts))
            fout.write(data)
            units += 1
        fin.close()
        fout.close()
        fin = fout = None
        if seq_bytes is None or units == 0:
            tmp_path.unlink(missing_ok=True)
            return None
        os.replace(str(tmp_path), str(raw_path))
        return units, inserted
    except Exception:
        try:
            if fin is not None:
                fin.close()
            if fout is not None:
                fout.close()
        except Exception:
            pass
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        return None


def _ffmpeg_input_fmt(codec: str) -> str:
    if codec in ("hevc", "h265"):
        return "hevc"
    if codec in ("h264", "avc"):
        return "h264"
    if codec == "av1":
        # NVENC emits a low-overhead OBU stream; ffmpeg's demuxer for that is
        # "obu" ("av1" is the Annex-B demuxer and REJECTS low-overhead
        # streams with "No sequence header available" — verified empirically).
        return "obu"
    raise ValueError(f"Unsupported codec: {codec}")


def _av1_encode_supported(gpu_id: int) -> bool | None:
    """NVENC AV1 encode exists on Ada (sm_89) and Blackwell (sm_120+) only —
    Ampere/Turing can DECODE AV1 but cannot encode it. Returns None if the
    capability can't be determined (then we let NVENC itself decide)."""
    try:
        major, minor = torch.cuda.get_device_capability(int(gpu_id))
        return (major, minor) >= (8, 9)
    except Exception:
        return None


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
            **NOWINDOW,
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

        # AV1 preflight: fail fast with a clear message on GPUs that cannot
        # encode AV1, instead of a cryptic NVENC creation error mid-run.
        if self.codec == "av1":
            sup = _av1_encode_supported(self.gpu_id)
            if sup is False:
                try:
                    cap = torch.cuda.get_device_capability(self.gpu_id)
                    cap_s = f"sm_{cap[0]}{cap[1]}"
                except Exception:
                    cap_s = "this GPU"
                raise RuntimeError(
                    f"AV1 encoding requires an RTX 40-series (Ada) or newer "
                    f"NVENC; {cap_s} cannot encode AV1 (it can only decode it). "
                    f"Choose HEVC or H.264 in the Encoder panel."
                )

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

        # Recovery sidecar (field event 7/22/2026): a 3h40m run under VRAM
        # paging was HARD-KILLED before close() ran, so the REMUX FAILED
        # banner never printed -- the raw .hevc survived only because cleanup
        # also never ran. To make recovery survive a hard kill, write a
        # PowerShell recovery script to disk NOW (next to the raw bitstream),
        # and delete it only on a successful remux. If the process dies at any
        # point, the user finds <output>.RECOVER.ps1 with the exact wrap
        # command already filled in.
        self._recovery_script_path: Optional[str] = None
        if self._needs_remux:
            try:
                self._write_recovery_script()
            except Exception:
                pass  # recovery aid must never break the encode

        # NVENC options
        gop = max(1, int(round(self.fps * 2.0)))

        # Minimum-viable base options. Codec / preset / QP only —
        # everything else (bf, aq, lookahead, multipass, tune) defaults
        # to whatever NVENC + the card pick. Keeps the encoder workload
        # as light as possible to avoid hardware saturation.
        # AV1's constant-QP scale is qindex 0-255, not the 0-51 QP scale H.264/
        # HEVC use. Map the UI's HEVC-style QP by the standard rough 4x
        # equivalence so the same slider value means a similar quality across
        # codecs (e.g. QP18 -> qindex 72). Logged so runs are auditable.
        effective_qp = self.qp
        if self.codec == "av1":
            effective_qp = min(255, self.qp * 4)
            print(f"[Encoder] AV1 rate control: QP{self.qp} (HEVC scale) -> "
                  f"qindex {effective_qp} (AV1 scale)")

        base_opts = {
            "codec": self.codec,
            "preset": self.preset,
            "fps": self.fps_str,
            "gop": str(gop),
            "idrperiod": str(gop),
            "rc": "constqp",
            "constqp": str(effective_qp),
        }

        # Optional/advanced quality options — only populated if a caller
        # explicitly set the corresponding kwarg. Defaults to all sentinels
        # = empty dict (NVENC picks its own defaults). Future free-form
        # ffmpeg-style override CLI will populate this via the kwargs.
        quality_opts: dict[str, str] = {}
        # CM-091 (Batch 30): AV1 seek fix. NVENC's AV1 default writes the
        # sequence-header OBU ONCE at stream start, so mid-stream keyframes
        # are not self-contained: linear playback works, but any seek lands
        # on a header-less keyframe and dav1d-family decoders (ffmpeg, VLC,
        # mpv) fail with "Error parsing OBU data" (field-confirmed on a
        # 2.5h 4K file: frame 0 = [TD, SEQ_HDR, FRAME], every later
        # keyframe = [TD, FRAME]). Request in-band repetition so every
        # keyframe carries its own sequence header, like SVT/libaom output.
        # Two candidate option names -- PyNvVideoCodec builds differ; the
        # create-with-fallback path drops whichever a build rejects, and we
        # warn below if neither survived.
        if self.codec == "av1":
            quality_opts["repeatseqhdr"] = "1"
            quality_opts["repeatspspps"] = "1"
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
        try:
            self._encoder, used_opts = _create_nvenc_with_fallback(
                self.width, self.height, "ARGB", enc_opts, base_opts
            )
        except Exception as e:
            if self.codec == "av1":
                # Capability probe passed (or was inconclusive) but NVENC still
                # refused — most likely an older driver / PyNvVideoCodec build
                # without AV1 encode support.
                raise RuntimeError(
                    "NVENC refused to create an AV1 encoder. AV1 encoding needs "
                    "an RTX 40-series (Ada) or newer GPU AND a recent NVIDIA "
                    "driver / PyNvVideoCodec build. Choose HEVC or H.264, or "
                    f"update the driver. (Underlying error: {e})"
                ) from e
            raise
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
        # CM-091: the repeatseqhdr/repeatspspps request above is belt &
        # braces only -- PyNvVideoCodec 2.1 ACCEPTS both keys but silently
        # ignores them for AV1 (field-verified 2026-07-29: banner showed
        # repeatseqhdr=1 yet the stream still carried ONE sequence header).
        # The authoritative fix runs at remux time: _av1_fix_seq_headers()
        # rewrites the raw IVF so every temporal unit carries the sequence
        # header, making every keyframe a true seek point. Its insertion
        # count is the ground truth a create-time banner cannot give.
        if self.codec == "av1":
            print("[Encoder] AV1 seek fix armed: sequence headers will be "
                  "verified and repeated in-band at the remux stage (CM-091).")

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

    def _recovery_command_str(self) -> str:
        """The exact PowerShell ffmpeg command that wraps the raw bitstream
        into a playable file (video only — the always-works fallback).
        Shared by the sidecar script and the REMUX FAILED banner so both
        show the SAME command."""
        fixed = str(Path(self.output_path).with_suffix("")) + "-FIXED" + \
            Path(self.output_path).suffix
        fmt = _ffmpeg_input_fmt(self.codec)
        # For AV1 the bitstream may be IVF-wrapped (auto-probes); a plain -i
        # works for that, while HEVC/H.264 want the explicit demuxer. Use the
        # demuxer for non-av1 and let av1 auto-probe.
        fmt_arg = "" if self.codec == "av1" else f'-f {fmt} '
        # Per-codec MP4 fourcc tag (hvc1/avc1/av01) — a wrong tag makes some
        # players reject the file.
        _tag = {"hevc": "hvc1", "h265": "hvc1", "h264": "avc1",
                "avc": "avc1", "av1": "av01"}.get(self.codec, "hvc1")
        return (
            f'ffmpeg -hide_banner -fflags +genpts -r {self.fps_str} '
            f'{fmt_arg}-i "{self._raw_path}" -c:v copy -tag:v {_tag} '
            f'-video_track_timescale 90000 -movflags +faststart "{fixed}"'
        )

    def _write_recovery_script(self) -> None:
        """Write a self-contained PowerShell recovery script next to the raw
        bitstream, so a hard kill (OOM/wedge) before close() still leaves the
        user a one-double-click fix. Deleted on successful remux."""
        script_path = str(Path(self.output_path).with_suffix("")) + \
            "-RECOVER.ps1"
        source_hint = ""
        if self.mux_audio and self.input_path:
            source_hint = (
                f'# To also re-add audio, use the source file:\n'
                f'#   {self.input_path}\n'
                f'# (add:  -i "<source>" -map 0:v:0 -map 1:a? -c:a copy)\n'
            )
        body = (
            "# ChitraMaya encode-recovery script (auto-generated).\n"
            "# If this file still exists, the run did NOT finalize the output\n"
            "# (crash, out-of-memory, or forced close). Every encoded frame is\n"
            f"# preserved in the raw bitstream below. Run this script in\n"
            "# PowerShell to produce a playable *-FIXED file. Verify it plays\n"
            "# at the right SPEED — if not, change -r to the source frame rate.\n"
            f"# ffmpeg must be on PATH.\n{source_hint}\n"
            f"& {self._recovery_command_str().replace('ffmpeg', 'ffmpeg', 1)}\n"
        )
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(body)
        self._recovery_script_path = script_path

    def _remove_recovery_script(self) -> None:
        if self._recovery_script_path:
            try:
                p = Path(self._recovery_script_path)
                if p.exists():
                    p.unlink()
            except Exception:
                pass

    def close(self) -> None:
        """Flush, close file, remux with audio if needed."""
        if self._closed:
            return
        self._closed = True

        self.flush()
        self._file.close()

        if self._needs_remux:
            self._remux_ok = self._remux()

        # Cleanup raw bitstream + recovery script only if remux succeeded
        if self._needs_remux and self._remux_ok:
            raw = Path(self._raw_path)
            if raw.exists():
                try:
                    raw.unlink()
                except Exception:
                    pass
            self._remove_recovery_script()
        elif self._needs_remux and not self._remux_ok:
            print(f"[Encoder] Raw bitstream kept for debugging: {self._raw_path}")

        if self._needs_remux and not self._remux_ok:
            # Do NOT print a cheerful "Done" over a broken output (field
            # event: a 19h45m AV1 run ended with a failed remux and a
            # misleading Done line). Every encoded frame is safe in the raw
            # bitstream; print the EXACT command to wrap it, ready to paste
            # into a PowerShell window, plus the sidecar script that survives
            # even a hard kill.
            print(f"[Encoder] *** REMUX FAILED -- {self.output_path} is NOT a "
                  f"playable file. ***")
            print(f"[Encoder] All encoded frames are preserved in: {self._raw_path}")
            print(f"[Encoder] Recover now -- paste this into PowerShell:")
            print(f"    {self._recovery_command_str()}")
            if self._recovery_script_path:
                print(f"[Encoder] (Or run the ready-made script: "
                      f"{self._recovery_script_path})")
            print(f"[Encoder] If playback speed is wrong, change -r "
                  f"{self.fps_str} to the source frame rate. Re-add audio with "
                  f"a second -i \"<source>\" -map 0:v:0 -map 1:a? -c:a copy.")
        else:
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
                **NOWINDOW,
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

        # AV1 container sniff (field event 7/18/2026, 19h45m run nearly lost):
        # PyNvVideoCodec's NVENC AV1 emits an IVF-WRAPPED bitstream ("DKIF"
        # magic), not raw low-overhead OBU — forcing -f obu on IVF fails with
        # "No sequence header available" and the remux dies AFTER the whole
        # encode. Sniff the first 4 bytes: IVF -> let ffmpeg auto-probe (it
        # reads IVF natively); anything else -> keep the explicit demuxer.
        input_fmt_args = ["-f", input_fmt]
        if self.codec == "av1":
            try:
                with open(raw_path, "rb") as _rf:
                    magic = _rf.read(4)
                if magic == b"DKIF":
                    input_fmt_args = []            # IVF: auto-probe
                    print("[Encoder] AV1 raw bitstream is IVF-wrapped (DKIF); "
                          "using ffmpeg auto-probe for remux")
                    # CM-091: make every keyframe a real seek point BEFORE
                    # wrapping into MP4 (see _av1_fix_seq_headers docstring).
                    _fix = _av1_fix_seq_headers(raw_path)
                    if _fix is None:
                        print("[Encoder] WARNING: AV1 seek fix could not run "
                              "(unexpected stream shape); output will play "
                              "linearly but seeking may fail in ffmpeg/VLC/"
                              "mpv-class players. Prefer HEVC, or repair "
                              "later with Fix-AV1-SeekHeaders.py.")
                    else:
                        _units, _ins = _fix
                        if _ins == 0:
                            print(f"[Encoder] AV1 seek fix: stream already "
                                  f"carries in-band sequence headers "
                                  f"({_units} units checked) -- encoder/"
                                  f"driver honored repetition; nothing to do.")
                        else:
                            print(f"[Encoder] AV1 seek fix: sequence header "
                                  f"inserted into {_ins} of {_units} temporal "
                                  f"units; every keyframe is now a valid "
                                  f"seek target.")
            except Exception:
                pass

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
            elif self.codec == "av1":
                tag_args = ["-tag:v", "av01"]

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
                         *input_fmt_args, "-i", str(raw_path),
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
                   *input_fmt_args, "-i", str(raw_path)]
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