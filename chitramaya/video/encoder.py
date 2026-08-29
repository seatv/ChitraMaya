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

# CM-093 X1: PyNvVideoCodec is NVIDIA-only and absent on the Intel Arc /
# XPU build. Guarded import so the module loads anywhere; the Encoder
# constructor raises a clear message instead (QSV encode is CM-093 X3).
try:
    import PyNvVideoCodec as nvc
except Exception:
    nvc = None


# CM-122: consecutive empty Encode() returns that mean the NVENC session is
# dead, not merely latent. Legit output latency (priming + bf + rc_lookahead)
# is bounded by a few dozen frames; 600 (=10s @ 59.94) is far beyond any of
# it and still fails within seconds of a real mid-run death.
_NVENC_DEAD_AFTER = 600


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


def _ps_quote(s: str) -> str:
    """Quote a path as a PowerShell single-quoted literal. Single quotes
    are the only escape needed ('' inside '...'); $, backtick, and double
    quotes are all inert inside single quotes, so arbitrary Windows / UNC
    paths survive verbatim."""
    return "'" + str(s).replace("'", "''") + "'"


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


def _finalize_timeout_s(*paths) -> int:
    """Timeout (seconds) for a finalize/remux ffmpeg call, scaled to the
    bytes it must move.

    The flat 900s (15 min) died in the field on a ~50 GB output (The-Idol,
    2026-08): the finalize remux is stream-copy, but ``-movflags
    +faststart`` makes ffmpeg write the whole file and then REWRITE it a
    second time to move the moov atom to the front -- two full passes of
    disk I/O over the entire output. At spinning-disk / SMR / USB / network
    speeds, two passes over tens of GB take far more than 15 minutes, and
    killing a remux that is still making disk progress converts a slow
    finalize into a failed run (every frame already safely encoded).

    Budget: 25 MB/s effective throughput (deliberately pessimistic), two
    passes, +300s slack, floored at the old 900s. 50 GB -> ~72 min ceiling;
    the short clips that dominate testing keep the familiar 15 min.
    """
    total = 0
    for p in paths:
        try:
            if p:
                total += os.path.getsize(str(p))
        except OSError:
            pass
    return max(900, int(total / (25 * 1024 * 1024)) * 2 + 300)


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
        # CM-121 (field, 2026-08-26): MPEG-TS sources (HLS stream rips,
        # often wearing .mkv extensions) carry PROGRAMS, and ffprobe lists
        # each stream twice -- once under the program, once standalone --
        # so stdout is the value printed twice. float() on that blob raised
        # and this probe silently returned None, which disabled A/V
        # start-offset restoration for EVERY TS source (a constant ~67 ms
        # audio-early error in the field). Parse the first usable line.
        for line in (out.stdout or "").splitlines():
            line = line.strip()
            if line and line.upper() != "N/A":
                return float(line)
        return None
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
        # CM-093 X1: encoding is NVENC-only until X3 (ffmpeg/QSV backend).
        # Fail at construction with a clear story, not deep in NVENC setup.
        if nvc is None:
            raise RuntimeError(
                "Encoding requires NVENC (PyNvVideoCodec), which is not "
                "available on this system (non-NVIDIA GPU?). Analysis, "
                "detection and Test Frame previews work; encoding on Intel "
                "Arc (QSV) arrives with CM-093 X3."
            )
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

        # CM-122 NVENC liveness canary (field event 2026-08-28): an NVENC
        # session died silently 16.5 minutes into an 11-hour run (metrics
        # CSV: encoder_util 61% -> 0% at one sample boundary, never
        # returned; no VRAM spike, no clock/thermal event). The 2.2 wheel's
        # Encode() swallowed the error and returned empty packet lists for
        # 217,020 consecutive calls, so the .hevc stopped growing at frame
        # 12,792 while the pipeline "encoded" for 10.7 more hours and
        # reported success. Empty returns are normal for the first few
        # frames (NVENC priming/lookahead latency: a handful of frames,
        # bounded by bf + rc_lookahead), but hundreds in a row after bytes
        # have flowed means the session is dead. Fail fast and loud: the
        # raw bitstream plus the RECOVER script are already on disk, so an
        # abort here costs minutes, not the rest of the run.
        self._bytes_written = 0
        self._empty_streak = 0
        self._nvenc_dead = False

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

    @staticmethod
    def _bitstream_bytes(ret) -> bytes:
        """Normalize Encode()/EndEncode() output across PyNvVideoCodec
        API generations.

        PyNvVideoCodec 2.1 returns a bytes-like blob (possibly empty).
        PyNvVideoCodec 2.2 changed the API: it returns a LIST of packet
        dicts {"data", "picture_type", "timestamp"}; NVIDIA's own 2.2
        samples concatenate bytes(p["data"]) before writing. Field event
        2026-08-16: requirements' open bound (>=2.1) pulled 2.2.0 into a
        rebuilt dev venv and every frame write raised
        TypeError: 'dict' object cannot be interpreted as an integer --
        first drain died, .hevc got no headers, remux unrecoverable.
        This helper accepts both shapes so the encoder works against
        either wheel; frozen releases bundle whichever was installed at
        build time.
        """
        if not ret:
            return b""
        if isinstance(ret, (bytes, bytearray, memoryview)):
            return bytes(ret)
        if isinstance(ret, (list, tuple)):
            parts = []
            for p in ret:
                if isinstance(p, dict):
                    parts.append(bytes(p["data"]))
                else:
                    parts.append(bytes(p))
            return b"".join(parts)
        # Unknown-but-buffer-like (future API drift): let bytes() try.
        return bytes(ret)

    def encode_frame(self, frame: torch.Tensor) -> None:
        """Encode a single BGRA HWC4 uint8 CUDA tensor."""
        if self._closed:
            return

        frame = frame.contiguous()
        if frame.device.type == "cuda":
            torch.cuda.synchronize(device=frame.device)

        payload = self._bitstream_bytes(self._encoder.Encode(frame))
        if payload:
            self._file.write(payload)
            self._bytes_written += len(payload)
            self._empty_streak = 0
        else:
            self._empty_streak += 1
            if self._empty_streak == _NVENC_DEAD_AFTER:
                self._nvenc_dead = True
                mb = self._bytes_written / 1e6
                # frames_encoded has not been incremented for THIS frame yet
                # and includes streak-1 earlier empties.
                n_payload = self._frames_encoded - (self._empty_streak - 1)
                secs = n_payload / max(self.fps, 1.0)
                print(
                    f"[Encoder] FATAL (CM-122): NVENC returned no bitstream for "
                    f"{self._empty_streak} consecutive frames -- the encode "
                    f"session is dead. Bitstream on disk: {mb:.1f} MB "
                    f"(~{secs:.1f}s of video from {n_payload} frames)."
                )
                print(
                    "[Encoder] The raw bitstream and its RECOVER script are "
                    "preserved next to the output. Aborting now instead of "
                    "silently encoding nothing for the rest of the run."
                )
                raise RuntimeError(
                    "NVENC session died mid-run (no bitstream for "
                    f"{self._empty_streak} consecutive frames, CM-122). "
                    "Partial bitstream preserved; re-run the job. If this "
                    "repeats at the same spot, report it -- if it repeats at "
                    "random spots, suspect drivers/power events on this GPU."
                )
        self._frames_encoded += 1

    def flush(self) -> None:
        """Flush remaining frames from encoder."""
        try:
            tail = self._bitstream_bytes(self._encoder.EndEncode())
            if tail:
                self._file.write(tail)
                self._bytes_written += len(tail)
        except Exception as e:
            print(f"[Encoder] Flush error: {e}")
        print(f"[Encoder] Flushed ({self._frames_encoded} frames, "
              f"{self._bytes_written / 1e6:.1f} MB bitstream)")
        # CM-122 end-of-run honesty check: catches a session death too close
        # to the end to trip the streak canary. Even ultra-static content at
        # high QP averages far more than 2 KB/frame; a lower average means
        # frames were submitted that produced no bytes.
        if self._frames_encoded > 0:
            avg = self._bytes_written / self._frames_encoded
            if avg < 2000:
                print(
                    f"[Encoder] WARNING (CM-122): bitstream averages only "
                    f"{avg:.0f} bytes/frame across {self._frames_encoded} "
                    f"frames -- far below any plausible encode. Part of this "
                    f"run likely produced NO video; verify the output's video "
                    f"duration before trusting it."
                )

    def _mp4_tag(self) -> str:
        """Per-codec MP4 fourcc tag (hvc1/avc1/av01) — a wrong tag makes
        some players reject the file."""
        return {"hevc": "hvc1", "h265": "hvc1", "h264": "avc1",
                "avc": "avc1", "av1": "av01"}.get(self.codec, "hvc1")

    def _recovery_command_str(self) -> str:
        """The exact PowerShell ffmpeg command that wraps the raw bitstream
        into a playable file (video only — the always-works paste-in
        fallback shown by the REMUX FAILED banner). The sidecar script goes
        further and re-adds audio automatically."""
        fixed = str(Path(self.output_path).with_suffix("")) + "-FIXED" + \
            Path(self.output_path).suffix
        fmt = _ffmpeg_input_fmt(self.codec)
        # For AV1 the bitstream may be IVF-wrapped (auto-probes); a plain -i
        # works for that, while HEVC/H.264 want the explicit demuxer. Use the
        # demuxer for non-av1 and let av1 auto-probe.
        fmt_arg = "" if self.codec == "av1" else f'-f {fmt} '
        # Tag / timescale / faststart are mp4-container concerns, matching
        # the gating _remux() applies (an mkv output gets none of them).
        mp4_args = ""
        if self._container == "mp4":
            mp4_args = f'-tag:v {self._mp4_tag()} -video_track_timescale 90000 '
            if self.mp4_faststart:
                mp4_args += "-movflags +faststart "
        return (
            f'ffmpeg -hide_banner -fflags +genpts -r {self.fps_str} '
            f'{fmt_arg}-i "{self._raw_path}" -c:v copy {mp4_args}"{fixed}"'
        )

    def _write_recovery_script(self) -> None:
        """Write a self-contained PowerShell recovery script next to the raw
        bitstream, so a hard kill (OOM/wedge) before close() still leaves the
        user a one-double-click fix. Deleted on successful remux.

        The script does the COMPLETE job -- it wraps the raw bitstream into a
        container AND re-adds audio from the source file (the source path is
        known right here at generation time; making the user hand-edit an
        ffmpeg command was our failure landing on their desk). Two steps,
        mirroring the production remux:

          1. raw bitstream -> temp video container (gives the stream real
             timestamps; -shortest is unreliable directly on a raw
             elementary-stream input, which is why the audio mux happens on
             the containerized copy).
          2. temp container + source audio -> *-FIXED file, with -shortest
             trimming the full-length audio to the partial video.

        A Test-Path guard falls back to a video-only *-FIXED file when the
        source has moved, with instructions to edit $source and re-run.
        Written as UTF-8 with BOM: Windows PowerShell 5.1 assumes ANSI for
        BOM-less scripts and would mangle non-ASCII paths."""
        script_path = str(Path(self.output_path).with_suffix("")) + \
            "-RECOVER.ps1"
        out = Path(self.output_path)
        fixed = str(out.with_suffix("")) + "-FIXED" + out.suffix
        vtmp = fixed + ".vtmp" + out.suffix
        fmt_args = "" if self.codec == "av1" else \
            f"-f {_ffmpeg_input_fmt(self.codec)} "

        is_mp4 = self._container == "mp4"
        tag_args = f"-tag:v {self._mp4_tag()} " if is_mp4 else ""
        ts_args = "-video_track_timescale 90000 " if is_mp4 else ""
        fs_args = "-movflags +faststart " if (is_mp4 and self.mp4_faststart) \
            else ""

        has_source = bool(self.mux_audio and self.input_path)
        src_line = (
            f"$source = {_ps_quote(self.input_path)}\n" if has_source else ""
        )

        av1_note = ""
        if self.codec == "av1":
            av1_note = (
                "# NOTE (AV1 only): the recovered file plays linearly, but\n"
                "# SEEKING may fail in ffmpeg/VLC/mpv-class players because\n"
                "# the in-band sequence-header repair (CM-091) runs only in\n"
                "# the normal finalize step. Repair later with\n"
                "# Fix-AV1-SeekHeaders.py if seeking matters.\n"
            )

        header = (
            "# ChitraMaya encode-recovery script (auto-generated).\n"
            "# If this file still exists, the run did NOT finalize the output\n"
            "# (crash, out-of-memory, or forced close). Every encoded frame is\n"
            "# preserved in the raw bitstream next to this script. Just RUN\n"
            "# THIS SCRIPT in PowerShell: it wraps the video AND re-adds the\n"
            "# audio from the source file automatically.\n"
            "# ffmpeg must be on PATH.\n"
            "# Verify the *-FIXED file plays at the right SPEED -- if not,\n"
            "# edit the -r value below to the source frame rate and re-run.\n"
            f"{av1_note}\n"
            f"$raw    = {_ps_quote(self._raw_path)}\n"
            f"{src_line}"
            f"$fixed  = {_ps_quote(fixed)}\n"
            f"$vtmp   = {_ps_quote(vtmp)}\n"
            "\n"
            "# Step 1: wrap the raw bitstream into a real video container.\n"
            f"& ffmpeg -hide_banner -y -fflags +genpts "
            f"-analyzeduration 10M -probesize 50M -r {self.fps_str} "
            f"{fmt_args}-i $raw -map 0:v:0 -c:v copy {tag_args}{ts_args}"
            f"$vtmp\n"
            "if ($LASTEXITCODE -ne 0) {\n"
            "    Write-Host 'RECOVERY FAILED: could not wrap the raw "
            "bitstream (see ffmpeg output above).'\n"
            "    exit 1\n"
            "}\n"
            "\n"
        )

        if has_source:
            body = header + (
                "# Step 2: re-add audio from the source. The video is partial\n"
                "# (the run died mid-encode), so -shortest trims the audio to\n"
                "# match. The tiny source A/V start offset (usually < 50 ms)\n"
                "# is not restored here; a normal completed run does that.\n"
                "$done = $false\n"
                "if (Test-Path -LiteralPath $source) {\n"
                f"    & ffmpeg -hide_banner -y -i $source -i $vtmp "
                f"-map 1:v:0 -c:v copy {tag_args}-map 0:a? -c:a copy "
                f"-shortest {fs_args}{ts_args}$fixed\n"
                "    if ($LASTEXITCODE -eq 0) {\n"
                "        Remove-Item -LiteralPath $vtmp -ErrorAction "
                "SilentlyContinue\n"
                "        $done = $true\n"
                "        Write-Host \"Recovered WITH audio: $fixed\"\n"
                "    } else {\n"
                "        Write-Host 'Audio mux failed; keeping the "
                "video-only recovery instead.'\n"
                "    }\n"
                "} else {\n"
                "    Write-Host 'Source file not found -- producing a "
                "VIDEO-ONLY recovery.'\n"
                "    Write-Host 'If the source moved, edit the $source "
                "line at the top of this script and re-run.'\n"
                "}\n"
                "if (-not $done) {\n"
                "    Move-Item -LiteralPath $vtmp -Destination $fixed -Force\n"
                "    Write-Host \"Recovered (video only): $fixed\"\n"
                "}\n"
                "Write-Host 'Once the *-FIXED file plays correctly, the raw "
                "bitstream and this script can be deleted.'\n"
            )
        else:
            # No audio was requested (or no source path known): single step,
            # straight to the *-FIXED file.
            body = header + (
                "Move-Item -LiteralPath $vtmp -Destination $fixed -Force\n"
                "Write-Host \"Recovered: $fixed\"\n"
                "Write-Host 'Once the *-FIXED file plays correctly, the raw "
                "bitstream and this script can be deleted.'\n"
            )
        # utf-8-sig: the BOM is what makes Windows PowerShell 5.1 parse the
        # file as UTF-8 (BOM-less .ps1 is read as ANSI and non-ASCII paths
        # in $raw/$source would be corrupted).
        with open(script_path, "w", encoding="utf-8-sig", newline="\r\n") as f:
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
            if self._recovery_script_path:
                print(f"[Encoder] Recover now -- run the ready-made script "
                      f"(it re-adds the audio automatically):")
                print(f"    {self._recovery_script_path}")
                print(f"[Encoder] Or, for a video-only wrap, paste this into "
                      f"PowerShell:")
            else:
                print(f"[Encoder] Recover now -- paste this into PowerShell "
                      f"(video only):")
            print(f"    {self._recovery_command_str()}")
            print(f"[Encoder] If playback speed is wrong, change -r "
                  f"{self.fps_str} to the source frame rate.")
        else:
            print(f"[Encoder] Done: {self.output_path}")

    def _run_ffmpeg(self, cmd: List[str], label: str = "ffmpeg",
                    timeout_s: int = 900,
                    watch_path: Optional[str] = None,
                    stall_s: int = 600) -> bool:
        """Run one ffmpeg command; return True on rc==0. Shared by both remux
        passes. UTF-8 decoding avoids cp1252 crashes on non-ASCII filenames.
        ``timeout_s``: pass _finalize_timeout_s(...) for whole-file remux
        steps (v1.50.00, the 50GB Idol lesson); the 900s default suits
        everything else.

        ``watch_path`` (CM-124, field 2026-08-28): a wall-clock timeout,
        however scaled, still killed a remux that was MAKING DISK PROGRESS
        at ~4 MB/s (slow HDD pair + faststart second pass; a complete
        229,812-frame bitstream lost its finalize at 59% written). When
        watch_path is set, the command is instead supervised by progress:
        the output's size (plus its ffmpeg faststart '.tmp' sibling) is
        polled every 5s, and the process is killed only after ``stall_s``
        seconds with NO byte growth -- a slow disk gets as long as it
        needs, while a truly wedged ffmpeg still dies in minutes. The
        scaled ``timeout_s`` becomes an estimate: passing it just logs a
        courtesy note that the remux is slow but alive."""
        print(f"[Encoder] {label}: {' '.join(cmd)}")
        if watch_path is None:
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    timeout=timeout_s,
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
            except subprocess.TimeoutExpired:
                print(f"[Encoder] {label} TIMED OUT after {timeout_s}s with no "
                      f"result. Every encoded frame is preserved in the raw "
                      f"bitstream -- use the recovery command printed below, "
                      f"ideally targeting a faster disk.")
                return False
            except Exception as e:
                print(f"[Encoder] {label} error: {e}")
                return False

        # Progress-supervised mode (CM-124).
        import tempfile
        import time as _time

        def _progress() -> int:
            total = 0
            for p in (watch_path, watch_path + ".tmp"):
                try:
                    total += os.path.getsize(p)
                except OSError:
                    pass
            return total

        err_path = None
        err_f = None
        try:
            err_f = tempfile.NamedTemporaryFile(
                prefix="cm_ffmpeg_", suffix=".stderr", delete=False)
            err_path = err_f.name
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                    stderr=err_f, **NOWINDOW)
            start = last_change = _time.monotonic()
            last_bytes = _progress()
            hard_cap_s = max(int(timeout_s) * 10, 6 * 3600)
            slow_noted = False
            while True:
                rc = proc.poll()
                if rc is not None:
                    break
                _time.sleep(5)
                cur = _progress()
                now = _time.monotonic()
                if cur != last_bytes:
                    last_bytes = cur
                    last_change = now
                if now - last_change > stall_s:
                    proc.kill()
                    try:
                        proc.wait(timeout=30)
                    except Exception:
                        pass
                    print(f"[Encoder] {label} KILLED: no disk progress for "
                          f"{int(now - last_change)}s (output stuck at "
                          f"{last_bytes / 1e6:.1f} MB). Every encoded frame "
                          f"is preserved in the raw bitstream -- use the "
                          f"recovery script.")
                    return False
                if now - start > hard_cap_s:
                    proc.kill()
                    try:
                        proc.wait(timeout=30)
                    except Exception:
                        pass
                    print(f"[Encoder] {label} KILLED after {int(now - start)}s "
                          f"(absolute safety cap). Raw bitstream preserved -- "
                          f"use the recovery script on a faster disk.")
                    return False
                if not slow_noted and now - start > timeout_s \
                        and last_bytes > 0:
                    slow_noted = True
                    print(f"[Encoder] {label}: past the {int(timeout_s)}s "
                          f"estimate but still making disk progress "
                          f"({last_bytes / 1e6:.1f} MB written) -- letting it "
                          f"finish (CM-124).")
            err_f.close()
            if rc != 0:
                print(f"[Encoder] {label} failed (rc={rc})")
                try:
                    with open(err_path, "r", encoding="utf-8",
                              errors="replace") as f:
                        for line in f.read().strip().split("\n")[-5:]:
                            print(f"  {line}")
                except Exception:
                    pass
                return False
            elapsed = _time.monotonic() - start
            print(f"[Encoder] {label} OK ({elapsed:.0f}s, "
                  f"{last_bytes / 1e6:.1f} MB)")
            return True
        except Exception as e:
            print(f"[Encoder] {label} error: {e}")
            return False
        finally:
            if err_f is not None:
                try:
                    err_f.close()
                except Exception:
                    pass
            if err_path:
                try:
                    os.unlink(err_path)
                except Exception:
                    pass

    def _discard_partial_output(self) -> None:
        """CM-124: a killed/failed final remux leaves a half-written output
        that looks like a deliverable (field 2026-08-28: a 3.45GB partial
        .mp4 sat next to the complete 5.79GB .hevc and read as 'the
        result'). Delete it -- the raw bitstream + recovery script are the
        salvage, and a missing file is clearer than a broken one."""
        try:
            p = Path(self.output_path)
            if p.exists():
                sz = p.stat().st_size
                p.unlink()
                print(f"[Encoder] Removed unplayable partial output "
                      f"({sz / 1e6:.1f} MB): {self.output_path}")
        except Exception as e:
            print(f"[Encoder] Could not remove partial output: {e}")

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

        # CM-120r2: HEAD-SKIP compensation. When the decode stage discarded
        # the stream's head (mid-GOP capture start; measured by the pipeline
        # from the first delivered frame's absolute PTS), the encoded video
        # begins later in the content than the audio does -- delay the video
        # by the same amount so the timelines line up again. Set by the
        # pipeline as _head_skip_seconds; absent/0 on healthy files.
        _head = 0.0
        try:
            _head = float(getattr(self, "_head_skip_seconds", 0.0) or 0.0)
        except Exception:
            _head = 0.0
        if _head > 0:
            video_delay += _head
            print(f"[Encoder] Head-skip compensation: +{_head:.3f}s video "
                  f"delay (decoder discarded the stream head; audio stays "
                  f"aligned).")

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

        # v1.50.00 rider (from the lada A/B "pop" investigation): propagate
        # the SOURCE's color tags into the output. Restored files used to
        # ship with color_space=unknown while sources carry bt709 -- players
        # then GUESS, and can render output vs source subtly differently.
        # With -c:v copy the reliable channel is the codec VUI via the
        # *_metadata bitstream filter, applied on the command that wraps the
        # raw stream. Conservative: only known name->code mappings; unknown
        # sources keep today's behavior.
        color_args: List[str] = []
        try:
            if self.codec in ("hevc", "h265", "h264", "avc") and self.input_path:
                import subprocess as _sp
                _pr = _sp.run(
                    [_derive_ffprobe(self.ffmpeg_path), "-v", "error",
                     "-select_streams", "v:0", "-show_entries",
                     "stream=color_space,color_transfer,color_primaries,"
                     "color_range", "-of", "default=nw=1",
                     str(self.input_path)],
                    capture_output=True, text=True, timeout=15,
                )
                _cv = {}
                for _l in (_pr.stdout or "").splitlines():
                    if "=" in _l:
                        k, v = _l.split("=", 1)
                        _cv[k.strip()] = v.strip().lower()
                _PRI = {"bt709": 1, "bt470bg": 5, "smpte170m": 6, "bt2020": 9}
                _TRC = {"bt709": 1, "smpte170m": 6, "bt2020-10": 14,
                        "smpte2084": 16, "arib-std-b67": 18}
                _MAT = {"bt709": 1, "bt470bg": 5, "smpte170m": 6,
                        "bt2020nc": 9}
                _parts = []
                if _cv.get("color_primaries") in _PRI:
                    _parts.append(f"colour_primaries="
                                  f"{_PRI[_cv['color_primaries']]}")
                if _cv.get("color_transfer") in _TRC:
                    _parts.append(f"transfer_characteristics="
                                  f"{_TRC[_cv['color_transfer']]}")
                if _cv.get("color_space") in _MAT:
                    _parts.append(f"matrix_coefficients="
                                  f"{_MAT[_cv['color_space']]}")
                if _cv.get("color_range") in ("tv", "pc"):
                    _parts.append(f"video_full_range_flag="
                                  f"{1 if _cv['color_range'] == 'pc' else 0}")
                if _parts:
                    _flt = ("hevc_metadata"
                            if self.codec in ("hevc", "h265")
                            else "h264_metadata")
                    color_args = ["-bsf:v", f"{_flt}=" + ":".join(_parts)]
                    print(f"[Encoder] color tags: propagating source "
                          f"{','.join(sorted(v for v in _cv.values() if v and v != 'unknown'))} "
                          f"into the output VUI")
        except Exception:
            color_args = []

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
                step1 += color_args + tag_args + timescale_args + [str(temp_video)]
                # v1.50.00: scale finalize timeouts with the bytes moved
                # (the 50GB Idol lesson -- faststart = TWO passes of I/O).
                # CM-124: supervised by disk progress; the scaled value is
                # now just the "slow but alive" courtesy-note threshold.
                _t_s = _finalize_timeout_s(raw_path)
                if not self._run_ffmpeg(step1, "video-container",
                                        timeout_s=_t_s,
                                        watch_path=str(temp_video)):
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
                # CM-124: the remux also READS the audio source end to end,
                # so its bytes belong in the estimate too.
                ok = self._run_ffmpeg(
                    step2, "remux",
                    timeout_s=_finalize_timeout_s(
                        temp_video,
                        self.input_path if has_audio_source else None),
                    watch_path=str(out_path))
                if not ok:
                    self._discard_partial_output()
                return ok

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
            cmd += ["-map", "0:v:0", "-c:v", "copy"] + color_args + tag_args
            if has_audio_source:
                cmd += ["-map", "1:a?", "-c:a", "copy"]
            cmd += faststart_args + timescale_args + dur_args + extra_args + [str(out_path)]
            # v1.50.00: size-scaled timeout (the 50GB Idol lesson).
            # CM-124: progress-supervised; source audio bytes included.
            ok = self._run_ffmpeg(
                cmd, "remux",
                timeout_s=_finalize_timeout_s(
                    raw_path,
                    self.input_path if has_audio_source else None),
                watch_path=str(out_path))
            if not ok:
                self._discard_partial_output()
            return ok
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

# ═══════════════════════════════════════════════════════════════════════════
# CM-093 X3: ffmpeg-subprocess encoder backend for non-NVIDIA machines
# ═══════════════════════════════════════════════════════════════════════════

def nvenc_available() -> bool:
    """True when PyNvVideoCodec imported (NVIDIA build). The pipeline uses
    this to choose Encoder (NVENC) vs FfmpegEncoder (QSV/software)."""
    return nvc is not None


class FfmpegEncoder:
    """CM-093 X3: encoding for machines without NVENC (Intel Arc, CPU-only).

    Same construction signature and encode_frame/flush/close contract as
    Encoder, so the pipeline swaps classes and nothing else changes. Video
    goes to an ffmpeg subprocess over stdin as raw BGRA frames; the encoder
    is chosen by a runtime probe:

      1. Intel QSV (hevc_qsv / h264_qsv / av1_qsv) -- Arc's hardware media
         engine via libvpl (the gyan.dev 'full' ffmpeg builds include it).
      2. AMD AMF (hevc_amf / h264_amf / av1_amf) -- Radeon's hardware
         encoder (Batch 34, ROCm edition; av1_amf is RDNA4+).
      3. Software fallback (libx265 / libx264 / libsvtav1) with a loud
         banner when no hardware encoder initializes.

    The video-only stream is written to <output>.venc.mp4 as a FRAGMENTED
    mp4 (crash-tolerant: a killed run still leaves decodable video), then
    close() remuxes audio from the source with the same A/V start-offset
    handling as the NVENC path, faststart, and the right fourcc tag.
    ffmpeg-managed AV1 carries in-band sequence headers natively, so there
    is no CM-091 seek-fix analog here.

    QP semantics match the UI's HEVC 0-51 scale: QSV gets it as
    -global_quality (ICQ), x264/x265 as -crf, SVT-AV1 as -crf (0-63 scale,
    same value = same intent). Preset P1-P7 maps to the speed ladders.
    """

    _QSV = {"hevc": "hevc_qsv", "h265": "hevc_qsv",
            "h264": "h264_qsv", "avc": "h264_qsv", "av1": "av1_qsv"}
    # Batch 34 (ROCm edition): AMD's hardware encoder. Probed between QSV
    # and software -- on Radeon the QSV probe fails and, pre-Batch-34, the
    # ladder silently landed on CPU x265, making the GPU look slow through
    # no fault of its own. av1_amf exists on RDNA4 only; the probe decides.
    _AMF = {"hevc": "hevc_amf", "h265": "hevc_amf",
            "h264": "h264_amf", "avc": "h264_amf", "av1": "av1_amf"}
    _SW = {"hevc": "libx265", "h265": "libx265",
           "h264": "libx264", "avc": "libx264", "av1": "libsvtav1"}
    _P2SPEED = {"P1": "veryfast", "P2": "faster", "P3": "fast", "P4": "medium",
                "P5": "slow", "P6": "slower", "P7": "veryslow"}
    _P2SVT = {"P1": "10", "P2": "9", "P3": "8", "P4": "7",
              "P5": "6", "P6": "4", "P7": "2"}
    # AMF exposes three quality tiers, not a 7-step ladder; map the UI's
    # P1-P7 onto them (P1-P3 speed, P4-P5 balanced, P6-P7 quality).
    _P2AMF = {"P1": "speed", "P2": "speed", "P3": "speed",
              "P4": "balanced", "P5": "balanced",
              "P6": "quality", "P7": "quality"}

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
        **_ignored,   # NVENC-only quality kwargs arrive here harmlessly
    ) -> None:
        self.output_path = str(output_path)
        self.input_path = str(input_path) if input_path else None
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.fps_str = _fps_to_rational(fps)
        self.codec = str(codec).lower()
        self.preset = str(preset).upper() if str(preset).upper() in self._P2SPEED else "P5"
        self.qp = int(qp)
        self.mux_audio = bool(mux_audio)
        self.mp4_faststart = bool(mp4_faststart)
        self.ffmpeg_path = str(ffmpeg_path)
        self._frames_encoded = 0
        self._closed = False
        self._proc = None

        if self.codec not in self._QSV:
            raise ValueError(f"FfmpegEncoder: unsupported codec {self.codec!r}")

        self._venc_path = self.output_path + ".venc.mp4"
        self._stderr_path = self.output_path + ".venc.stderr.log"

        # Pick the encoder: QSV -> AMF -> software (Batch 34 ladder). Each
        # rung is a real 2-frame init probe on THIS machine, so a missing
        # driver/GPU falls through cleanly rather than dying mid-run.
        qsv_name = self._QSV[self.codec]
        amf_name = self._AMF[self.codec]
        sw_name = self._SW[self.codec]
        if self._probe_encoder(qsv_name):
            self._enc_name = qsv_name
            self._enc_args = ["-global_quality", str(self.qp),
                              "-preset", self._P2SPEED[self.preset]]
            hw = "Intel QSV hardware"
        elif self._probe_encoder(amf_name):
            self._enc_name = amf_name
            # CQP to match the UI's constant-QP semantics. AV1's qindex
            # scale is 0-255; map the HEVC-style slider by the same rough
            # 4x equivalence the NVENC path uses (QP18 -> qindex 72).
            _q = self.qp
            if self.codec == "av1":
                _q = min(255, self.qp * 4)
                print(f"[Encoder] AV1 rate control: QP{self.qp} (HEVC scale)"
                      f" -> qindex {_q} (AV1 scale)")
            self._enc_args = ["-rc", "cqp",
                              "-qp_i", str(_q), "-qp_p", str(_q),
                              "-quality", self._P2AMF[self.preset]]
            hw = "AMD AMF hardware"
        elif self._probe_encoder(sw_name):
            self._enc_name = sw_name
            if sw_name == "libsvtav1":
                self._enc_args = ["-crf", str(min(63, self.qp)),
                                  "-preset", self._P2SVT[self.preset]]
            else:
                self._enc_args = ["-crf", str(self.qp),
                                  "-preset", self._P2SPEED[self.preset]]
            hw = "SOFTWARE (CPU)"
            print(f"[Encoder] WARNING: no hardware encoder available "
                  f"({qsv_name} and {amf_name} both failed to initialize) -- "
                  f"falling back to {sw_name} on the CPU. Expect slow "
                  f"encodes. Intel: check the GPU driver and that ffmpeg has "
                  f"libvpl. AMD: check the Adrenalin driver (AMF).")
        else:
            raise RuntimeError(
                f"No usable encoder for {self.codec}: none of {qsv_name}, "
                f"{amf_name}, {sw_name} works in this ffmpeg build. Install "
                f"the ffmpeg 'full' build (gyan.dev)."
            )

        cmd = [
            self.ffmpeg_path, "-hide_banner", "-y", "-loglevel", "warning",
            "-f", "rawvideo", "-pix_fmt", "bgra",
            "-s", f"{self.width}x{self.height}", "-r", self.fps_str,
            "-i", "-",
            "-an",
            "-c:v", self._enc_name, *self._enc_args,
            # Hardware encoders (QSV and AMF) both prefer NV12 input;
            # software x264/x265/svt take planar yuv420p.
            "-pix_fmt", ("nv12" if self._enc_name.endswith(("_qsv", "_amf"))
                         else "yuv420p"),
            # Crash-tolerant temp container: fragmented mp4 stays decodable
            # if the run dies; close() rewrites it properly with faststart.
            # -flush_packets 1 forces the header+fragments out of ffmpeg's
            # I/O buffer promptly, so even an early kill leaves a readable
            # file (container-tested: without it, moov can sit unflushed).
            "-movflags", "+frag_keyframe+empty_moov",
            "-flush_packets", "1",
            self._venc_path,
        ]
        print(f"[Encoder] ffmpeg backend: {self._enc_name} ({hw}) "
              f"{self.width}x{self.height} @ {self.fps:.2f}fps "
              f"{self.codec}/{self.preset}/QP{self.qp}")
        self._stderr_f = open(self._stderr_path, "w", encoding="utf-8")
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=self._stderr_f, **NOWINDOW,
        )

        # Recovery sidecar (same rationale as the NVENC path): a hard kill
        # leaves <output>.venc.mp4 -- fragmented, so every flushed frame IS
        # decodable -- but audio-less and without faststart. Write the
        # finish-the-job script now; deleted on a successful remux.
        self._recovery_script_path: Optional[str] = None
        try:
            self._write_recovery_script()
        except Exception:
            pass  # recovery aid must never break the encode

    def _mp4_tag(self) -> str:
        return {"hevc": "hvc1", "h265": "hvc1", "h264": "avc1",
                "avc": "avc1", "av1": "av01"}.get(self.codec, "hvc1")

    def _write_recovery_script(self) -> None:
        """Write a complete PowerShell recovery script next to the .venc
        temp file. Unlike the NVENC path there is no raw elementary stream
        to wrap -- the fragmented mp4 already carries timestamps -- so this
        is a single remux: video from the .venc, audio from the source,
        -shortest trimming the audio to the partial video. Test-Path guard
        falls back to a video-only rewrap if the source moved. UTF-8 BOM
        for Windows PowerShell 5.1 (see the NVENC twin)."""
        script_path = str(Path(self.output_path).with_suffix("")) + \
            "-RECOVER.ps1"
        out = Path(self.output_path)
        fixed = str(out.with_suffix("")) + "-FIXED" + out.suffix
        tag_args = f"-tag:v {self._mp4_tag()} "
        ts_args = "-video_track_timescale 90000 "
        fs_args = "-movflags +faststart " if self.mp4_faststart else ""

        has_source = bool(self.mux_audio and self.input_path)
        src_line = (
            f"$source = {_ps_quote(self.input_path)}\n" if has_source else ""
        )
        vid_only_cmd = (
            f"& ffmpeg -hide_banner -y -i $venc -map 0:v:0 -c:v copy "
            f"{tag_args}{fs_args}{ts_args}$fixed"
        )

        header = (
            "# ChitraMaya encode-recovery script (auto-generated).\n"
            "# If this file still exists, the run did NOT finalize the output\n"
            "# (crash, out-of-memory, or forced close). Every encoded frame is\n"
            "# preserved in the .venc temp file next to this script. Just RUN\n"
            "# THIS SCRIPT in PowerShell: it finalizes the video AND re-adds\n"
            "# the audio from the source file automatically.\n"
            "# ffmpeg must be on PATH.\n"
            "\n"
            f"$venc   = {_ps_quote(self._venc_path)}\n"
            f"{src_line}"
            f"$fixed  = {_ps_quote(fixed)}\n"
            "\n"
        )
        if has_source:
            body = header + (
                "# The video is partial (the run died mid-encode), so\n"
                "# -shortest trims the full-length audio to match. The tiny\n"
                "# source A/V start offset (usually < 50 ms) is not restored\n"
                "# here; a normal completed run does that.\n"
                "$done = $false\n"
                "if (Test-Path -LiteralPath $source) {\n"
                f"    & ffmpeg -hide_banner -y -i $source -i $venc "
                f"-map 1:v:0 -c:v copy {tag_args}-map 0:a? -c:a copy "
                f"-shortest {fs_args}{ts_args}$fixed\n"
                "    if ($LASTEXITCODE -eq 0) {\n"
                "        $done = $true\n"
                "        Write-Host \"Recovered WITH audio: $fixed\"\n"
                "    } else {\n"
                "        Write-Host 'Audio mux failed; producing a "
                "video-only recovery instead.'\n"
                "    }\n"
                "} else {\n"
                "    Write-Host 'Source file not found -- producing a "
                "VIDEO-ONLY recovery.'\n"
                "    Write-Host 'If the source moved, edit the $source "
                "line at the top of this script and re-run.'\n"
                "}\n"
                "if (-not $done) {\n"
                f"    {vid_only_cmd}\n"
                "    if ($LASTEXITCODE -eq 0) { Write-Host \"Recovered "
                "(video only): $fixed\" }\n"
                "}\n"
                "Write-Host 'Once the *-FIXED file plays correctly, the "
                ".venc file and this script can be deleted.'\n"
            )
        else:
            body = header + (
                f"{vid_only_cmd}\n"
                "if ($LASTEXITCODE -eq 0) { Write-Host \"Recovered: "
                "$fixed\" }\n"
                "Write-Host 'Once the *-FIXED file plays correctly, the "
                ".venc file and this script can be deleted.'\n"
            )
        with open(script_path, "w", encoding="utf-8-sig", newline="\r\n") as f:
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

    # -- probe --------------------------------------------------------------

    def _probe_encoder(self, enc_name: str) -> bool:
        """2-frame null encode: proves the encoder initializes on THIS
        machine (presence in -encoders does not imply a working driver)."""
        try:
            r = subprocess.run(
                [self.ffmpeg_path, "-hide_banner", "-v", "error",
                 "-f", "lavfi", "-i", "color=c=black:s=256x128:r=30:d=0.1",
                 "-c:v", enc_name, "-f", "null", "-"],
                capture_output=True, timeout=30, **NOWINDOW,
            )
            return r.returncode == 0
        except Exception:
            return False

    # -- streaming ----------------------------------------------------------

    def encode_frame(self, frame: "torch.Tensor") -> None:
        """BGRA HWC4 uint8 tensor, any device (cuda/xpu/cpu)."""
        if self._closed or self._proc is None:
            return
        if frame.device.type != "cpu":
            from chitramaya.device import sync as _dev_sync
            _dev_sync(frame.device)
            frame = frame.to("cpu")
        try:
            self._proc.stdin.write(frame.contiguous().numpy().tobytes())
        except (BrokenPipeError, OSError):
            raise RuntimeError(
                f"ffmpeg encoder died mid-run ({self._enc_name}). "
                f"Last stderr: {self._stderr_tail()}"
            ) from None
        self._frames_encoded += 1

    def flush(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            rc = self._proc.wait(timeout=900)
            if rc != 0:
                print(f"[Encoder] WARNING: ffmpeg exited rc={rc}: "
                      f"{self._stderr_tail()}")
        except subprocess.TimeoutExpired:
            self._proc.kill()
            print("[Encoder] WARNING: ffmpeg flush timed out; killed.")
        finally:
            try:
                self._stderr_f.close()
            except Exception:
                pass
        print(f"[Encoder] Flushed ({self._frames_encoded} frames)")

    def _stderr_tail(self, n: int = 400) -> str:
        try:
            with open(self._stderr_path, "r", encoding="utf-8",
                      errors="replace") as f:
                return f.read()[-n:].strip() or "(empty)"
        except Exception:
            return "(stderr unavailable)"

    # -- finalize -----------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.flush()

        venc = Path(self._venc_path)
        if not venc.exists() or self._frames_encoded == 0:
            print(f"[Encoder] Nothing to finalize ({self._frames_encoded} frames).")
            self._remove_recovery_script()   # nothing recoverable either
            return

        tag = {"hevc": "hvc1", "h265": "hvc1", "h264": "avc1",
               "avc": "avc1", "av1": "av01"}.get(self.codec, "hvc1")

        # A/V start-offset handling: identical intent to the NVENC path.
        video_delay = 0.0
        audio_delay = 0.0
        has_audio = bool(self.mux_audio and self.input_path
                         and Path(self.input_path).exists())
        if has_audio:
            ffprobe = _derive_ffprobe(self.ffmpeg_path)
            v_start = _probe_stream_start_seconds(ffprobe, self.input_path, "v:0")
            a_start = _probe_stream_start_seconds(ffprobe, self.input_path, "a:0")
            if v_start is not None and a_start is not None:
                rel = v_start - a_start
                if rel > 1e-4:
                    video_delay = rel
                elif rel < -1e-4:
                    audio_delay = -rel
        try:
            _head = float(getattr(self, "_head_skip_seconds", 0.0) or 0.0)
        except Exception:
            _head = 0.0
        if _head > 0:
            video_delay += _head
            print(f"[Encoder] Head-skip compensation: +{_head:.3f}s video delay.")
        duration = video_delay + (self._frames_encoded / self.fps
                                  if self.fps > 0 else 0)

        cmd = [self.ffmpeg_path, "-hide_banner", "-y", "-loglevel", "warning"]
        if has_audio:
            if audio_delay:
                cmd += ["-itsoffset", f"{audio_delay:.6f}"]
            cmd += ["-i", self.input_path]
            if video_delay:
                cmd += ["-itsoffset", f"{video_delay:.6f}"]
            cmd += ["-i", str(venc),
                    "-map", "1:v:0", "-c:v", "copy", "-tag:v", tag,
                    "-map", "0:a?", "-c:a", "copy"]
        else:
            cmd += ["-i", str(venc),
                    "-map", "0:v:0", "-c:v", "copy", "-tag:v", tag]
        if self.mp4_faststart:
            cmd += ["-movflags", "+faststart"]
        cmd += ["-video_track_timescale", "90000"]
        if duration > 0:
            cmd += ["-t", f"{duration:.3f}"]
        cmd += [self.output_path]

        print(f"[Encoder] remux: {' '.join(cmd)}")
        # v1.50.00: size-scaled timeout (the 50GB Idol lesson -- faststart
        # rewrites the whole output a second time; flat 900s killed a remux
        # that was still making disk progress).
        _t_s = _finalize_timeout_s(venc)
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               encoding="utf-8", errors="replace",
                               timeout=_t_s, **NOWINDOW)
            ok = (r.returncode == 0)
            if not ok:
                for line in (r.stderr or "").strip().split("\n")[-5:]:
                    print(f"  {line}")
        except subprocess.TimeoutExpired:
            ok = False
            print(f"[Encoder] remux TIMED OUT after {_t_s}s with no result.")
        except Exception as e:
            ok = False
            print(f"[Encoder] remux error: {e}")

        if ok:
            print("[Encoder] remux OK")
            try:
                venc.unlink()
                Path(self._stderr_path).unlink(missing_ok=True)
            except Exception:
                pass
            self._remove_recovery_script()
            print(f"[Encoder] Done: {self.output_path}")
        else:
            print(f"[Encoder] *** REMUX FAILED -- {self.output_path} is NOT "
                  f"complete. Video-only stream preserved at: {venc} "
                  f"(fragmented mp4; playable).")
            if self._recovery_script_path:
                print(f"[Encoder] Recover now -- run the ready-made script "
                      f"(it re-adds the audio automatically): "
                      f"{self._recovery_script_path}")
            else:
                print(f"[Encoder] Re-wrap manually with: "
                      f'ffmpeg -i "{venc}" -c copy -movflags +faststart out.mp4')


# (Encoder, FfmpegEncoder, nvenc_available are imported explicitly by the
# pipeline; no __all__ needed.)
