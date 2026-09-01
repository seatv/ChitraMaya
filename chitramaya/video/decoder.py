# gRestorer/video/decoder.py
# --------------------------------------------------------------------------
# CHANGES vs original:
#   [CHANGE 4] Added per-frame PTS extraction in read_batch()
#   [CHANGE 4] New method: read_batch_with_pts() returns (frames, pts_list)
#   [CHANGE 4] ffmpeg CPU path: PTS estimated from frame count + fps
# --------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, List, Optional, Tuple

import json
import os
import shlex
import subprocess
from chitramaya.winproc import NOWINDOW

import torch

# CM-093 X1: PyNvVideoCodec is NVIDIA-only and absent on the Intel Arc /
# XPU build. Guarded import so this module (and everything that imports
# it) loads anywhere; the backend selection below routes to the existing
# ffmpeg CPU decode when nvc is unavailable.
try:
    import PyNvVideoCodec as nvc
except Exception:  # ImportError normally; any failure means no NVDEC
    nvc = None


@dataclass
class VideoMetadata:
    width: int
    height: int
    bit_depth: int
    num_frames: int
    fps: Optional[float]
    duration: Optional[float]
    bitrate: Optional[float]
    codec_name: Optional[str]
    # CM-120r2: the video stream's start_time (seconds) from ffprobe.
    # Used to measure decoder HEAD SKIP: stream captures often start
    # mid-GOP, and the hardware decode path silently discards the
    # reference-broken head frames -- the first frame we ever see can be
    # a second or more into the source. Comparing the first delivered
    # frame's PTS against this start time quantifies the skip so the
    # remux can re-delay the video to match the untouched audio.
    start_time: Optional[float] = None


class Decoder:
    """
    GPU-first video decoder.

    - Primary backend: PyNvVideoCodec ThreadedDecoder (NVDEC) outputting RGBP (planar) or RGB (packed) in device memory.
    - Fallback backend: ffmpeg decode to raw NV12 (default) or RGB24 (env override).
      CM-093 X2: the ffmpeg path probes hardware decode first (QSV, then
      D3D11VA) and falls back to CPU decode -- see _probe_hwaccel. Control
      with CM_HW_DECODE = auto (default) | qsv | d3d11va | off.

    Extra:
    - ffmpeg_input_args can be provided to tune the ffmpeg fallback (must not include -i).
    """

    # CM-093 X2: per-file hwaccel probe results, so interactive paths
    # (Test Frame / FOI previews) don't re-pay the probe on every open.
    _HWACCEL_CACHE: dict = {}

    def __init__(
        self,
        input_path: str,
        gpu_id: int = 0,
        batch_size: int = 80,
        trim_negative_pts: bool = True,
        output_format: str = "RGBP",          # "RGBP" or "RGB"
        ffmpeg_input_args: str = "",          # injected BEFORE -i (CPU fallback)
    ) -> None:
        self.input_path = str(Path(input_path))
        self.gpu_id = int(gpu_id)
        self.batch_size = int(batch_size)
        self.output_format = str(output_format or "RGBP").upper()
        self.ffmpeg_input_args = str(ffmpeg_input_args or "")

        # Probe once up front
        self._probe_meta: VideoMetadata | None = None
        try:
            self._probe_meta = self._ffprobe()
        except Exception:
            self._probe_meta = None

        # Escape hatch (force CPU decode)
        self._force_cpu = os.environ.get("GR_FORCE_CPU_DECODE", "").strip() in ("1", "true", "True", "YES", "yes")

        self.backend: str = "nvdec"
        self.metadata: VideoMetadata

        self._decoder: Any = None  # NVDEC decoder
        self._ffmpeg_proc: subprocess.Popen | None = None
        self._ffmpeg_frame_size: int = 0

        self._raw_num_frames: int = 0
        self._frames_read: int = 0
        self._prefetch: List[Any] = []
        self._trim_prefix: int = 0
        self._trim_negative_pts: bool = bool(trim_negative_pts)

        # [CHANGE 4] PTS tracking for prefetched frames
        self._prefetch_pts: List[Optional[int]] = []

        # CM-120r3: MPEG-TS auto-remux. PyNvVideoCodec's own demuxer
        # mishandles TS stream captures two ways (field-proven 2026-08-27
        # on two files): it silently DROPS ~5s of frames (head + interior
        # -- reference-broken frames at capture start and rip seams), and
        # it SYNTHESIZES uniform frame timestamps, hiding the loss from
        # every timestamp-based safety net. The same bitstreams remuxed
        # losslessly into MP4 decode COMPLETELY (229,812/229,812) with
        # REAL timestamps. So: when the NVDEC path is about to open an
        # MPEG-TS container, stream-copy it to a temporary MP4 first
        # (seconds, no quality change) and decode that instead. The
        # original path still feeds the audio mux; the temp is deleted
        # in close().
        self._container_format: str = str(getattr(self, "_container_format", "") or "")
        self._decode_path: str = self.input_path
        self._ts_remux_path: Optional[str] = None

        # Init backend
        skip_nvdec, skip_reason = self._should_skip_nvdec_preflight()

        if self._force_cpu:
            self.backend = "ffmpeg-cpu"
            self._init_ffmpeg_cpu_backend()
        elif nvc is None:
            # CM-093 X1: non-NVIDIA build (e.g. Intel Arc). The ffmpeg CPU
            # fallback is a first-class backend here until X2 brings an
            # accelerated path.
            print("[Decoder] PyNvVideoCodec not available (non-NVIDIA build); "
                  "using ffmpeg decode (hw decode probed at open).")
            self.backend = "ffmpeg-cpu"
            self._init_ffmpeg_cpu_backend()
        elif skip_nvdec:
            print(f"[Decoder] Preflight: {skip_reason}; using ffmpeg decode (hw decode probed at open).")
            self.backend = "ffmpeg-cpu"
            self._init_ffmpeg_cpu_backend()
        else:
            if "mpegts" in (self._container_format or ""):
                self._prepare_ts_remux()
            try:
                out_color = nvc.OutputColorType.RGBP
                if self.output_format == "RGB":
                    out_color = nvc.OutputColorType.RGB

                self._decoder = nvc.ThreadedDecoder(
                    enc_file_path=self._decode_path,
                    buffer_size=self._threaded_buffer_size(),
                    gpu_id=self.gpu_id,
                    output_color_type=out_color,
                    use_device_memory=True,
                    need_scanned_stream_metadata=True,
                )
                self.backend = "nvdec"
            except Exception as e:
                if self._looks_like_nvdec_unsupported(e):
                    print(
                        f"[Decoder] NVDEC unsupported for this stream on GPU {self.gpu_id}; "
                        f"falling back to ffmpeg CPU decode. ({e})"
                    )
                    self.backend = "ffmpeg-cpu"
                    self._decoder = None
                    self._init_ffmpeg_cpu_backend()
                else:
                    raise
        # Extract metadata
        if self.backend == "nvdec":
            meta = None
            try:
                meta = self._decoder.get_scanned_stream_metadata()
            except Exception:
                meta = self._decoder.get_stream_metadata()
            self.metadata = VideoMetadata(
                width=int(getattr(meta, "width", 0) or 0),
                height=int(getattr(meta, "height", 0) or 0),
                bit_depth=int(getattr(meta, "bit_depth", 8) or 8),
                num_frames=int(getattr(meta, "num_frames", 0) or 0),
                fps=float(getattr(meta, "average_fps", getattr(meta, "fps", 0)) or 0) or None,
                duration=getattr(meta, "duration_in_seconds", None),
                bitrate=float(getattr(meta, "bitrate", 0) or 0) or None,
                codec_name=getattr(meta, "codec_name", None),
            )
            if self._probe_meta:
                pm = self._probe_meta
                if not self.metadata.width:
                    self.metadata.width = pm.width
                if not self.metadata.height:
                    self.metadata.height = pm.height
                if not self.metadata.fps:
                    self.metadata.fps = pm.fps
                if not self.metadata.duration:
                    self.metadata.duration = pm.duration
                if not self.metadata.bitrate:
                    self.metadata.bitrate = pm.bitrate
                if not self.metadata.codec_name:
                    self.metadata.codec_name = pm.codec_name
                if not self.metadata.num_frames:
                    self.metadata.num_frames = pm.num_frames
                if self.metadata.start_time is None:
                    self.metadata.start_time = pm.start_time
        else:
            # _init_ffmpeg_cpu_backend() fills self.metadata
            pass

        self._raw_num_frames = int(self.metadata.num_frames or 0)

        # Optional: trim negative-PTS preroll for NVDEC backend
        if self._trim_negative_pts and self.backend == "nvdec":
            try:
                self._prime_to_first_nonneg_pts()
            except Exception as e:
                if self._looks_like_nvdec_unsupported(e):
                    print(
                        f"[Decoder] NVDEC failed during preroll trim; falling back to ffmpeg CPU decode. ({e})"
                    )
                    self.backend = "ffmpeg-cpu"
                    self._decoder = None
                    self._init_ffmpeg_cpu_backend()
                    self._raw_num_frames = int(self.metadata.num_frames or 0)
                else:
                    raise

        fps_s = f"{self.metadata.fps:.2f}" if self.metadata.fps else "?"
        nf_s = str(self.metadata.num_frames) if self.metadata.num_frames else "?"
        print(f"[Decoder] Initialized ({self.backend}): {self.metadata.width}x{self.metadata.height}, {nf_s} frames, {fps_s} fps")
        if self.backend == "nvdec":
            print(f"[Decoder] Output: {self.output_format} on GPU {self.gpu_id} (ThreadedDecoder, buffer={self._threaded_buffer_size()})")
        else:
            print(f"[Decoder] Output: {self.output_pix_fmt} on CPU (ffmpeg)")

    @property
    def num_frames(self) -> int:
        if self._raw_num_frames <= 0:
            return 0
        return max(0, self._raw_num_frames - self._trim_prefix)

    def is_complete(self) -> bool:
        if self.backend != "nvdec":
            return self._ffmpeg_proc is None
        if self._raw_num_frames <= 0:
            return False
        return self._frames_read >= self._raw_num_frames

    def read_batch(self) -> List[Any]:
        """Original API: returns list of frames (surfaces or tensors)."""
        if self.backend != "nvdec":
            n = self.batch_size
            out: List[torch.Tensor] = []
            for _ in range(n):
                fr = self._ffmpeg_read_frame()
                if fr is None:
                    self.close()
                    break
                self._frames_read += 1
                out.append(fr)
            return out

        n = self.batch_size
        if self._raw_num_frames > 0:
            remaining_raw = self._raw_num_frames - self._frames_read
            if remaining_raw <= 0 and not self._prefetch:
                return []
            if remaining_raw > 0:
                n = min(n, remaining_raw)

        if self._prefetch:
            frames = self._prefetch
            self._prefetch = []
            self._prefetch_pts = []                    # [CHANGE 4]
            if len(frames) > n:
                out = frames[:n]
                self._prefetch = frames[n:]
                return out
            return frames

        frames = self._decoder.get_batch_frames(n)
        if not frames:
            return []
        self._frames_read += len(frames)
        return frames

    # [CHANGE 4] -------------------------------------------------------
    def read_batch_with_pts(self) -> Tuple[List[Any], List[Optional[int]]]:
        """Enhanced API: returns (frames, pts_list) where pts_list[i] is the
        PTS of frames[i] (nanoseconds), or None if unavailable.

        For NVDEC: extracts PTS via frame.pts()
        For ffmpeg-cpu: synthesizes PTS from frame count and fps.
        """
        if self.backend != "nvdec":
            n = self.batch_size
            out_frames: List[torch.Tensor] = []
            out_pts: List[Optional[int]] = []
            fps = float(self.metadata.fps or 30.0) or 30.0
            for _ in range(n):
                fr = self._ffmpeg_read_frame()
                if fr is None:
                    self.close()
                    break
                # Synthesize PTS from frame count (nanoseconds)
                synth_pts = int(self._frames_read * (1_000_000_000.0 / fps))
                self._frames_read += 1
                out_frames.append(fr)
                out_pts.append(synth_pts)
            return out_frames, out_pts

        # NVDEC path
        n = self.batch_size
        if self._raw_num_frames > 0:
            remaining_raw = self._raw_num_frames - self._frames_read
            if remaining_raw <= 0 and not self._prefetch:
                return [], []
            if remaining_raw > 0:
                n = min(n, remaining_raw)

        if self._prefetch:
            frames = self._prefetch
            pts_list = self._prefetch_pts
            self._prefetch = []
            self._prefetch_pts = []
            if len(frames) > n:
                out_f = frames[:n]
                out_p = pts_list[:n] if pts_list else [None] * n
                self._prefetch = frames[n:]
                self._prefetch_pts = pts_list[n:] if pts_list else []
                return out_f, out_p
            # Pad PTS if needed
            while len(pts_list) < len(frames):
                pts_list.append(None)
            return frames, pts_list

        frames = self._decoder.get_batch_frames(n)
        if not frames:
            return [], []
        self._frames_read += len(frames)

        # Extract PTS from each frame
        pts_list = []
        for fr in frames:
            pts_list.append(self._frame_pts(fr))

        return frames, pts_list
    # -------------------------------------------------------------------

    def close(self) -> None:
        if self.backend != "nvdec":
            try:
                if self._ffmpeg_proc is not None:
                    try:
                        if self._ffmpeg_proc.stdout:
                            self._ffmpeg_proc.stdout.close()
                    except Exception:
                        pass
                    try:
                        if self._ffmpeg_proc.stderr:
                            self._ffmpeg_proc.stderr.close()
                    except Exception:
                        pass
                    try:
                        self._ffmpeg_proc.terminate()
                    except Exception:
                        pass
                    try:
                        self._ffmpeg_proc.wait(timeout=2)
                    except Exception:
                        pass
            finally:
                self._ffmpeg_proc = None
            self._cleanup_ts_remux()
            return

        for attr in ("_decoder", "_demuxer", "_reader", "_ctx", "_stream"):
            if hasattr(self, attr):
                try:
                    setattr(self, attr, None)
                except Exception:
                    pass
        self._cleanup_ts_remux()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _threaded_buffer_size(self) -> int:
        env = os.environ.get("GR_NVDEC_BUFFER_SIZE", "").strip()
        if env:
            try:
                v = int(env)
                if v > 0:
                    return v
            except Exception:
                pass

        # ThreadedDecoder buffer_size is a prefetch queue depth, not a consumer batch size.
        # Keep it modest by default to avoid excessive GPU memory use on 4K streams while
        # still giving the background decoder room to stay ahead of inference.
        return max(8, min(max(self.batch_size, 16), 32))

    @staticmethod
    def _looks_like_nvdec_unsupported(e: Exception) -> bool:
        msg = str(e)
        return (
            ("Resolution not supported" in msg)
            or ("Error code : 801" in msg)
            or ("PyNvVCExceptionUnsupported" in msg)
        )

    def _frame_pts(self, frame: Any) -> Optional[int]:
        """Per-frame presentation timestamp, across PyNvVideoCodec API
        generations.

        CM-120 (field, 2026-08-26): PyNvVideoCodec 2.2 renamed the decoded
        frame's time accessor -- DecodedFrame now exposes a ``timestamp``
        field (visible in its repr) and ``frame.pts()`` raises
        AttributeError. Because this method swallowed every exception, the
        2.2 wheel bump (v1.60) silently returned None for EVERY frame,
        which blinded the whole PTS safety net downstream: no timecodes,
        no VFR warning, no visibility when NVDEC dropped 150 frames of a
        TS-flavored stream rip and the A/V sync shifted by 5 seconds.
        Try the 2.2 name first, then the legacy ones; a float value that
        looks like seconds is scaled to nanoseconds so the downstream
        ns-based math (preroll trim, timecodes, gap detection) keeps
        working whatever the wheel emits.
        """
        for name in ("timestamp", "pts"):
            try:
                v = getattr(frame, name)
            except AttributeError:
                continue
            try:
                if callable(v):
                    v = v()
                if v is None:
                    continue
                if hasattr(v, "value"):
                    v = v.value
                if isinstance(v, float):
                    # Heuristic: a float under ~11.5 days is seconds, not
                    # ns/ticks -- scale to the nanoseconds the pipeline
                    # assumes. Integer values pass through untouched.
                    if abs(v) < 1_000_000.0:
                        return int(round(v * 1_000_000_000.0))
                    return int(round(v))
                return int(v)
            except Exception:
                continue
        return None

    def _prime_to_first_nonneg_pts(self) -> None:
        if self._raw_num_frames <= 0:
            return

        scan_batch = max(8, min(128, self.batch_size))
        while True:
            remaining = self._raw_num_frames - self._frames_read
            if remaining <= 0:
                return

            n = min(scan_batch, remaining)
            frames = self._decoder.get_batch_frames(n)
            if not frames:
                return

            self._frames_read += len(frames)

            first_ok = None
            for i, fr in enumerate(frames):
                pts = self._frame_pts(fr)
                if pts is None:
                    self._prefetch = frames
                    self._prefetch_pts = [self._frame_pts(f) for f in frames]  # [CHANGE 4]
                    return
                if pts >= 0:
                    first_ok = i
                    break

            if first_ok is None:
                self._trim_prefix += len(frames)
                continue

            self._trim_prefix += first_ok
            self._prefetch = frames[first_ok:]
            self._prefetch_pts = [self._frame_pts(f) for f in frames[first_ok:]]  # [CHANGE 4]

            if self._trim_prefix > 0:
                presented = max(0, self._raw_num_frames - self._trim_prefix)
                print(f"[Decoder] Trimmed {self._trim_prefix} negative-PTS preroll frames (presented={presented})")
            return

    def _prepare_ts_remux(self) -> None:
        """CM-120r3: losslessly remux an MPEG-TS source to a temp MP4 and
        decode that instead (see the field notes at the top of __init__).

        Stream copy only -- video and audio bytes are untouched; the TS
        packaging (which PyNvVideoCodec's demuxer mishandles) is replaced
        with MP4. Cost: seconds at disk speed, ~source-sized temp file,
        deleted in close(). On ANY failure the original file is decoded
        as before -- the run must never break because a remux could not.
        """
        import tempfile
        import time as _time

        src = Path(self.input_path)
        stem = src.stem
        # CM-127 (field 2026-08-29): the write probe used the full source
        # stem, so a long filename pushed the probe path past Windows'
        # 260-char limit and the remux was silently skipped ("no writable
        # location") -- re-exposing the CM-120 head-drop on exactly the
        # files streaming sites name most verbosely. The probe needs no
        # stem at all, and the cache name shortens (stable hash suffix, so
        # reuse still works) when the stem is long.
        if len(stem) > 60:
            import hashlib
            _h = hashlib.md5(stem.encode("utf-8", "surrogatepass")).hexdigest()[:8]
            cache_stem = stem[:51] + "~" + _h
        else:
            cache_stem = stem
        candidates = [src.parent, Path(tempfile.gettempdir())]
        tmp_path: Optional[Path] = None
        for d in candidates:
            try:
                probe = d / "cm120.writetest.tmp"
                with open(probe, "wb") as _f:
                    _f.write(b"x")
                probe.unlink()
                tmp_path = d / (cache_stem + ".cm120-tsremux.mp4")
                break
            except Exception:
                continue
        if tmp_path is None:
            print("[Decoder] MPEG-TS remux skipped: no writable location; "
                  "decoding the TS directly (frames may be dropped -- CM-120).")
            return

        # Reuse a fresh leftover (crashed prior run / repeated Test Frames).
        try:
            if (tmp_path.exists() and tmp_path.stat().st_size > 0
                    and tmp_path.stat().st_mtime >= src.stat().st_mtime):
                print(f"[Decoder] MPEG-TS remux: reusing {tmp_path.name}")
                self._ts_remux_path = str(tmp_path)
                self._decode_path = str(tmp_path)
                try:
                    self._probe_meta = self._ffprobe(str(tmp_path))
                except Exception:
                    pass
                return
        except Exception:
            pass

        print("[Decoder] MPEG-TS source detected: PyNvVideoCodec's demuxer "
              "drops frames and synthesizes timestamps on TS captures "
              "(CM-120). Losslessly remuxing to a temporary MP4 first "
              "(stream copy; no quality change)...")
        # CM-125 (field 2026-08-28, Austin power blip): write to a .part
        # name and atomic-rename on success. Writing the final name
        # directly meant a power cut / hard kill mid-remux left a
        # TRUNCATED file with a fresh mtime -- the reuse check above would
        # adopt it, the re-probe below would report the truncated frame
        # count as truth, and the next run would silently process a
        # shortened movie. A .part never matches the reuse name, and
        # os.replace is atomic on the same volume.
        part_path = Path(str(tmp_path) + ".part")
        try:
            part_path.unlink(missing_ok=True)  # stale orphan from a crash
        except Exception:
            pass
        cmd = [
            "ffmpeg", "-hide_banner", "-y", "-loglevel", "error",
            "-i", str(src),
            "-map", "0:v:0", "-map", "0:a?",
            "-c", "copy", "-movflags", "+faststart",
            str(part_path),
        ]
        try:
            sz = src.stat().st_size
        except OSError:
            sz = 0
        timeout_s = max(300, int(sz / (25 * 1024 * 1024)) * 2 + 120)
        t0 = _time.perf_counter()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               encoding="utf-8", errors="replace",
                               timeout=timeout_s, **NOWINDOW)
            if r.returncode != 0:
                tail = (r.stderr or "").strip().splitlines()[-3:]
                print("[Decoder] MPEG-TS remux FAILED (decoding the TS "
                      "directly; frames may be dropped -- CM-120):")
                for ln in tail:
                    print(f"  {ln}")
                try:
                    part_path.unlink(missing_ok=True)
                except Exception:
                    pass
                return
        except Exception as e:
            print(f"[Decoder] MPEG-TS remux error ({e}); decoding the TS "
                  f"directly (frames may be dropped -- CM-120).")
            try:
                part_path.unlink(missing_ok=True)
            except Exception:
                pass
            return

        # CM-125: publish the finished remux under its reusable name only
        # now that ffmpeg exited cleanly.
        try:
            os.replace(str(part_path), str(tmp_path))
        except Exception as e:
            print(f"[Decoder] MPEG-TS remux rename failed ({e}); decoding "
                  f"the TS directly (frames may be dropped -- CM-120).")
            try:
                part_path.unlink(missing_ok=True)
            except Exception:
                pass
            return

        dt = _time.perf_counter() - t0
        try:
            mb = tmp_path.stat().st_size / (1024 * 1024)
        except OSError:
            mb = 0.0
        print(f"[Decoder] MPEG-TS remux ready: {tmp_path.name} "
              f"({mb:.0f} MB in {dt:.1f}s); decoding the remux. The "
              f"original file still supplies the audio at finalize.")
        self._ts_remux_path = str(tmp_path)
        self._decode_path = str(tmp_path)
        # Re-probe the REMUX: its normalized timeline (start times shifted
        # by the earliest stream) is what the decoder will see, so
        # metadata.start_time and num_frames must describe THIS file for
        # the CM-120r2 head-skip math to be consistent.
        try:
            self._probe_meta = self._ffprobe(str(tmp_path))
        except Exception:
            pass

    def _cleanup_ts_remux(self) -> None:
        if self._ts_remux_path:
            try:
                Path(self._ts_remux_path).unlink(missing_ok=True)
            except Exception:
                pass
            self._ts_remux_path = None

    def _ffprobe(self, path: Optional[str] = None) -> VideoMetadata:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,codec_name,bit_rate,nb_frames,start_time",
            "-show_entries", "format=duration,format_name",
            "-of", "json",
            str(path or self.input_path),
        ]
        # ffprobe -of json always emits UTF-8. Without encoding=, Python falls
        # back to the system codepage (cp1252 on Windows) which crashes on any
        # non-ASCII filename or metadata. errors='replace' is defensive against
        # any malformed bytes in stderr.
        p = subprocess.run(cmd, capture_output=True, text=True,
                           encoding='utf-8', errors='replace', **NOWINDOW)
        if p.returncode != 0:
            raise RuntimeError(f"ffprobe failed:\n{p.stderr}")

        j = json.loads(p.stdout or "{}")
        s = (j.get("streams") or [{}])[0]
        f = j.get("format") or {}

        w = int(s.get("width") or 0)
        h = int(s.get("height") or 0)

        fps = None
        afr = s.get("avg_frame_rate")
        if afr and afr != "0/0":
            try:
                fps = float(Fraction(afr))
            except Exception:
                fps = None

        codec = s.get("codec_name") or None

        bitrate = None
        try:
            bitrate = float(s.get("bit_rate")) if s.get("bit_rate") else None
        except Exception:
            bitrate = None

        duration = None
        try:
            duration = float(f.get("duration")) if f.get("duration") else None
        except Exception:
            duration = None

        # CM-120r3: remember the container so __init__ can detect MPEG-TS
        # stream captures (the PyNvVideoCodec failure class).
        try:
            self._container_format = str(f.get("format_name") or "")
        except Exception:
            self._container_format = ""


        num_frames = 0
        try:
            if s.get("nb_frames"):
                num_frames = int(s["nb_frames"])
        except Exception:
            num_frames = 0

        # CM-120r2: video stream start_time (seconds) for head-skip math.
        start_time = None
        try:
            st = s.get("start_time")
            if st is not None and str(st).upper() != "N/A":
                start_time = float(st)
        except Exception:
            start_time = None

        if (not num_frames) and duration and fps:
            num_frames = int(round(duration * fps))

        return VideoMetadata(
            width=w,
            height=h,
            bit_depth=8,
            num_frames=num_frames,
            fps=fps,
            duration=duration,
            bitrate=bitrate,
            codec_name=codec,
            start_time=start_time,
        )

    def _init_ffmpeg_cpu_backend(self) -> None:
        self.metadata = self._probe_meta or self._ffprobe()
        if not self.metadata.width or not self.metadata.height:
            raise RuntimeError("ffprobe did not return width/height; cannot CPU-decode")

        w = int(self.metadata.width)
        h = int(self.metadata.height)

        force_rgb24 = os.getenv("GR_CPU_DECODE_RGB24", "0").strip().lower() in {"1", "true", "yes"}
        if force_rgb24:
            self._ffmpeg_frame_size = w * h * 3
            self.ffmpeg_pix_fmt = "rgb24"
            _out_desc = f"RGB24(HWC,u8)  {w}x{h}  (GR_CPU_DECODE_RGB24=1)"
        else:
            self._ffmpeg_frame_size = w * h * 3 // 2
            self.ffmpeg_pix_fmt = "nv12"
            _out_desc = f"NV12(Y+UV,u8)  {w}x{h}"

        self.output_pix_fmt = self.ffmpeg_pix_fmt

        extra = self.ffmpeg_input_args.strip()
        extra_tokens: List[str] = []
        if extra:
            extra_tokens = shlex.split(extra)
            for t in extra_tokens:
                if t == "-i" or t.startswith("-i"):
                    raise ValueError("dec-ffmpeg-input-args must not include -i")

        # CM-093 X2: hardware-accelerated decode on the ffmpeg path. The
        # probe picks QSV (Intel fixed-function decode) or D3D11VA (any
        # WDDM GPU), else plain CPU decode. Decoded frames are downloaded
        # to system memory by ffmpeg either way, so the stdout byte stream
        # (NV12/RGB24 rawvideo) and everything downstream are IDENTICAL --
        # this changes who does the decoding, not what arrives. The win is
        # CPU relief: 4K AV1/HEVC software decode saturates older CPUs.
        # (X2b: the Backend line prints AFTER the probe so it can tell the
        # truth about who decodes; self.backend stays "ffmpeg-cpu" -- that
        # string is an internal id the pipeline keys prefetch behavior on.)
        hw_tokens = self._probe_hwaccel()
        _hw_desc = ("CPU software" if not hw_tokens
                    else f"{hw_tokens[1]} hardware")
        print(f"[Decoder] Backend: ffmpeg ({_hw_desc} decode)  "
              f"output={_out_desc}")

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            *hw_tokens,
            "-fflags", "+genpts",
            *extra_tokens,
            "-i", self.input_path,
            "-an", "-sn", "-dn",
            "-fps_mode", "passthrough",  # CM-131: -vsync removed in ffmpeg 9; fps_mode exists since 5.1
            "-f", "rawvideo",
            "-pix_fmt", self.ffmpeg_pix_fmt,
            "pipe:1",
        ]
        self._ffmpeg_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=10**8,
            **NOWINDOW,
        )

    def _probe_hwaccel(self) -> List[str]:
        """CM-093 X2: pick ffmpeg -hwaccel tokens for THIS input file.

        Order: qsv -> d3d11va -> none. Each candidate is proven by actually
        decoding 2 frames of the real file (codec/profile support varies per
        file, so a global capability check is not enough).

        CRITICAL probe detail: when -hwaccel init fails, ffmpeg logs
        "Failed setup ..." and SILENTLY continues with software decode,
        exiting 0 -- so a return-code check alone would report hardware
        decode while the CPU quietly does the work. The probe therefore
        runs at -v warning and treats any setup-failure/fallback message in
        stderr as a probe failure.

        CM_HW_DECODE env: auto (default) | qsv | d3d11va (force one
        candidate) | off/0/cpu/none (skip probing, plain CPU decode --
        also the A/B switch for benchmarking).

        Results are cached per input path for the process lifetime.
        """
        env = os.getenv("CM_HW_DECODE", "auto").strip().lower()
        if env in {"0", "off", "cpu", "none", "false"}:
            print("[Decoder] hw decode: disabled (CM_HW_DECODE)")
            return []

        cached = Decoder._HWACCEL_CACHE.get(self.input_path)
        if cached is not None:
            return list(cached)

        # X2b: qsv needs an explicit system-memory output format. ffmpeg 8
        # defaults "-hwaccel qsv" to GPU-RESIDENT frames ("defaulting
        # hwaccel_output_format to qsv"), which collide with the rawvideo
        # download chain -- field result on Arc A580: qsv probe exited -40
        # while d3d11va (which defaults to downloading) passed. Requesting
        # nv12 output forces the decode-then-download path we want.
        candidates = [
            ("qsv", ["-hwaccel", "qsv", "-hwaccel_output_format", "nv12"]),
            ("d3d11va", ["-hwaccel", "d3d11va"]),
        ]
        if env in {"qsv", "d3d11va"}:
            candidates = [c for c in candidates if c[0] == env]

        _FALLBACK_MARKERS = ("failed setup", "falling back", "fall back",
                             "device creation failed", "no device available")
        chosen: List[str] = []
        for name, tokens in candidates:
            # The probe mirrors the PRODUCTION output chain (rawvideo in the
            # same pix_fmt), not a null sink: ffmpeg 8 defaults qsv to
            # GPU-resident frames, which a null sink happily accepts but a
            # rawvideo download path may not. Decoding 2 frames through the
            # real chain proves the whole path, not just decoder init.
            probe_cmd = [
                "ffmpeg", "-hide_banner", "-v", "warning",
                *tokens,
                "-i", self.input_path,
                "-an", "-sn", "-dn",
                "-fps_mode", "passthrough",  # CM-131: -vsync removed in ffmpeg 9; fps_mode exists since 5.1
                "-frames:v", "2",
                "-f", "rawvideo",
                "-pix_fmt", self.ffmpeg_pix_fmt,
                "-y", os.devnull,
            ]
            try:
                r = subprocess.run(
                    probe_cmd, stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE, text=True,
                    timeout=60, **NOWINDOW,
                )
                err = (r.stderr or "").lower()
                soft_fallback = any(m in err for m in _FALLBACK_MARKERS)
                if r.returncode == 0 and not soft_fallback:
                    chosen = list(tokens)
                    print(f"[Decoder] hw decode: {name} "
                          f"(GPU fixed-function decode; frames downloaded "
                          f"to system memory)")
                    break
                reason = ("software-fallback detected" if soft_fallback
                          else f"exit code {r.returncode}")
                print(f"[Decoder] hw decode probe: {name} not usable "
                      f"({reason})")
            except Exception as e:
                print(f"[Decoder] hw decode probe: {name} failed ({e})")

        if not chosen:
            print("[Decoder] hw decode: none usable; CPU decode")
        Decoder._HWACCEL_CACHE[self.input_path] = list(chosen)
        return chosen

    def _ffmpeg_read_frame(self) -> torch.Tensor | None:
        if self._ffmpeg_proc is None or self._ffmpeg_proc.stdout is None:
            return None

        try:
            buf_t = torch.empty((self._ffmpeg_frame_size,), dtype=torch.uint8, pin_memory=True)
        except Exception:
            buf_t = torch.empty((self._ffmpeg_frame_size,), dtype=torch.uint8)

        view = memoryview(buf_t.numpy())
        got = 0
        while got < self._ffmpeg_frame_size:
            n = self._ffmpeg_proc.stdout.readinto(view[got:])
            if not n:
                return None
            got += int(n)

        h = int(self.metadata.height)
        w = int(self.metadata.width)
        if getattr(self, "ffmpeg_pix_fmt", "nv12") == "rgb24":
            return buf_t.view(h, w, 3)
        return buf_t.view(h * 3 // 2, w)

    def _should_skip_nvdec_preflight(self) -> tuple[bool, str]:
        """
        Skip NVDEC entirely for stream classes that crash the native decoder on
        this GPU. This is a PREDICTIVE gate, not an attempt-and-recover path:
        oversize H.264 does not fail gracefully on pre-Blackwell NVDEC -- it
        hard-crashes the process (Windows 0xC0000409 STATUS_STACK_BUFFER_OVERRUN,
        a native fast-fail) on the first decode call, AFTER the decoder appears
        to construct successfully. A native crash cannot be caught from Python,
        so the only safe option is to not attempt it on hardware that crashes.

        Confirmed: RTX 3060 Ti (Ampere, sm_86) crashes on 4320-wide H.264.

        Capability rule:
        - H.264/AVC with width or height > 4096:
            * allow NVDEC only on Blackwell or newer (compute capability major
              >= 12), where NVIDIA Video Codec SDK 13.0 documents H.264 decode
              up to 8192x8192.
            * skip NVDEC (use CPU) on everything older.

        NOTE: allowing the attempt on Blackwell is documented-but-unverified on
        our hardware; the first 5060 run is the real test. If it also crashes,
        that is a clean process exit on an internal card (no eGPU link at risk),
        and this rule should then be tightened to skip there too.
        """
        pm = self._probe_meta
        if pm is None:
            return False, ""

        codec = (pm.codec_name or "").strip().lower()
        width = int(pm.width or 0)
        height = int(pm.height or 0)

        if codec in ("h264", "avc", "avc1") and (width > 4096 or height > 4096):
            cap_major = self._cuda_capability_major()
            if cap_major is not None and cap_major >= 12:
                # Blackwell+: documented support for >4096 H.264 decode. Allow
                # the attempt (the only place we deliberately do so).
                return False, ""
            return True, (
                f"H.264 {width}x{height} > 4096 not supported by NVDEC on this "
                f"GPU (compute capability major={cap_major}); needs Blackwell+ "
                f"(major>=12)"
            )

        return False, ""

    def _cuda_capability_major(self) -> Optional[int]:
        """Compute-capability MAJOR for self.gpu_id, or None if unavailable.

        Reading the capability is a safe property query (no decoder is created),
        so it cannot trigger the native crash. Ampere=8, Ada=8, Blackwell=12.
        """
        try:
            import torch
            if torch.cuda.is_available():
                major, _minor = torch.cuda.get_device_capability(self.gpu_id)
                return int(major)
        except Exception:
            pass
        return None
