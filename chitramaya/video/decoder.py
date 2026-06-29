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

import torch
import PyNvVideoCodec as nvc


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


class Decoder:
    """
    GPU-first video decoder.

    - Primary backend: PyNvVideoCodec ThreadedDecoder (NVDEC) outputting RGBP (planar) or RGB (packed) in device memory.
    - Fallback backend: ffmpeg CPU decode to raw NV12 (default) or RGB24 (env override).

    Extra:
    - ffmpeg_input_args can be provided to tune CPU fallback (must not include -i).
    """

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

        # Init backend
        skip_nvdec, skip_reason = self._should_skip_nvdec_preflight()

        if self._force_cpu:
            self.backend = "ffmpeg-cpu"
            self._init_ffmpeg_cpu_backend()
        elif skip_nvdec:
            print(f"[Decoder] Preflight: {skip_reason}; using ffmpeg CPU decode.")
            self.backend = "ffmpeg-cpu"
            self._init_ffmpeg_cpu_backend()
        else:
            try:
                out_color = nvc.OutputColorType.RGBP
                if self.output_format == "RGB":
                    out_color = nvc.OutputColorType.RGB

                self._decoder = nvc.ThreadedDecoder(
                    enc_file_path=self.input_path,
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
            return

        for attr in ("_decoder", "_demuxer", "_reader", "_ctx", "_stream"):
            if hasattr(self, attr):
                try:
                    setattr(self, attr, None)
                except Exception:
                    pass

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
        try:
            p = frame.pts()
            if p is None:
                return None
            if hasattr(p, "value"):
                return int(p.value)
            return int(p)
        except Exception:
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

    def _ffprobe(self) -> VideoMetadata:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,codec_name,bit_rate,nb_frames",
            "-show_entries", "format=duration",
            "-of", "json",
            self.input_path,
        ]
        # ffprobe -of json always emits UTF-8. Without encoding=, Python falls
        # back to the system codepage (cp1252 on Windows) which crashes on any
        # non-ASCII filename or metadata. errors='replace' is defensive against
        # any malformed bytes in stderr.
        p = subprocess.run(cmd, capture_output=True, text=True,
                           encoding='utf-8', errors='replace')
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

        num_frames = 0
        try:
            if s.get("nb_frames"):
                num_frames = int(s["nb_frames"])
        except Exception:
            num_frames = 0

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
            print(f"[Decoder] Backend: ffmpeg-cpu  output=RGB24(HWC,u8)  {w}x{h}  (GR_CPU_DECODE_RGB24=1)")
        else:
            self._ffmpeg_frame_size = w * h * 3 // 2
            self.ffmpeg_pix_fmt = "nv12"
            print(f"[Decoder] Backend: ffmpeg-cpu  output=NV12(Y+UV,u8)  {w}x{h}")

        self.output_pix_fmt = self.ffmpeg_pix_fmt

        extra = self.ffmpeg_input_args.strip()
        extra_tokens: List[str] = []
        if extra:
            extra_tokens = shlex.split(extra)
            for t in extra_tokens:
                if t == "-i" or t.startswith("-i"):
                    raise ValueError("dec-ffmpeg-input-args must not include -i")

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-fflags", "+genpts",
            *extra_tokens,
            "-i", self.input_path,
            "-an", "-sn", "-dn",
            "-vsync", "0",
            "-f", "rawvideo",
            "-pix_fmt", self.ffmpeg_pix_fmt,
            "pipe:1",
        ]
        self._ffmpeg_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=10**8,
        )

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
