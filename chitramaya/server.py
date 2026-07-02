"""
ChitraMaya Web Server — Flask + PyWebView

Thin HTTP adapter over SwapServer. Follows Tilester's architecture:
- Flask bound to 127.0.0.1
- PyWebView wraps the browser window
- Single active long-running operation at a time
- All GPU state lives in SwapServer (persistent across requests)
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import socket
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from flask import Flask, Response, jsonify, render_template, request
from werkzeug.serving import WSGIRequestHandler

logger = logging.getLogger(__name__)

# ── Flask App ─────────────────────────────────────────────────────────────

app = Flask(__name__,
            template_folder=str(Path(__file__).parent / "templates"),
            static_folder=str(Path(__file__).parent / "static"))

# Suppress Flask request logging in production
WSGIRequestHandler.log_request = lambda *args, **kwargs: None

# ── Operation gate (one long-running op at a time) ────────────────────────

_op_lock = threading.Lock()
_op_state: dict[str, Any] | None = None


def _set_op(name: str, **kwargs):
    global _op_state
    _op_state = {"name": name, "started": time.time(), **kwargs}


def _clear_op():
    global _op_state
    _op_state = None


# ── SwapServer (persistent GPU state) ─────────────────────────────────────

class SwapServer:
    """Holds all GPU state: engines, current frame, detections, embeddings.

    Lives for the lifetime of the app. Flask routes call methods on a
    global instance of this class.
    """

    def __init__(self, models_dir: str | Path, gpu_id: int = 0):
        import torch
        torch.set_grad_enabled(False)

        self.models_dir = Path(models_dir)
        self.gpu_id = gpu_id

        # Mosaic restoration pipeline (lazy; rebuilt only when load-affecting
        # config changes). Cached across requests so TRT engines don't reload.
        self._mosaic_pipeline = None
        self._mosaic_pipeline_key: tuple | None = None

        # Video state
        self.video_path: str | None = None
        self.decoder = None
        self.video_info = None

        # Current frame (on GPU)
        self._current_frame = None      # (C, H, W) uint8 CUDA tensor
        self._current_frame_num: int = -1
        self._current_frame_orig_h: int = 0
        self._current_frame_orig_w: int = 0

        # Processing state
        self._cancel_flag = threading.Event()
        self._progress: dict[str, Any] = {}

        # Output
        self.output_dir: str = ""
        self.temp_dir: str = ""
        self.preview_path: str | None = None

        # Load saved folder paths from config
        self._load_ui_config()

    def _load_ui_config(self):
        """Load saved folder paths from unified config file."""
        try:
            config_path = Path.cwd() / "ChitraMaya-config.json"
            if config_path.exists():
                data = json.loads(config_path.read_text(encoding="utf-8"))
                self.faces_dir = data.get("facesDir", data.get("faces_dir", ""))
                self.output_dir = data.get("outputDir", data.get("output_dir", ""))
                self.temp_dir = data.get("tempDir", data.get("temp_dir", ""))
                logger.info("Loaded config: faces=%s output=%s temp=%s",
                            self.faces_dir, self.output_dir, self.temp_dir)
        except Exception as e:
            logger.warning("Failed to load config: %s", e)

    def _save_ui_config(self):
        """Save folder paths to unified config file (merges with existing)."""
        try:
            config_path = Path.cwd() / "ChitraMaya-config.json"
            data = {}
            if config_path.exists():
                data = json.loads(config_path.read_text(encoding="utf-8"))
            data["facesDir"] = self.faces_dir
            data["outputDir"] = self.output_dir
            data["tempDir"] = self.temp_dir
            # Clean up old keys
            for old_key in ("faces_dir", "output_dir", "temp_dir"):
                data.pop(old_key, None)
            config_path.write_text(
                json.dumps(data, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning("Failed to save config: %s", e)

    # ── Video ──

    def load_video(self, path: str) -> dict:
        """Load video — get metadata via ffprobe, serve via /video endpoint.

        Does NOT initialize NVDEC decoder here. The decoder is only
        created when we need GPU frame extraction (detect, swap).
        HTML5 player handles playback directly via /video streaming.
        """
        self.video_path = path

        if self.decoder is not None:
            try:
                self.decoder.close()
            except Exception:
                pass
            self.decoder = None

        # Get metadata via ffprobe (no GPU needed)
        self.video_info = self._probe_video(path)

        # NOTE: output_dir is NOT pinned to the input directory here. The job
        # payload (params.output_dir) is authoritative at submit-time; mosaic_full
        # falls back to self.output_dir, then to the input's parent. Pinning here
        # caused a blank output box to silently write next to the input even after
        # the user set a directory.

        # Clear detection state
        self.detected_kpss = []
        self.detected_embeddings = []
        self.detected_crops = []
        self.assignments = {}

        return {
            "info": self.video_info,
        }

    @staticmethod
    def _probe_video(path: str) -> dict:
        """Get video metadata via ffprobe (no GPU needed)."""
        import subprocess, json as _json
        try:
            cmd = [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_streams", "-show_format",
                str(path),
            ]
            # ffprobe -of json always emits UTF-8. Without encoding=, Python falls
            # back to the system codepage (cp1252 on Windows) which crashes on any
            # non-ASCII filename or metadata.
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10,
                                    encoding='utf-8', errors='replace')
            data = _json.loads(result.stdout)

            # Find video stream
            vstream = None
            for s in data.get("streams", []):
                if s.get("codec_type") == "video":
                    vstream = s
                    break

            if not vstream:
                return {"path": path, "width": 0, "height": 0, "fps": 30.0,
                        "num_frames": 0, "duration": 0}

            # Parse FPS
            fps = 30.0
            r_rate = vstream.get("r_frame_rate", "30/1")
            try:
                num, den = r_rate.split("/")
                fps = float(num) / float(den)
            except Exception:
                pass

            # Parse duration
            duration = float(data.get("format", {}).get("duration", 0))
            num_frames = int(vstream.get("nb_frames", 0))
            if num_frames == 0 and duration > 0 and fps > 0:
                num_frames = int(duration * fps)

            return {
                "path": path,
                "width": int(vstream.get("width", 0)),
                "height": int(vstream.get("height", 0)),
                "fps": fps,
                "num_frames": num_frames,
                "duration": duration,
            }
        except Exception as e:
            logger.warning("ffprobe failed for %s: %s", path, e)
            return {"path": path, "width": 0, "height": 0, "fps": 30.0,
                    "num_frames": 0, "duration": 0}

    def _ensure_decoder(self):
        """Lazily initialize NVDEC decoder (only when GPU frame extraction needed)."""
        if self.decoder is not None:
            return
        if not self.video_path:
            return
        from chitramaya.video.decoder import Decoder
        self.decoder = Decoder(
            input_path=self.video_path,
            gpu_id=self.gpu_id,
            batch_size=1,
            output_format="RGBP",
        )

    def seek(self, frame_num: int) -> str:
        """Seek to frame and return image as base64 JPEG."""
        if not self.video_path:
            return ""
        time_sec = frame_num / (self.video_info.get("fps", 30.0)) if self.video_info else 0
        return self._extract_frame_at_time(time_sec)

    def _extract_frame_at_time(self, time_sec: float) -> str:
        """Extract a single frame at the given time using ffmpeg, load to GPU.

        Returns base64 JPEG of the frame. Caches the frame tensor on GPU
        for subsequent detection/swap operations.
        """
        import torch

        if not self.video_path:
            return ""

        info = self.video_info or {}
        w = info.get("width", 0)
        h = info.get("height", 0)
        if w == 0 or h == 0:
            return ""

        # Use ffmpeg to extract a single RGB24 frame
        import subprocess
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", f"{time_sec:.4f}",
            "-i", self.video_path,
            "-vframes", "1",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}",
            "pipe:1",
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, timeout=10,
            )
            if result.returncode != 0 or len(result.stdout) != w * h * 3:
                logger.warning("ffmpeg frame extraction failed at t=%.3f (got %d bytes, expected %d)",
                               time_sec, len(result.stdout), w * h * 3)
                return ""

            # Load to GPU as CHW uint8 tensor
            frame_np = np.frombuffer(result.stdout, dtype=np.uint8).reshape(h, w, 3)
            frame = torch.from_numpy(frame_np.copy()).to(f'cuda:{self.gpu_id}').permute(2, 0, 1)

            # Cache for detection/swap
            self._current_frame_num = int(time_sec * info.get("fps", 30.0))
            self._current_frame_orig_h = h
            self._current_frame_orig_w = w

            self._current_frame = frame

            return self._tensor_to_b64(frame)

        except Exception as e:
            logger.exception("Frame extraction failed at t=%.3f", time_sec)
            return ""

    # ── Source Faces ──

    # ── Detection ──

    # ── Assignment ──

    # ── Live Preview ──

    # ── Processing ──

    def get_progress(self) -> dict:
        return dict(self._progress)

    def cancel(self):
        self._cancel_flag.set()

    # ── Utility ──

    @staticmethod
    def _tensor_to_b64(tensor, quality: int = 90, max_dim: int = 0) -> str:
        """Convert CHW uint8 CUDA tensor to base64 JPEG string."""
        arr = tensor.permute(1, 2, 0).cpu().numpy()
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        if max_dim > 0:
            h, w = arr.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                arr = cv2.resize(arr, (int(w * scale), int(h * scale)))

        _, buf = cv2.imencode('.jpg', arr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buf).decode('ascii')

    @staticmethod
    def _image_to_b64_thumb(img_rgb: np.ndarray, size: int = 128) -> str:
        """Convert RGB numpy image to base64 JPEG thumbnail."""
        h, w = img_rgb.shape[:2]
        scale = size / max(h, w)
        thumb = cv2.resize(img_rgb, (int(w * scale), int(h * scale)))
        thumb_bgr = cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode('.jpg', thumb_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buf).decode('ascii')

    def reset(self):
        """Reset all state for new project."""
        if self.decoder:
            try:
                self.decoder.close()
            except Exception:
                pass

        # Release mosaic pipeline (GPU VRAM); cheap to recreate if needed.
        if self._mosaic_pipeline is not None:
            try:
                self._mosaic_pipeline.close()
            except Exception:
                pass
            self._mosaic_pipeline = None
            self._mosaic_pipeline_key = None

        self.video_path = None
        self.decoder = None
        self.video_info = None
        self._current_frame = None
        self._current_frame_num = -1
        self.preview_path = None
        self._cancel_flag.clear()
        self._progress = {}

    # ── Mosaic restoration ────────────────────────────────────────────

    def _ensure_mosaic_pipeline(self, mosaic_cfg, encoder_dict: dict | None = None):
        """Lazy/cached mosaic pipeline.

        Rebuilds only when load-affecting config changes (model paths, fp16,
        use_trt, max_clip_size, detect_only, batch_size). Non-load-affecting
        tweaks (detection_score, crossfade, blend_frames, color_match,
        encoder, denoise) are applied to the cached pipeline in place — no
        model reload.
        """
        from chitramaya.mosaic.pipeline import MosaicPipeline

        # If TRT restoration is requested but the requested clip size has no
        # compiled sub-engine set, snap to the nearest AVAILABLE compiled size
        # (prefer the largest that is <= requested; else the smallest available).
        # This prevents a stale UI/config clip size from hard-failing with a
        # FileNotFoundError. When nothing is compiled the request is left
        # unchanged so the build raises the clean "compile first" error.
        if (mosaic_cfg.mosaic_compile_trt and mosaic_cfg.restoration_model
                and not mosaic_cfg.mosaic_detect_only):
            import dataclasses
            from chitramaya.mosaic.models.basicvsrpp.engine_paths import (
                list_basicvsrpp_compiled_clip_sizes,
            )
            avail = list_basicvsrpp_compiled_clip_sizes(
                mosaic_cfg.restoration_model, bool(mosaic_cfg.mosaic_fp16),
            )
            req = int(mosaic_cfg.mosaic_max_clip_size)
            if avail and req not in avail:
                le = [n for n in avail if n <= req]
                snapped = max(le) if le else min(avail)
                logger.warning(
                    "[mosaic] Max clip %d has no compiled TRT engines for %s "
                    "(fp16=%s); snapping to %d (available: %s)",
                    req, mosaic_cfg.restoration_model,
                    mosaic_cfg.mosaic_fp16, snapped, avail,
                )
                mosaic_cfg = dataclasses.replace(
                    mosaic_cfg, mosaic_max_clip_size=snapped,
                )

        key = (
            mosaic_cfg.detection_model,
            mosaic_cfg.restoration_model,
            bool(mosaic_cfg.mosaic_fp16),
            bool(mosaic_cfg.mosaic_compile_trt),
            int(mosaic_cfg.mosaic_max_clip_size),
            bool(mosaic_cfg.mosaic_detect_only),
            int(mosaic_cfg.mosaic_detection_batch_size),
        )
        pipeline_cfg = mosaic_cfg.to_pipeline_config(encoder=encoder_dict)

        if self._mosaic_pipeline is None or self._mosaic_pipeline_key != key:
            if self._mosaic_pipeline is not None:
                logger.info("[mosaic] Config changed — rebuilding pipeline")
                try:
                    self._mosaic_pipeline.close()
                except Exception:
                    logger.exception("Error closing old mosaic pipeline")
            self._mosaic_pipeline = MosaicPipeline(pipeline_cfg, gpu_id=self.gpu_id)
            self._mosaic_pipeline_key = key
            return self._mosaic_pipeline

        # Cached pipeline — apply runtime-only config delta in place.
        self._mosaic_pipeline.config = pipeline_cfg
        d = max(0, int(pipeline_cfg.temporal_overlap))
        if pipeline_cfg.blend_frames < 0:
            bf = (d // 3) if pipeline_cfg.crossfade else 0
        else:
            bf = min(int(pipeline_cfg.blend_frames), d)
        if not pipeline_cfg.crossfade:
            bf = 0
        self._mosaic_pipeline._discard_margin = d
        self._mosaic_pipeline._blend_frames = bf
        # Detector score is read at detect-time from self.score_threshold,
        # so live-mutation is safe.
        if self._mosaic_pipeline._detector is not None:
            self._mosaic_pipeline._detector.score_threshold = pipeline_cfg.detection_score
        return self._mosaic_pipeline

    def _mosaic_progress_cb(self, *, frame_num, total_frames, detections,
                            restorations, fps_win, fps_avg, buffered, mode):
        """Pipeline progress callback that writes into self._progress."""
        if self._cancel_flag.is_set():
            return
        remaining = (total_frames - frame_num) / fps_win if fps_win > 0 else 0
        self._progress.update({
            "frame": frame_num,
            "total": total_frames,
            "fps": round(fps_win, 1),
            "fps_avg": round(fps_avg, 1),
            "eta": f"{int(remaining)}s",
            "detections": int(detections),
            "restorations": int(restorations),
            "buffered": int(buffered),
            "mosaic_mode": str(mode),
        })

    def mosaic_full(self, params: dict) -> str:
        """Process full video with mosaic restoration. Runs in BG thread."""
        from chitramaya.models import MosaicConfig

        if not self.video_path:
            raise RuntimeError("No video loaded")

        mosaic_cfg = MosaicConfig.from_dict(params.get("mosaic", params))
        encoder_dict = params.get("encoder", {})
        if not mosaic_cfg.detection_model:
            raise RuntimeError("Detection model path not set")
        if not mosaic_cfg.restoration_model and not mosaic_cfg.mosaic_detect_only:
            raise RuntimeError("Restoration model path not set")

        # Output path. Priority: payload output_dir (the value in the UI box at
        # submit-time) -> persisted self.output_dir -> input's parent directory.
        payload_out = str(params.get("output_dir", "") or "").strip()
        if payload_out:
            self.output_dir = payload_out
        input_path = Path(self.video_path)
        out_dir = self.output_dir or str(input_path.parent)
        os.makedirs(out_dir, exist_ok=True)
        suffix = "-detect" if mosaic_cfg.mosaic_detect_only else "-restored"
        output_path = str(Path(out_dir) / f"{input_path.stem}{suffix}.mp4")

        info = self.video_info or {}
        total_frames = info.get("num_frames", 0)
        self._cancel_flag.clear()
        self._progress = {
            "status": "processing",
            "frame": 0,
            "total": total_frames,
            "fps": 0,
            "eta": "loading models...",
            "detections": 0,
            "restorations": 0,
            "buffered": 0,
            "output_path": output_path,
            "mosaic_mode": "detect-only" if mosaic_cfg.mosaic_detect_only else "restore",
        }

        pipeline = self._ensure_mosaic_pipeline(mosaic_cfg, encoder_dict=encoder_dict)

        try:
            result = pipeline.process_file(
                self.video_path, output_path,
                progress_cb=self._mosaic_progress_cb,
                use_tqdm=False,
                cancel_flag=self._cancel_flag,
            )
        except Exception as e:
            logger.exception("mosaic_full failed")
            self._progress["status"] = "error"
            self._progress["error"] = str(e)
            return ""

        if self._cancel_flag.is_set():
            self._progress["status"] = "cancelled"
        else:
            self._progress["status"] = "complete"
            self._progress["frame"] = result.frames
            self._progress["detections"] = result.detections
            self._progress["restorations"] = result.restorations
            if result.diag_path:
                self._progress["diag_path"] = result.diag_path

        logger.info("mosaic_full: %d frames, %d det, %d res → %s",
                    result.frames, result.detections, result.restorations, output_path)
        return output_path

    def mosaic_segment(self, start_time: float, end_time: float, params: dict) -> str:
        """Process a video segment for preview. Mirrors swap_segment.

        Extracts via ffmpeg (lossless, NVDEC-compatible), then runs the
        pipeline on the temp file. Output goes to self.preview_path.
        """
        import subprocess
        import tempfile
        from chitramaya.models import MosaicConfig

        if not self.video_path:
            raise RuntimeError("No video loaded")

        seg_duration = end_time - start_time
        if seg_duration < 0.5:
            raise RuntimeError(f"Segment too short ({seg_duration:.1f}s). Minimum is 0.5 seconds.")

        mosaic_cfg = MosaicConfig.from_dict(params.get("mosaic", params))
        encoder_dict = params.get("encoder", {})
        if not mosaic_cfg.detection_model:
            raise RuntimeError("Detection model path not set")
        if not mosaic_cfg.restoration_model and not mosaic_cfg.mosaic_detect_only:
            raise RuntimeError("Restoration model path not set")

        # Temp files. Priority: payload temp_dir -> persisted self.temp_dir ->
        # the OS temp directory.
        payload_temp = str(params.get("temp_dir", "") or "").strip()
        if payload_temp:
            self.temp_dir = payload_temp
        temp_dir = self.temp_dir if self.temp_dir else tempfile.gettempdir()
        os.makedirs(temp_dir, exist_ok=True)
        seg_input = os.path.join(temp_dir, "chitramaya_mosaic_seg_input.mp4")
        self.preview_path = os.path.join(temp_dir, "chitramaya_mosaic_preview.mp4")

        info = self.video_info or {}
        vid_fps = float(info.get("fps", 30.0))

        self._cancel_flag.clear()
        self._progress = {
            "status": "processing",
            "frame": 0,
            "total": int(seg_duration * vid_fps),
            "fps": 0,
            "eta": "extracting...",
            "detections": 0,
            "restorations": 0,
            "buffered": 0,
            "mosaic_mode": "detect-only" if mosaic_cfg.mosaic_detect_only else "restore",
        }

        # ── Phase 1: ffmpeg extract ─────────────────────────────────────
        logger.info("Extracting mosaic segment %.2f → %.2f (%.1fs)",
                    start_time, end_time, seg_duration)
        extract_cmd = [
            "ffmpeg", "-hide_banner", "-y", "-loglevel", "error",
            "-ss", f"{start_time:.3f}",
            "-i", self.video_path,
            "-t", f"{seg_duration:.3f}",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "0",
            "-an",
            seg_input,
        ]
        result_proc = subprocess.run(extract_cmd, capture_output=True, text=True, timeout=120,
                                     encoding='utf-8', errors='replace')
        if result_proc.returncode != 0:
            raise RuntimeError(f"Segment extraction failed: {result_proc.stderr[-200:]}")
        seg_path = Path(seg_input)
        if not seg_path.exists() or seg_path.stat().st_size < 1024:
            raise RuntimeError("Segment extraction produced no output")

        # ── Phase 2: run pipeline on the segment ────────────────────────
        # The pipeline writes diagnostics next to the output unless we
        # disable it for previews (clutter in temp dir). Disable here.
        pipeline = self._ensure_mosaic_pipeline(mosaic_cfg, encoder_dict=encoder_dict)
        original_write_diag = pipeline.config.write_diagnostics
        pipeline.config.write_diagnostics = False
        self._progress["eta"] = "—"

        try:
            result = pipeline.process_file(
                seg_input, self.preview_path,
                progress_cb=self._mosaic_progress_cb,
                use_tqdm=False,
                cancel_flag=self._cancel_flag,
            )
        except Exception as e:
            logger.exception("mosaic_segment failed")
            self._progress["status"] = "error"
            self._progress["error"] = str(e)
            return ""
        finally:
            pipeline.config.write_diagnostics = original_write_diag

        if self._cancel_flag.is_set():
            self._progress["status"] = "cancelled"
        else:
            preview_p = Path(self.preview_path)
            if not preview_p.exists() or preview_p.stat().st_size < 1024:
                self._progress["status"] = "error"
                self._progress["error"] = "Preview file was not created"
            else:
                self._progress["status"] = "complete"
                self._progress["frame"] = result.frames
                self._progress["detections"] = result.detections
                self._progress["restorations"] = result.restorations

        logger.info("mosaic_segment: %d frames, %d det, %d res → %s",
                    result.frames, result.detections, result.restorations, self.preview_path)
        return self.preview_path


# ── Global server instance ────────────────────────────────────────────────

_server: SwapServer | None = None


def _get_server() -> SwapServer:
    global _server
    if _server is None:
        models_dir = os.environ.get("CHITRAMAYA_MODELS_DIR", "./models")
        _server = SwapServer(models_dir=models_dir)
    return _server


# ── Flask Routes ──────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("ui.html")


# ── Video ──

@app.route("/api/load-video", methods=["POST"])
def api_load_video():
    data = request.get_json(force=True)
    path = data.get("path", "")
    if not path or not Path(path).exists():
        return jsonify({"error": f"File not found: {path}"}), 400
    try:
        result = _get_server().load_video(path)
        return jsonify(result)
    except Exception as e:
        logger.exception("load-video failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/seek", methods=["POST"])
def api_seek():
    data = request.get_json(force=True)
    frame = data.get("frame", 0)
    try:
        b64 = _get_server().seek(int(frame))
        return jsonify({"image": b64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Faces ──

# ── Preview ──

# ── Processing ──

# ── Mosaic restoration endpoints ─────────────────────────────────────────

def _run_mosaic_threaded(op_name: str, data: dict, fn):
    """Shared launcher for mosaic-{segment,full}. ``fn`` takes the server.

    Wraps the same lock + op-tracking + lockfile pattern as swap-* with the
    addition of SessionLock so a CLI batch starting concurrently can warn
    the user (or vice versa).
    """
    from chitramaya.mosaic.session import SessionLock

    if not _op_lock.acquire(blocking=False):
        print(f"[ChitraMaya] {op_name} BLOCKED: operation lock held")
        return jsonify({"error": "Operation already in progress"}), 409

    def _run():
        server = _get_server()
        try:
            _set_op(op_name)
            server._cancel_flag.clear()
            paths = [server.video_path] if server.video_path else []
            with SessionLock(mode="ui", paths=paths):
                print(f"[ChitraMaya] {op_name} starting")
                fn(server, data)
                print(f"[ChitraMaya] {op_name} finished: status={server._progress.get('status')}")
        except Exception as e:
            print(f"[ChitraMaya] {op_name} EXCEPTION: {e}")
            logger.exception(f"{op_name} thread failed")
            server._progress["status"] = "error"
            server._progress["error"] = str(e)
        finally:
            _clear_op()
            _op_lock.release()
            print(f"[ChitraMaya] {op_name} lock released")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return jsonify({"status": "started"})


@app.route("/api/mosaic-segment", methods=["POST"])
def api_mosaic_segment():
    data = request.get_json(force=True)

    def _do(server, data):
        params = data.get("params", {})
        start_time = float(data.get("start_time", 0))
        end_time = float(data.get("end_time", 0))
        server.mosaic_segment(start_time, end_time, params)

    return _run_mosaic_threaded("mosaic-segment", data, _do)


@app.route("/api/mosaic-full", methods=["POST"])
def api_mosaic_full():
    data = request.get_json(force=True)

    def _do(server, data):
        params = data.get("params", {})
        server.mosaic_full(params)

    return _run_mosaic_threaded("mosaic-full", data, _do)


@app.route("/api/session-status", methods=["GET"])
def api_session_status():
    """Return info about any concurrent ChitraMaya session (CLI/UI/batch).

    UI calls this on startup and shows a warning if a session is running
    in another process (e.g. CLI batch).
    """
    from chitramaya.mosaic.session import read_session
    info = read_session()
    if info is None:
        return jsonify({"running": False})
    return jsonify({
        "running": True,
        "pid": info.pid,
        "mode": info.mode,
        "started_at": info.started_at,
        "age_sec": round(info.age_sec, 1),
        "paths": info.paths,
        "is_us": info.is_us,
    })


@app.route("/api/list-mosaic-models", methods=["GET"])
def api_list_mosaic_models():
    """Scan the project's ``models/`` directory for mosaic models.

    Detection models are ``.pt`` (Ultralytics YOLO); restoration models
    are ``.pth`` (BasicVSR++ checkpoints). Heuristic by extension — if
    the user keeps face-swap models in the same dir, this is liberal
    and will include those too. UI can filter further by filename if
    that ever becomes an issue.
    """
    models_dir = Path("./models")
    detection: list[dict] = []
    restoration: list[dict] = []

    from chitramaya.mosaic.models.basicvsrpp.engine_paths import (
        list_basicvsrpp_compiled_clip_sizes,
    )

    if models_dir.exists() and models_dir.is_dir():
        for p in sorted(models_dir.iterdir()):
            if not p.is_file():
                continue
            ext = p.suffix.lower()
            if ext == ".pt":
                detection.append({"path": str(p).replace("\\", "/"), "label": p.stem})
            elif ext == ".pth":
                mp = str(p)
                # Compiled clip sizes per precision, so the UI can default and
                # constrain Max Clip to what actually exists on disk.
                restoration.append({
                    "path": mp.replace("\\", "/"),
                    "label": p.stem,
                    "engines": {
                        "fp16": list_basicvsrpp_compiled_clip_sizes(mp, True),
                        "fp32": list_basicvsrpp_compiled_clip_sizes(mp, False),
                    },
                })

    return jsonify({"detection": detection, "restoration": restoration})


@app.route("/api/progress", methods=["GET"])
def api_progress():
    return jsonify(_get_server().get_progress())


@app.route("/api/cancel", methods=["POST"])
def api_cancel():
    _get_server().cancel()
    return jsonify({"ok": True})


# ── Config ──

@app.route("/api/config", methods=["GET"])
def api_get_config():
    from chitramaya.config import load_config
    try:
        cfg = load_config()
        return jsonify(cfg.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/config", methods=["POST"])
def api_set_config():
    data = request.get_json(force=True)
    from chitramaya.config import load_config, save_config
    try:
        cfg = load_config()
        cfg.update(data)
        save_config(cfg)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/set-output-dir", methods=["POST"])
def api_set_output_dir():
    data = request.get_json(force=True)
    path = data.get("path", "")
    if path:
        os.makedirs(path, exist_ok=True)
    server = _get_server()
    server.output_dir = path
    server._save_ui_config()
    return jsonify({"ok": True, "path": path})


@app.route("/api/set-temp-dir", methods=["POST"])
def api_set_temp_dir():
    data = request.get_json(force=True)
    path = data.get("path", "")
    if path:
        os.makedirs(path, exist_ok=True)
    server = _get_server()
    server.temp_dir = path
    server._save_ui_config()
    return jsonify({"ok": True, "path": path})


@app.route("/api/new-project", methods=["POST"])
def api_new_project():
    _get_server().reset()
    return jsonify({"ok": True})


@app.route("/api/clear-preview", methods=["POST"])
def api_clear_preview():
    """Delete preview temp file and clear preview state."""
    server = _get_server()
    if server.preview_path and Path(server.preview_path).exists():
        try:
            os.unlink(server.preview_path)
        except Exception:
            pass
    server.preview_path = None
    return jsonify({"ok": True})


@app.route("/api/ui-config", methods=["GET"])
def api_ui_config():
    """Get saved UI config for startup initialization."""
    server = _get_server()
    return jsonify({
        "facesDir": server.faces_dir,
        "outputDir": server.output_dir,
        "tempDir": server.temp_dir,
    })


@app.route("/api/save-config", methods=["POST"])
def api_save_config():
    """Save UI config. Writes ONLY known UI keys — no legacy data."""
    data = request.get_json(force=True)
    try:
        config_path = Path.cwd() / "ChitraMaya-config.json"
        config_path.write_text(
            json.dumps(data, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        # Update server state for folders
        server = _get_server()
        if "facesDir" in data:
            server.faces_dir = data["facesDir"]
        if "outputDir" in data:
            server.output_dir = data["outputDir"]
        if "tempDir" in data:
            server.temp_dir = data["tempDir"]
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/load-config", methods=["GET"])
def api_load_config():
    """Load UI config. Filters to known UI keys only."""
    try:
        config_path = Path.cwd() / "ChitraMaya-config.json"
        if not config_path.exists():
            return jsonify({})
        data = json.loads(config_path.read_text(encoding="utf-8"))
        # Filter to known UI keys — ignore legacy SwapConfig keys
        known_prefixes = ("ctrl", "skip", "facesDir", "outputDir", "tempDir")
        known_exact = {"debug", "perf_test"}
        filtered = {k: v for k, v in data.items()
                    if k.startswith(known_prefixes) or k in known_exact}
        return jsonify(filtered)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/default-config", methods=["GET"])
def api_default_config():
    """Return default config derived from MosaicConfig dataclass.

    Single source of truth for the mosaic UI's default control values.
    """
    from chitramaya.models import MosaicConfig

    m = MosaicConfig()

    return jsonify({
        "facesDir": "",
        "outputDir": "",
        "tempDir": "",

        # Detection
        "ctrlMosaicDetModel": m.detection_model,
        "ctrlMosaicDetScore": str(int(m.mosaic_detection_score * 100)),
        "ctrlMosaicDetBatch": str(m.mosaic_detection_batch_size),
        "ctrlMosaicDetectOnly": m.mosaic_detect_only,

        # Restoration
        "ctrlMosaicRestModel": m.restoration_model,
        "ctrlMosaicMaxClip": str(m.mosaic_max_clip_size),
        "ctrlMosaicFp16": m.mosaic_fp16,
        "ctrlMosaicCompileTrt": m.mosaic_compile_trt,

        # Encoder
        "ctrlCodec": "hevc",
        "ctrlPreset": "P5",
        "ctrlQP": "18",

        # Transport
        "skipBackward": "5",
        "skipForward": "5",

        # Developer flags
        "debug": False,
        "perf_test": False,
    })


# ── Serve video for player ──

@app.route("/video")
def serve_video():
    """Serve current video file for HTML5 player with Range support."""
    server = _get_server()
    if not server.video_path or not Path(server.video_path).exists():
        return Response("No video loaded", status=404)

    return _serve_file(server.video_path)


@app.route("/preview-video")
def serve_preview_video():
    """Serve preview video for HTML5 player."""
    server = _get_server()
    if not server.preview_path or not Path(server.preview_path).exists():
        return Response("No preview available", status=404)

    return _serve_file(server.preview_path)


def _serve_file(filepath: str) -> Response:
    """Serve a file with Range request support (needed for video seeking)."""
    path = Path(filepath)
    size = path.stat().st_size
    mime = f'video/{path.suffix.lstrip(".")}'
    if path.suffix.lower() == '.mp4':
        mime = 'video/mp4'
    elif path.suffix.lower() == '.mkv':
        mime = 'video/x-matroska'

    range_header = request.headers.get('Range')

    if range_header:
        # Parse range: "bytes=start-end"
        byte_range = range_header.replace('bytes=', '').split('-')
        start = int(byte_range[0])
        end = int(byte_range[1]) if byte_range[1] else size - 1
        end = min(end, size - 1)
        length = end - start + 1

        def generate_range():
            with open(str(path), 'rb') as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        return Response(
            generate_range(),
            status=206,
            mimetype=mime,
            headers={
                'Content-Range': f'bytes {start}-{end}/{size}',
                'Accept-Ranges': 'bytes',
                'Content-Length': str(length),
                'Cache-Control': 'no-store',
            },
        )
    else:
        def generate():
            with open(str(path), 'rb') as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    yield chunk

        return Response(
            generate(),
            mimetype=mime,
            headers={
                'Accept-Ranges': 'bytes',
                'Content-Length': str(size),
                'Cache-Control': 'no-store',
            },
        )


# ── Launch ────────────────────────────────────────────────────────────────

def find_free_port(start: int = 5100, end: int = 5200) -> int:
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port in {start}-{end}")


def run(models_dir: str = "./models", gpu_id: int = 0, debug: bool = False, console: bool = False):
    """Launch the ChitraMaya server with PyWebView or browser fallback.

    Args:
        models_dir: Path to models directory.
        gpu_id: CUDA device ID.
        debug: Enable Flask debug mode.
        console: Open WebView2 DevTools console on startup.
    """
    global _server
    _server = SwapServer(models_dir=models_dir, gpu_id=gpu_id)

    port = find_free_port()
    url = f"http://127.0.0.1:{port}"

    # Start Flask in a thread
    t = threading.Thread(
        target=lambda: app.run(host='127.0.0.1', port=port, debug=False, use_reloader=False),
        daemon=True,
    )
    t.start()
    time.sleep(0.5)

    print(f"[ChitraMaya] Server running at {url}")

    # Try PyWebView
    try:
        import webview

        class Api:
            def __init__(self):
                self._window = None

            def toggle_fullscreen(self):
                win = getattr(self, "_window", None)
                if not win:
                    return {"ok": False}
                try:
                    win.toggle_fullscreen()
                    return {"ok": True}
                except Exception as e:
                    return {"ok": False, "error": str(e)}

            @staticmethod
            def select_video():
                types = ('Video Files (*.mp4;*.mkv;*.avi;*.mov;*.wmv;*.webm)',)
                try:
                    result = webview.windows[0].create_file_dialog(
                        webview.FileDialog.OPEN, file_types=types,
                    )
                except AttributeError:
                    # Fallback for older pywebview
                    result = webview.windows[0].create_file_dialog(
                        webview.OPEN_DIALOG, file_types=types,
                    )
                if result and len(result) > 0:
                    return str(result[0])
                return None

            @staticmethod
            def select_folder():
                try:
                    result = webview.windows[0].create_file_dialog(
                        webview.FileDialog.FOLDER,
                    )
                except AttributeError:
                    result = webview.windows[0].create_file_dialog(
                        webview.FOLDER_DIALOG,
                    )
                if result and len(result) > 0:
                    return str(result[0])
                return None

        api = Api()
        window = webview.create_window(
            "ChitraMaya",
            url,
            js_api=api,
            width=1600,
            height=900,
            min_size=(1200, 700),
        )
        api._window = window

        def on_loaded():
            window.evaluate_js("""
                window.__chitramayaPyWebViewReady = true;
                if (window.__chitramayaOnPyWebViewReady) window.__chitramayaOnPyWebViewReady();
            """)

        # ── Native Drag/Drop (pywebview DOM handlers) ─────────
        # Browser File API only gives filename, not full path.
        # pywebview's DOM handlers provide pywebviewFullPath.
        import re as _re

        VIDEO_RE = _re.compile(r"\.(mp4|mkv|webm|avi|mov|ts|m2ts|wmv)$", _re.IGNORECASE)

        def on_drop(evt):
            try:
                files = evt.get("dataTransfer", {}).get("files", [])
                if not files:
                    return

                paths = []
                for f in files:
                    p = f.get("pywebviewFullPath")
                    if p:
                        paths.append(p)

                if not paths:
                    return

                logger.info("Dropped file(s): %s", paths)

                video_path = next((p for p in paths if VIDEO_RE.search(p)), None)
                if video_path:
                    window.evaluate_js(
                        f"window.chitramayaSetVideoPath({json.dumps(video_path, ensure_ascii=False)});"
                    )
            except Exception:
                logger.exception("Drop handler failed")

        def on_drag_over(_evt):
            return

        def attach_dom_handlers():
            import threading as _th
            from webview.dom import DOMEventHandler

            def _attempt(n):
                try:
                    doc = window.dom.document
                    if doc is None:
                        raise ValueError("document is None")
                    doc.events.dragover += DOMEventHandler(
                        on_drag_over, prevent_default=True,
                        stop_propagation=True, stop_immediate_propagation=True,
                    )
                    doc.events.drop += DOMEventHandler(
                        on_drop, prevent_default=True,
                        stop_propagation=True, stop_immediate_propagation=True,
                    )
                    logger.info("pywebview DOM drag/drop handlers attached")
                except (TypeError, ValueError, AttributeError):
                    if n < 10:
                        _th.Timer(0.5, _attempt, args=(n + 1,)).start()
                    else:
                        logger.warning("Failed to attach DOM handlers after %d attempts", n)
                except Exception:
                    logger.exception("Failed to attach DOM handlers")
            _attempt(1)

        window.events.loaded += on_loaded
        window.events.loaded += attach_dom_handlers
        webview.start(debug=console)

    except ImportError:
        print("[ChitraMaya] pywebview not available, opening in browser")
        webbrowser.open(url)
        t.join()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ChitraMaya Server")
    parser.add_argument("--models-dir", type=str, default="./models")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--debug", action="store_true", help="Flask debug mode")
    parser.add_argument("--console", action="store_true", help="Open WebView2 DevTools console")
    args = parser.parse_args()
    run(models_dir=args.models_dir, gpu_id=args.gpu, debug=args.debug, console=args.console)
