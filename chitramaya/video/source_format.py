"""CM-246: the container a source is REALLY in, when ffprobe's auto-detection
gets it wrong.

Field case (10-09, STZY-016 .ts, 1.5 GB): ffprobe reported ``png_pipe`` -- a
1x1 PNG glued in front of a plain MPEG-TS (a header disguise seen on shared
files). libavformat scores the PNG signature at byte 0 (99) above the TS
sync bytes behind it, so every ffprobe/ffmpeg call that opened the file
without ``-f`` saw a one-pixel PNG: the audio probe found no audio stream,
the sidecar said "source has no audio", and two five-hour runs finalized
video-only. (The NVDEC demuxer coped and decoded all 119,592 frames, so the
picture was fine and nothing complained.)

The rule: when auto-detection names an IMAGE format for a file whose
extension says video, re-probe with the demuxer forced from the extension;
if that finds a video stream, every ffmpeg/ffprobe call on the source gets
``-f <demuxer>`` in front of ``-i``. Files that probe correctly get nothing
(``input_args`` returns ``[]``), so the common path is byte-identical to
before.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from chitramaya.winproc import NOWINDOW
except Exception:  # pragma: no cover
    NOWINDOW: dict = {}

# auto-detection results that mean "ffprobe is looking at a picture"
IMAGE_FORMATS = {
    "png_pipe", "jpeg_pipe", "bmp_pipe", "gif", "gif_pipe", "webp_pipe", "tiff_pipe",
    "image2", "image2pipe", "jpegxl_pipe", "jpegxl_anim", "psd_pipe", "pgm_pipe",
    "ppm_pipe", "pam_pipe", "pbm_pipe", "pgmyuv_pipe", "svg_pipe", "qoi_pipe",
    "photocd_pipe", "cri_pipe", "dds_pipe", "dpx_pipe", "exr_pipe", "pictor_pipe",
    "sgi_pipe", "sunrast_pipe", "xbm_pipe", "xpm_pipe", "xwd_pipe", "pfm_pipe",
    "phm_pipe", "hdr_pipe", "vbn_pipe", "gem_pipe", "avif", "heif",
}

# the demuxer to force, by extension
DEMUXER_BY_EXT: Dict[str, str] = {
    ".ts": "mpegts", ".m2ts": "mpegts", ".mts": "mpegts", ".tsv": "mpegts",
    ".mp4": "mp4", ".m4v": "mp4", ".mov": "mov", ".3gp": "mp4",
    ".mkv": "matroska", ".webm": "matroska",
    ".avi": "avi", ".wmv": "asf", ".asf": "asf", ".flv": "flv",
    ".mpg": "mpeg", ".mpeg": "mpeg", ".vob": "mpeg", ".m2v": "mpegvideo",
}

# path -> (size, mtime, demuxer or "")
_CACHE: Dict[str, Tuple[int, float, str]] = {}
_SAID: set = set()


def _stat_key(path: str) -> Tuple[int, float]:
    try:
        st = os.stat(path)
        return int(st.st_size), float(st.st_mtime)
    except OSError:
        return -1, -1.0


def _probe_format(ffprobe: str, path: str, forced: str = "") -> Tuple[str, bool]:
    """(format_name, has a video stream that is not a still image)."""
    cmd = [ffprobe, "-v", "error"]
    if forced:
        cmd += ["-f", forced]
    cmd += ["-show_entries", "format=format_name", "-show_entries",
            "stream=codec_type,codec_name,width,height", "-of", "json", str(path)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=60, **NOWINDOW)
    except Exception:
        return "", False
    if p.returncode != 0:
        return "", False
    try:
        j = json.loads(p.stdout or "{}")
    except Exception:
        return "", False
    fmt = str((j.get("format") or {}).get("format_name") or "")
    has_video = False
    for s in (j.get("streams") or []):
        if s.get("codec_type") == "video" and str(s.get("codec_name") or "") not in (
                "png", "mjpeg", "bmp", "gif", "webp", "tiff", "jpegxl"):
            if int(s.get("width") or 0) > 1 and int(s.get("height") or 0) > 1:
                has_video = True
                break
    return fmt, has_video


def forced_demuxer(path: str, ffprobe: str = "ffprobe", *, say=print) -> str:
    """The demuxer to force for ``path``, or "" when auto-detection is right.
    Probes once per (path, size, mtime); prints one line the first time a
    file needs forcing."""
    path = str(path)
    key = _stat_key(path)
    hit = _CACHE.get(path)
    if hit is not None and hit[0] == key[0] and hit[1] == key[1]:
        return hit[2]
    result = ""
    ext = Path(path).suffix.lower()
    want = DEMUXER_BY_EXT.get(ext, "")
    if want:
        auto_fmt, auto_video = _probe_format(ffprobe, path)
        auto_names = {n.strip() for n in auto_fmt.split(",") if n.strip()}
        looks_like_image = bool(auto_names & IMAGE_FORMATS) or (auto_fmt and not auto_video)
        if looks_like_image:
            forced_fmt, forced_video = _probe_format(ffprobe, path, want)
            if forced_video:
                result = want
                if path not in _SAID:
                    _SAID.add(path)
                    say(f"[Source] ffprobe auto-detection reports '{auto_fmt or '?'}' for {Path(path).name} "
                        f"-- a picture header in front of the stream; forcing -f {want} for every "
                        f"ffmpeg/ffprobe call on this file (CM-246)")
    _CACHE[path] = (key[0], key[1], result)
    return result


def input_args(path: str, ffprobe: str = "ffprobe", *, say=print) -> List[str]:
    """``["-f", demuxer]`` to put in front of ``-i path`` (or the bare path for
    ffprobe), or ``[]`` when the file probes correctly on its own."""
    f = forced_demuxer(path, ffprobe, say=say)
    return ["-f", f] if f else []


def describe(path: str) -> Optional[str]:
    """What the run report says: None when nothing was forced."""
    hit = _CACHE.get(str(path))
    return hit[2] if (hit and hit[2]) else None


__all__ = ["forced_demuxer", "input_args", "describe", "IMAGE_FORMATS", "DEMUXER_BY_EXT"]
