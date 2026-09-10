# tools/probe_decode_path.py
"""CM-171: which layer of the ffmpeg decode path is the wall?

On the AMD and Intel editions frames come from
    ffmpeg -hwaccel d3d11va|qsv -i F -f rawvideo -pix_fmt nv12 pipe:1
through a Windows anonymous pipe into chitramaya.video.decoder.Decoder, which
reads one frame at a time into a freshly pinned torch buffer. Measured ceilings
(09-08): ~35 fps at 1080p on an RX 9060 XT, ~10 fps at 4K -- both about
120 MB/s, i.e. a byte-throughput bound, while the VCN sat at ~60%.

This times the SAME file through four layers, N frames each, so the wall has a
name before anyone writes a fix:

  L1  ffmpeg decode + readback only     ffmpeg ... -f null -        (no pipe)
  L2  L1 + the pipe, read into a plain  bytearray in big chunks     (no torch)
  L3  L1 + pipe + ONE pinned buffer     reused (torch, no per-frame pin)
  L4  the shipped Decoder class         (per-frame pinned alloc, as in the app)
  L5  CPU software decode through L2    (-hwaccel none) for comparison

Each layer prints frames/s and MB/s. Reading the table:
  L1 slow            -> the hardware readback (hwdownload) is the wall; CPU decode
                        (L5) may beat it at 1080p -> pick hw decode by resolution
  L1 fast, L2 slow   -> the pipe is the wall -> wide named pipe / shared memory,
                        or in-process decode (PyAV)
  L2 fast, L3 slow   -> Python read loop -> reader thread + bigger reads
  L3 fast, L4 slow   -> the per-frame pin_memory allocation -> pinned ring buffer

    python tools/probe_decode_path.py --input clip.mp4 --frames 600
    python tools/probe_decode_path.py --input clip.mp4 --frames 600 --hwaccel qsv
    python tools/probe_decode_path.py --input clip.mp4 --frames 300 --skip L4 --skip L5

ASCII-only output. Needs ffmpeg/ffprobe on PATH (or --ffmpeg / --ffprobe).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from typing import List, Optional, Tuple

try:
    from chitramaya.winproc import NOWINDOW
except Exception:  # pragma: no cover
    NOWINDOW = {}


def _probe(ffprobe: str, path: str) -> Tuple[int, int, float, str]:
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,codec_name", "-of", "json", path],
        capture_output=True, text=True, encoding="utf-8", errors="replace", **NOWINDOW).stdout
    st = json.loads(out)["streams"][0]
    num, den = st["r_frame_rate"].split("/")
    fps = float(num) / float(den or 1)
    return int(st["width"]), int(st["height"]), fps, str(st.get("codec_name", "?"))


def _hw_tokens(hwaccel: str) -> List[str]:
    if hwaccel == "none":
        return []
    if hwaccel == "qsv":
        return ["-hwaccel", "qsv", "-hwaccel_output_format", "nv12"]
    return ["-hwaccel", hwaccel]


def _ffmpeg_cmd(ffmpeg: str, path: str, hwaccel: str, frames: int, sink: List[str]) -> List[str]:
    return [ffmpeg, "-hide_banner", "-loglevel", "error", *_hw_tokens(hwaccel),
            "-fflags", "+genpts", "-i", path, "-an", "-sn", "-dn",
            "-fps_mode", "passthrough", "-frames:v", str(frames), *sink]


def _report(label: str, frames: int, frame_bytes: int, dt: float, note: str = "") -> None:
    fps = frames / dt if dt > 0 else 0.0
    mbs = frames * frame_bytes / dt / 1e6 if dt > 0 else 0.0
    print(f"[decode-probe] {label:<4s} {frames:6d} frames in {dt:7.2f} s  ->  "
          f"{fps:7.1f} fps  {mbs:8.1f} MB/s  {note}")


def layer1(ffmpeg: str, path: str, hwaccel: str, frames: int, frame_bytes: int) -> None:
    cmd = _ffmpeg_cmd(ffmpeg, path, hwaccel, frames, ["-f", "null", "-"])
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", **NOWINDOW)
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        print(f"[decode-probe] L1 ffmpeg failed: {(r.stderr or '').strip()[:200]}")
        return
    _report("L1", frames, frame_bytes, dt, f"ffmpeg -hwaccel {hwaccel} -> -f null (decode + readback, no pipe)")


def _pipe_reader(ffmpeg: str, path: str, hwaccel: str, frames: int, frame_bytes: int,
                 label: str, reader) -> None:
    cmd = _ffmpeg_cmd(ffmpeg, path, hwaccel, frames, ["-f", "rawvideo", "-pix_fmt", "nv12", "pipe:1"])
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            bufsize=10 ** 8, **NOWINDOW)
    t0 = time.perf_counter()
    got_frames = 0
    try:
        got_frames = reader(proc.stdout, frames, frame_bytes)
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.wait(timeout=60)
    dt = time.perf_counter() - t0
    _report(label, got_frames, frame_bytes, dt, f"ffmpeg -hwaccel {hwaccel} -> pipe -> {reader.__doc__}")


def read_bytearray(stdout, frames: int, frame_bytes: int) -> int:
    """plain bytearray, readinto in 4 MB chunks (no torch)"""
    buf = bytearray(frame_bytes)
    view = memoryview(buf)
    n = 0
    while n < frames:
        got = 0
        while got < frame_bytes:
            k = stdout.readinto(view[got:got + min(4 << 20, frame_bytes - got)])
            if not k:
                return n
            got += k
        n += 1
    return n


def make_read_pinned_once(torch):
    buf_t = None

    def read_pinned_once(stdout, frames: int, frame_bytes: int) -> int:
        """ONE pinned torch buffer reused for every frame"""
        nonlocal buf_t
        if buf_t is None:
            try:
                buf_t = torch.empty((frame_bytes,), dtype=torch.uint8, pin_memory=True)
            except Exception:
                buf_t = torch.empty((frame_bytes,), dtype=torch.uint8)
        view = memoryview(buf_t.numpy())
        n = 0
        while n < frames:
            got = 0
            while got < frame_bytes:
                k = stdout.readinto(view[got:])
                if not k:
                    return n
                got += k
            n += 1
        return n
    return read_pinned_once


def make_read_pinned_per_frame(torch):
    def read_pinned_per_frame(stdout, frames: int, frame_bytes: int) -> int:
        """a NEW pinned torch buffer per frame (what Decoder._ffmpeg_read_frame does)"""
        n = 0
        while n < frames:
            try:
                buf_t = torch.empty((frame_bytes,), dtype=torch.uint8, pin_memory=True)
            except Exception:
                buf_t = torch.empty((frame_bytes,), dtype=torch.uint8)
            view = memoryview(buf_t.numpy())
            got = 0
            while got < frame_bytes:
                k = stdout.readinto(view[got:])
                if not k:
                    return n
                got += k
            n += 1
        return n
    return read_pinned_per_frame


def main() -> int:
    ap = argparse.ArgumentParser(description="CM-171 decode-path layer probe")
    ap.add_argument("--input", required=True)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--hwaccel", default="d3d11va", choices=["d3d11va", "qsv", "none"],
                    help="hardware decoder for L1-L4 (default d3d11va; L5 always uses none)")
    ap.add_argument("--skip", action="append", default=[], choices=["L1", "L2", "L3", "L4", "L5"])
    ap.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    ap.add_argument("--ffprobe", default=shutil.which("ffprobe") or "ffprobe")
    args = ap.parse_args()

    w, h, fps, codec = _probe(args.ffprobe, args.input)
    frame_bytes = w * h * 3 // 2
    print(f"[decode-probe] {os.path.basename(args.input)}: {w}x{h} {codec} {fps:.2f} fps  "
          f"nv12 frame = {frame_bytes / 1e6:.2f} MB  realtime needs {fps * frame_bytes / 1e6:.0f} MB/s")
    print(f"[decode-probe] frames per layer: {args.frames}   hwaccel: {args.hwaccel}")

    torch = None
    try:
        import torch as _torch
        torch = _torch
        dev = "cuda" if _torch.cuda.is_available() else ("xpu" if getattr(_torch, "xpu", None) and _torch.xpu.is_available() else "cpu")
        print(f"[decode-probe] torch {_torch.__version__} device for pinning: {dev}")
        # warm the allocator so L3 does not pay torch's first-allocation cost
        try:
            _w = _torch.empty((1 << 20,), dtype=_torch.uint8, pin_memory=True)
            del _w
        except Exception:
            pass
    except Exception as e:
        print(f"[decode-probe] torch unavailable ({e}); L3/L4 skipped")

    if "L1" not in args.skip:
        layer1(args.ffmpeg, args.input, args.hwaccel, args.frames, frame_bytes)
    if "L2" not in args.skip:
        _pipe_reader(args.ffmpeg, args.input, args.hwaccel, args.frames, frame_bytes, "L2", read_bytearray)
    if torch is not None and "L3" not in args.skip:
        _pipe_reader(args.ffmpeg, args.input, args.hwaccel, args.frames, frame_bytes, "L3", make_read_pinned_once(torch))
    if torch is not None and "L4" not in args.skip:
        _pipe_reader(args.ffmpeg, args.input, args.hwaccel, args.frames, frame_bytes, "L4", make_read_pinned_per_frame(torch))
    if "L5" not in args.skip:
        _pipe_reader(args.ffmpeg, args.input, "none", args.frames, frame_bytes, "L5", read_bytearray)

    print("[decode-probe] read: L1 slow = hardware readback is the wall (compare L5); "
          "L1 fast + L2 slow = the pipe; L2 fast + L3 slow = the Python read loop; "
          "L3 fast + L4 slow = per-frame pinned allocation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
