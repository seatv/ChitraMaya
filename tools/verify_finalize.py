# tools/verify_finalize.py
"""CM-187: prove the one-pass finalize on THIS machine's ffmpeg.

Synthesizes a 12-second test title (colour bars + a tone) whose audio
leads the video by a known offset, encodes it the way each edition's
encoder writes its output (a raw P-only HEVC elementary stream for the
NVENC path; a fragmented .venc.mp4 with B-frames for the AMD/Intel path),
then finalizes each with the SAME helper functions the encoders call
(chitramaya.video.finalize) and checks the result:

  1. the file plays and has every frame;
  2. the index (moov) sits at the FRONT, written in ONE pass -- no
     +faststart rewrite;
  3. the A/V start offset is restored: raw path -> a sample-accurate
     edit list on the audio; container path -> the video starts at the
     source offset;
  4. the audio-delayed direction (video before audio) also works;
  5. a too-small index reservation is detected and the retry (index at
     the end) succeeds.

Runs from an installed build with nothing but the bundled ffmpeg:
    ChitraMaya-cli.exe -verify-finalize [--keep] [--workdir DIR]
Exit 0 = PASS, 1 = FAIL. Needs libx265 in the bundled ffmpeg (the gyan
builds have it); falls back to libx264/H.264 raw when it is missing.
"""
from __future__ import annotations

import argparse
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from chitramaya.video.finalize import (
    AudioSidecar, build_onepass_raw_cmd, build_onepass_container_cmd,
    estimate_moov_bytes, moov_too_small, probe_av_start, first_presented_time,
)
from chitramaya.winproc import NOWINDOW

FPS = 30
DUR = 12.0
FRAMES = int(FPS * DUR)


def _run(cmd: List[str], timeout: int = 300) -> Tuple[int, str]:
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout, **NOWINDOW)
    return r.returncode, (r.stderr or "")


def _has_encoder(ffmpeg: str, name: str) -> bool:
    rc, _ = _run([ffmpeg, "-hide_banner", "-h", f"encoder={name}"])
    return rc == 0


def _atoms(path: str) -> List[str]:
    out = []
    with open(path, "rb") as f:
        pos = 0
        while True:
            h = f.read(8)
            if len(h) < 8:
                break
            size, typ = struct.unpack(">I4s", h)
            out.append(typ.decode("latin1"))
            if size == 1:
                size = struct.unpack(">Q", f.read(8))[0]
            if size == 0:
                break
            pos += size
            f.seek(pos)
    return out


def _nb_frames(ffprobe: str, path: str) -> int:
    r = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0",
                        "-count_packets", "-show_entries", "stream=nb_read_packets",
                        "-of", "csv=p=0", path], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=120, **NOWINDOW)
    try:
        return int((r.stdout or "0").strip().split(",")[0])
    except ValueError:
        return -1


def _stream_start(ffprobe: str, path: str, sel: str) -> Optional[float]:
    r = subprocess.run([ffprobe, "-v", "error", "-select_streams", sel,
                        "-show_entries", "stream=start_time", "-of", "csv=p=0", path],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=60, **NOWINDOW)
    for ln in (r.stdout or "").splitlines():
        ln = ln.strip().strip(",")
        if ln and ln.upper() != "N/A":
            try:
                return float(ln)
            except ValueError:
                pass
    return None


def main() -> int:
    ap = argparse.ArgumentParser(prog="ChitraMaya -verify-finalize")
    ap.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    ap.add_argument("--ffprobe", default=shutil.which("ffprobe") or "ffprobe")
    ap.add_argument("--workdir", default="")
    ap.add_argument("--keep", action="store_true", help="keep the work folder")
    ap.add_argument("--offset", type=float, default=0.100,
                    help="source A/V start offset in seconds (video starts later)")
    args = ap.parse_args()
    ff, fp = args.ffmpeg, args.ffprobe
    work = args.workdir or tempfile.mkdtemp(prefix="cm_verify_finalize_")
    os.makedirs(work, exist_ok=True)
    W = lambda n: os.path.join(work, n)  # noqa: E731
    fails: List[str] = []

    def check(cond: bool, msg: str) -> None:
        print(("  PASS  " if cond else "  FAIL  ") + msg)
        if not cond:
            fails.append(msg)

    print(f"[verify-finalize] ffmpeg: {ff}")
    print(f"[verify-finalize] work:   {work}")
    off = float(args.offset)

    # ---- 1. synthetic source with a known A/V start offset --------------
    rc, err = _run([ff, "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", f"testsrc2=size=640x360:rate={FPS}",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
                    "-t", f"{DUR}", "-c:v", "libx264", "-preset", "ultrafast",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", W("base.mp4")])
    if rc != 0:
        print(err); print("[verify-finalize] could not synthesize the test source"); return 1
    rc, err = _run([ff, "-hide_banner", "-loglevel", "error", "-y", "-i", W("base.mp4"),
                    "-itsoffset", f"{off:.6f}", "-i", W("base.mp4"),
                    "-map", "1:v:0", "-map", "0:a:0", "-c", "copy",
                    "-video_track_timescale", "90000", W("src.mp4")])
    if rc != 0:
        print(err); return 1
    vs, as_ = _stream_start(fp, W("src.mp4"), "v:0"), _stream_start(fp, W("src.mp4"), "a:0")
    print(f"[verify-finalize] source: video starts {vs}, audio {as_} (offset {off:.3f})")
    check(vs is not None and abs(vs - off) < 0.002, "test source carries the A/V offset")

    # ---- 2. audio sidecar (the run-start extraction) --------------------
    sc = AudioSidecar(ff, fp, W("src.mp4"), W("out.audio.mka")).start()
    sc.wait(120)
    check(sc.ready, f"audio sidecar extracted ({os.path.getsize(sc.path) if sc.ready else 0} bytes)")

    # ---- 3. NVENC-shaped raw stream (P-only) ----------------------------
    if _has_encoder(ff, "libx265"):
        codec, fmt, tag, enc = "hevc", "hevc", "hvc1", ["-c:v", "libx265", "-preset", "ultrafast",
                                                        "-x265-params", "log-level=error:bframes=0"]
    else:
        codec, fmt, tag, enc = "h264", "h264", "avc1", ["-c:v", "libx264", "-preset", "ultrafast", "-bf", "0"]
        print("[verify-finalize] libx265 not in this ffmpeg; using H.264 raw")
    rc, err = _run([ff, "-hide_banner", "-loglevel", "error", "-y", "-i", W("src.mp4"),
                    "-an", *enc, "-f", fmt, W(f"raw.{fmt}")])
    if rc != 0:
        print(err); return 1
    moov = estimate_moov_bytes(FRAMES, DUR + off)
    cmd = build_onepass_raw_cmd(
        ff, fps_str=str(FPS), input_fmt_args=["-f", fmt], raw_path=W(f"raw.{fmt}"),
        sidecar=sc.path, audio_itsoffset=-off, color_args=[], tag_args=["-tag:v", tag],
        timescale_args=["-video_track_timescale", "90000"], extra_args=[],
        moov_bytes=moov, out_path=W("one_raw.mp4"))
    print("[verify-finalize] raw one-pass: " + " ".join(cmd))
    rc, err = _run(cmd)
    check(rc == 0, "raw one-pass finalize exit 0" + ("" if rc == 0 else f" ({err.strip().splitlines()[-1:]})"))
    if rc == 0:
        n = _nb_frames(fp, W("one_raw.mp4"))
        check(n >= FRAMES, f"all frames present ({n} >= {FRAMES})")
        at = _atoms(W("one_raw.mp4"))
        check(at[:2] == ["ftyp", "moov"], f"index at the front in one pass (atoms: {' '.join(at[:4])})")
        v0, a0, skip = probe_av_start(fp, W("one_raw.mp4"))
        want_skip = int(round(off * 48000))
        # newer ffmpeg folds the AAC encoder priming (~1024 samples) into the
        # same skip; either form presents the same audio.
        check(skip is not None and -96 <= (skip - want_skip) <= 1200,
              f"audio edit list restores the offset (skip {skip} samples; expected {want_skip} (+priming))")
        vp = first_presented_time(ff, W("one_raw.mp4"), "v")
        ap_ = first_presented_time(ff, W("one_raw.mp4"), "a")
        check(vp is not None and ap_ is not None and abs(vp - ap_) < 0.003,
              f"video and (trimmed) audio both present from 0 (video {vp}, audio {ap_})")

    # ---- 4. AMD/Intel-shaped container (fragmented mp4, B-frames) -------
    rc, err = _run([ff, "-hide_banner", "-loglevel", "error", "-y", "-i", W("src.mp4"),
                    "-an", "-r", str(FPS), "-c:v", "libx264", "-preset", "ultrafast",
                    "-movflags", "frag_keyframe+empty_moov", "-video_track_timescale", "90000",
                    W("out.venc.mp4")])
    if rc != 0:
        print(err); return 1
    cmd = build_onepass_container_cmd(
        ff, video_path=W("out.venc.mp4"), video_itsoffset=off, sidecar=sc.path,
        audio_itsoffset=0.0, tag_args=["-tag:v", "avc1"],
        timescale_args=["-video_track_timescale", "90000"], extra_args=[],
        duration_s=DUR + off, moov_bytes=moov, out_path=W("one_venc.mp4"))
    print("[verify-finalize] container one-pass: " + " ".join(cmd))
    rc, err = _run(cmd)
    check(rc == 0, "container one-pass finalize exit 0")
    if rc == 0:
        n = _nb_frames(fp, W("one_venc.mp4"))
        check(n >= FRAMES - 1, f"all frames present ({n} ~ {FRAMES})")
        at = _atoms(W("one_venc.mp4"))
        check(at[:2] == ["ftyp", "moov"], f"index at the front in one pass (atoms: {' '.join(at[:4])})")
        v0 = first_presented_time(ff, W("one_venc.mp4"), "v")
        a0 = first_presented_time(ff, W("one_venc.mp4"), "a")
        check(v0 is not None and a0 is not None and abs((v0 - a0) - off) < 0.005,
              f"video starts {off:.3f}s after the audio (video {v0}, audio {a0})")

    # ---- 5. audio-delayed direction (video before audio in the source) --
    cmd = build_onepass_raw_cmd(
        ff, fps_str=str(FPS), input_fmt_args=["-f", fmt], raw_path=W(f"raw.{fmt}"),
        sidecar=sc.path, audio_itsoffset=+0.050, color_args=[], tag_args=["-tag:v", tag],
        timescale_args=["-video_track_timescale", "90000"], extra_args=[],
        moov_bytes=moov, out_path=W("one_adelay.mp4"))
    rc, err = _run(cmd)
    check(rc == 0, "audio-delayed finalize exit 0")
    if rc == 0:
        a0 = first_presented_time(ff, W("one_adelay.mp4"), "a")
        v0 = first_presented_time(ff, W("one_adelay.mp4"), "v")
        # Within one AAC frame (21.3 ms): newer ffmpeg keeps the encoder
        # priming as a negative first pts in the sidecar and the delay lands
        # ~21 ms short -- identical to the source-based command on the same
        # ffmpeg (measured), so not a regression; well inside lip-sync tolerance.
        check(a0 is not None and v0 is not None and 0.025 <= (a0 - v0) <= 0.055,
              f"audio presented ~50 ms after the video (audio {a0}, video {v0}; 25-55 ms accepted)")

    # ---- 6. too-small index reservation -> detected -> retry without ----
    cmd = build_onepass_raw_cmd(
        ff, fps_str=str(FPS), input_fmt_args=["-f", fmt], raw_path=W(f"raw.{fmt}"),
        sidecar=sc.path, audio_itsoffset=-off, color_args=[], tag_args=["-tag:v", tag],
        timescale_args=["-video_track_timescale", "90000"], extra_args=[],
        moov_bytes=1000, out_path=W("small.mp4"))
    rc, err = _run(cmd)
    check(rc != 0 and moov_too_small(err), "too-small reservation is detected from ffmpeg's message")
    retry = list(cmd)
    i = retry.index("-moov_size"); del retry[i:i + 2]
    rc, err = _run(retry)
    check(rc == 0 and _atoms(W("small.mp4"))[0] == "ftyp", "retry without the reservation succeeds")

    print()
    if fails:
        print(f"[verify-finalize] FAIL ({len(fails)} check(s)):")
        for f in fails:
            print("   - " + f)
    else:
        print("[verify-finalize] PASS -- one-pass finalize verified on this ffmpeg.")
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    else:
        print(f"[verify-finalize] work folder kept: {work}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
