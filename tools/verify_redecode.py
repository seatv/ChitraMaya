# ChitraMaya/tools/verify_redecode.py
"""CM-191 A/B check: are two restorations of the same clip the same picture?

Usage (installed build):
    ChitraMaya-cli.exe -verify-redecode A.mp4 B.mp4 [--max-frames N]
    ChitraMaya-cli.exe -verify-redecode --dumps DIR_A DIR_B

Mode 1 decodes both videos with the bundled ffmpeg to raw yuv420p, frame by
frame, and counts identical frames / differing frames / the largest
per-pixel difference and where it first appears. NVENC is deterministic
for identical input frames and settings, so two runs whose PRE-ENCODE
frames were identical normally decode identical; a small nonzero
difference on a handful of frames points at the encoder's rate control,
a large or growing one at the paste-back. Exit 0 when every frame is
identical, 1 otherwise.

Mode 2 compares two pre-encode dump folders written by the debug switch
(GR_CORRUPT_DUMP_FRAMES=0,1,2,... GR_CORRUPT_DUMP_DIR=<dir>) -- the
bit-exact proof, independent of the encoder: it compares the checksum32
lines of the .txt sidecars frame for frame.

ASCII only on stdout. Weights travel; content never does: this tool
prints numbers, never pixels.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
from typing import List, Optional, Tuple


def _nowindow() -> dict:
    try:
        from chitramaya.winproc import NOWINDOW
        return dict(NOWINDOW)
    except Exception:
        return {}


def _probe_dims(ffprobe: str, path: str) -> Tuple[int, int, int]:
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames",
         "-of", "csv=p=0", path],
        capture_output=True, text=True, **_nowindow()).stdout.strip()
    parts = [p for p in out.split(",") if p != ""]
    w = int(parts[0]); h = int(parts[1])
    n = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
    return w, h, n


def _open_raw(ffmpeg: str, path: str, max_frames: int) -> subprocess.Popen:
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", path,
           "-map", "0:v:0", "-an", "-sn", "-dn"]
    if max_frames > 0:
        cmd += ["-frames:v", str(max_frames)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            bufsize=0, **_nowindow())


def _read_exact(fh, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = fh.read(n - len(buf))
        if not chunk:
            break
        buf += chunk
    return bytes(buf)


def compare_videos(ffmpeg: str, ffprobe: str, a: str, b: str, max_frames: int = 0) -> int:
    wa, ha, na = _probe_dims(ffprobe, a)
    wb, hb, nb = _probe_dims(ffprobe, b)
    print(f"[verify-redecode] A: {os.path.basename(a)} {wa}x{ha} frames={na or '?'}")
    print(f"[verify-redecode] B: {os.path.basename(b)} {wb}x{hb} frames={nb or '?'}")
    if (wa, ha) != (wb, hb):
        print("  FAIL  frame size differs")
        return 1
    frame_bytes = wa * ha * 3 // 2
    pa = _open_raw(ffmpeg, a, max_frames)
    pb = _open_raw(ffmpeg, b, max_frames)
    try:
        import numpy as np
    except Exception:
        np = None  # type: ignore
    same = 0
    diff = 0
    first_diff: Optional[int] = None
    max_abs = 0
    worst_frame = -1
    i = 0
    while True:
        fa = _read_exact(pa.stdout, frame_bytes)
        fb = _read_exact(pb.stdout, frame_bytes)
        if len(fa) < frame_bytes and len(fb) < frame_bytes:
            break
        if len(fa) < frame_bytes or len(fb) < frame_bytes:
            print(f"  FAIL  frame count differs (one stream ended at frame {i})")
            diff += 1
            break
        if fa == fb:
            same += 1
        else:
            diff += 1
            if first_diff is None:
                first_diff = i
            if np is not None:
                d = int(np.abs(np.frombuffer(fa, dtype=np.uint8).astype(np.int16)
                               - np.frombuffer(fb, dtype=np.uint8).astype(np.int16)).max())
                if d > max_abs:
                    max_abs = d
                    worst_frame = i
        i += 1
        if i % 2000 == 0:
            print(f"  ... {i} frames compared ({diff} differ)")
    for p in (pa, pb):
        try:
            p.stdout.close()
            p.wait(timeout=10)
        except Exception:
            pass
    print(f"[verify-redecode] compared {i} frames: identical {same}, differing {diff}")
    if diff:
        print(f"  first difference at frame {first_diff}; largest per-pixel "
              f"difference {max_abs} (frame {worst_frame})")
        if max_abs <= 2:
            print("  NOTE  differences of 1-2 levels on a few frames are the encoder's rate "
                  "control, not the paste-back; use --dumps for the bit-exact check.")
        print("FAIL -- outputs are not identical")
        return 1
    print("PASS -- every decoded frame identical")
    return 0


def compare_dumps(dir_a: str, dir_b: str) -> int:
    def sums(d: str) -> dict:
        out = {}
        for p in glob.glob(os.path.join(d, "preencode_f*.txt")):
            m = re.search(r"preencode_f(\d+)\.txt$", os.path.basename(p))
            if not m:
                continue
            txt = open(p, "r", encoding="utf-8", errors="replace").read()
            c = re.search(r"checksum32=(\d+)", txt)
            out[int(m.group(1))] = c.group(1) if c else None
        return out
    sa, sb = sums(dir_a), sums(dir_b)
    print(f"[verify-redecode] dumps: A {len(sa)} frames, B {len(sb)} frames")
    common = sorted(set(sa) & set(sb))
    only_a = sorted(set(sa) - set(sb)); only_b = sorted(set(sb) - set(sa))
    bad = [k for k in common if sa[k] != sb[k]]
    if only_a or only_b:
        print(f"  NOTE  frames only in A: {len(only_a)}; only in B: {len(only_b)}")
    print(f"[verify-redecode] {len(common)} frames compared by checksum: "
          f"identical {len(common) - len(bad)}, differing {len(bad)}")
    if bad:
        print(f"  first difference at frame {bad[0]}")
        print("FAIL -- pre-encode frames differ (paste-back is not identical)")
        return 1
    if not common:
        print("FAIL -- nothing to compare")
        return 1
    print("PASS -- pre-encode frames bit-identical")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="ChitraMaya -verify-redecode")
    ap.add_argument("paths", nargs="*", help="A.mp4 B.mp4 (or two dump folders with --dumps)")
    ap.add_argument("--dumps", action="store_true", help="compare two pre-encode dump folders")
    ap.add_argument("--max-frames", type=int, default=0, help="compare only the first N frames")
    ap.add_argument("--ffmpeg", default=os.environ.get("CHITRAMAYA_FFMPEG") or shutil.which("ffmpeg") or "ffmpeg")
    ap.add_argument("--ffprobe", default=os.environ.get("CHITRAMAYA_FFPROBE") or shutil.which("ffprobe") or "ffprobe")
    args = ap.parse_args()
    if len(args.paths) != 2:
        ap.print_usage()
        print("need exactly two paths")
        return 2
    a, b = args.paths
    if args.dumps or (os.path.isdir(a) and os.path.isdir(b)):
        return compare_dumps(a, b)
    for p in (a, b):
        if not os.path.isfile(p):
            print(f"not found: {p}")
            return 2
    return compare_videos(args.ffmpeg, args.ffprobe, a, b, max_frames=int(args.max_frames))


if __name__ == "__main__":
    sys.exit(main())
