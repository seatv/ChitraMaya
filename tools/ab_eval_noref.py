# tools/ab_eval_noref.py
"""No-reference A/B comparison of two restorations of the SAME real clip.

Companion to tools/ab_eval.py. ab_eval needs a pristine original (the
paired-clip exam); this tool is for the case where none exists -- real
studio content restored by two contenders (e.g. lada generic vs a
ChitraMaya fine-tune) with everything else identical. It cannot say which
output is more CORRECT (that needs the pristine). It measures, inside the
region the restorers actually touched:

  grid      how much of the source's mosaic block-grid survives in each
            output (the question "how much mosaic is left to restore").
            The mosaic pitch is estimated from the SOURCE's gradient
            periodicity; the same spectral peak is then read in A and B
            and reported as a fraction of the source peak (0 = grid gone,
            1 = untouched) plus its peak-over-floor SNR (grid visible when
            well above ~3).
  sharp     Laplacian variance relative to the source (is one output softer)
  disagree  mean |A-B| inside the region and the share of region pixels
            where A and B differ by more than --thresh
  flicker   frame-to-frame change inside the region for src, A and B
            (motion inflates all three equally; compare A and B to src)

The region is derived per frame from where EITHER output differs from
the source (the compositor pastes back through the mask, so untouched
pixels are identical up to encoder noise) -- no detector needed, and a
detection miss cancels out of the comparison by construction.

Usage (frame numbers are 0-based output frame indices, as in misses JSON):

    python tools/ab_eval_noref.py --src H:\\Exam\\clip.mp4 ^
        --a lada=H:\\ExamResults\\clip-restored-lada.mp4 ^
        --b ft_v0=H:\\ExamResults\\clip-restored-cm.mp4 ^
        --start 12600 --end 13350 --out H:\\ExamResults\\noref-12600 ^
        [--sheet 12] [--every 1] [--thresh 8] [--src-offset auto|N]

Outputs in --out: ab_noref.csv (per frame), ab_noref.json (summary),
ab_noref_sheet.png (src | A | B | |A-B| heat for the frames where the two
outputs disagree most). Requires ffmpeg on PATH (or CHITRAMAYA_FFMPEG),
numpy, opencv (a ChitraMaya dependency). ASCII-only output.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


# --- ffmpeg helpers ---

def _ffmpeg_exe() -> str:
    return os.environ.get("CHITRAMAYA_FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"


def _ffprobe_exe() -> str:
    ff = _ffmpeg_exe()
    cand = Path(ff).with_name("ffprobe" + Path(ff).suffix)
    if cand.exists():
        return str(cand)
    return shutil.which("ffprobe") or "ffprobe"


_FPS_ARGS = None


def _fps_args() -> list:
    """-fps_mode passthrough (ffmpeg >= 5.1) or the older -vsync 0."""
    global _FPS_ARGS
    if _FPS_ARGS is None:
        _FPS_ARGS = ["-vsync", "0"]
        try:
            p = subprocess.run([_ffmpeg_exe(), "-version"], capture_output=True,
                               text=True, **_nowindow())
            first = (p.stdout or "").splitlines()[0] if p.stdout else ""
            tok = first.split()[2] if len(first.split()) > 2 else ""
            head = tok.lstrip("n").split("-")[0].split(".")
            major = int(head[0]) if head and head[0].isdigit() else 0
            minor = int(head[1]) if len(head) > 1 and head[1].isdigit() else 0
            if (major, minor) >= (5, 1):
                _FPS_ARGS = ["-fps_mode", "passthrough"]
        except Exception:
            pass
    return list(_FPS_ARGS)


def _nowindow() -> dict:
    try:
        from chitramaya.winproc import NOWINDOW  # type: ignore
        return dict(NOWINDOW)
    except Exception:
        return {}


def _probe(path: str) -> tuple:
    p = subprocess.run(
        [_ffprobe_exe(), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames,r_frame_rate",
         "-of", "json", str(path)],
        capture_output=True, text=True, **_nowindow(),
    )
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {p.stderr[-300:]}")
    st = json.loads(p.stdout)["streams"][0]
    w, h = int(st["width"]), int(st["height"])
    try:
        n = int(st.get("nb_frames") or 0)
    except Exception:
        n = 0
    return w, h, n


class FrameReader:
    """Streams gray (or rgb) frames start..end (inclusive, 0-based decode
    order) from one video via ffmpeg select+rawvideo. Frame-accurate: the
    select filter counts decoded frames, no keyframe seeking involved."""

    def __init__(self, path: str, start: int, end: int, w: int, h: int,
                 pix: str = "gray"):
        self.path, self.w, self.h = str(path), w, h
        self.ch = 3 if pix == "rgb24" else 1
        self.frame_bytes = w * h * self.ch
        sel = f"select='between(n,{max(0, start)},{end})'"
        cmd = [_ffmpeg_exe(), "-v", "error", "-nostdin", "-i", self.path,
               "-vf", sel] + _fps_args() + [
               "-f", "rawvideo", "-pix_fmt", pix, "-"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, **_nowindow())

    def read(self):
        buf = b""
        need = self.frame_bytes
        out = self.proc.stdout
        while need > 0:
            chunk = out.read(need)
            if not chunk:
                return None
            buf += chunk
            need -= len(chunk)
        a = np.frombuffer(buf, dtype=np.uint8)
        if self.ch == 3:
            return a.reshape(self.h, self.w, 3)
        return a.reshape(self.h, self.w)

    def close(self):
        try:
            self.proc.stdout.close()
        except Exception:
            pass
        try:
            self.proc.kill()
        except Exception:
            pass


def grab_frames(path: str, frames: list, w: int, h: int) -> dict:
    """Decode a specific set of frames as RGB (second pass, sheet only)."""
    if not frames:
        return {}
    frames = sorted(set(int(f) for f in frames))
    expr = "+".join(f"eq(n,{f})" for f in frames)
    cmd = [_ffmpeg_exe(), "-v", "error", "-nostdin", "-i", str(path),
           "-vf", f"select='{expr}'"] + _fps_args() + [
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.run(cmd, capture_output=True, **_nowindow())
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg grab failed for {path}: "
                           f"{p.stderr.decode(errors='replace')[-300:]}")
    a = np.frombuffer(p.stdout, dtype=np.uint8)
    n = len(a) // (w * h * 3)
    a = a[: n * w * h * 3].reshape(n, h, w, 3)
    return {f: a[i] for i, f in enumerate(frames[:n])}


# --- region + metrics ---

def region_mask(src: np.ndarray, a: np.ndarray, b: np.ndarray, thresh: int,
                min_area: int = 400) -> np.ndarray:
    """Pixels where either output differs from the source beyond encoder
    noise, cleaned with morphology. Returns uint8 mask (0/1)."""
    d = cv2.max(cv2.absdiff(a, src), cv2.absdiff(b, src))
    # local mean of the difference: a restored patch changes MANY pixels a
    # little, encoder noise changes FEW pixels a little -> average first
    d_s = cv2.blur(d, (9, 9))
    m = (d_s > max(1.0, thresh * 0.5)).astype(np.uint8)
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k5)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k15)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == i] = 1
    keep = cv2.dilate(keep, k5, iterations=1)
    return keep


def _bbox(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _profile_spectrum(grad: np.ndarray, axis: int):
    """Sum a masked gradient image along one axis -> 1D profile -> |FFT|.
    axis=0 sums rows (profile across x), axis=1 sums columns."""
    prof = grad.sum(axis=axis).astype(np.float64)
    prof = prof - prof.mean()
    n = len(prof)
    if n < 16:
        return None
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(prof * win))
    return spec


def grid_metrics(src_g: np.ndarray, a_g: np.ndarray, b_g: np.ndarray,
                 mask: np.ndarray, pitch_min: int = 4, pitch_max: int = 64):
    """Estimate the mosaic pitch from the source's gradient periodicity and
    read the same spectral peak in A and B.

    Returns dict with pitch, axis, snr_src, snr_a, snr_b, res_a, res_b
    (res = peak amplitude relative to the source peak) or None if the
    source shows no usable grid."""
    bb = _bbox(mask)
    if bb is None:
        return None
    x0, y0, x1, y1 = bb
    if (x1 - x0) < 24 or (y1 - y0) < 24:
        return None
    m = mask[y0:y1, x0:x1].astype(np.float32)
    best = None
    imgs = {}
    for name, img in (("src", src_g), ("a", a_g), ("b", b_g)):
        crop = img[y0:y1, x0:x1].astype(np.float32)
        gx = np.abs(np.diff(crop, axis=1, prepend=crop[:, :1])) * m
        gy = np.abs(np.diff(crop, axis=0, prepend=crop[:1, :])) * m
        imgs[name] = (gx, gy)
    for axis_name, axis in (("x", 0), ("y", 1)):
        gsrc = imgs["src"][0] if axis_name == "x" else imgs["src"][1]
        spec = _profile_spectrum(gsrc, axis)
        if spec is None:
            continue
        n = gsrc.shape[1] if axis_name == "x" else gsrc.shape[0]
        # frequency bin k <-> pitch n/k ; restrict to plausible pitches
        k_lo = max(2, int(math.ceil(n / pitch_max)))
        k_hi = min(len(spec) - 1, int(math.floor(n / pitch_min)))
        if k_hi <= k_lo + 2:
            continue
        band = spec[k_lo:k_hi + 1]
        k_pk = int(np.argmax(band)) + k_lo
        # a block grid is a comb: strong harmonics at 2f, 3f, 4f. If the
        # argmax landed on a harmonic, step down to the fundamental (the
        # lowest sub-multiple that still carries a real peak).
        for d in (4, 3, 2):
            kd = int(round(k_pk / d))
            if kd >= max(k_lo, 2) and spec[max(1, kd - 1):kd + 2].max() >= 0.45 * spec[k_pk]:
                k_pk = int(np.argmax(spec[max(1, kd - 1):kd + 2])) + max(1, kd - 1)
                break
        floor = float(np.median(band)) + 1e-9
        snr = float(spec[k_pk]) / floor
        if best is None or snr > best["snr_src"]:
            best = {"axis": axis_name, "k": k_pk, "n": n,
                    "pitch": n / k_pk, "snr_src": snr,
                    "amp_src": float(spec[k_pk]), "floor_src": floor,
                    "k_lo": k_lo, "k_hi": k_hi}
    if best is None:
        return None
    axis = 0 if best["axis"] == "x" else 1
    out = {"pitch": round(best["pitch"], 1), "axis": best["axis"],
           "snr_src": round(best["snr_src"], 2)}
    for name in ("a", "b"):
        g = imgs[name][0] if best["axis"] == "x" else imgs[name][1]
        spec = _profile_spectrum(g, axis)
        if spec is None:
            out[f"res_{name}"] = float("nan")
            out[f"snr_{name}"] = float("nan")
            continue
        k = best["k"]
        lo, hi = max(1, k - 1), min(len(spec) - 1, k + 1)
        amp = float(spec[lo:hi + 1].max())
        band = spec[best["k_lo"]:best["k_hi"] + 1]
        floor = float(np.median(band)) + 1e-9
        out[f"res_{name}"] = round(amp / (best["amp_src"] + 1e-9), 4)
        out[f"snr_{name}"] = round(amp / floor, 2)
    return out


def sharpness(img: np.ndarray, mask: np.ndarray) -> float:
    bb = _bbox(mask)
    if bb is None:
        return float("nan")
    x0, y0, x1, y1 = bb
    crop = img[y0:y1, x0:x1].astype(np.float32)
    lap = cv2.Laplacian(crop, cv2.CV_32F, ksize=3)
    m = mask[y0:y1, x0:x1] > 0
    if m.sum() < 16:
        return float("nan")
    return float(lap[m].var())


def masked_mad(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    m = mask > 0
    if m.sum() == 0:
        return float("nan")
    return float(cv2.absdiff(x, y)[m].mean())


# --- source alignment ---

def pick_offset(a_frame: np.ndarray, src_frames: list, offsets: list) -> tuple:
    """Choose the source offset whose frame matches A best OUTSIDE the
    restored region (whole-frame median abs diff; the region is a small
    share of the frame so the median is background)."""
    best = None
    small_a = cv2.resize(a_frame, (a_frame.shape[1] // 4, a_frame.shape[0] // 4),
                         interpolation=cv2.INTER_AREA)
    for off, sf in zip(offsets, src_frames):
        if sf is None:
            continue
        small_s = cv2.resize(sf, (small_a.shape[1], small_a.shape[0]),
                             interpolation=cv2.INTER_AREA)
        d = float(np.median(cv2.absdiff(small_a, small_s)))
        mean = float(cv2.absdiff(small_a, small_s).mean())
        score = d * 1000 + mean  # median first, mean breaks ties
        if best is None or score < best[1]:
            best = (off, score, d, mean)
    return best


# --- contact sheet ---

def _fit(img: np.ndarray, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((height, height, 3), np.uint8)
    nw = max(8, int(round(w * height / h)))
    return cv2.resize(img, (nw, height), interpolation=cv2.INTER_AREA)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 8 + 9 * len(text)), 18),
                  (0, 0, 0), -1)
    cv2.putText(out, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def build_sheet(rows: list, out_path: Path, labels: tuple, tile_h: int = 240):
    """rows: list of dicts with frame, bbox, src, a, b (RGB crops), stats."""
    tiles = []
    for r in rows:
        src, a, b = r["src"], r["a"], r["b"]
        heat = cv2.absdiff(a, b).max(axis=2)
        heat = np.clip(heat.astype(np.float32) * 4.0, 0, 255).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        s = r["stats"]
        t1 = _label(_fit(src, tile_h), f"f{r['frame']} src grid={s.get('pitch', 'na')}px snr={s.get('snr_src', 'na')}")
        t2 = _label(_fit(a, tile_h), f"{labels[0]} res={s.get('res_a', 'na')} sharp={s.get('sharp_a_rel', 'na')}")
        t3 = _label(_fit(b, tile_h), f"{labels[1]} res={s.get('res_b', 'na')} sharp={s.get('sharp_b_rel', 'na')}")
        t4 = _label(_fit(heat, tile_h), f"|A-B| x4  mad={s.get('mad_ab', 'na')} area>{s.get('thresh', '')}={s.get('disagree_frac', 'na')}")
        row = np.concatenate([t1, t2, t3, t4], axis=1)
        tiles.append(row)
    if not tiles:
        return
    W = max(t.shape[1] for t in tiles)
    padded = []
    for t in tiles:
        if t.shape[1] < W:
            pad = np.zeros((t.shape[0], W - t.shape[1], 3), np.uint8)
            t = np.concatenate([t, pad], axis=1)
        padded.append(t)
        padded.append(np.full((4, W, 3), 40, np.uint8))
    sheet = np.concatenate(padded, axis=0)
    cv2.imwrite(str(out_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


# --- main ---

def _split_label(spec: str, default: str) -> tuple:
    if "=" in spec:
        lab, path = spec.split("=", 1)
        return lab.strip() or default, path
    return default, spec


def _nanmean(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(v)) if v else float("nan")


def _nanmedian(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.median(v)) if v else float("nan")


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "na"
    return f"{x:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="ab_noref",
        description="No-reference A/B of two restorations of the same real clip "
                    "(how much mosaic grid is left, sharpness, disagreement, flicker).")
    ap.add_argument("--src", required=True, help="the mosaic'd source clip both runs restored")
    ap.add_argument("--a", required=True, metavar="LABEL=PATH", help="restoration A")
    ap.add_argument("--b", required=True, metavar="LABEL=PATH", help="restoration B")
    ap.add_argument("--start", type=int, default=0, help="first output frame (0-based)")
    ap.add_argument("--end", type=int, default=None, help="last output frame (inclusive)")
    ap.add_argument("--every", type=int, default=1, help="analyse every Nth frame (default 1)")
    ap.add_argument("--thresh", type=int, default=8,
                    help="gray-level difference that counts as 'touched' (default 8)")
    ap.add_argument("--src-offset", default="auto",
                    help="source frame = output frame + offset; 'auto' picks from -8..8 "
                         "by background match (stream-copy cuts with gap-fill need this)")
    ap.add_argument("--pitch-max", type=int, default=64,
                    help="largest plausible mosaic pitch in px (default 64); "
                         "peaks beyond it are region-scale, not grid, and the "
                         "frame's grid reading is marked not readable")
    ap.add_argument("--min-snr", type=float, default=3.0,
                    help="source grid SNR below which a frame's grid reading is not trusted")
    ap.add_argument("--sheet", type=int, default=8, metavar="N",
                    help="contact sheet of the N frames where A and B disagree most (0=off)")
    ap.add_argument("--out", required=True, help="output folder")
    args = ap.parse_args()

    if cv2 is None:
        print("[ab-noref] ERROR: opencv (cv2) is required")
        return 2

    lab_a, path_a = _split_label(args.a, "A")
    lab_b, path_b = _split_label(args.b, "B")
    for p in (args.src, path_a, path_b):
        if not Path(p).exists():
            print(f"[ab-noref] ERROR: not found: {p}")
            return 2

    w, h, n_src = _probe(args.src)
    wa, ha, n_a = _probe(path_a)
    wb, hb, n_b = _probe(path_b)
    if (wa, ha) != (w, h) or (wb, hb) != (w, h):
        print(f"[ab-noref] ERROR: size mismatch src={w}x{h} A={wa}x{ha} B={wb}x{hb}")
        return 2
    n_out = min(x for x in (n_a, n_b) if x > 0) if (n_a or n_b) else 0
    start = max(0, args.start)
    end = args.end if args.end is not None else (n_out - 1 if n_out else start + 599)
    if end < start:
        print("[ab-noref] ERROR: --end before --start")
        return 2
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ab-noref] src={args.src} ({w}x{h}, {n_src or '?'} frames)")
    print(f"[ab-noref] A={lab_a}: {path_a}")
    print(f"[ab-noref] B={lab_b}: {path_b}")
    print(f"[ab-noref] frames {start}..{end} every {args.every}, thresh={args.thresh}")

    # source offset handling: read a window of source frames around start
    if args.src_offset == "auto":
        offsets = list(range(-8, 9))
    else:
        offsets = [int(args.src_offset)]
    src_first = start + min(offsets)
    rd_src = FrameReader(args.src, max(0, src_first), end + max(offsets), w, h)
    rd_a = FrameReader(path_a, start, end, w, h)
    rd_b = FrameReader(path_b, start, end, w, h)

    # prime the source window
    src_win = collections.deque()
    src_idx_next = max(0, src_first)  # decode index of the next frame rd_src yields
    win_len = len(offsets)
    for _ in range(win_len):
        f = rd_src.read()
        src_win.append(f)
        src_idx_next += 1

    a0 = rd_a.read()
    b0 = rd_b.read()
    if a0 is None or b0 is None:
        print("[ab-noref] ERROR: could not read the first output frame (range beyond file?)")
        return 2

    # choose offset
    cand_offsets = [start + o - max(0, src_first) for o in offsets]  # positions in window
    src_cands = []
    for pos in cand_offsets:
        src_cands.append(src_win[pos] if 0 <= pos < len(src_win) else None)
    pick = pick_offset(a0, src_cands, offsets)
    if pick is None:
        print("[ab-noref] ERROR: could not align source to outputs")
        return 2
    off = pick[0]
    print(f"[ab-noref] source offset = {off:+d} (background median diff {pick[2]:.1f}, mean {pick[3]:.2f})"
          + ("" if pick[2] <= 2 else "  WARNING: poor background match, check alignment"))

    # from now on: output frame k <-> source frame k + off.
    # src_win[0] currently holds source frame max(0, src_first); keep the deque
    # rolling so that the frame for k is at position (k + off) - head_idx.
    head_idx = max(0, src_first)

    rows = []
    prev = None  # (mask, src_g, a_g, b_g)
    t0 = time.time()
    k = start
    a_g, b_g = a0, b0
    n_done = 0
    while True:
        want = k + off
        # advance source window so that head_idx <= want and want is inside
        while head_idx < want - 0:
            if src_win:
                src_win.popleft()
            head_idx += 1
            if len(src_win) < win_len:
                f = rd_src.read()
                if f is not None:
                    src_win.append(f)
        while len(src_win) <= (want - head_idx):
            f = rd_src.read()
            if f is None:
                break
            src_win.append(f)
        pos = want - head_idx
        src_g = src_win[pos] if 0 <= pos < len(src_win) else None
        if src_g is None:
            print(f"[ab-noref] source ended at output frame {k}; stopping")
            break

        if (k - start) % args.every == 0:
            mask = region_mask(src_g, a_g, b_g, args.thresh)
            area = int(mask.sum())
            row = {"frame": k, "src_frame": want, "region_px": area,
                   "region_frac": round(area / float(w * h), 5)}
            if area > 0:
                gm = grid_metrics(src_g, a_g, b_g, mask, pitch_max=args.pitch_max)
                if gm is not None:
                    row.update({"pitch": gm["pitch"], "axis": gm["axis"],
                                "snr_src": gm["snr_src"],
                                "snr_a": gm["snr_a"], "snr_b": gm["snr_b"],
                                "res_a": gm["res_a"], "res_b": gm["res_b"],
                                "grid_ok": int(gm["snr_src"] >= args.min_snr
                                               and gm["pitch"] <= args.pitch_max)})
                sh_s = sharpness(src_g, mask)
                sh_a = sharpness(a_g, mask)
                sh_b = sharpness(b_g, mask)
                row.update({"sharp_src": round(sh_s, 1), "sharp_a": round(sh_a, 1),
                            "sharp_b": round(sh_b, 1),
                            "sharp_a_rel": round(sh_a / (sh_s + 1e-6), 3),
                            "sharp_b_rel": round(sh_b / (sh_s + 1e-6), 3)})
                d_ab = cv2.absdiff(a_g, b_g)
                m = mask > 0
                row["mad_ab"] = round(float(d_ab[m].mean()), 3)
                row["disagree_frac"] = round(float((d_ab[m] > args.thresh).mean()), 4)
                row["mad_a_src"] = round(masked_mad(a_g, src_g, mask), 3)
                row["mad_b_src"] = round(masked_mad(b_g, src_g, mask), 3)
                if prev is not None and (k - prev[4]) == args.every:
                    both = (mask > 0) & (prev[0] > 0)
                    if both.sum() > 0:
                        row["flick_src"] = round(float(cv2.absdiff(src_g, prev[1])[both].mean()), 3)
                        row["flick_a"] = round(float(cv2.absdiff(a_g, prev[2])[both].mean()), 3)
                        row["flick_b"] = round(float(cv2.absdiff(b_g, prev[3])[both].mean()), 3)
                bb = _bbox(mask)
                row["bbox"] = list(bb) if bb else None
            rows.append(row)
            prev = (mask, src_g, a_g, b_g, k)
            n_done += 1
            if n_done % 50 == 0:
                el = time.time() - t0
                print(f"[ab-noref] frame {k}/{end} ({n_done} analysed, {n_done / max(el, 1e-6):.1f} fps)")

        if k >= end:
            break
        a_g = rd_a.read()
        b_g = rd_b.read()
        k += 1
        if a_g is None or b_g is None:
            print(f"[ab-noref] outputs ended at frame {k - 1}; stopping")
            break

    for r in (rd_src, rd_a, rd_b):
        r.close()

    if not rows:
        print("[ab-noref] no frames analysed")
        return 1

    # --- summary ---
    touched = [r for r in rows if r.get("region_px", 0) > 0]
    gridok = [r for r in touched if r.get("grid_ok") == 1]
    summ = {
        "frames_analysed": len(rows),
        "frames_touched": len(touched),
        "frames_grid_ok": len(gridok),
        "region_frac_mean": _nanmean([r["region_frac"] for r in touched]),
        "pitch_median_px": _nanmedian([r["pitch"] for r in gridok]),
        "grid_res_a_mean": _nanmean([r["res_a"] for r in gridok]),
        "grid_res_b_mean": _nanmean([r["res_b"] for r in gridok]),
        "grid_res_a_median": _nanmedian([r["res_a"] for r in gridok]),
        "grid_res_b_median": _nanmedian([r["res_b"] for r in gridok]),
        "grid_snr_src_mean": _nanmean([r["snr_src"] for r in gridok]),
        "grid_snr_a_mean": _nanmean([r["snr_a"] for r in gridok]),
        "grid_snr_b_mean": _nanmean([r["snr_b"] for r in gridok]),
        "grid_visible_a_frames": sum(1 for r in gridok if r["snr_a"] >= args.min_snr),
        "grid_visible_b_frames": sum(1 for r in gridok if r["snr_b"] >= args.min_snr),
        "sharp_a_rel_mean": _nanmean([r["sharp_a_rel"] for r in touched]),
        "sharp_b_rel_mean": _nanmean([r["sharp_b_rel"] for r in touched]),
        "mad_ab_mean": _nanmean([r["mad_ab"] for r in touched]),
        "disagree_frac_mean": _nanmean([r["disagree_frac"] for r in touched]),
        "mad_a_src_mean": _nanmean([r["mad_a_src"] for r in touched]),
        "mad_b_src_mean": _nanmean([r["mad_b_src"] for r in touched]),
        "flick_src_mean": _nanmean([r.get("flick_src") for r in touched]),
        "flick_a_mean": _nanmean([r.get("flick_a") for r in touched]),
        "flick_b_mean": _nanmean([r.get("flick_b") for r in touched]),
    }
    A, B = lab_a, lab_b
    print("")
    print(f"[ab-noref] ===== {A} vs {B}: frames {start}..{end} (every {args.every}) =====")
    print(f"[ab-noref] analysed={summ['frames_analysed']} touched={summ['frames_touched']} "
          f"grid_readable={summ['frames_grid_ok']} (source grid SNR >= {args.min_snr})")
    print(f"[ab-noref] region: mean {100 * summ['region_frac_mean']:.2f}% of frame; "
          f"mosaic pitch median {_fmt(summ['pitch_median_px'], 1)} px")
    print(f"[ab-noref] GRID LEFT (fraction of source grid peak; 0 = gone, 1 = untouched):")
    print(f"[ab-noref]   {A:>10}: mean {_fmt(summ['grid_res_a_mean'])} median {_fmt(summ['grid_res_a_median'])} "
          f" grid-still-visible frames {summ['grid_visible_a_frames']}/{summ['frames_grid_ok']}")
    print(f"[ab-noref]   {B:>10}: mean {_fmt(summ['grid_res_b_mean'])} median {_fmt(summ['grid_res_b_median'])} "
          f" grid-still-visible frames {summ['grid_visible_b_frames']}/{summ['frames_grid_ok']}")
    print(f"[ab-noref] SHARPNESS (Laplacian var / source; >1 = sharper than the mosaic'd source):")
    print(f"[ab-noref]   {A:>10}: {_fmt(summ['sharp_a_rel_mean'])}    {B:>10}: {_fmt(summ['sharp_b_rel_mean'])}")
    print(f"[ab-noref] DISAGREEMENT inside region: mean |A-B| {_fmt(summ['mad_ab_mean'], 2)} levels; "
          f"{100 * summ['disagree_frac_mean']:.1f}% of region pixels differ by > {args.thresh}")
    print(f"[ab-noref] CHANGE vs source inside region: {A} {_fmt(summ['mad_a_src_mean'], 2)}  "
          f"{B} {_fmt(summ['mad_b_src_mean'], 2)} levels")
    print(f"[ab-noref] FLICKER (frame-to-frame |dt| inside region): src {_fmt(summ['flick_src_mean'], 2)}  "
          f"{A} {_fmt(summ['flick_a_mean'], 2)}  {B} {_fmt(summ['flick_b_mean'], 2)}")
    # verdict lines (AS IS; no correctness claim)
    if summ["frames_grid_ok"] > 0:
        ra, rb = summ["grid_res_a_median"], summ["grid_res_b_median"]
        if not math.isnan(ra) and not math.isnan(rb):
            if abs(ra - rb) < 0.03:
                who = "no meaningful difference in residual grid"
            else:
                who = f"{B if rb < ra else A} leaves less mosaic grid (median {min(ra, rb):.3f} vs {max(ra, rb):.3f})"
            print(f"[ab-noref] READING: {who}. "
                  f"Sharper: {A if summ['sharp_a_rel_mean'] > summ['sharp_b_rel_mean'] else B}. "
                  f"(Flicker: lower with lower sharpness = smoother, not steadier; "
                  f"read it against the sharpness line.)")
    print("[ab-noref] NOTE: no pristine -> these say what changed, not what is correct. "
          "Eyes/headset decide; CompareChitraRuns with -Clean decides correctness on paired clips.")

    # --- files ---
    csv_path = out_dir / "ab_noref.csv"
    cols = ["frame", "src_frame", "region_px", "region_frac", "pitch", "axis", "snr_src",
            "snr_a", "snr_b", "res_a", "res_b", "grid_ok", "sharp_src", "sharp_a", "sharp_b",
            "sharp_a_rel", "sharp_b_rel", "mad_ab", "disagree_frac", "mad_a_src", "mad_b_src",
            "flick_src", "flick_a", "flick_b", "bbox"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            rr = dict(r)
            if rr.get("bbox"):
                rr["bbox"] = " ".join(str(v) for v in rr["bbox"])
            wr.writerow(rr)
    js = {"src": args.src, "a": {"label": A, "path": path_a}, "b": {"label": B, "path": path_b},
          "start": start, "end": end, "every": args.every, "thresh": args.thresh,
          "src_offset": off, "summary": summ}
    (out_dir / "ab_noref.json").write_text(json.dumps(js, indent=2), encoding="utf-8")
    print(f"[ab-noref] wrote {csv_path}")
    print(f"[ab-noref] wrote {out_dir / 'ab_noref.json'}")

    # --- contact sheet: frames where A and B disagree most ---
    if args.sheet > 0 and touched:
        top = sorted(touched, key=lambda r: r.get("mad_ab", 0), reverse=True)[: args.sheet]
        top = sorted(top, key=lambda r: r["frame"])
        fr_out = [r["frame"] for r in top]
        fr_src = [r["src_frame"] for r in top]
        try:
            rgb_a = grab_frames(path_a, fr_out, w, h)
            rgb_b = grab_frames(path_b, fr_out, w, h)
            rgb_s = grab_frames(args.src, fr_src, w, h)
            sheet_rows = []
            for r in top:
                if r["frame"] not in rgb_a or r["frame"] not in rgb_b or r["src_frame"] not in rgb_s:
                    continue
                x0, y0, x1, y1 = r["bbox"]
                mg = 24
                x0, y0 = max(0, x0 - mg), max(0, y0 - mg)
                x1, y1 = min(w, x1 + mg), min(h, y1 + mg)
                st = dict(r)
                st["thresh"] = args.thresh
                sheet_rows.append({"frame": r["frame"], "bbox": (x0, y0, x1, y1),
                                   "src": rgb_s[r["src_frame"]][y0:y1, x0:x1],
                                   "a": rgb_a[r["frame"]][y0:y1, x0:x1],
                                   "b": rgb_b[r["frame"]][y0:y1, x0:x1],
                                   "stats": st})
            sheet_path = out_dir / "ab_noref_sheet.png"
            build_sheet(sheet_rows, sheet_path, (A, B))
            print(f"[ab-noref] wrote {sheet_path} ({len(sheet_rows)} frames: src | {A} | {B} | |A-B| heat)")
        except Exception as e:  # sheet is a convenience; never fail the run on it
            print(f"[ab-noref] contact sheet skipped: {e}")

    print(f"[ab-noref] done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
