# tools/ab_eval.py
"""Paired A/B evaluation of mosaic restorations against a ground-truth
original (Batch 44; first field run: the lada-vs-ChitraMaya parity
measurement, 2026-08-16).

The doctrine this implements (see ChitraMaya-NextRelease-ToDo):
restoration is GENERATIVE -- no ground truth is recoverable from a
mosaic, so quality measurement needs a PAIRED clip: take a pristine
original, mosaic it (Add Mosaic), restore it with each contender, then
compare each restore back to the original. Two hard-won rules are baked
in here:

  1) ALIGN FIRST. A trimmed head (the field case: original led the
     restores by exactly 3 frames) makes unaligned PSNR pure garbage
     (~22 dB instead of ~37 dB). This tool scans a shift range and
     locks to the minimum-difference offset before any metric runs.
  2) MASK TO THE REGION, PER FRAME. Global metrics are dominated by the
     ~97% of the frame the restorer never touched, and on real footage
     the region MOVES. The ROI is per frame: |censored - original| when
     the censored input is given (--censored: the mosaic itself, the
     answer key, independent of restore quality -- Batch T6); else the
     misses JSON's `detection_rois` (--misses, needs --det-dump-rois);
     else |original - restore| per frame (a perfectly restored region is
     invisible to that one). --roi static keeps the legacy persistent
     mask for mosaics that do not move (PurpleRain).

Usage:

    python -m tools.ab_eval --original PATH/orig.mp4 \
        --censored PATH/orig-censored.mp4 \
        --restored lada=PATH/lada_out.mp4 \
        --restored chitramaya=PATH/cm_out.mp4 \
        [--misses PATH/out.misses.json]   # detection_rois source
        [--shift auto|N] [--scale 960x540] [--out-dir DIR]
        [--side-by-side] [--contact-sheet N]

Outputs: a metrics table on stdout, ab_eval_results.json in --out-dir,
and (on request) a labeled synced side-by-side MP4 plus a PNG contact
sheet of the highest-divergence frames. Requires only ffmpeg + numpy
(scikit-image, already a ChitraMaya dependency, enables SSIM).
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np


# ── decode helpers (ffmpeg → numpy, analysis resolution) ──────────────

def _decode(path: str, w: int, h: int, pix: str = "gray",
            memmap_dir: str = None, tag: str = None) -> np.ndarray:
    """Decode a video to a (N,H,W[,3]) uint8 array at analysis scale.

    memmap_dir=None decodes into RAM. With a directory, ffmpeg writes the
    raw frames to a file there and the array is a read-only np.memmap the
    OS pages in as needed -- the clip no longer has to fit in RAM (Batch
    T5: a 48 s 4K60 clip at 1080p analysis is 6 GB per video). The raw
    files are left behind for re-runs; delete the folder when done.
    """
    ch = 3 if pix == "rgb24" else 1
    if memmap_dir:
        Path(memmap_dir).mkdir(parents=True, exist_ok=True)
        # Name the raw file by the caller's TAG (original / restore label),
        # never by the video's stem: two restores of the same clip share a
        # stem, and the second decode would clobber the first (field case
        # 2026-09-06: v0 and v1 outputs both named *-censored-restored.mp4).
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag or Path(path).stem)
        raw = Path(memmap_dir) / f"{safe}.{w}x{h}.{pix}.raw"
        if raw.exists():
            try:
                raw.unlink()
            except OSError as e:
                raise RuntimeError(
                    f"cannot replace {raw} ({e}); is another ab_eval still "
                    f"using this --memmap folder?")
        p = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(path),
             "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", pix,
             str(raw)],
            capture_output=True,
        )
        if p.returncode != 0:
            raise RuntimeError(f"ffmpeg decode failed for {path}: "
                               f"{p.stderr.decode(errors='replace')[-400:]}")
        n = raw.stat().st_size // (w * h * ch)
        if n == 0:
            raise RuntimeError(f"no frames decoded from {path}")
        shape = (n, h, w, 3) if ch == 3 else (n, h, w)
        return np.memmap(str(raw), dtype=np.uint8, mode="r", shape=shape)
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", pix, "-"],
        capture_output=True,
    )
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed for {path}: "
                           f"{p.stderr.decode(errors='replace')[-400:]}")
    a = np.frombuffer(p.stdout, dtype=np.uint8)
    n = len(a) // (w * h * ch)
    if n == 0:
        raise RuntimeError(f"no frames decoded from {path}")
    a = a[: n * w * h * ch]
    return (a.reshape(n, h, w, 3) if ch == 3 else a.reshape(n, h, w))


def _probe_dims(path: str) -> tuple:
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    parts = p.stdout.strip().split(",")
    w, h = int(parts[0]), int(parts[1])
    num, den = parts[2].split("/")
    return w, h, float(num) / float(den or 1)


def _probe_frames(path: str) -> int:
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-count_packets", "-show_entries", "stream=nb_read_packets",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    return int(p.stdout.strip().split(",")[0])


# ── alignment ─────────────────────────────────────────────────────────

CHUNK = 64  # frames per block for every whole-video float operation


def _mad_chunked(x: np.ndarray, y: np.ndarray) -> float:
    """Mean |x-y| over two equal-length uint8 videos, CHUNK frames at a time
    (never materialises a float32 copy of the whole video)."""
    n = len(x)
    if n == 0:
        return float("inf")
    tot = 0.0
    for i in range(0, n, CHUNK):
        a = np.asarray(x[i:i + CHUNK], dtype=np.float32)
        b = np.asarray(y[i:i + CHUNK], dtype=np.float32)
        tot += float(np.abs(a - b).sum())
    return tot / float(x.size)


def find_shift(orig: np.ndarray, restored: np.ndarray, max_shift: int = 10):
    """Return the shift S >= 0 (original leads by S frames) minimizing
    mean abs difference. Negative shifts (restored leading) are scanned
    too and returned as negative values. Chunked: a 4K60 clip no longer
    needs ~13 GB for the scan."""
    best = (0, float("inf"))
    for s in range(-max_shift, max_shift + 1):
        if s >= 0:
            m = min(len(orig) - s, len(restored))
            if m < 10:
                continue
            d = _mad_chunked(orig[s:s + m], restored[:m])
        else:
            m = min(len(orig), len(restored) + s)
            if m < 10:
                continue
            d = _mad_chunked(orig[:m], restored[-s:-s + m])
        if d < best[1]:
            best = (s, d)
    return best


def _aligned(orig, rest, s):
    if s >= 0:
        m = min(len(orig) - s, len(rest))
        return orig[s:s + m], rest[:m]
    m = min(len(orig), len(rest) + s)
    return orig[:m], rest[-s:-s + m]


# ── ROI masks ─────────────────────────────────────────────────────────

try:
    from scipy.ndimage import uniform_filter as _uniform_filter  # via scikit-image
except Exception:  # pragma: no cover
    _uniform_filter = None


def _box_blur(img: np.ndarray, k: int) -> np.ndarray:
    """k x k mean filter. scipy's uniform_filter when available (fast),
    else an integral image in float64 (float32 sums lose precision on a
    1080p frame)."""
    if _uniform_filter is not None:
        return _uniform_filter(img.astype(np.float32), size=k, mode="nearest")
    r = k // 2
    x = np.pad(img.astype(np.float64), r, mode="edge")
    ii = np.zeros((x.shape[0] + 1, x.shape[1] + 1), dtype=np.float64)
    ii[1:, 1:] = x.cumsum(axis=0).cumsum(axis=1)
    h, w = img.shape
    out = (ii[k:k + h, k:k + w] - ii[:h, k:k + w] - ii[k:k + h, :w] + ii[:h, :w])
    return (out / float(k * k)).astype(np.float32)


def region_from_diff(a: np.ndarray, b: np.ndarray, thresh: float) -> np.ndarray:
    """2-D bool region where two gray frames differ beyond encoder noise:
    |a-b| averaged over ~9x9 (a mosaic/restore changes MANY pixels a little,
    encoder noise changes FEW), thresholded at thresh/2, then smoothed with
    a ~15x15 majority pass (fills pinholes, drops specks) and grown ~4 px.
    Computed at half resolution (2x2 means) for speed; the ROI edge is
    accurate to ~2 px, which the growth step covers."""
    h, w = a.shape
    h2, w2 = h // 2, w // 2
    d = np.abs(a[:h2 * 2, :w2 * 2].astype(np.int16)
               - b[:h2 * 2, :w2 * 2].astype(np.int16)).astype(np.float32)
    d = d.reshape(h2, 2, w2, 2).mean(axis=(1, 3))
    m = _box_blur(d, 5) > max(1.0, thresh * 0.5)
    m = _box_blur(m.astype(np.float32), 7) > 0.5
    m = _box_blur(m.astype(np.float32), 5) > 0.05
    full = np.repeat(np.repeat(m, 2, axis=0), 2, axis=1)
    if full.shape != (h, w):
        out = np.zeros((h, w), dtype=bool)
        out[:full.shape[0], :full.shape[1]] = full
        return out
    return full


class RoiMasks:
    """Per-frame ROI masks stored as packed bits (1 bit/pixel: 260 KB per
    1080p frame, ~750 MB for 2,906 frames) built ONCE, read via at(i).
    Sources:
      from_censored   |censored - original|  -> the mosaic itself (answer key)
      from_divergence |original - any restore| per frame (fallback; hides a
                      region a restore reproduced perfectly)
      from_misses     detection_rois boxes from a misses JSON
    A static 2-D mask (PurpleRain-style) is also accepted by the helpers."""

    def __init__(self, n: int, h: int, w: int, source: str):
        self.n, self.h, self.w, self.source = n, h, w, source
        self._rows = w
        self._bits = np.zeros((n, h, (w + 7) // 8), dtype=np.uint8)
        self._cov = np.zeros(n, dtype=np.float32)
        self._union = np.zeros((h, w), dtype=bool)
        self.ndim = 3

    def _set(self, i: int, m: np.ndarray):
        self._bits[i] = np.packbits(m, axis=1)
        self._cov[i] = float(m.mean())
        self._union |= m

    def at(self, i: int) -> np.ndarray:
        return np.unpackbits(self._bits[i], axis=1, count=self.w).astype(bool)

    def block(self, i: int, j: int) -> np.ndarray:
        return np.stack([self.at(k) for k in range(i, j)], axis=0)

    def union(self) -> np.ndarray:
        return self._union

    def coverage(self) -> float:
        return float(self._cov.mean()) if self.n else 0.0

    def empty_frames(self) -> int:
        return int((self._cov == 0).sum())

    def nbytes(self) -> int:
        return int(self._bits.nbytes)

    @classmethod
    def from_pairs(cls, n, h, w, source, pair_fn, thresh: float,
                   progress_every: int = 500):
        """pair_fn(i) -> list of (a, b) gray frames; region = union of
        region_from_diff over the pairs."""
        rm = cls(n, h, w, source)
        for i in range(n):
            m = None
            for a, b in pair_fn(i):
                r = region_from_diff(np.asarray(a), np.asarray(b), thresh)
                m = r if m is None else (m | r)
            rm._set(i, m if m is not None else np.zeros((h, w), dtype=bool))
            if progress_every and (i + 1) % progress_every == 0:
                print(f"[ab-eval] ROI masks {i + 1}/{n}")
        return rm

    @classmethod
    def from_misses(cls, misses_path: str, n_frames: int, src_w: int,
                    src_h: int, w: int, h: int):
        data = json.loads(Path(misses_path).read_text(encoding="utf-8"))
        rois = data.get("detection_rois")
        if not rois:
            raise KeyError(
                "misses JSON has no detection_rois -- re-run the restore with "
                "--det-dump-rois (or detDumpRois in ChitraMaya-config.json)")
        sx, sy = w / float(src_w), h / float(src_h)
        rm = cls(n_frames, h, w, "detection_rois")
        by_frame = {}
        for k, boxes in rois.items():
            i = int(k)
            if 0 <= i < n_frames:
                by_frame[i] = boxes
        for i in range(n_frames):
            m = np.zeros((h, w), dtype=bool)
            for (t, l, b, r) in by_frame.get(i, []):
                y0 = max(0, int(t * sy)); y1 = min(h, int((b + 1) * sy) + 1)
                x0 = max(0, int(l * sx)); x1 = min(w, int((r + 1) * sx) + 1)
                m[y0:y1, x0:x1] = True
            rm._set(i, m)
        return rm


def masks_from_divergence(orig: np.ndarray, restores: list,
                          thresh: float = 12.0, persist: float = 0.20):
    """Legacy STATIC 2-D mask (pixel differs from the original in >= persist
    of frames). Correct only for mosaics that do not move (PurpleRain);
    kept for --roi static."""
    n = len(orig)
    acc = None
    for r in restores:
        cnt = np.zeros(orig.shape[1:], dtype=np.float32)
        for i in range(0, n, CHUNK):
            a = np.asarray(orig[i:i + CHUNK], dtype=np.float32)
            b = np.asarray(r[i:i + CHUNK], dtype=np.float32)
            cnt += (np.abs(a - b) > thresh).sum(axis=0)
        d = cnt / float(max(n, 1))
        acc = d if acc is None else np.maximum(acc, d)
    return acc > persist


def _mask_at(mask, i: int) -> np.ndarray:
    """Frame i's 2-D mask from a RoiMasks, a static (H,W) mask or an (N,H,W) stack."""
    if isinstance(mask, RoiMasks):
        return mask.at(i)
    return mask if mask.ndim == 2 else mask[i]


def _mask_block(mask, i: int, j: int) -> np.ndarray:
    """(j-i, H, W) bool block of frame masks."""
    if isinstance(mask, RoiMasks):
        return mask.block(i, j)
    if mask.ndim == 2:
        return np.broadcast_to(mask, (j - i,) + mask.shape)
    return mask[i:j]


def _mask_any(mask) -> np.ndarray:
    """2-D union of the mask over all frames."""
    if isinstance(mask, RoiMasks):
        return mask.union()
    return mask if mask.ndim == 2 else mask.any(axis=0)


def _mask_coverage(mask) -> float:
    if isinstance(mask, RoiMasks):
        return mask.coverage()
    return float(mask.mean())


# ── metrics ───────────────────────────────────────────────────────────

def psnr_frames(x: np.ndarray, y: np.ndarray, mask: np.ndarray = None):
    out = []
    for i in range(len(x)):
        xf = np.asarray(x[i], dtype=np.float32)
        yf = np.asarray(y[i], dtype=np.float32)
        if mask is None:
            se = float(((xf - yf) ** 2).mean())
        else:
            mi = _mask_at(mask, i)
            if not mi.any():
                out.append(np.nan)
                continue
            se = float(((xf - yf) ** 2)[mi].mean())
        out.append(10 * np.log10(255.0 ** 2 / max(se, 1e-6)))
    return np.array(out)


def texture_corr(orig: np.ndarray, rest: np.ndarray, mask: np.ndarray):
    """Correlation of gradient magnitude with the original inside the
    mask -- a download-free perceptual/texture-fidelity proxy."""
    def gm(v):
        gy, gx = np.gradient(np.asarray(v, dtype=np.float32), axis=(0, 1))
        return np.sqrt(gx * gx + gy * gy)
    out = []
    for i in range(len(orig)):
        mi = _mask_at(mask, i)
        if mi.sum() < 64:
            out.append(np.nan)
            continue
        a, b = gm(orig[i])[mi], gm(rest[i])[mi]
        sa, sb = a.std(), b.std()
        out.append(float(np.corrcoef(a, b)[0, 1]) if sa > 0 and sb > 0
                   else np.nan)
    return np.array(out)


def _temporal_pairs(n: int):
    """Yield (start, stop) blocks such that frame pairs (t-1, t) are all
    covered exactly once: block [i, j) contributes diffs for t in i+1..j-1,
    and the next block starts at j-1 so the seam pair is not lost."""
    i = 0
    while i < n - 1:
        j = min(n, i + CHUNK)
        yield i, j
        i = j - 1


def motion_fidelity(orig: np.ndarray, rest: np.ndarray, mask: np.ndarray):
    """Mean |d/dt(restore) - d/dt(original)| inside the mask: how
    faithfully the restore reproduces the original's MOTION (lower is
    better). Complements flicker, which needs no reference. Chunked."""
    tot, cnt = 0.0, 0
    for i, j in _temporal_pairs(len(orig)):
        do = np.diff(np.asarray(orig[i:j], dtype=np.float32), axis=0)
        dr = np.diff(np.asarray(rest[i:j], dtype=np.float32), axis=0)
        mb = _mask_block(mask, i, j)
        m = mb[1:] & mb[:-1]
        vals = np.abs(dr - do)[m]
        tot += float(vals.sum())
        cnt += int(vals.size)
    return tot / cnt if cnt else float("nan")


def flicker(rest: np.ndarray, mask: np.ndarray):
    """No-reference temporal roughness inside the mask (lower=smoother). Chunked."""
    tot, cnt = 0.0, 0
    for i, j in _temporal_pairs(len(rest)):
        d = np.abs(np.diff(np.asarray(rest[i:j], dtype=np.float32), axis=0))
        mb = _mask_block(mask, i, j)
        m = mb[1:] & mb[:-1]
        vals = d[m]
        tot += float(vals.sum())
        cnt += int(vals.size)
    return tot / cnt if cnt else float("nan")


def cross_mad_frames(a: np.ndarray, b: np.ndarray, mask=None) -> np.ndarray:
    """Per-frame mean |a-b| between two uint8 videos, chunked; inside the
    mask when one is given (NaN for frames with an empty mask)."""
    out = np.empty(len(a), dtype=np.float32)
    for i in range(0, len(a), CHUNK):
        x = np.asarray(a[i:i + CHUNK], dtype=np.float32)
        y = np.asarray(b[i:i + CHUNK], dtype=np.float32)
        d = np.abs(x - y)
        if mask is None:
            out[i:i + len(x)] = d.mean(axis=tuple(range(1, x.ndim)))
        else:
            mb = _mask_block(mask, i, i + len(x))
            for k in range(len(x)):
                mk = mb[k]
                out[i + k] = float(d[k][mk].mean()) if mk.any() else np.nan
    return out


def ssim_sampled(orig: np.ndarray, rest: np.ndarray, mask: np.ndarray,
                 every: int = 5):
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        return None
    flat = _mask_any(mask)
    ys, xs = np.where(flat)
    if ys.size == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    vals = []
    for i in range(0, len(orig), every):
        vals.append(ssim(np.asarray(orig[i, y0:y1, x0:x1]),
                         np.asarray(rest[i, y0:y1, x0:x1]),
                         data_range=255))
    return float(np.mean(vals))


# ── deliverables ──────────────────────────────────────────────────────

def write_png(path: str, img: np.ndarray) -> None:
    hgt, wid = img.shape[:2]
    raw = b"".join(b"\x00" + img[i].tobytes() for i in range(hgt))
    def chunk(t, d):
        c = t + d
        return struct.pack(">I", len(d)) + c + struct.pack(">I", zlib.crc32(c))
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", wid, hgt, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    Path(path).write_bytes(png)


def _drawtext_font() -> str:
    """Return a drawtext fontfile= prefix that works on this platform.

    Field crash 2026-08-16 (Windows, gyan ffmpeg): drawtext WITHOUT an
    explicit fontfile makes fontconfig fail ("Cannot load default config
    file") and the process dies with 0xC0000005. The
    worked-in-every-test-on-Linux trap, tool edition. Fix: point at a
    font that ships with the OS; if none is found, the caller drops the
    labels entirely rather than crashing.
    """
    import os
    candidates = []
    if os.name == "nt":
        windir = os.environ.get("WINDIR", "C:/Windows")
        candidates = [f"{windir}/Fonts/arialbd.ttf",
                      f"{windir}/Fonts/arial.ttf",
                      f"{windir}/Fonts/segoeui.ttf"]
    else:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    for c in candidates:
        if Path(c).exists():
            # ffmpeg filter syntax: colon inside a quoted value is escaped
            esc = c.replace("\\", "/").replace(":", "\\:")
            return f"fontfile='{esc}':"
    return ""


def side_by_side(original: str, labeled: list, shift: int, fps: float,
                 out_path: str, pane_w: int = 1280, pane_h: int = 720):
    """Labeled synced hstack: ORIGINAL | <label> | <label> ...

    Tries labeled panes first; if drawtext is unavailable/crashy on this
    ffmpeg build, retries without labels (pane order is still
    ORIGINAL, then --restored order) instead of failing the run.
    """
    inputs = ["-i", original]
    for _, p in labeled:
        inputs += ["-i", p]
    sel = f"select='gte(n\\,{shift})'," if shift > 0 else ""

    def build_cmd(with_labels: bool):
        font = _drawtext_font() if with_labels else ""
        filters, tags = [], []
        def pane(idx, first, text):
            base = (f"[{idx}:v]{sel if first else ''}setpts=N/{fps}/TB,"
                    f"scale={pane_w}:{pane_h}")
            if with_labels:
                base += (f",drawtext={font}text='{text}':x=20:y=20:"
                         f"fontsize=48:fontcolor=white:box=1:"
                         f"boxcolor=black@0.5")
            return base + f"[v{idx}]"
        filters.append(pane(0, True, "ORIGINAL"))
        tags.append("[v0]")
        for i, (label, _) in enumerate(labeled, start=1):
            safe = label.upper().replace("'", "").replace(":", "")
            filters.append(pane(i, False, safe))
            tags.append(f"[v{i}]")
        filters.append("".join(tags) + f"hstack={len(tags)}")
        return (["ffmpeg", "-v", "error", "-y"] + inputs
                + ["-filter_complex", ";".join(filters),
                   "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                   out_path])

    try:
        subprocess.run(build_cmd(with_labels=True), check=True)
    except subprocess.CalledProcessError:
        print("[ab-eval] NOTE: drawtext failed on this ffmpeg build; "
              "rendering WITHOUT labels (pane order: ORIGINAL, then "
              "--restored order).")
        subprocess.run(build_cmd(with_labels=False), check=True)


def contact_sheet(original: str, labeled: list, shift: int,
                  frames: list, bboxes: list, out_path: str, tile_h: int = 320):
    """Rows: original + each restore; columns: the given frame numbers,
    each cropped to THAT frame's ROI bounding box at source resolution
    (per-frame boxes: the region moves) and scaled to a common tile
    height so the rows line up."""
    def grab(path, n, bbox, extra_shift=0):
        x0, y0, x1, y1 = bbox
        w, h = max(2, x1 - x0), max(2, y1 - y0)
        p = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path,
             "-vf", f"select='eq(n\\,{n + extra_shift})',"
                    f"crop={w}:{h}:{x0}:{y0},scale=-2:{tile_h}",
             "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            capture_output=True)
        a = np.frombuffer(p.stdout, dtype=np.uint8)
        tw = a.size // (tile_h * 3)
        if tw == 0 or a.size != tw * tile_h * 3:
            return np.zeros((tile_h, 8, 3), dtype=np.uint8)
        return a.reshape(tile_h, tw, 3)
    rows = [np.concatenate([grab(original, n, bb, shift)
                            for n, bb in zip(frames, bboxes)], axis=1)]
    for _, p in labeled:
        cols = [grab(p, n, bb) for n, bb in zip(frames, bboxes)]
        row = np.concatenate(cols, axis=1)
        if row.shape[1] != rows[0].shape[1]:   # a failed grab: pad/trim
            fixed = np.zeros_like(rows[0])
            m = min(row.shape[1], fixed.shape[1])
            fixed[:, :m] = row[:, :m]
            row = fixed
        rows.append(row)
    sheet = np.concatenate(rows, axis=0)
    while sheet.shape[1] > 4000:          # keep the PNG manageable
        sheet = sheet[::2, ::2]
    write_png(out_path, sheet)


# ── main ──────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Paired A/B evaluation of restorations vs a pristine "
                    "original (align first, mask to the mosaic region).")
    ap.add_argument("--original", required=True,
                    help="Pristine pre-mosaic clip (ground truth).")
    ap.add_argument("--restored", action="append", required=True,
                    metavar="LABEL=PATH",
                    help="A restored output to evaluate; repeatable. "
                         "'label=path' or a bare path (label = file stem).")
    ap.add_argument("--misses", default=None,
                    help="misses JSON containing detection_rois (from a run "
                         "with --det-dump-rois). Without it the mosaic "
                         "region is derived from output divergence.")
    ap.add_argument("--shift", default="auto",
                    help="Frame offset of the original vs the restores: "
                         "'auto' (scan +/-10) or an integer.")
    ap.add_argument("--scale", default="960x540",
                    help="Analysis resolution WxH (default 960x540).")
    ap.add_argument("--out-dir", default=".",
                    help="Where results and deliverables are written.")
    ap.add_argument("--side-by-side", action="store_true",
                    help="Also render a labeled synced comparison MP4.")
    ap.add_argument("--contact-sheet", type=int, default=0, metavar="N",
                    help="Also render a PNG contact sheet of the N frames "
                         "where the restores diverge most from each other "
                         "(needs >=2 restores) or from the original.")
    ap.add_argument("--censored", default=None, metavar="PATH",
                    help="The censored (mosaic'd) clip the restores consumed. "
                         "ROI per frame = where it differs from the original "
                         "= the mosaic itself (the answer key; no detector, "
                         "independent of restore quality). RECOMMENDED.")
    ap.add_argument("--roi", default="auto", choices=["auto", "perframe", "static"],
                    help="ROI fallback when neither --censored nor --misses is "
                         "given: perframe = |original - any restore| per frame "
                         "(moving regions OK); static = legacy persistent mask "
                         "(PurpleRain-style, mosaics that do not move). "
                         "auto = perframe.")
    ap.add_argument("--roi-thresh", type=float, default=8.0,
                    help="gray-level difference that counts as 'touched' when "
                         "deriving the ROI (default 8)")
    ap.add_argument("--memmap", default=None, metavar="DIR",
                    help="Decode to raw files in DIR and page them from disk "
                         "instead of holding every video in RAM (a 48 s 4K60 "
                         "clip at 1080p analysis is ~6 GB per video). Used "
                         "automatically (in --out-dir/decode) if the in-RAM "
                         "decode runs out of memory.")
    args = ap.parse_args()

    w, h = (int(v) for v in args.scale.lower().split("x"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labeled = []
    for spec in args.restored:
        if "=" in spec:
            label, path = spec.split("=", 1)
        else:
            label, path = Path(spec).stem, spec
        labeled.append((label, path))

    src_w, src_h, fps = _probe_dims(labeled[0][1])
    print(f"[ab-eval] source {src_w}x{src_h} @ {fps:g} fps; "
          f"analysis at {w}x{h}")

    est_gb = 0.0
    try:
        n_est = _probe_frames(labeled[0][1])
        est_gb = n_est * w * h * (1 + len(labeled)) / 2 ** 30
    except Exception:
        pass
    memmap_dir = args.memmap
    if memmap_dir:
        print(f"[ab-eval] decoding to raw files in {memmap_dir} (disk-paged)")
    elif est_gb:
        print(f"[ab-eval] in-RAM decode needs ~{est_gb:.1f} GB for "
              f"{1 + len(labeled)} videos; pass --memmap DIR to page from disk")
    if len({lab for lab, _ in labeled}) != len(labeled):
        print("[ab-eval] ERROR: --restored labels must be unique "
              "(use label=path for each)")
        return 2
    if args.censored and not Path(args.censored).exists():
        print(f"[ab-eval] ERROR: --censored not found: {args.censored}")
        return 2
    cens = None

    def _decode_all(mm):
        o = _decode(args.original, w, h, memmap_dir=mm, tag="original")
        rs = {label: _decode(p, w, h, memmap_dir=mm, tag=label)
              for label, p in labeled}
        c = _decode(args.censored, w, h, memmap_dir=mm, tag="censored") \
            if args.censored else None
        return o, rs, c

    try:
        orig, rests, cens = _decode_all(memmap_dir)
    except MemoryError:
        memmap_dir = str(out_dir / "decode")
        print(f"[ab-eval] out of RAM during decode -> retrying disk-paged "
              f"in {memmap_dir}")
        orig, rests, cens = _decode_all(memmap_dir)
    if memmap_dir:
        gb = (orig.nbytes + sum(r.nbytes for r in rests.values())
              + (cens.nbytes if cens is not None else 0)) / 2 ** 30
        print(f"[ab-eval] raw frames on disk: {gb:.1f} GB in {memmap_dir} "
              f"(delete when done)")

    first = next(iter(rests.values()))
    if args.shift == "auto":
        shift, mad = find_shift(orig, first)
        print(f"[ab-eval] alignment: original leads by {shift} frame(s) "
              f"(MAD {mad:.2f} at best shift)")
    else:
        shift = int(args.shift)
        print(f"[ab-eval] alignment: forced shift {shift}")

    orig_a, _ = _aligned(orig, first, shift)
    n = len(orig_a)
    rests_a = {}
    for label, r in rests.items():
        oa, ra = _aligned(orig, r, shift)
        m = min(len(oa), n)
        rests_a[label] = ra[:m]
    orig_a = orig_a[:min(n, min(len(r) for r in rests_a.values()))]
    n = len(orig_a)
    rests_a = {k: v[:n] for k, v in rests_a.items()}
    cens_a = None
    if cens is not None:
        _, cens_a = _aligned(orig, cens, shift)   # censored shares the restores' timeline
        cens_a = cens_a[:n]
        if len(cens_a) < n:
            n = len(cens_a)
            orig_a = orig_a[:n]
            rests_a = {k: v[:n] for k, v in rests_a.items()}
    print(f"[ab-eval] comparing {n} aligned frames")

    if args.misses:
        mask = RoiMasks.from_misses(args.misses, n, src_w, src_h, w, h)
        print(f"[ab-eval] ROI: detection_rois from {args.misses} "
              f"(mean coverage {_mask_coverage(mask) * 100:.2f}% of frame)")
    elif cens_a is not None:
        print("[ab-eval] ROI: |censored - original| per frame (the mosaic "
              "itself) ...")
        mask = RoiMasks.from_pairs(
            n, h, w, "censored",
            lambda i: [(orig_a[i], cens_a[i])], args.roi_thresh)
        print(f"[ab-eval] ROI: mean coverage {_mask_coverage(mask) * 100:.2f}% "
              f"of frame; {mask.empty_frames()} of {n} frames without mosaic; "
              f"masks {mask.nbytes() / 2 ** 20:.0f} MB packed")
    elif args.roi == "static":
        mask = masks_from_divergence(orig_a, list(rests_a.values()))
        print(f"[ab-eval] ROI: legacy STATIC divergence mask "
              f"(coverage {_mask_coverage(mask) * 100:.2f}% of frame) -- only "
              f"valid when the mosaic does not move")
    else:
        print("[ab-eval] ROI: |original - restore| per frame (union over "
              "restores) ... NOTE: a region a restore reproduced perfectly is "
              "invisible to this method; pass --censored for the answer key")
        rl = list(rests_a.values())
        mask = RoiMasks.from_pairs(
            n, h, w, "divergence",
            lambda i: [(orig_a[i], r[i]) for r in rl], args.roi_thresh)
        print(f"[ab-eval] ROI: mean coverage {_mask_coverage(mask) * 100:.2f}% "
              f"of frame; {mask.empty_frames()} of {n} frames with no region")
    if _mask_coverage(mask) == 0:
        print("[ab-eval] WARNING: ROI is EMPTY in every frame -- nothing "
              "differs beyond the threshold. Check the alignment (shift line) "
              "and that --original is the PRISTINE clip, not the censored one. "
              "ROI metrics will read n/a; global PSNR still applies.")

    results = {"aligned_frames": n, "shift": shift,
               "roi_source": (mask.source if isinstance(mask, RoiMasks) else "static"),
               "roi_coverage": _mask_coverage(mask),
               "restores": {}}
    print()
    hdr = (f"{'restore':<14} {'PSNR-ROI':>9} {'PSNR-glob':>10} "
           f"{'SSIM-ROI':>9} {'texture':>8} {'motion':>7} {'flicker':>8}")
    print(hdr)
    print("-" * len(hdr))
    for label, r in rests_a.items():
        p_roi = psnr_frames(orig_a, r, mask)
        p_glob = psnr_frames(orig_a, r)
        tex = texture_corr(orig_a, r, mask)
        mot = motion_fidelity(orig_a, r, mask)
        fli = flicker(r, mask)
        s_roi = ssim_sampled(orig_a, r, mask)
        row = {"psnr_roi_mean": float(np.nanmean(p_roi)),
               "psnr_roi_median": float(np.nanmedian(p_roi)),
               "psnr_global_mean": float(np.nanmean(p_glob)),
               "ssim_roi_sampled": s_roi,
               "texture_corr_mean": float(np.nanmean(tex)),
               "motion_deviation": mot,
               "flicker": fli}
        results["restores"][label] = row
        print(f"{label:<14} {row['psnr_roi_mean']:>8.2f}d "
              f"{row['psnr_global_mean']:>9.2f}d "
              f"{(f'{s_roi:.4f}' if s_roi is not None else 'n/a'):>9} "
              f"{row['texture_corr_mean']:>8.4f} {mot:>7.3f} {fli:>8.3f}")

    if len(rests_a) >= 2:
        labels = list(rests_a.keys())
        xd = cross_mad_frames(rests_a[labels[0]], rests_a[labels[1]])
        xr = cross_mad_frames(rests_a[labels[0]], rests_a[labels[1]], mask)
        results["cross_output_mad_mean"] = float(xd.mean())
        results["cross_output_mad_roi_mean"] = float(np.nanmean(xr)) if np.isfinite(xr).any() else None
        print(f"\n[ab-eval] {labels[0]} vs {labels[1]}: mean abs difference "
              f"{xd.mean():.2f}/255 whole frame, "
              f"{(np.nanmean(xr) if np.isfinite(xr).any() else float('nan')):.2f}/255 inside the ROI")
        div_source = np.where(np.isfinite(xr), xr, 0.0)
    else:
        only = next(iter(rests_a.values()))
        dv = cross_mad_frames(orig_a, only, mask)
        div_source = np.where(np.isfinite(dv), dv, 0.0)

    res_path = out_dir / "ab_eval_results.json"
    res_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[ab-eval] results: {res_path}")

    # Deliverables are best-effort: a renderer failure must never take
    # the metrics (already printed and saved above) down with it.
    if args.contact_sheet > 0:
        try:
            top = sorted(int(i)
                         for i in np.argsort(-div_source)[:args.contact_sheet])
            sx, sy = src_w / float(w), src_h / float(h)
            bboxes = []
            for i in top:
                mi = _mask_at(mask, i)
                ys, xs = np.where(mi)
                if ys.size == 0:
                    flat = _mask_any(mask)
                    ys, xs = np.where(flat)
                if ys.size == 0:
                    raise ValueError("empty ROI, nothing to crop")
                mg = 16
                bboxes.append((max(0, int((xs.min() - mg) * sx)),
                               max(0, int((ys.min() - mg) * sy)),
                               min(src_w, int((xs.max() + mg) * sx)),
                               min(src_h, int((ys.max() + mg) * sy))))
            sheet = out_dir / "ab_eval_contact_sheet.png"
            contact_sheet(args.original, labeled, shift, top, bboxes, str(sheet))
            print(f"[ab-eval] contact sheet (frames {top}): {sheet}")
        except Exception as e:
            print(f"[ab-eval] WARNING: contact sheet failed ({e}); "
                  f"metrics are unaffected.")

    if args.side_by_side:
        try:
            sbs = out_dir / "ab_eval_side_by_side.mp4"
            side_by_side(args.original, labeled, shift, fps, str(sbs))
            print(f"[ab-eval] side-by-side: {sbs}")
        except Exception as e:
            print(f"[ab-eval] WARNING: side-by-side failed ({e}); "
                  f"metrics are unaffected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
