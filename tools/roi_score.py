# tools/roi_score.py
"""No-reference quality scoring of restorations INSIDE the pasted-back regions,
for a matrix of runs of the same source (e.g. 3 scalers x 5 Max Clip Lengths).

There is no ground truth on real footage, so nothing here says "correct"; it
says sharper / softer, more or less mosaic grid left, steadier or more
flickery, and how far each run strays from the others on the same pixels.
Everything is measured only where the pipeline touched the frame, which is
what makes the numbers comparable: the untouched 97 % of the frame is
identical in every run and would otherwise drown the signal.

Where the regions come from (pick one):
  --mask  <Mask Preview output>  the "pseudo" run paints every pasted region
          magenta (RGB 255,0,255 at 70 %), so its frames ARE the per-frame
          mask; no detector, no source, no config needed. Detection settings
          must match the scored runs (the tool compares the run.json files
          when given).
  --run-json <x.run.json>         a run made with detDumpRois=true carries
          debug.detection_rois (frame -> [[top,left,bottom,right], ...]).

Three steps:

  1) extract the regions once:
     python -m tools.roi_score rois --mask D:\\Results\\ToKeep\\IPZZ-664-410-446K-mask.mp4 --out D:\\Results\\ToKeep\\rois.json
     (or: python -m tools.roi_score rois --run-json ...-mask.run.json --out rois.json)

  2) score every output (decode is in-process through OpenCV, no pipe; the
     reference run is decoded alongside each file for the disagreement
     metric; --every 3 scores a third of the frames and is plenty):
     python -m tools.roi_score score --rois D:\\Results\\ToKeep\\rois.json --out D:\\Results\\ToKeep\\scores ^
         --ref D:\\Results\\ToKeep\\180\\IPZZ-664-410-446K-restored-RC9-RTX-180.mp4 ^
         D:\\Results\\ToKeep\\90\\*.mp4 D:\\Results\\ToKeep\\120\\*.mp4 D:\\Results\\ToKeep\\180\\*.mp4 ^
         D:\\Results\\ToKeep\\240\\*.mp4 D:\\Results\\ToKeep\\300\\*.mp4

  3) the matrix:
     python -m tools.roi_score report D:\\Results\\ToKeep\\scores --out D:\\Results\\ToKeep\\matrix.md

Per-frame metrics (inside the regions, luma unless noted):
  sharp      variance of the Laplacian -- fine detail / edge energy
  hf         share of gradient energy above a coarse scale -- texture vs smooth
  grid_snr   peak-over-floor of the gradient autocorrelation at block pitches
             (4..64 px): a periodic peak = mosaic block grid still visible
  temporal   mean |frame - previous frame| where both frames have regions --
             motion inflates it for every run alike; the report subtracts
             the per-frame median across runs so what is left is the run's
             own flicker (clip joins show up as spikes)
  diff_ref   mean |frame - reference run's frame| (BGR) -- how far this run
             strays from the reference on the same pixels; 0 for the reference

report writes, per metric, a scaler x MCL table (mean over frames) plus a
join-flicker count (frames whose temporal excess is a spike), as Markdown
and CSV. Requires ffmpeg/ffprobe on PATH (or CHITRAMAYA_FFMPEG), numpy,
opencv (ChitraMaya dependencies). ASCII-only output.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


# ── ffmpeg helpers ─────────────────────────────────────────────────────

def _ffmpeg() -> str:
    env = os.environ.get("CHITRAMAYA_FFMPEG")
    if env and Path(env).is_file():
        return env
    if env and Path(env).is_dir():
        for n in ("ffmpeg.exe", "ffmpeg"):
            p = Path(env) / n
            if p.is_file():
                return str(p)
    return shutil.which("ffmpeg") or "ffmpeg"


def _ffprobe() -> str:
    f = _ffmpeg()
    p = Path(f)
    cand = p.with_name("ffprobe" + p.suffix) if p.name.lower().startswith("ffmpeg") else None
    if cand and cand.is_file():
        return str(cand)
    return shutil.which("ffprobe") or "ffprobe"


def probe(path: str) -> Tuple[int, int, int, float]:
    """(width, height, frame_count_or_-1, fps)"""
    if not Path(path).is_file():
        raise SystemExit(f"file not found: {path}")
    cmd = [_ffprobe(), "-v", "error", "-select_streams", "v:0", "-show_entries",
           "stream=width,height,nb_frames,r_frame_rate,avg_frame_rate", "-of", "json", path]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    st = json.loads(out)["streams"][0]
    w, h = int(st["width"]), int(st["height"])
    n = int(st.get("nb_frames") or -1)
    fr = st.get("avg_frame_rate") or st.get("r_frame_rate") or "0/1"
    a, b = fr.split("/")
    fps = float(a) / float(b) if float(b) else 0.0
    return w, h, n, fps


class FfmpegReader:
    """Stream BGR frames from an ffmpeg rawvideo pipe. On Windows the anonymous
    pipe caps at ~120 MB/s (CM-171) = ~5 fps at 4K, so this is the fallback,
    not the default."""

    def __init__(self, path: str, w: int, h: int, hwaccel: Optional[str] = None, every: int = 1):
        self.path, self.w, self.h, self.every = path, w, h, max(1, int(every))
        cmd = [_ffmpeg(), "-v", "error", "-nostdin"]
        if hwaccel:
            cmd += ["-hwaccel", hwaccel]
        cmd += ["-i", path]
        if self.every > 1:
            cmd += ["-vf", f"select=not(mod(n\\,{self.every}))", "-fps_mode", "passthrough"]
        cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10 ** 8)
        self.nbytes = w * h * 3
        self.buf = bytearray(self.nbytes)
        self.index = -self.every

    def read(self) -> Optional[np.ndarray]:
        view, got = memoryview(self.buf), 0
        while got < self.nbytes:
            n = self.proc.stdout.readinto(view[got:])
            if not n:
                return None
            got += n
        self.index += self.every
        return np.frombuffer(self.buf, dtype=np.uint8).reshape(self.h, self.w, 3).copy()

    def close(self):
        for fn in (lambda: self.proc.stdout.close(), lambda: self.proc.kill()):
            try:
                fn()
            except Exception:
                pass


class CvReader:
    """In-process decode through OpenCV's bundled ffmpeg: no pipe, multi-threaded
    CPU decode, hardware decode when the build offers it (--hw). Frames come
    out BGR, which is what the metrics want."""

    def __init__(self, path: str, w: int, h: int, hw: bool = False, every: int = 1):
        if cv2 is None:
            raise SystemExit("opencv is required (pip install opencv-python-headless)")
        self.path, self.w, self.h, self.every = path, w, h, max(1, int(every))
        self.cap = None
        if hw:
            try:
                cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG,
                                       [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])
                if cap.isOpened():
                    self.cap = cap
            except Exception:
                self.cap = None
        if self.cap is None:
            self.cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise SystemExit(f"OpenCV could not open {path} -- try --reader ffmpeg")
        self.index = -1
        self._first = True

    def read(self) -> Optional[np.ndarray]:
        if not self._first:
            for _ in range(self.every - 1):          # decode but do not convert the skipped frames
                if not self.cap.grab():
                    return None
                self.index += 1
        self._first = False
        ok, frm = self.cap.read()
        if not ok or frm is None:
            return None
        self.index += 1
        if frm.shape[0] != self.h or frm.shape[1] != self.w:
            frm = cv2.resize(frm, (self.w, self.h))
        return frm

    def close(self):
        try:
            self.cap.release()
        except Exception:
            pass


def open_reader(path: str, w: int, h: int, a, every: int):
    if getattr(a, "reader", "cv") == "ffmpeg":
        return FfmpegReader(path, w, h, hwaccel=a.hwaccel, every=every)
    return CvReader(path, w, h, hw=bool(a.hwaccel), every=every)


# ── 1) regions ─────────────────────────────────────────────────────────

def magenta_mask(bgr: np.ndarray) -> np.ndarray:
    """Pixels painted by the pseudo restorer (RGB 255,0,255 at 70 % over
    anything): R and B pulled up, G pulled down. Thresholds leave the feather
    band's half-strength pixels in."""
    b = bgr[:, :, 0].astype(np.int16)
    g = bgr[:, :, 1].astype(np.int16)
    r = bgr[:, :, 2].astype(np.int16)
    return ((r > 140) & (b > 140) & (g < 110) & ((r + b) // 2 - g > 70)).astype(np.uint8)


def boxes_from_mask(m: np.ndarray, min_area: int, pad: int) -> List[List[int]]:
    if cv2 is None:
        raise SystemExit("opencv is required for --mask (pip install opencv-python-headless)")
    if m.max() == 0:
        return []
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m2 = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, _, stats, _ = cv2.connectedComponentsWithStats(m2, connectivity=8)
    h, w = m.shape
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        t, l = max(0, y - pad), max(0, x - pad)
        bt, rt = min(h - 1, y + bh - 1 + pad), min(w - 1, x + bw - 1 + pad)
        out.append([int(t), int(l), int(bt), int(rt)])
    return out


def cmd_rois(a) -> int:
    rois: Dict[str, List[List[int]]] = {}
    meta = {"tool": "roi_score", "format": "frame -> [[top,left,bottom,right], ...] (pixel coords, inclusive)"}
    if a.run_json:
        data = json.loads(Path(a.run_json).read_text(encoding="utf-8"))
        dbg = (data.get("debug") or {}).get("detection_rois")
        if not dbg:
            raise SystemExit(f"{a.run_json} has no debug.detection_rois -- run with detDumpRois=true, or use --mask")
        for k, boxes in dbg.items():
            rois[str(int(k))] = [[int(v) for v in bx] for bx in boxes]
        meta.update({"source": a.run_json, "method": "detection_rois"})
        w = h = n = 0
    else:
        if not a.mask:
            raise SystemExit("give --mask <mask preview mp4> or --run-json <x.run.json>")
        w, h, n, fps = probe(a.mask)
        print(f"[rois] {a.mask}: {w}x{h}, {n if n > 0 else '?'} frames, {fps:.3f} fps; magenta regions, every {a.every}")
        rd = open_reader(a.mask, w, h, a, a.every)
        t0 = time.time()
        frames_with = 0
        while True:
            f = rd.read()
            if f is None:
                break
            ds = a.downscale
            if ds > 1:
                small = cv2.resize(f, (w // ds, h // ds), interpolation=cv2.INTER_AREA)
                bx = boxes_from_mask(magenta_mask(small), max(1, a.min_area // (ds * ds)), max(1, a.pad // ds))
                bx = [[min(h - 1, t * ds), min(w - 1, l * ds), min(h - 1, (b + 1) * ds - 1), min(w - 1, (r + 1) * ds - 1)]
                      for t, l, b, r in bx]
            else:
                bx = boxes_from_mask(magenta_mask(f), a.min_area, a.pad)
            if a.limit and rd.index >= a.limit:
                break
            if bx:
                rois[str(rd.index)] = bx
                frames_with += 1
            if rd.index % (600 * a.every) == 0 and rd.index:
                el = time.time() - t0
                fps_now = rd.index / max(el, 1e-6)
                eta = (n - rd.index) / fps_now / 60 if n > 0 and fps_now > 0 else float("nan")
                print(f"[rois] frame {rd.index}  regions on {frames_with}  {fps_now:.1f} fps  ETA {eta:.0f} min")
        rd.close()
        meta.update({"source": a.mask, "method": "magenta", "width": w, "height": h,
                     "frames_scanned": rd.index + 1 if rd.index >= 0 else 0, "every": a.every})
    meta["frames_with_regions"] = len(rois)
    meta["boxes"] = sum(len(v) for v in rois.values())
    Path(a.out).write_text(json.dumps({"meta": meta, "rois": rois}), encoding="utf-8")
    print(f"[rois] wrote {a.out}: {meta['frames_with_regions']} frames with regions, {meta['boxes']} boxes")
    return 0


# ── 2) metrics ─────────────────────────────────────────────────────────

def _luma(bgr: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return (0.114 * bgr[:, :, 0] + 0.587 * bgr[:, :, 1] + 0.299 * bgr[:, :, 2]).astype(np.float32)


def _laplacian_var(y: np.ndarray) -> float:
    if y.shape[0] < 3 or y.shape[1] < 3:
        return float("nan")
    if cv2 is not None:
        lap = cv2.Laplacian(y, cv2.CV_32F, ksize=1)[1:-1, 1:-1]
        mean, std = cv2.meanStdDev(lap)
        return float(std[0][0] ** 2)
    lap = (-4 * y[1:-1, 1:-1] + y[:-2, 1:-1] + y[2:, 1:-1] + y[1:-1, :-2] + y[1:-1, 2:])
    return float(lap.var())


def _hf_share(y: np.ndarray) -> float:
    """Gradient energy at 1 px vs at 4 px: fine texture share in [0,1]."""
    if y.shape[0] < 9 or y.shape[1] < 9:
        return float("nan")
    if cv2 is not None:
        g1 = cv2.mean(cv2.absdiff(y[:, 1:], y[:, :-1]))[0] + cv2.mean(cv2.absdiff(y[1:, :], y[:-1, :]))[0]
        g4 = (cv2.mean(cv2.absdiff(y[:, 4:], y[:, :-4]))[0] + cv2.mean(cv2.absdiff(y[4:, :], y[:-4, :]))[0]) / 4.0
    else:
        g1 = np.abs(np.diff(y, axis=1)).mean() + np.abs(np.diff(y, axis=0)).mean()
        g4 = (np.abs(y[:, 4:] - y[:, :-4]).mean() + np.abs(y[4:, :] - y[:-4, :]).mean()) / 4.0
    return float(g1 / (g1 + g4 + 1e-6))


def _grid_snr(y: np.ndarray, lo: int = 4, hi: int = 64) -> float:
    """Mosaic-grid residual. The gradient magnitude of a block grid is a comb
    (spikes at every block edge, flat inside), so its power spectrum has a
    sharp peak at 1/pitch. The spectrum is averaged over all rows (and, for the
    other axis, all columns) so natural texture averages flat; returns the
    peak over the median in the pitch band 4..64 px, max of the two axes.
    ~1-1.5 = no grid; >3 = a grid is there."""
    best = 0.0
    for axis in (1, 0):
        g = np.abs(np.diff(y, axis=axis))              # gradient across `axis`
        if axis == 0:
            g = g.T                                    # rows now run along the differenced axis
        n = g.shape[1]
        if n < 3 * lo + 8 or g.shape[0] < 4:
            continue
        g = g - g.mean(axis=1, keepdims=True)
        g = g * np.hanning(n).astype(np.float32)[None, :]
        spec = (np.abs(np.fft.rfft(g, axis=1)) ** 2).mean(axis=0)
        freqs = np.fft.rfftfreq(n, d=1.0)
        band = (freqs >= 1.0 / min(hi, n // 3)) & (freqs <= 1.0 / lo)
        if band.sum() < 6:
            continue
        seg = spec[band]
        floor = np.median(seg) + 1e-9
        best = max(best, float(seg.max() / floor))
    return best


def _grid_snr_tiled(y: np.ndarray, tile: int = 256, max_side: int = 1024) -> float:
    """Residual mosaic usually survives in a patch, not over the whole region;
    a clean majority would hide it in the averaged spectrum. So: the max of
    _grid_snr over non-overlapping tiles (plus the whole crop). Big 4K crops
    are first reduced to <= max_side on the long side (block pitches of 16-40
    px stay inside the 4-64 px band at half size)."""
    h, w = y.shape
    scale = max(h, w) / float(max_side)
    if scale > 1.0 and cv2 is not None:
        y = cv2.resize(y, (max(16, int(w / scale)), max(16, int(h / scale))), interpolation=cv2.INTER_AREA)
        h, w = y.shape
    best = _grid_snr(y)
    if h <= tile and w <= tile:
        return best
    for t in range(0, h, tile):
        for l in range(0, w, tile):
            sub = y[t:t + tile, l:l + tile]
            if sub.shape[0] >= 48 and sub.shape[1] >= 48:
                best = max(best, _grid_snr(sub))
    return best


def _union_mask(boxes: List[List[int]], h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    for t, l, b, r in boxes:
        m[max(0, t):min(h, b + 1), max(0, l):min(w, r + 1)] = True
    return m


def _box_diff(a_bgr: np.ndarray, b_bgr: np.ndarray, boxes: List[List[int]]) -> float:
    """Mean |a-b| over the union of boxes (boxes may overlap; counted once each)."""
    if not boxes:
        return float("nan")
    tot, cnt = 0.0, 0
    for t, l, b, r in boxes:
        ca = a_bgr[t:b + 1, l:r + 1]
        cb = b_bgr[t:b + 1, l:r + 1]
        if ca.size == 0 or cb.size == 0:
            continue
        d = cv2.absdiff(ca, cb) if cv2 is not None else np.abs(ca.astype(np.int16) - cb.astype(np.int16))
        tot += float(d.sum())
        cnt += d.size
    return tot / cnt if cnt else float("nan")


def _intersect(a: List[int], b: List[int]) -> Optional[List[int]]:
    t, l = max(a[0], b[0]), max(a[1], b[1])
    bt, r = min(a[2], b[2]), min(a[3], b[3])
    return [t, l, bt, r] if bt > t and r > l else None


def score_frame(bgr: np.ndarray, boxes: List[List[int]], prev_bgr: Optional[np.ndarray],
                prev_boxes: Optional[List[List[int]]], ref_bgr: Optional[np.ndarray]) -> Dict[str, float]:
    sharp, hf, grid, wts, area = [], [], [], [], 0
    for t, l, b, r in boxes:
        crop = bgr[t:b + 1, l:r + 1]
        if crop.size == 0:
            continue
        y = _luma(crop)
        px = crop.shape[0] * crop.shape[1]
        area += px
        wts.append(px)
        sharp.append(_laplacian_var(y))
        hf.append(_hf_share(y))
        grid.append(_grid_snr_tiled(y))

    def wmean(vals):
        v = np.array(vals, dtype=np.float64)
        w = np.array(wts, dtype=np.float64)
        ok = ~np.isnan(v)
        return float((v[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() and w[ok].sum() > 0 else float("nan")

    out = {"n_boxes": len(boxes), "area": area,
           "sharp": wmean(sharp) if sharp else float("nan"),       # area-weighted: a crumb cannot dilute the region
           "hf": wmean(hf) if hf else float("nan"),
           "grid_snr": float(np.nanmax(grid)) if grid else float("nan"),   # max: a grid anywhere counts
           "temporal": float("nan"), "diff_ref": float("nan")}
    if prev_bgr is not None and prev_boxes:
        inter = [x for x in (_intersect(a, b) for a in boxes for b in prev_boxes) if x]
        out["temporal"] = _box_diff(bgr, prev_bgr, inter)
    if ref_bgr is not None:
        out["diff_ref"] = _box_diff(bgr, ref_bgr, boxes)
    return out


LABEL_RE = re.compile(r"-(RTX|SPAN|ESRGAN|ESR)-?(\d{2,3})?\.mp4$", re.IGNORECASE)


def label_of(path: str) -> Tuple[str, str]:
    """(scaler, mcl) from the file name; 'ESR' and 'ESRGAN' are the same scaler."""
    m = LABEL_RE.search(Path(path).name)
    if not m:
        return Path(path).stem, "?"
    sc = m.group(1).upper()
    if sc == "ESR":
        sc = "ESRGAN"
    mcl = m.group(2)
    if not mcl and Path(path).parent.name.isdigit():   # the 180 pair was named without the suffix
        mcl = Path(path).parent.name
    return sc, (mcl or "?")


def _settings_of(run_json: Optional[Path]) -> Dict[str, object]:
    if not run_json or not run_json.is_file():
        return {}
    try:
        d = json.loads(run_json.read_text(encoding="utf-8"))
    except Exception:
        return {}
    eff = d.get("effective") or {}
    det = eff.get("detection") or d.get("detection") or {}
    keys = {}
    for k in ("imgsz", "image_size", "conf", "score", "iou", "batch", "model"):
        if k in det:
            keys[k] = det[k]
    return keys


def _find_run_json(video: str) -> Optional[Path]:
    p = Path(video)
    cands = [p.with_suffix(".run.json"), p.parent / "Support" / (p.stem + ".run.json"),
             p.parent.parent / "Support" / (p.stem + ".run.json")]
    # the 180 pair was named without the -180 suffix in the field
    stem2 = re.sub(r"-180$", "", p.stem)
    cands += [p.parent.parent / "Support" / (stem2 + ".run.json"), p.parent / "Support" / (stem2 + ".run.json")]
    for c in cands:
        if c.is_file():
            return c
    return None


def merge_boxes(boxes: List[List[int]], gap: int) -> List[List[int]]:
    """Union boxes that touch or lie within `gap` px of each other (a feathered
    region split into fragments by the quarter-res threshold). Iterates until
    nothing merges."""
    bx = [list(b) for b in boxes]
    changed = True
    while changed and len(bx) > 1:
        changed = False
        out: List[List[int]] = []
        while bx:
            cur = bx.pop()
            i = 0
            while i < len(bx):
                o = bx[i]
                if (cur[0] <= o[2] + gap and o[0] <= cur[2] + gap and
                        cur[1] <= o[3] + gap and o[1] <= cur[3] + gap):
                    cur = [min(cur[0], o[0]), min(cur[1], o[1]), max(cur[2], o[2]), max(cur[3], o[3])]
                    bx.pop(i)
                    changed = True
                else:
                    i += 1
            out.append(cur)
        bx = out
    return bx


def cmd_score(a) -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    rois_doc = json.loads(Path(a.rois).read_text(encoding="utf-8"))
    rois: Dict[int, List[List[int]]] = {int(k): v for k, v in rois_doc["rois"].items()}
    if a.merge > 0:
        before = sum(len(v) for v in rois.values())
        rois = {k: merge_boxes(v, a.merge) for k, v in rois.items()}
        after = sum(len(v) for v in rois.values())
        print(f"[score] regions: {len(rois)} frames, {before} boxes -> {after} after merging within {a.merge} px")
    files: List[str] = []
    for pat in a.files:
        hits = sorted(glob.glob(pat))
        files += hits if hits else [pat]
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_settings = _settings_of(Path(a.mask_run_json)) if a.mask_run_json else {}
    for f in files:
        if not Path(f).is_file():
            print(f"[score] missing: {f}")
            continue
        sc, mcl = label_of(f)
        stem = Path(f).stem
        csv_path = out_dir / f"{stem}.csv"
        if csv_path.is_file() and not a.force:
            print(f"[score] exists, skipping (use --force): {csv_path.name}")
            continue
        w, h, n, fps = probe(f)
        rj = _find_run_json(f)
        st = _settings_of(rj)
        if ref_settings and st and any(st.get(k) != ref_settings.get(k) for k in ref_settings):
            print(f"[score] WARNING detection settings differ from the mask run: {st} vs {ref_settings}")
        print(f"[score] {Path(f).name}  [{sc} / MCL {mcl}]  {w}x{h}  {n if n > 0 else '?'} frames  every {a.every}")
        rd = open_reader(f, w, h, a, a.every)
        rref = None
        is_ref = bool(a.ref) and Path(a.ref).resolve() == Path(f).resolve()
        if a.ref and not is_ref:
            rw, rh, _, _ = probe(a.ref)
            if (rw, rh) != (w, h):
                print(f"[score] WARNING reference size {rw}x{rh} differs; diff_ref skipped")
            else:
                rref = open_reader(a.ref, w, h, a, a.every)
        prev, prev_boxes = None, None
        rows = []
        t0 = time.time()
        while True:
            frm = rd.read()
            if frm is None:
                break
            reff = rref.read() if rref is not None else None
            i = rd.index
            boxes = rois.get(i, [])
            if boxes:
                m = score_frame(frm, boxes, prev, prev_boxes, reff)
                if is_ref:
                    m["diff_ref"] = 0.0
                m["frame"] = i
                rows.append(m)
            prev, prev_boxes = frm, boxes
            if a.limit and i >= a.limit:
                break
            if i and i % (600 * a.every) == 0:
                el = time.time() - t0
                fps_now = i / max(el, 1e-6)
                eta = (n - i) / fps_now / 60 if n > 0 and fps_now > 0 else float("nan")
                print(f"[score]   frame {i}  scored {len(rows)}  {fps_now:.1f} fps  ETA {eta:.0f} min")
        rd.close()
        if rref is not None:
            rref.close()
        cols = ["frame", "n_boxes", "area", "sharp", "hf", "grid_snr", "temporal", "diff_ref"]
        with open(csv_path, "w", newline="", encoding="ascii") as fh:
            wr = csv.DictWriter(fh, fieldnames=cols)
            wr.writeheader()
            for r in rows:
                wr.writerow({k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k]) for k in cols})
        summ = {"file": f, "scaler": sc, "mcl": mcl, "frames_scored": len(rows), "every": a.every,
                "run_json": str(rj) if rj else None, "settings": st,
                "mean": {k: float(np.nanmean([r[k] for r in rows])) if rows else None
                         for k in ("sharp", "hf", "grid_snr", "temporal", "diff_ref")},
                "median": {k: float(np.nanmedian([r[k] for r in rows])) if rows else None
                           for k in ("sharp", "hf", "grid_snr", "temporal", "diff_ref")}}
        (out_dir / f"{stem}.summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
        print(f"[score]   done: {len(rows)} frames in {time.time() - t0:.0f} s -> {csv_path.name}")
    return 0


# ── 3) report ──────────────────────────────────────────────────────────

def cmd_report(a) -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    d = Path(a.dir)
    runs = []
    for sj in sorted(d.glob("*.summary.json")):
        s = json.loads(sj.read_text(encoding="utf-8"))
        csvp = d / (sj.name[: -len(".summary.json")] + ".csv")
        per = {}
        with open(csvp, newline="", encoding="ascii") as fh:
            for r in csv.DictReader(fh):
                per[int(r["frame"])] = {k: (float(r[k]) if r[k] not in ("", "nan") else float("nan"))
                                        for k in ("sharp", "hf", "grid_snr", "temporal", "diff_ref")}
        s["per"] = per
        runs.append(s)
    if not runs:
        raise SystemExit(f"no *.summary.json in {d}")
    scalers = sorted({r["scaler"] for r in runs})
    mcls = sorted({r["mcl"] for r in runs}, key=lambda x: (int(x) if x.isdigit() else 10 ** 6))

    # temporal excess: per frame, run minus the median over all runs (motion cancels)
    frames = sorted(set().union(*[set(r["per"].keys()) for r in runs]))
    med_t = {}
    for fr in frames:
        vals = [r["per"][fr]["temporal"] for r in runs if fr in r["per"] and not np.isnan(r["per"][fr]["temporal"])]
        if len(vals) >= 3:
            med_t[fr] = float(np.median(vals))
    for r in runs:
        ex = np.array([r["per"][fr]["temporal"] - med_t[fr] for fr in r["per"] if fr in med_t
                       and not np.isnan(r["per"][fr]["temporal"])])
        if ex.size:
            mad = np.median(np.abs(ex - np.median(ex))) + 1e-6
            r["t_excess_mean"] = float(ex.mean())
            r["t_spikes"] = int((ex > np.median(ex) + a.spike_k * 1.4826 * mad).sum())
            r["t_spikes_per_1000"] = 1000.0 * r["t_spikes"] / ex.size
        else:
            r["t_excess_mean"], r["t_spikes"], r["t_spikes_per_1000"] = float("nan"), 0, float("nan")

    def cell(sc, mcl):
        for r in runs:
            if r["scaler"] == sc and r["mcl"] == mcl:
                return r
        return None

    metrics = [("sharp", "Sharpness (Laplacian variance, mean; higher = more edge energy)", "mean"),
               ("hf", "Fine-texture share (0..1, mean; higher = more 1-px texture)", "mean"),
               ("grid_snr", "Mosaic grid residual (spectral comb SNR, mean; ~1-1.5 none, >3 visible)", "mean"),
               ("temporal", "Frame-to-frame change in regions (mean abs, raw -- motion included)", "mean"),
               ("t_excess_mean", "Temporal excess over the per-frame median of all runs (flicker)", "run"),
               ("t_spikes_per_1000", f"Flicker spikes per 1000 frames (excess > {a.spike_k} MAD; clip joins)", "run"),
               ("diff_ref", "Disagreement with the reference run (mean abs BGR, 0..255)", "mean")]
    md = [f"# ROI score matrix -- {len(runs)} runs, {len(frames)} frames with regions", ""]
    csv_rows = []
    for key, title, where in metrics:
        md += [f"## {title}", "", "| scaler \\ MCL | " + " | ".join(mcls) + " |", "|---|" + "---|" * len(mcls)]
        for sc in scalers:
            cells = []
            for mcl in mcls:
                r = cell(sc, mcl)
                if r is None:
                    cells.append("--")
                    continue
                v = r["mean"].get(key) if where == "mean" else r.get(key)
                cells.append("--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}")
                csv_rows.append({"metric": key, "scaler": sc, "mcl": mcl, "value": cells[-1]})
            md.append(f"| {sc} | " + " | ".join(cells) + " |")
        md.append("")
    md += ["## Runs", "", "| scaler | MCL | frames | file |", "|---|---|---|---|"]
    for r in sorted(runs, key=lambda r: (r["scaler"], int(r["mcl"]) if r["mcl"].isdigit() else 0)):
        md.append(f"| {r['scaler']} | {r['mcl']} | {r['frames_scored']} | {Path(r['file']).name} |")
    md += ["", "Read with care: these are no-reference numbers inside the pasted regions. Sharpness and texture reward "
           "detail whether real or invented; grid residual is the one that says 'mosaic still there'; the flicker "
           "columns are the ones that should move with Max Clip Length; disagreement says where to look, not who is right."]
    Path(a.out).write_text("\n".join(md), encoding="utf-8")
    with open(Path(a.out).with_suffix(".csv"), "w", newline="", encoding="ascii") as fh:
        wr = csv.DictWriter(fh, fieldnames=["metric", "scaler", "mcl", "value"])
        wr.writeheader()
        wr.writerows(csv_rows)
    print("\n".join(md))
    print(f"\n[report] wrote {a.out} and {Path(a.out).with_suffix('.csv')}")
    return 0


# ── CLI ────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("rois", help="extract per-frame regions from a Mask Preview output or a run.json")
    r.add_argument("--mask", help="Mask Preview (pseudo) output mp4 -- magenta regions")
    r.add_argument("--run-json", help="run.json with debug.detection_rois (detDumpRois=true)")
    r.add_argument("--out", required=True)
    r.add_argument("--every", type=int, default=1, help="scan every Nth frame (default 1)")
    r.add_argument("--min-area", type=int, default=400, help="ignore blobs below this many pixels")
    r.add_argument("--pad", type=int, default=4, help="grow each box by this many pixels")
    r.add_argument("--downscale", type=int, default=4, help="find regions at 1/N resolution (default 4; boxes are scaled back)")
    r.add_argument("--reader", choices=["cv", "ffmpeg"], default="cv", help="cv = in-process OpenCV decode (default); ffmpeg = rawvideo pipe (slow on Windows at 4K)")
    r.add_argument("--hwaccel", default=None, help="cv: any value turns hardware decode on; ffmpeg: the -hwaccel name (cuda, d3d11va)")
    r.add_argument("--limit", type=int, default=0, help="stop after this many frames (timing runs)")
    r.set_defaults(fn=cmd_rois)

    s = sub.add_parser("score", help="score outputs inside the regions")
    s.add_argument("--rois", required=True, help="rois.json from the rois step")
    s.add_argument("--out", required=True, help="folder for per-file CSV + summary")
    s.add_argument("--ref", default=None, help="reference run for the disagreement metric")
    s.add_argument("--mask-run-json", default=None, help="the mask run's run.json, to check detection settings")
    s.add_argument("--every", type=int, default=1, help="score every Nth frame (must match the rois step if >1)")
    s.add_argument("--reader", choices=["cv", "ffmpeg"], default="cv")
    s.add_argument("--hwaccel", default=None, help="cv: any value turns hardware decode on; ffmpeg: the -hwaccel name")
    s.add_argument("--limit", type=int, default=0, help="stop after this many frames per file (timing runs)")
    s.add_argument("--merge", type=int, default=24, help="merge boxes closer than this many px (default 24; 0 = off)")
    s.add_argument("--force", action="store_true", help="re-score files that already have a CSV")
    s.add_argument("files", nargs="+", help="output videos (globs allowed)")
    s.set_defaults(fn=cmd_score)

    p = sub.add_parser("report", help="scaler x MCL matrix from a score folder")
    p.add_argument("dir")
    p.add_argument("--out", required=True, help="matrix.md (a .csv is written beside it)")
    p.add_argument("--spike-k", type=float, default=4.0, help="spike threshold in MADs (default 4)")
    p.set_defaults(fn=cmd_report)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
