# tools/mosaic_forensics.py
"""Measure what a STUDIO mosaic actually is, from a paired clip (Batch T7).

Recipe v2 for the restoration trainer is supposed to be MEASURED from real
censoring, not assumed. Given the pristine clip and its censored twin (a
studio's mosaic over the same frames), this tool finds each mosaic region
per frame (|censored - pristine|) and measures, per region:

  pitch        block size in px (gradient-profile FFT, fundamental), x and y
  ratio        pitch relative to the region (short side, sqrt(area))
  anchoring    is the block grid fixed in SCREEN space while the region moves
               (grid phase constant in frame coordinates), or does it travel
               with the region (phase constant relative to the region box)?
  blocks       flatness inside cells (0 = perfectly flat mean blocks) and how
               far each cell's value sits from the pristine cell mean
  edge         how the mosaic edge falls off: hard (<= 3 px after the encoder)
               or feathered (wider)
  shape        fill ratio of the region inside its own bounding box
               (~1.0 rectangle, ~0.79 ellipse, less = irregular)
  margin       (with --det-model) how far the mosaic extends beyond the
               anatomy box lada_nsfw finds on the pristine, per side
  distance     how far the real mosaic is from an IDEAL mean-block
               pixelation at the measured pitch/phase (mean abs diff inside
               the region) -- encoder + method, together

Outputs in --out: forensics.csv (one row per region per analysed frame),
forensics.json (aggregates + verdicts + the recipe-v2 numbers they imply),
forensics_sheet.png (pristine | censored | ideal re-pixelation | diff heat
for the N frames that fit the model WORST). Numbers only leave the machine;
the sheet is local. ASCII-only stdout.

Usage:
    python tools/mosaic_forensics.py --pristine H:\\Exam\\clean.mp4 \\
        --censored H:\\Exam\\clean-censored.mp4 --out H:\\Exam\\forensics \\
        [--det-model models\\lada_nsfw_detection_model_v1.3.pt --device 0] \\
        [--start 0 --end -1 --every 5] [--sheet 8] [--thresh 8]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None
try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover
    ndi = None

from tools.ab_eval import region_from_diff  # noqa: E402  (T6 ROI: |a-b| per frame)


# --- helpers ---------------------------------------------------------------

def _gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _profile_peak(grad: np.ndarray, axis: int, pitch_min: int, pitch_max: int):
    """|FFT| of the gradient profile summed along `axis`; return (pitch, snr,
    k, n) for the fundamental of the strongest comb in the plausible band."""
    prof = grad.sum(axis=axis).astype(np.float64)
    n = len(prof)
    if n < 16:
        return None
    prof = prof - prof.mean()
    spec = np.abs(np.fft.rfft(prof * np.hanning(n)))
    k_lo = max(2, int(math.ceil(n / pitch_max)))
    k_hi = min(len(spec) - 1, int(math.floor(n / pitch_min)))
    if k_hi <= k_lo + 2:
        return None
    band = spec[k_lo:k_hi + 1]
    k = int(np.argmax(band)) + k_lo
    for d in (4, 3, 2):  # step down to the fundamental if a harmonic won
        kd = int(round(k / d))
        if kd >= max(k_lo, 2) and spec[max(1, kd - 1):kd + 2].max() >= 0.45 * spec[k]:
            k = int(np.argmax(spec[max(1, kd - 1):kd + 2])) + max(1, kd - 1)
            break
    floor = float(np.median(band)) + 1e-9
    return (n / k, float(spec[k]) / floor, k, n)


def _refine_pitch(profile: np.ndarray, rough: float) -> float:
    """Sharpen an FFT pitch estimate: block boundaries are local maxima of
    the gradient profile; the median spacing between consecutive maxima is
    the pitch to sub-pixel precision (the FFT bin is coarse on short
    profiles: 140 px / 11 bins = 12.7 for a true 12)."""
    n = len(profile)
    if rough < 3 or n < 3 * rough:
        return rough
    prof = profile.astype(np.float64)
    prof = prof - np.convolve(prof, np.ones(int(rough) * 2 + 1) / (int(rough) * 2 + 1), mode="same")
    min_gap = max(2, int(rough * 0.6))
    idx = []
    last = -min_gap
    for i in range(1, n - 1):
        if prof[i] >= prof[i - 1] and prof[i] > prof[i + 1] and prof[i] > 0:
            if i - last >= min_gap:
                idx.append(i); last = i
            elif prof[i] > prof[last]:
                idx[-1] = i; last = i
    if len(idx) < 3:
        return rough
    gaps = np.diff(np.array(idx, dtype=np.float64))
    gaps = gaps[(gaps > rough * 0.6) & (gaps < rough * 1.5)]
    if gaps.size < 2:
        return rough
    return float(np.median(gaps))


def _phase(grad_profile: np.ndarray, pitch: int) -> int:
    """Grid phase 0..pitch-1 maximising comb energy over the profile."""
    if pitch < 2:
        return 0
    best, best_v = 0, -1.0
    for ph in range(pitch):
        v = float(grad_profile[ph::pitch].sum())
        if v > best_v:
            best, best_v = ph, v
    return best


def _circ_std(vals: np.ndarray, period: float) -> float:
    """Circular std (in px) of values living on a ring of length `period`."""
    if len(vals) < 2:
        return float("nan")
    ang = 2 * np.pi * (np.asarray(vals, dtype=np.float64) / period)
    R = float(np.hypot(np.cos(ang).mean(), np.sin(ang).mean()))
    R = min(max(R, 1e-9), 1.0)
    return float(math.sqrt(-2.0 * math.log(R)) * period / (2 * np.pi))


def _ideal_pixelate(pristine_gray: np.ndarray, mask: np.ndarray, x0: int, y0: int,
                    px: int, py: int, phx: int, phy: int) -> np.ndarray:
    """Mean-block pixelation of the pristine on a grid of pitch (px,py) whose
    boundaries sit at x = x0 + phx (mod px), y = y0 + phy (mod py), applied
    inside the mask only. Returns the full-size gray frame."""
    out = pristine_gray.copy()
    h, w = pristine_gray.shape
    ys, xs = np.where(mask)
    if xs.size == 0:
        return out
    bx0, bx1, by0, by1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    # first boundary at or before bx0 that is congruent to x0+phx mod px
    gx0 = bx0 - ((bx0 - (x0 + phx)) % px)
    gy0 = by0 - ((by0 - (y0 + phy)) % py)
    src = pristine_gray.astype(np.float32)
    for yy in range(gy0, by1, py):
        for xx in range(gx0, bx1, px):
            ya, yb = max(0, yy), min(h, yy + py)
            xa, xb = max(0, xx), min(w, xx + px)
            if yb <= ya or xb <= xa:
                continue
            m = mask[ya:yb, xa:xb]
            if not m.any():
                continue
            cell = src[ya:yb, xa:xb]
            val = float(cell.mean())
            sub = out[ya:yb, xa:xb]
            sub[m] = np.uint8(round(min(255.0, max(0.0, val))))
    return out


def _edge_ramp(cen: np.ndarray, pri: np.ndarray, ideal: np.ndarray, mask: np.ndarray) -> tuple:
    """How the mosaic blends into the pristine across its edge. Per pixel the
    mix is alpha = (censored - pristine) / (ideal - pristine), valid where
    the ideal block differs from the pristine by > 20 levels. Around the
    alpha half-level contour, ramp_px = distance over which the median alpha
    goes from 0.9 (inside) to 0.1 (outside): 1-2 px = hard edge, larger =
    feathered. Returns (valid_fraction, ramp_px)."""
    if ndi is None:
        return (float("nan"), float("nan"))
    c = cen.astype(np.float32); p = pri.astype(np.float32); i = ideal.astype(np.float32)
    den = i - p
    valid = np.abs(den) > 20
    alpha = np.zeros_like(c)
    alpha[valid] = np.clip((c[valid] - p[valid]) / den[valid], -0.25, 1.25)
    valid_frac = float(valid[mask].mean()) if mask.any() else 0.0
    core = mask & (alpha > 0.5)
    core = ndi.binary_closing(core, iterations=2)
    if core.sum() < 50 or valid_frac < 0.05:
        return (valid_frac, float("nan"))
    din = ndi.distance_transform_edt(core)
    dout = ndi.distance_transform_edt(~core)
    grown = ndi.binary_dilation(mask, iterations=10)
    prof = {}
    for r in range(1, 11):
        a = alpha[(din > r - 1) & (din <= r) & valid]
        b = alpha[(dout > r - 1) & (dout <= r) & valid & grown]
        prof[-r] = float(np.median(a)) if a.size >= 10 else float("nan")
        prof[r] = float(np.median(b)) if b.size >= 10 else float("nan")
    # ramp = how many 1-px rings around the half-level contour sit in the
    # transition band 0.2 < alpha < 0.8 (a hard edge: 1-2; sigma-2.5
    # feathering: ~4-5). Rings beyond a ring that is already flat are not
    # counted, so a misaligned interior cannot inflate the number.
    band = 0
    for r in range(1, 11):
        v = prof[-r]
        if np.isnan(v) or v >= 0.8:
            break
        band += 1
    for r in range(1, 11):
        v = prof[r]
        if np.isnan(v) or v <= 0.2:
            break
        band += 1
    return (valid_frac, float(band + 1))


def _auto_offset(cap_a, cap_b, start: int, n_b: int, scan: int = 4):
    """Small alignment search: which censored frame matches the pristine
    probe frame best outside the mosaic (whole-frame median abs diff at
    quarter scale). The probe sits `scan` frames after start so negative
    offsets are testable; two probes are averaged."""
    scores = {}
    for probe in (start + scan, start + scan + 7):
        cap_a.set(cv2.CAP_PROP_POS_FRAMES, probe)
        ok, fa = cap_a.read()
        if not ok:
            continue
        ga = _gray(fa)
        ga = cv2.resize(ga, (ga.shape[1] // 4, ga.shape[0] // 4), interpolation=cv2.INTER_AREA)
        for off in range(-scan, scan + 1):
            pos = probe + off
            if pos < 0 or pos >= n_b:
                continue
            cap_b.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, fb = cap_b.read()
            if not ok:
                continue
            gb = cv2.resize(_gray(fb), (ga.shape[1], ga.shape[0]), interpolation=cv2.INTER_AREA)
            d = float(np.median(cv2.absdiff(ga, gb)))
            scores[off] = scores.get(off, 0.0) + d
    if not scores:
        return 0
    return min(scores.items(), key=lambda kv: kv[1])[0]


# --- main ------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(prog="mosaic_forensics",
                                 description="Measure a studio mosaic from a pristine/censored pair.")
    ap.add_argument("--pristine", required=True)
    ap.add_argument("--censored", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1, help="last frame inclusive (-1 = end)")
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--thresh", type=float, default=8.0, help="ROI difference threshold (gray levels)")
    ap.add_argument("--pitch-min", type=int, default=4)
    ap.add_argument("--pitch-max", type=int, default=96)
    ap.add_argument("--max-frame-cover", type=float, default=0.6,
                    help="skip frames whose pristine/censored difference covers more than this "
                         "fraction of the picture (scene cut, tail-frame mismatch); default 0.6")
    ap.add_argument("--min-snr", type=float, default=3.0)
    ap.add_argument("--min-region-px", type=int, default=2000)
    ap.add_argument("--det-model", default=None, help="lada_nsfw .pt for the anatomy/margin measurement")
    ap.add_argument("--det-imgsz", type=int, default=640)
    ap.add_argument("--det-conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--sheet", type=int, default=8)
    ap.add_argument("--offset", default="auto", help="censored = pristine + offset frames; auto scans -3..3")
    args = ap.parse_args()

    if cv2 is None:
        print("[forensics] ERROR: opencv (cv2) is required")
        return 2
    for p in (args.pristine, args.censored):
        if not Path(p).exists():
            print(f"[forensics] ERROR: not found: {p}")
            return 2
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap_p = cv2.VideoCapture(args.pristine)
    cap_c = cv2.VideoCapture(args.censored)
    n_p = int(cap_p.get(cv2.CAP_PROP_FRAME_COUNT))
    n_c = int(cap_c.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap_p.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_p.get(cv2.CAP_PROP_FRAME_HEIGHT))
    end = args.end if args.end >= 0 else min(n_p, n_c) - 1
    print(f"[forensics] pristine {w}x{h} {n_p} frames; censored {n_c} frames; "
          f"analysing {args.start}..{end} every {args.every}")

    off = _auto_offset(cap_p, cap_c, args.start, n_c) if args.offset == "auto" else int(args.offset)
    print(f"[forensics] censored offset = {off:+d} frames")

    detector = None
    if args.det_model:
        try:
            import torch
            from chitramaya.mosaic.detector.core import Detector
            from chitramaya.mosaic.pipeline import _tensor_boxes_to_list_xyxy
            dev = "cpu" if str(args.device).lower() == "cpu" else f"cuda:{args.device}"
            if dev != "cpu" and not torch.cuda.is_available():
                dev = "cpu"
            detector = Detector(model_path=str(args.det_model), device=dev,
                                imgsz=int(args.det_imgsz), conf_thres=float(args.det_conf),
                                iou_thres=0.7, fp16=(dev != "cpu"))
            print(f"[forensics] anatomy detector: {Path(args.det_model).name} @{args.det_imgsz} "
                  f"conf {args.det_conf} on {dev}")
        except Exception as e:
            print(f"[forensics] detector unavailable ({e}); margin measurement skipped")
            detector = None

    rows = []
    anchoring = []      # (frame, region, pitch_x, phase_abs_x, bx0, pitch_y, phase_abs_y, by0)
    sheet_cands = []    # (distance, frame, bbox)
    t0 = time.time()
    frames_done = 0
    for fi in range(args.start, end + 1, args.every):
        cap_p.set(cv2.CAP_PROP_POS_FRAMES, fi)
        cap_c.set(cv2.CAP_PROP_POS_FRAMES, fi + off)
        okp, fp = cap_p.read()
        okc, fc = cap_c.read()
        if not okp or not okc:
            print(f"[forensics] read failed at frame {fi}; stopping")
            break
        gp, gc = _gray(fp), _gray(fc)
        mask = region_from_diff(gp, gc, args.thresh)
        if not mask.any():
            rows.append({"frame": fi, "region": -1, "note": "no mosaic"})
            frames_done += 1
            continue
        # T9b nit (first run 09-07): a pristine/censored pair can differ over
        # the WHOLE frame at a scene boundary or from two extra tail frames
        # (frame 2900 of the exam clip) -- that is not a mosaic region and it
        # dominated the worst-fit sheet. Reject frames whose diff support
        # covers most of the picture.
        _cover = float(mask.mean())
        if _cover > args.max_frame_cover:
            rows.append({"frame": fi, "region": -1,
                         "note": f"whole-frame difference ({100.0 * _cover:.0f}% of picture) -- "
                                 f"not a mosaic region; skipped"})
            frames_done += 1
            continue
        if ndi is not None:
            lab, nreg = ndi.label(mask)
        else:
            lab, nreg = mask.astype(np.int32), 1

        det_boxes = None
        if detector is not None:
            try:
                import torch
                t = torch.from_numpy(np.ascontiguousarray(fp))
                det = detector.detect_batch([t])[0]
                det_boxes = _tensor_boxes_to_list_xyxy(det.boxes, w=w, h=h)
            except Exception as e:
                print(f"[forensics] detector failed at frame {fi}: {e}")
                det_boxes = None

        diff_full = cv2.absdiff(gc, gp)
        for r in range(1, nreg + 1):
            rm = lab == r
            area = int(rm.sum())
            if area < args.min_region_px:
                continue
            ys, xs = np.where(rm)
            bx0, bx1, by0, by1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
            bw, bh = bx1 - bx0, by1 - by0
            row = {"frame": fi, "region": r, "area_px": area, "bbox": f"{bx0} {by0} {bx1} {by1}",
                   "region_w": bw, "region_h": bh, "short_side": min(bw, bh),
                   "fill_ratio": round(area / float(bw * bh), 3)}

            # --- pitch (x and y) from the censored crop's gradients inside the mask ---
            crop = gc[by0:by1, bx0:bx1].astype(np.float32)
            m = rm[by0:by1, bx0:bx1].astype(np.float32)
            gx = np.abs(np.diff(crop, axis=1, prepend=crop[:, :1])) * m
            gy = np.abs(np.diff(crop, axis=0, prepend=crop[:1, :])) * m
            pk_x = _profile_peak(gx, 0, args.pitch_min, args.pitch_max)
            pk_y = _profile_peak(gy, 1, args.pitch_min, args.pitch_max)
            px = py = None
            # T9b nit: refinement could walk a peak past --pitch-max (a 97 px
            # "pitch" slipped through a 96 px cap on 09-07); clamp the refined
            # value to the search window the user asked for.
            if pk_x and pk_x[1] >= args.min_snr:
                fx = _refine_pitch(gx.sum(axis=0), pk_x[0])
                if args.pitch_min <= fx <= args.pitch_max:
                    row["pitch_x"] = round(fx, 2); row["snr_x"] = round(pk_x[1], 2)
                    px = int(round(fx))
            if pk_y and pk_y[1] >= args.min_snr:
                fy = _refine_pitch(gy.sum(axis=1), pk_y[0])
                if args.pitch_min <= fy <= args.pitch_max:
                    row["pitch_y"] = round(fy, 2); row["snr_y"] = round(pk_y[1], 2)
                    py = int(round(fy))
            # sanity: a pitch wider than ~40% of the region's short side is the
            # region itself, not a grid
            lim = max(4, min(bw, bh) * 0.4)
            if px is not None and px > lim:
                row.pop("pitch_x", None); row.pop("snr_x", None); px = None
            if py is not None and py > lim:
                row.pop("pitch_y", None); row.pop("snr_y", None); py = None
            if px is None and py is None:
                row["note"] = "no readable grid"
                rows.append(row)
                continue
            if px is None:
                px = py
            if py is None:
                py = px
            pitch = 0.5 * (row.get("pitch_x", px) + row.get("pitch_y", py))
            row["pitch"] = round(pitch, 2)
            row["pitch_over_short"] = round(pitch / max(1, min(bw, bh)), 4)
            row["pitch_over_sqrt_area"] = round(pitch / math.sqrt(area), 4)

            # --- grid phase in ABSOLUTE frame coordinates ---
            phx = _phase(gx.sum(axis=0), px)
            phy = _phase(gy.sum(axis=1), py)
            phx_abs = (bx0 + phx) % px
            phy_abs = (by0 + phy) % py
            row["phase_abs_x"] = phx_abs; row["phase_abs_y"] = phy_abs
            anchoring.append((fi, r, px, phx_abs, bx0, py, phy_abs, by0))

            # --- block statistics on cells fully inside the mask ---
            flat, dev_from_mean, cells = [], [], 0
            gx0 = bx0 - ((bx0 - (bx0 + phx)) % px)
            gy0 = by0 - ((by0 - (by0 + phy)) % py)
            for yy in range(gy0, by1, py):
                for xx in range(gx0, bx1, px):
                    ya, yb, xa, xb = yy, yy + py, xx, xx + px
                    if ya < 0 or xa < 0 or yb > h or xb > w:
                        continue
                    if not rm[ya:yb, xa:xb].all():
                        continue
                    cc = gc[ya:yb, xa:xb].astype(np.float32)
                    cp = gp[ya:yb, xa:xb].astype(np.float32)
                    flat.append(float(np.abs(cc - cc.mean()).mean()))
                    dev_from_mean.append(float(abs(cc.mean() - cp.mean())))
                    cells += 1
                    if cells >= 400:
                        break
                if cells >= 400:
                    break
            row["cells"] = cells
            if cells:
                row["block_flatness"] = round(float(np.median(flat)), 2)
                row["block_dev_from_pristine_mean"] = round(float(np.median(dev_from_mean)), 2)

            # --- ideal mean-block pixelation at the measured grid ---
            ideal = _ideal_pixelate(gp, rm, bx0, by0, px, py, phx, phy)

            # --- edge (alpha ramp across the half-level contour) ---
            vf, ramp = _edge_ramp(gc, gp, ideal, rm)
            row["edge_level"] = round(vf, 2)
            row["edge_ramp_px"] = round(ramp, 1) if not np.isnan(ramp) else ""

            # --- distance from the ideal ---
            dist = float(cv2.absdiff(gc, ideal)[rm].mean())
            raw = float(diff_full[rm].mean())
            row["ideal_distance"] = round(dist, 2)
            row["raw_distance"] = round(raw, 2)   # censored vs pristine (how strong the mosaic is)

            # --- margin beyond anatomy boxes (optional) ---
            if det_boxes:
                best = None
                for (l, t, rr, b) in det_boxes:
                    ix0, iy0, ix1, iy1 = max(l, bx0), max(t, by0), min(rr, bx1), min(b, by1)
                    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
                    if inter <= 0:
                        continue
                    if best is None or inter > best[0]:
                        best = (inter, l, t, rr, b)
                if best:
                    _, l, t, rr, b = best
                    dw, dh = max(1, rr - l), max(1, b - t)
                    row["anat_box"] = f"{int(l)} {int(t)} {int(rr)} {int(b)}"
                    row["margin_left_frac"] = round((l - bx0) / dw, 3)
                    row["margin_right_frac"] = round((bx1 - rr) / dw, 3)
                    row["margin_top_frac"] = round((t - by0) / dh, 3)
                    row["margin_bottom_frac"] = round((by1 - b) / dh, 3)
                    row["mosaic_over_anat_area"] = round(area / float(dw * dh), 3)
                    row["pitch_over_anat_short"] = round(pitch / max(1, min(dw, dh)), 4)
            rows.append(row)
            sheet_cands.append((dist, fi, (bx0, by0, bx1, by1), px, py, phx, phy))
        frames_done += 1
        if frames_done % 50 == 0:
            el = time.time() - t0
            print(f"[forensics] frame {fi}/{end} ({frames_done} analysed, {frames_done / max(el, 1e-6):.1f} fps)")

    cap_p.release(); cap_c.release()
    good = [r for r in rows if "pitch" in r]
    if not good:
        print("[forensics] no readable mosaic regions found; nothing to summarise")
        return 1

    # --- aggregates ---
    def med(key):
        v = [r[key] for r in good if key in r and r[key] != ""]
        return float(np.median(v)) if v else float("nan")

    def q(key, p):
        v = [r[key] for r in good if key in r and r[key] != ""]
        return float(np.percentile(v, p)) if v else float("nan")

    # anchoring verdict: phase constant in frame coords vs constant relative to region
    verdict = {}
    for axis, (ip, iph, ib) in {"x": (2, 3, 4), "y": (5, 6, 7)}.items():
        by_pitch = {}
        for a in anchoring:
            by_pitch.setdefault(a[ip], []).append(a)
        # use the most common integer pitch so phases are comparable
        pbest = max(by_pitch.items(), key=lambda kv: len(kv[1]))
        p, items = pbest
        if len(items) < 4:
            verdict[axis] = {"pitch": p, "n": len(items), "verdict": "insufficient"}
            continue
        ph_abs = np.array([a[iph] for a in items], dtype=np.float64)
        ph_rel = np.array([(a[iph] - a[ib]) % p for a in items], dtype=np.float64)
        s_abs = _circ_std(ph_abs, p)
        s_rel = _circ_std(ph_rel, p)
        moved = float(np.std([a[ib] for a in items]))
        v = "frame-anchored" if s_abs < s_rel * 0.7 else ("region-anchored" if s_rel < s_abs * 0.7 else "undetermined")
        if moved < p:
            v += " (region barely moved; weak evidence)"
        # T9b nit (read of the first run, 09-07): the region-anchored half of
        # the test measures phase against the diff-region's bbox corner, which
        # is the mosaic's own corner only when the mask is rectangular
        # (fill ~1). A polygon/segmentation mask (fill 0.43 on the exam clip)
        # moves its bbox corner independently of the grid, so "undetermined"
        # there is the test failing, not the studio being ambiguous.
        _fills = [r["fill_ratio"] for r in rows if "fill_ratio" in r]
        _fill_med = float(np.median(_fills)) if _fills else float("nan")
        if _fills and _fill_med < 0.85:
            v += (f" (mask fill median {_fill_med:.2f} -- not rectangular; the bbox-corner "
                  f"test is unreliable, needs fill ~1)")
        verdict[axis] = {"pitch": p, "n": len(items), "phase_std_frame_px": round(s_abs, 2),
                         "phase_std_region_px": round(s_rel, 2), "region_travel_std_px": round(moved, 1),
                         "verdict": v}

    summary = {
        "frames_analysed": frames_done, "regions_measured": len(good),
        "pitch_px": {"median": med("pitch"), "p10": q("pitch", 10), "p90": q("pitch", 90)},
        "pitch_over_short_side": {"median": med("pitch_over_short"), "p10": q("pitch_over_short", 10), "p90": q("pitch_over_short", 90)},
        "pitch_over_sqrt_area": {"median": med("pitch_over_sqrt_area"), "p10": q("pitch_over_sqrt_area", 10), "p90": q("pitch_over_sqrt_area", 90)},
        "region_short_side_px": {"median": med("short_side"), "p10": q("short_side", 10), "p90": q("short_side", 90)},
        "fill_ratio": {"median": med("fill_ratio")},
        "block_flatness": {"median": med("block_flatness")},
        "block_dev_from_pristine_mean": {"median": med("block_dev_from_pristine_mean")},
        "edge_ramp_px": {"median": med("edge_ramp_px")},
        "ideal_distance": {"median": med("ideal_distance"), "p90": q("ideal_distance", 90)},
        "raw_distance": {"median": med("raw_distance")},
        "anchoring": verdict,
    }
    if any("margin_left_frac" in r for r in good):
        summary["margin_frac_of_anatomy_box"] = {
            k: med(k) for k in ("margin_left_frac", "margin_right_frac", "margin_top_frac", "margin_bottom_frac")}
        summary["mosaic_over_anat_area"] = {"median": med("mosaic_over_anat_area")}
        summary["pitch_over_anat_short"] = {"median": med("pitch_over_anat_short")}

    # recipe-v2 numbers implied by the measurements
    sh = summary["pitch_over_short_side"]
    summary["recipe_v2_implied"] = {
        "block_over_region_short_side": [round(sh["p10"], 4), round(sh["median"], 4), round(sh["p90"], 4)],
        "grid_anchor": verdict.get("x", {}).get("verdict", "?"),
        "mask_fill_ratio": round(summary["fill_ratio"]["median"], 3),
        "edge_ramp_px": summary["edge_ramp_px"]["median"],
        "cell_flatness_levels": summary["block_flatness"]["median"],
        "encoder_plus_method_distance_levels": summary["ideal_distance"]["median"],
    }

    print("")
    print(f"[forensics] ===== {len(good)} regions in {frames_done} frames =====")
    print(f"[forensics] pitch: median {summary['pitch_px']['median']:.1f} px (p10 {summary['pitch_px']['p10']:.1f}, p90 {summary['pitch_px']['p90']:.1f})")
    print(f"[forensics] pitch / region short side: median {sh['median']:.3f} (p10 {sh['p10']:.3f}, p90 {sh['p90']:.3f}); "
          f"region short side median {summary['region_short_side_px']['median']:.0f} px")
    for ax in ("x", "y"):
        v = verdict.get(ax, {})
        print(f"[forensics] grid anchoring {ax}: {v.get('verdict', '?')} "
              f"(phase std frame {v.get('phase_std_frame_px', 'na')} px vs region {v.get('phase_std_region_px', 'na')} px, "
              f"region travel std {v.get('region_travel_std_px', 'na')} px, n={v.get('n', 0)})")
    print(f"[forensics] cells: flatness {summary['block_flatness']['median']:.2f} levels (0 = flat mean blocks); "
          f"cell mean vs pristine mean {summary['block_dev_from_pristine_mean']['median']:.2f} levels")
    print(f"[forensics] edge ramp: {summary['edge_ramp_px']['median']:.1f} px (<= 3 = hard edge after the encoder; larger = feathered)")
    print(f"[forensics] shape: fill ratio {summary['fill_ratio']['median']:.2f} (1.0 rectangle, 0.79 ellipse)")
    print(f"[forensics] distance from ideal mean-block pixelation: {summary['ideal_distance']['median']:.2f} levels "
          f"(mosaic strength vs pristine: {summary['raw_distance']['median']:.1f})")
    if "margin_frac_of_anatomy_box" in summary:
        mg = summary["margin_frac_of_anatomy_box"]
        print(f"[forensics] margin beyond anatomy box (fraction of box): L {mg['margin_left_frac']:.2f} "
              f"R {mg['margin_right_frac']:.2f} T {mg['margin_top_frac']:.2f} B {mg['margin_bottom_frac']:.2f}; "
              f"mosaic area / box area {summary['mosaic_over_anat_area']['median']:.2f}; "
              f"pitch / box short side {summary['pitch_over_anat_short']['median']:.3f}")
    print(f"[forensics] RECIPE V2 IMPLIED: {json.dumps(summary['recipe_v2_implied'])}")

    # --- files ---
    cols = ["frame", "region", "area_px", "bbox", "region_w", "region_h", "short_side", "fill_ratio",
            "pitch_x", "snr_x", "pitch_y", "snr_y", "pitch", "pitch_over_short", "pitch_over_sqrt_area",
            "phase_abs_x", "phase_abs_y", "cells", "block_flatness", "block_dev_from_pristine_mean",
            "edge_level", "edge_ramp_px", "ideal_distance", "raw_distance", "anat_box",
            "margin_left_frac", "margin_right_frac", "margin_top_frac", "margin_bottom_frac",
            "mosaic_over_anat_area", "pitch_over_anat_short", "note"]
    with open(out_dir / "forensics.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    (out_dir / "forensics.json").write_text(json.dumps(
        {"pristine": args.pristine, "censored": args.censored, "offset": off,
         "start": args.start, "end": end, "every": args.every, "summary": summary}, indent=2),
        encoding="utf-8")
    print(f"[forensics] wrote {out_dir / 'forensics.csv'} and forensics.json")

    # --- sheet: the frames that fit the ideal model WORST (most to learn) ---
    if args.sheet > 0 and sheet_cands:
        try:
            worst = sorted(sheet_cands, key=lambda t: -t[0])[: args.sheet]
            worst = sorted(worst, key=lambda t: t[1])
            cap_p = cv2.VideoCapture(args.pristine); cap_c = cv2.VideoCapture(args.censored)
            tiles = []
            for dist, fi, (bx0, by0, bx1, by1), px, py, phx, phy in worst:
                cap_p.set(cv2.CAP_PROP_POS_FRAMES, fi); cap_c.set(cv2.CAP_PROP_POS_FRAMES, fi + off)
                okp, fp = cap_p.read(); okc, fc = cap_c.read()
                if not (okp and okc):
                    continue
                mg = 24
                x0, y0 = max(0, bx0 - mg), max(0, by0 - mg)
                x1, y1 = min(w, bx1 + mg), min(h, by1 + mg)
                gp, gc = _gray(fp), _gray(fc)
                rm = region_from_diff(gp, gc, args.thresh)
                ideal = _ideal_pixelate(gp, rm, bx0, by0, px, py, phx, phy)
                heat = cv2.applyColorMap(np.clip(cv2.absdiff(gc, ideal) * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
                tile_h = 300
                def fit(img):
                    hh, ww = img.shape[:2]
                    return cv2.resize(img, (max(8, int(round(ww * tile_h / hh))), tile_h), interpolation=cv2.INTER_AREA)
                a = fit(fp[y0:y1, x0:x1]); b = fit(fc[y0:y1, x0:x1])
                c = fit(cv2.cvtColor(ideal[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)); d = fit(heat[y0:y1, x0:x1])
                for img, txt in ((a, f"f{fi} pristine"), (b, f"censored pitch {px}x{py} phase {phx},{phy}"),
                                 (c, "ideal mean-block at measured grid"), (d, f"|censored-ideal| x4  mad {dist:.1f}")):
                    cv2.rectangle(img, (0, 0), (min(img.shape[1], 8 + 9 * len(txt)), 18), (0, 0, 0), -1)
                    cv2.putText(img, txt, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
                tiles.append(np.concatenate([a, b, c, d], axis=1))
            cap_p.release(); cap_c.release()
            if tiles:
                W = max(t.shape[1] for t in tiles)
                padded = []
                for t in tiles:
                    if t.shape[1] < W:
                        t = np.concatenate([t, np.zeros((t.shape[0], W - t.shape[1], 3), np.uint8)], axis=1)
                    padded.append(t); padded.append(np.full((4, W, 3), 40, np.uint8))
                cv2.imwrite(str(out_dir / "forensics_sheet.png"), np.concatenate(padded, axis=0))
                print(f"[forensics] wrote {out_dir / 'forensics_sheet.png'} ({len(tiles)} worst-fit frames)")
        except Exception as e:
            print(f"[forensics] sheet skipped: {e}")
    print(f"[forensics] done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
