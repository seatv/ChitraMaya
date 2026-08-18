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
  2) MASK TO THE REGION. Global metrics are dominated by the ~87% of
     the frame the restorer never touched. Metrics here are reported
     both globally and inside the mosaic region -- taken from the
     misses JSON's `detection_rois` when the run was made with
     --det-dump-rois, else derived from output-vs-original divergence.

Usage:

    python -m tools.ab_eval --original PATH/orig.mp4 \
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
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np


# ── decode helpers (ffmpeg → numpy, analysis resolution) ──────────────

def _decode(path: str, w: int, h: int, pix: str = "gray") -> np.ndarray:
    """Decode a video to a (N,H,W[,3]) uint8 array at analysis scale."""
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", pix, "-"],
        capture_output=True,
    )
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed for {path}: "
                           f"{p.stderr.decode(errors='replace')[-400:]}")
    ch = 3 if pix == "rgb24" else 1
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


# ── alignment ─────────────────────────────────────────────────────────

def find_shift(orig: np.ndarray, restored: np.ndarray, max_shift: int = 10):
    """Return the shift S >= 0 (original leads by S frames) minimizing
    mean abs difference. Negative shifts (restored leading) are scanned
    too and returned as negative values."""
    best = (0, float("inf"))
    for s in range(-max_shift, max_shift + 1):
        if s >= 0:
            m = min(len(orig) - s, len(restored))
            if m < 10:
                continue
            d = float(np.abs(orig[s:s + m].astype(np.float32)
                             - restored[:m].astype(np.float32)).mean())
        else:
            m = min(len(orig), len(restored) + s)
            if m < 10:
                continue
            d = float(np.abs(orig[:m].astype(np.float32)
                             - restored[-s:-s + m].astype(np.float32)).mean())
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

def masks_from_misses(misses_path: str, n_frames: int, src_w: int, src_h: int,
                      w: int, h: int) -> np.ndarray:
    """Rasterize detection_rois from a misses JSON into per-frame boolean
    masks at analysis scale. Frame indices are RESTORED-video indices."""
    data = json.loads(Path(misses_path).read_text(encoding="utf-8"))
    rois = data.get("detection_rois")
    if not rois:
        raise KeyError(
            "misses JSON has no detection_rois -- re-run the restore with "
            "--det-dump-rois (or detDumpRois in ChitraMaya-config.json)")
    sx, sy = w / float(src_w), h / float(src_h)
    m = np.zeros((n_frames, h, w), dtype=bool)
    for k, boxes in rois.items():
        i = int(k)
        if not (0 <= i < n_frames):
            continue
        for (t, l, b, r) in boxes:
            y0 = max(0, int(t * sy)); y1 = min(h, int((b + 1) * sy) + 1)
            x0 = max(0, int(l * sx)); x1 = min(w, int((r + 1) * sx) + 1)
            m[i, y0:y1, x0:x1] = True
    return m


def masks_from_divergence(orig: np.ndarray, restores: list,
                          thresh: float = 12.0, persist: float = 0.20):
    """Fallback: static mask of pixels where any restore differs from the
    original persistently (the region that was actually regenerated)."""
    acc = None
    for r in restores:
        d = (np.abs(orig.astype(np.float32) - r.astype(np.float32))
             > thresh).mean(axis=0)
        acc = d if acc is None else np.maximum(acc, d)
    roi = acc > persist
    return np.broadcast_to(roi, orig.shape).copy()


# ── metrics ───────────────────────────────────────────────────────────

def psnr_frames(x: np.ndarray, y: np.ndarray, mask: np.ndarray = None):
    xf, yf = x.astype(np.float32), y.astype(np.float32)
    out = []
    for i in range(len(xf)):
        if mask is None:
            se = float(((xf[i] - yf[i]) ** 2).mean())
        else:
            mi = mask[i]
            if not mi.any():
                out.append(np.nan)
                continue
            se = float(((xf[i] - yf[i]) ** 2)[mi].mean())
        out.append(10 * np.log10(255.0 ** 2 / max(se, 1e-6)))
    return np.array(out)


def texture_corr(orig: np.ndarray, rest: np.ndarray, mask: np.ndarray):
    """Correlation of gradient magnitude with the original inside the
    mask -- a download-free perceptual/texture-fidelity proxy."""
    def gm(v):
        gy, gx = np.gradient(v.astype(np.float32), axis=(0, 1))
        return np.sqrt(gx * gx + gy * gy)
    out = []
    for i in range(len(orig)):
        mi = mask[i]
        if mi.sum() < 64:
            out.append(np.nan)
            continue
        a, b = gm(orig[i])[mi], gm(rest[i])[mi]
        sa, sb = a.std(), b.std()
        out.append(float(np.corrcoef(a, b)[0, 1]) if sa > 0 and sb > 0
                   else np.nan)
    return np.array(out)


def motion_fidelity(orig: np.ndarray, rest: np.ndarray, mask: np.ndarray):
    """Mean |d/dt(restore) - d/dt(original)| inside the mask: how
    faithfully the restore reproduces the original's MOTION (lower is
    better). Complements flicker, which needs no reference."""
    do = np.diff(orig.astype(np.float32), axis=0)
    dr = np.diff(rest.astype(np.float32), axis=0)
    m = mask[1:] & mask[:-1]
    vals = np.abs(dr - do)[m]
    return float(vals.mean()) if vals.size else float("nan")


def flicker(rest: np.ndarray, mask: np.ndarray):
    """No-reference temporal roughness inside the mask (lower=smoother)."""
    d = np.abs(np.diff(rest.astype(np.float32), axis=0))
    m = mask[1:] & mask[:-1]
    vals = d[m]
    return float(vals.mean()) if vals.size else float("nan")


def ssim_sampled(orig: np.ndarray, rest: np.ndarray, mask: np.ndarray,
                 every: int = 5):
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        return None
    flat = mask.any(axis=0)
    ys, xs = np.where(flat)
    if ys.size == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    vals = []
    for i in range(0, len(orig), every):
        vals.append(ssim(orig[i, y0:y1, x0:x1], rest[i, y0:y1, x0:x1],
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
                  frames: list, bbox: tuple, out_path: str):
    """Rows: original + each restore; columns: the given frame numbers;
    cropped to the ROI bounding box at source resolution."""
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    def grab(path, n, extra_shift=0):
        p = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path,
             "-vf", f"select='eq(n\\,{n + extra_shift})',"
                    f"crop={w}:{h}:{x0}:{y0}",
             "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            capture_output=True)
        a = np.frombuffer(p.stdout, dtype=np.uint8)
        if a.size != h * w * 3:
            return np.zeros((h, w, 3), dtype=np.uint8)
        return a.reshape(h, w, 3)
    rows = [np.concatenate([grab(original, n, shift) for n in frames], axis=1)]
    for _, p in labeled:
        rows.append(np.concatenate([grab(p, n) for n in frames], axis=1))
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

    orig = _decode(args.original, w, h)
    rests = {label: _decode(p, w, h) for label, p in labeled}

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
    print(f"[ab-eval] comparing {n} aligned frames")

    if args.misses:
        mask = masks_from_misses(args.misses, n, src_w, src_h, w, h)
        print(f"[ab-eval] ROI: detection_rois from {args.misses} "
              f"(mean coverage {mask.mean() * 100:.1f}% of frame)")
    else:
        mask = masks_from_divergence(orig_a, list(rests_a.values()))
        print(f"[ab-eval] ROI: derived from divergence "
              f"(coverage {mask[0].mean() * 100:.1f}% of frame) -- for the "
              f"true detected region, re-run the restore with "
              f"--det-dump-rois and pass --misses")

    results = {"aligned_frames": n, "shift": shift,
               "roi_source": "detection_rois" if args.misses else "derived",
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
        a, b = rests_a[labels[0]].astype(np.float32), \
            rests_a[labels[1]].astype(np.float32)
        xd = np.abs(a - b).mean(axis=(1, 2))
        results["cross_output_mad_mean"] = float(xd.mean())
        print(f"\n[ab-eval] {labels[0]} vs {labels[1]}: "
              f"mean abs difference {xd.mean():.2f}/255")
        div_source = xd
    else:
        only = next(iter(rests_a.values())).astype(np.float32)
        div_source = np.abs(orig_a.astype(np.float32) - only).mean(axis=(1, 2))

    res_path = out_dir / "ab_eval_results.json"
    res_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[ab-eval] results: {res_path}")

    # Deliverables are best-effort: a renderer failure must never take
    # the metrics (already printed and saved above) down with it.
    if args.contact_sheet > 0:
        try:
            top = sorted(int(i)
                         for i in np.argsort(-div_source)[:args.contact_sheet])
            flat = mask.any(axis=0)
            ys, xs = np.where(flat)
            sx, sy = src_w / float(w), src_h / float(h)
            bbox = (int(xs.min() * sx), int(ys.min() * sy),
                    int(xs.max() * sx), int(ys.max() * sy))
            sheet = out_dir / "ab_eval_contact_sheet.png"
            contact_sheet(args.original, labeled, shift, top, bbox, str(sheet))
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
