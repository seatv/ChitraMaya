"""seam_metrics.py -- numbers for a paste seam and for restored-region detail,
computed locally on dumped frames (nothing leaves the machine).

    python seam_metrics.py frame000740A.png frame000740B.png frame000740C.png ^
        --x 1835 --y0 300 --y1 1500

    --x      column of the seam (frame pixels; hover the PNG in any viewer)
    --y0/--y1  row range to average over (a stretch of skin the seam crosses,
             no hair/eyes if you can)
    --band   width of each side's strip in px (default 40)
    --gap    px skipped on either side of the seam line itself (default 6)
    --box    optional "x0,y0,x1,y1": a region INSIDE the restored area for the
             detail metric (default: the left strip's box)

Per image it prints:
  tone step   : mean(B,G,R) of the strip right of the seam minus left of it --
                the colour step the eye sees as a line; 0 = matched
  grad step   : the same for the strongest vertical gradient across the seam
                (how hard the edge is, independent of colour)
  detail (box): std of the Laplacian inside --box -- fine-texture energy;
                higher = more detail (compare across the three caps)
Reads PNG/JPG with Pillow or OpenCV, whichever is installed.
"""
import argparse
import sys

import numpy as np


def _load(path):
    try:
        from PIL import Image
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)[..., ::-1]  # -> BGR
    except Exception:
        import cv2
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        if im is None:
            raise SystemExit(f"cannot read {path}")
        return im.astype(np.float32)


def _lap_std(img, x0, y0, x1, y1):
    g = img[y0:y1, x0:x1].mean(axis=2)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return float("nan")
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return float(lap.std())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--x", type=int, required=True)
    ap.add_argument("--y0", type=int, required=True)
    ap.add_argument("--y1", type=int, required=True)
    ap.add_argument("--band", type=int, default=40)
    ap.add_argument("--gap", type=int, default=6)
    ap.add_argument("--box", type=str, default="")
    a = ap.parse_args()

    print(f"seam x={a.x}  rows {a.y0}-{a.y1}  strips {a.band}px, gap {a.gap}px")
    print(f"{'image':44s} {'left B,G,R':>22s} {'right B,G,R':>22s} {'tone step':>10s} {'grad step':>10s} {'detail':>8s}")
    for p in a.images:
        img = _load(p)
        H, W = img.shape[:2]
        lx0, lx1 = max(0, a.x - a.gap - a.band), max(0, a.x - a.gap)
        rx0, rx1 = min(W, a.x + a.gap), min(W, a.x + a.gap + a.band)
        y0, y1 = max(0, a.y0), min(H, a.y1)
        left = img[y0:y1, lx0:lx1].reshape(-1, 3).mean(axis=0)
        right = img[y0:y1, rx0:rx1].reshape(-1, 3).mean(axis=0)
        step = right - left
        # strongest column-to-column jump of the row-averaged luminance within +-gap of x
        lum = img[y0:y1].mean(axis=2).mean(axis=0)
        seg = lum[max(1, a.x - a.gap): min(W - 1, a.x + a.gap + 1)]
        grad = float(np.abs(np.diff(seg)).max()) if seg.size > 1 else float("nan")
        if a.box:
            bx0, by0, bx1, by1 = (int(v) for v in a.box.split(","))
        else:
            bx0, by0, bx1, by1 = lx0, y0, lx1, y1
        det = _lap_std(img, bx0, by0, bx1, by1)
        name = p.replace("\\", "/").split("/")[-1]
        print(f"{name[:44]:44s} {left[0]:6.1f},{left[1]:6.1f},{left[2]:6.1f}  "
              f"{right[0]:6.1f},{right[1]:6.1f},{right[2]:6.1f}  "
              f"{np.abs(step).mean():10.2f} {grad:10.2f} {det:8.2f}")
    print("\ntone step = mean |right - left| over B,G,R (levels); grad step = largest 1-px luminance jump at the seam;")
    print("detail = Laplacian std in --box (or the left strip). Compare rows, not absolute values.")
    return 0


if __name__ == "__main__":
    sys.exit(main())