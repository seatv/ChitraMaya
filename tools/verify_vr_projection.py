# tools/verify_vr_projection.py
"""Numeric validation of chitramaya.mosaic.vr_projection against ffmpeg v360.

Ground truth: ffmpeg's v360 filter (hequirect->fisheye and fisheye->hequirect).
Checks (CPU, no GPU needed):
  1. Forward warp matches `v360=hequirect:fisheye` (PSNR inside the valid circle).
  2. Inverse warp matches `v360=fisheye:hequirect` (PSNR away from edges).
  3. Roundtrip fwd+inv is near-identity in the central region.
  4. fisheye_box_to_hequirect_bbox really bounds the warped region.
Exit code 0 = all [OK]. ASCII-only output.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chitramaya.mosaic.vr_projection import (  # noqa: E402
    VRProjection, fisheye_box_to_hequirect_bbox,
)

EW = EH = 512  # eye size for the test (square eye, like real SBS content)
FAILS = []


def check(name, cond, detail=""):
    tag = "[OK]  " if cond else "[FAIL]"
    print(f"{tag} {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def make_test_eye() -> np.ndarray:
    """Deterministic hequirect eye: gradient + checkerboard + blocks."""
    y, x = np.mgrid[0:EH, 0:EW]
    img = np.zeros((EH, EW, 3), np.uint8)
    img[..., 0] = (x * 255 // EW).astype(np.uint8)
    img[..., 1] = (y * 255 // EH).astype(np.uint8)
    img[..., 2] = (((x // 32) + (y // 32)) % 2 * 200 + 30).astype(np.uint8)
    # a few solid blocks to give structure
    img[100:160, 300:380] = (255, 64, 32)
    img[350:420, 120:220] = (32, 255, 200)
    return img


def ffmpeg_v360(src: np.ndarray, filt: str) -> np.ndarray:
    with tempfile.TemporaryDirectory() as td:
        pin = os.path.join(td, "in.png")
        pout = os.path.join(td, "out.png")
        import cv2
        cv2.imwrite(pin, src)
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-i", pin, "-vf", filt, "-frames:v", "1", pout]
        subprocess.run(cmd, check=True)
        out = cv2.imread(pout, cv2.IMREAD_COLOR)
    return out


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    d = (a.astype(np.float64) - b.astype(np.float64)) ** 2
    m = mask.astype(bool)
    if d[m].size == 0:
        return 0.0
    mse = d[m].mean()
    if mse <= 0:
        return 99.0
    return 10.0 * np.log10(255.0 * 255.0 / mse)


def main() -> int:
    torch.manual_seed(0)
    eye = make_test_eye()
    frame = np.concatenate([eye, eye], axis=1)  # fake SBS (both halves equal)
    dev = torch.device("cpu")
    vp = VRProjection(EW, EH, dev)

    # circle mask (valid fisheye area), eroded away from the rim
    y, x = np.mgrid[0:EH, 0:EW]
    uf = (2.0 * (x + 0.5)) / EW - 1.0
    vf = (2.0 * (y + 0.5)) / EH - 1.0
    rad = np.hypot(uf, vf)
    circle_inner = rad < 0.92

    # ---- 1. forward vs ffmpeg ----
    t_frame = torch.from_numpy(frame)
    ours_full = vp.warp_frame_to_fisheye(t_frame).numpy()
    ours_fish = ours_full[:, :EW]
    ff_fish = ffmpeg_v360(eye, "v360=hequirect:fisheye")
    p1 = psnr(ours_fish, ff_fish, circle_inner)
    check("forward hequirect->fisheye matches ffmpeg v360", p1 > 30.0,
          f"PSNR {p1:.1f} dB inside circle (>30 required)")

    # both halves warped identically
    check("both SBS halves warped", np.array_equal(ours_full[:, :EW], ours_full[:, EW:]))

    # ---- 2. inverse vs ffmpeg ----
    import torch.nn.functional as F
    fish_t = torch.from_numpy(np.ascontiguousarray(ff_fish))
    xin = fish_t.permute(2, 0, 1).unsqueeze(0).float()
    inv = F.grid_sample(xin, vp.grid_inv(), mode="bilinear",
                        padding_mode="zeros", align_corners=False)
    ours_heq = inv.round().clamp(0, 255).to(torch.uint8).squeeze(0).permute(1, 2, 0).numpy()
    ff_heq = ffmpeg_v360(ff_fish, "v360=fisheye:hequirect")
    interior = np.zeros((EH, EW), bool)
    interior[EH // 8: EH - EH // 8, EW // 8: EW - EW // 8] = True
    p2 = psnr(ours_heq, ff_heq, interior)
    check("inverse fisheye->hequirect matches ffmpeg v360", p2 > 30.0,
          f"PSNR {p2:.1f} dB interior (>30 required)")

    # ---- 3. roundtrip near-identity (central region) ----
    p3 = psnr(ours_heq, eye, interior)
    check("roundtrip fwd+inv ~= identity in interior", p3 > 28.0,
          f"PSNR {p3:.1f} dB vs source (>28 required)")

    # ---- 4. bbox bound really bounds ----
    box = (200, 260, 280, 350)  # (t, l, b, r) in fisheye eye space
    bt, bl, bb, br = fisheye_box_to_hequirect_bbox(box, EW, EH)
    # inverse-warp an alpha canvas containing only the box; energy must stay in bbox
    canvas = torch.zeros((1, 1, EH, EW))
    canvas[0, 0, box[0]:box[2] + 1, box[1]:box[3] + 1] = 1.0
    wa = F.grid_sample(canvas, vp.grid_inv(), mode="bilinear",
                       padding_mode="zeros", align_corners=False)[0, 0].numpy()
    ys2, xs2 = np.nonzero(wa > 1e-3)
    inside = (ys2.size > 0 and ys2.min() >= bt and ys2.max() <= bb
              and xs2.min() >= bl and xs2.max() <= br)
    check("fisheye_box_to_hequirect_bbox bounds the warped region", inside,
          f"bbox=({bt},{bl},{bb},{br}) energy rows {ys2.min()}..{ys2.max()} cols {xs2.min()}..{xs2.max()}"
          if ys2.size else "no energy")

    # ---- 5. blend_projected end-to-end sanity ----
    store_frame = torch.from_numpy(frame.copy())
    crop = torch.full((box[2] - box[0] + 1, box[3] - box[1] + 1, 3), 255, dtype=torch.uint8)
    alpha = torch.ones((box[2] - box[0] + 1, box[3] - box[1] + 1), dtype=torch.float32)
    before = store_frame.clone()
    vp.blend_projected(frame_bgr_u8=store_frame, clip_img_u8=crop,
                       blend_alpha=alpha, box=box, model_dtype=torch.float32)
    diff = (store_frame.numpy().astype(int) - before.numpy().astype(int))
    changed = np.abs(diff).sum(axis=2) > 0
    ys3, xs3 = np.nonzero(changed)
    ok_region = (ys3.size > 0 and ys3.min() >= bt and ys3.max() <= bb
                 and xs3.min() >= bl and xs3.max() <= br)
    untouched_right = not changed[:, EW:].any()
    check("blend_projected writes only inside dest bbox (left eye)",
          ok_region and untouched_right,
          f"{ys3.size} px changed" if ys3.size else "no pixels changed")

    # right-eye box: same test with offset box
    box_r = (box[0], box[1] + EW, box[2], box[3] + EW)
    store_frame2 = torch.from_numpy(frame.copy())
    before2 = store_frame2.clone()
    vp.blend_projected(frame_bgr_u8=store_frame2, clip_img_u8=crop,
                       blend_alpha=alpha, box=box_r, model_dtype=torch.float32)
    diff2 = (store_frame2.numpy().astype(int) - before2.numpy().astype(int))
    changed2 = np.abs(diff2).sum(axis=2) > 0
    check("blend_projected right-eye box lands in right half",
          changed2[:, EW:].any() and not changed2[:, :EW].any())

    # scratch canvases zeroed after use
    check("scratch canvases zeroed after blend",
          float(vp._canvas_img.abs().sum()) == 0.0
          and float(vp._canvas_alpha.abs().sum()) == 0.0)

    print()
    if FAILS:
        print(f"RESULT: {len(FAILS)} check(s) FAILED: {FAILS}")
        return 1
    print("RESULT: all vr_projection checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
