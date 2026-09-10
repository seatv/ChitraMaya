# tools/verify_blend_mask_cm172.py
"""CM-172 check: the summed-area box filter must reproduce the legacy blend masks.

Compares, for many random region shapes:
  legacy create_blend_mask        (F.conv2d, uniform kernel, reflect pad)
  legacy create_support_blend_mask (F.avg_pool2d, zero pad)
against the shipped implementations in chitramaya.mosaic.utils.mask_utils,
which now build the same box means from cumulative sums (no conv2d, no
avg_pool2d -> no per-shape MIOpen compile on ROCm).

Reports max |diff| in alpha, the number of uint8 pixels that would differ in a
composite (round((clip - frame) * alpha + frame)), and per-shape wall time for
both paths. On the AMD box the timing column is the point: legacy pays a
MIOpen search/compile for every new (h, w, k); the new path does not.

    python tools/verify_blend_mask_cm172.py                 # cpu
    python tools/verify_blend_mask_cm172.py --device cuda   # NVIDIA or ROCm
    python tools/verify_blend_mask_cm172.py --device xpu
    python tools/verify_blend_mask_cm172.py --shapes 400 --seed 7

Tolerances. The two paths are not bit-identical and should not be: the legacy
conv2d accumulates up to k*k float32 products of 1/k^2 (k is 5% of the crop,
so ~1,600 taps on a 700 px region) and carries ~1e-5 of rounding; the
summed-area table sums exact 0/1 values in float64 and rounds once. When the
composite disagrees it is by one 8-bit level on a pixel whose blended value sat
within that rounding of a .5 boundary -- a few hundredths of a percent of the
region. Defaults: |alpha diff| <= 1e-4, differing pixels <= 0.1% of the region,
never more than 1 level apart. Exit code 0 on PASS, 1 otherwise.
"""
from __future__ import annotations

import argparse
import sys
import time

import torch
import torch.nn.functional as F

from chitramaya.mosaic.utils import mask_utils


# ----------------------------------------------------------------------------
# Legacy implementations (verbatim semantics of the pre-CM-172 code)
# ----------------------------------------------------------------------------
def _legacy_filter2D(img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    k = kernel.size(-1)
    b, c, h, w = img.size()
    if k % 2 == 1:
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode="reflect")
    else:
        img = F.pad(img, (k // 2, k // 2 - 1, k // 2, k // 2 - 1), mode="reflect")
    ph, pw = img.size()[-2:]
    img = img.view(b * c, 1, ph, pw)
    kernel = kernel.view(1, 1, k, k)
    return F.conv2d(img, kernel, padding=0).view(b, c, h, w)


def legacy_create_blend_mask(crop_mask: torch.Tensor) -> torch.Tensor:
    mask = crop_mask.squeeze()
    h, w = mask.shape
    border_ratio = 0.05
    h_inner, w_inner = int(h * (1.0 - border_ratio)), int(w * (1.0 - border_ratio))
    h_outer, w_outer = h - h_inner, w - w_inner
    border_size = min(h_outer, w_outer)
    if border_size < 5:
        return torch.ones_like(mask)
    blur_size = int(border_size)
    if blur_size % 2 == 0:
        blur_size += 1
    inner = torch.ones((h_inner, w_inner), device=mask.device, dtype=mask.dtype)
    pad_top = h_outer // 2
    pad_bottom = h_outer - pad_top
    pad_left = w_outer // 2
    pad_right = w_outer - pad_left
    blend = F.pad(inner, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
    mask4 = (mask > 0)
    blend = torch.maximum(mask4, blend)
    kernel = torch.tensor(1.0 / (blur_size ** 2), device=blend.device, dtype=blend.dtype).expand(1, blur_size, blur_size)
    blend = _legacy_filter2D(blend.unsqueeze(0).unsqueeze(0), kernel).squeeze(0).squeeze(0)
    return blend


def legacy_create_support_blend_mask(crop_mask: torch.Tensor, feather_px=None, *,
                                     min_feather_px: int = 2, max_feather_ratio: float = 0.05,
                                     passes: int = 1) -> torch.Tensor:
    mask = crop_mask.squeeze()
    support = (mask > 0).to(dtype=torch.float32)
    h, w = int(support.shape[0]), int(support.shape[1])
    if not bool(support.any()):
        return torch.zeros_like(support)
    if feather_px is None:
        feather_px = int(round(min(h, w) * float(max_feather_ratio)))
        feather_px = max(int(min_feather_px), feather_px)
    feather_px = int(max(0, feather_px))
    if feather_px <= 0:
        return support
    k = 2 * feather_px + 1
    alpha = support.unsqueeze(0).unsqueeze(0)
    for _ in range(max(1, int(passes))):
        alpha = F.avg_pool2d(alpha, kernel_size=k, stride=1, padding=feather_px)
    alpha = alpha.squeeze(0).squeeze(0)
    alpha = alpha * support
    amax = alpha.max()
    if float(amax) > 0.0:
        alpha = alpha / amax
    return alpha.clamp(0.0, 1.0)


# ----------------------------------------------------------------------------
def _random_region_mask(h: int, w: int, gen: torch.Generator, device) -> torch.Tensor:
    """A crop mask the way the compositor sees it: mostly-filled support with
    ragged edges and a few holes, uint8 0/255, shape (h, w)."""
    m = torch.zeros((h, w), dtype=torch.uint8)
    t = int(torch.randint(0, max(1, h // 8), (1,), generator=gen))
    l = int(torch.randint(0, max(1, w // 8), (1,), generator=gen))
    b = h - int(torch.randint(0, max(1, h // 8), (1,), generator=gen))
    r = w - int(torch.randint(0, max(1, w // 8), (1,), generator=gen))
    m[t:b, l:r] = 255
    noise = torch.rand((h, w), generator=gen) < 0.02
    m[noise] = 0
    return m.to(device)


def _sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize()


def _timed(fn, device):
    _sync(device)
    t0 = time.perf_counter()
    out = fn()
    _sync(device)
    return out, time.perf_counter() - t0


def _composite_u8(alpha: torch.Tensor, clip: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    a = alpha.to(torch.float32).unsqueeze(-1)
    out = (clip.float() - frame.float()) * a + frame.float()
    return torch.round(out).clamp(0, 255).to(torch.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(description="CM-172 blend-mask equivalence + timing check")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--shapes", type=int, default=200, help="number of random (h, w) shapes")
    ap.add_argument("--min-side", type=int, default=24)
    ap.add_argument("--max-side", type=int, default=1400)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tol", type=float, default=1e-4, help="max allowed |alpha diff|")
    ap.add_argument("--max-u8-frac", type=float, default=0.001, help="max allowed fraction of composite pixels differing per shape")
    ap.add_argument("--max-u8-level", type=int, default=1, help="max allowed 8-bit level difference on any differing pixel")
    args = ap.parse_args()

    device = torch.device(args.device)
    gen = torch.Generator().manual_seed(args.seed)
    print(f"[cm172] device={device} torch={torch.__version__} shapes={args.shapes}")

    worst_alpha = {"legacy": 0.0, "support": 0.0}
    worst_u8 = {"legacy": 0.0, "support": 0.0}
    worst_lvl = {"legacy": 0, "support": 0}
    t_old = {"legacy": 0.0, "support": 0.0}
    t_new = {"legacy": 0.0, "support": 0.0}
    slowest_old = {"legacy": (0.0, None), "support": (0.0, None)}
    failures = 0

    for i in range(args.shapes):
        h = int(torch.randint(args.min_side, args.max_side + 1, (1,), generator=gen))
        w = int(torch.randint(args.min_side, args.max_side + 1, (1,), generator=gen))
        mask_u8 = _random_region_mask(h, w, gen, device)
        mask_f = mask_u8.float()
        clip = torch.randint(0, 256, (h, w, 3), generator=gen).to(torch.uint8).to(device)
        frame = torch.randint(0, 256, (h, w, 3), generator=gen).to(torch.uint8).to(device)

        for name, f_old, f_new in (
            ("legacy", lambda: legacy_create_blend_mask(mask_f), lambda: mask_utils.create_blend_mask(mask_f)),
            ("support", lambda: legacy_create_support_blend_mask(mask_f), lambda: mask_utils.create_support_blend_mask(mask_f)),
        ):
            a_old, dt_old = _timed(f_old, device)
            a_new, dt_new = _timed(f_new, device)
            t_old[name] += dt_old
            t_new[name] += dt_new
            if dt_old > slowest_old[name][0]:
                slowest_old[name] = (dt_old, (h, w))

            if a_old.shape != a_new.shape:
                print(f"[cm172] FAIL {name}: shape mismatch {tuple(a_old.shape)} vs {tuple(a_new.shape)} for h={h} w={w}")
                failures += 1
                continue
            d = float((a_old.float() - a_new.float()).abs().max())
            worst_alpha[name] = max(worst_alpha[name], d)
            c_old = _composite_u8(a_old, clip, frame)
            c_new = _composite_u8(a_new, clip, frame)
            nd = int((c_old != c_new).any(dim=-1).sum())
            frac = nd / float(h * w)
            lvl = int((c_old.int() - c_new.int()).abs().max())
            worst_u8[name] = max(worst_u8[name], frac)
            worst_lvl[name] = max(worst_lvl[name], lvl)
            if d > args.tol or frac > args.max_u8_frac or lvl > args.max_u8_level:
                failures += 1
                print(f"[cm172] FAIL {name}: h={h} w={w} max|dalpha|={d:.3e} "
                      f"u8 pixels differing={nd} ({100.0 * frac:.3f}%) max level diff={lvl}")

        if (i + 1) % 50 == 0:
            print(f"[cm172] {i + 1}/{args.shapes} shapes checked")

    print("")
    print("[cm172] ---- equivalence ----")
    for name in ("legacy", "support"):
        print(f"[cm172] {name:8s} max|alpha diff|={worst_alpha[name]:.3e}  "
              f"worst share of composite pixels differing={100.0 * worst_u8[name]:.3f}%  "
              f"max 8-bit level difference={worst_lvl[name]}")
    print("[cm172] ---- timing (sum over all shapes; first-seen shapes on ROCm include MIOpen compile) ----")
    for name in ("legacy", "support"):
        so = slowest_old[name]
        print(f"[cm172] {name:8s} old={t_old[name]:8.2f}s  new={t_new[name]:8.3f}s  "
              f"slowest old call={so[0]:.2f}s at hxw={so[1]}")
    verdict = "PASS" if failures == 0 else f"FAIL ({failures} shape(s) outside tolerance)"
    print(f"[cm172] ==== {verdict} ====")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
