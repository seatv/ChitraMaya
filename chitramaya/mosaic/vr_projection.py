# chitramaya/mosaic/vr_projection.py
"""VR projection transforms for the mosaic pipeline (CM-045, Batch 19).

Some VR studios apply the mosaic in VIEWING space rather than to the raw
frame: the blocks look square in the headset but arrive warped/trapezoidal
in the raw half-equirect (hequirect) eye -- a pattern neither the YOLO
detector nor BasicVSR++ was trained on. The fix (borrowed conceptually from
zelefans' vr_remove_mosaic, reimplemented GPU-resident) is geometric:

  1. warp each eye hequirect -> fisheye, where those blocks become square;
  2. run detection / tracking / restoration in fisheye space;
  3. inverse-warp ONLY the restored regions (pixels + blend alpha) back to
     hequirect and blend them onto the pristine original frame.

Unlike the zelefans ffmpeg pipeline (whole-eye roundtrip + a re-encode per
stage), background pixels here are never resampled and nothing is re-encoded
in between: one quality generation total.

Math matches ffmpeg's v360 filter conventions (hequirect <-> fisheye,
FOV 180, yaw=pitch=roll=0), validated numerically against ffmpeg v360 in
tools/verify_vr_projection.py. Coordinate system (v360): x right, y down,
z forward; fisheye is equidistant (angle from forward axis proportional to
radius).

All transforms are precomputed torch.grid_sample grids: built once per
(eye_w, eye_h, device), ~2 x H x W x 2 float32 (about 74 MB total at a
2160x2160 eye). Per-frame warp cost is millisecond-class -- invisible next
to detection.

CPU-safe: everything runs on the CPU device too (used by the test harness).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

Box = Tuple[int, int, int, int]  # (t, l, b, r) inclusive, frame coords

_HALF_PI = math.pi / 2.0

# Supported modes for config validation (import-time constant shared with
# pipeline / cli_config so the choice lists cannot drift apart).
VR_PROJECTION_MODES = ("none", "fisheye")


# ---------------------------------------------------------------------------
# Direction math (v360 conventions, FOV fixed at 180)
# ---------------------------------------------------------------------------

def _fisheye_px_to_dir(u_px: torch.Tensor, v_px: torch.Tensor,
                       eye_w: int, eye_h: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fisheye pixel-center coords -> unit direction (x right, y down, z fwd).

    v360 fisheye_to_xyz with h_fov = v_fov = 180 (flat_range = 1):
      uf = (2*(i+0.5))/W - 1 ; vf likewise ; r = hypot(uf, vf)
      theta = pi/2 * (1 - r)  (equidistant; r=0 -> forward, r=1 -> 90 deg)
    Points with r > 1 lie outside the 180-degree circle; callers clamp.
    """
    uf = (2.0 * u_px) / float(eye_w) - 1.0
    vf = (2.0 * v_px) / float(eye_h) - 1.0
    phi = torch.atan2(vf, uf)
    r = torch.hypot(uf, vf)
    theta = _HALF_PI * (1.0 - r)
    ct = torch.cos(theta)
    x = ct * torch.cos(phi)
    y = ct * torch.sin(phi)
    z = torch.sin(theta)
    return x, y, z


def _dir_to_hequirect_px(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
                         eye_w: int, eye_h: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unit direction -> hequirect continuous pixel coords (v360 xyz_to_hequirect).

    hequirect: lon in [-90, 90] over width, lat in [-90, 90] over height.
    """
    lon = torch.atan2(x, z)
    lat = torch.asin(y.clamp(-1.0, 1.0))
    u = (lon / _HALF_PI + 1.0) * 0.5 * float(eye_w)
    v = (lat / _HALF_PI + 1.0) * 0.5 * float(eye_h)
    return u, v


def _hequirect_px_to_dir(u_px: torch.Tensor, v_px: torch.Tensor,
                         eye_w: int, eye_h: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hequirect pixel-center coords -> unit direction (v360 hequirect_to_xyz)."""
    lon = _HALF_PI * ((2.0 * u_px) / float(eye_w) - 1.0)
    lat = _HALF_PI * ((2.0 * v_px) / float(eye_h) - 1.0)
    cl = torch.cos(lat)
    x = cl * torch.sin(lon)
    y = torch.sin(lat)
    z = cl * torch.cos(lon)
    return x, y, z


def _dir_to_fisheye_px(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
                       eye_w: int, eye_h: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unit direction -> fisheye continuous pixel coords (v360 xyz_to_fisheye, FOV 180).

    radius = angle-from-forward / 90deg (equidistant), direction (x, y).
    """
    h = torch.hypot(x, y)
    lh = torch.where(h > 0, h, torch.ones_like(h))
    # angle from +z, normalized so 90 deg -> 0.5 (flat_range = 1 at FOV 180)
    ang = torch.atan2(h, z) / math.pi
    u = (ang * x / lh + 0.5) * float(eye_w)
    v = (ang * y / lh + 0.5) * float(eye_h)
    return u, v


def fisheye_box_to_hequirect_bbox(box: Box, eye_w: int, eye_h: int,
                                  margin: int = 4, samples_per_side: int = 32) -> Box:
    """Bounding box in hequirect space of a fisheye-space box.

    Samples the box perimeter densely (the mapping is smooth, so the image
    extremes of a rectangle lie on its boundary), maps each point
    fisheye -> direction -> hequirect, and takes min/max with a safety
    margin. Perimeter points outside the fisheye 180-degree circle are
    radially clamped just inside it before mapping.
    Returns (t, l, b, r) inclusive, clamped to the eye.
    """
    t, l, b, r = (float(v) for v in box)
    n = int(samples_per_side)
    xs = torch.linspace(l + 0.5, r + 0.5, n, dtype=torch.float64)
    ys = torch.linspace(t + 0.5, b + 0.5, n, dtype=torch.float64)
    u = torch.cat([xs, xs, torch.full((n,), l + 0.5, dtype=torch.float64),
                   torch.full((n,), r + 0.5, dtype=torch.float64)])
    v = torch.cat([torch.full((n,), t + 0.5, dtype=torch.float64),
                   torch.full((n,), b + 0.5, dtype=torch.float64), ys, ys])

    # Radially clamp points outside the fisheye circle (invalid directions).
    uf = (2.0 * u) / float(eye_w) - 1.0
    vf = (2.0 * v) / float(eye_h) - 1.0
    rad = torch.hypot(uf, vf)
    scale = torch.where(rad > 0.999, 0.999 / rad.clamp(min=1e-9), torch.ones_like(rad))
    uf = uf * scale
    vf = vf * scale
    u = (uf + 1.0) * 0.5 * float(eye_w)
    v = (vf + 1.0) * 0.5 * float(eye_h)

    x, y, z = _fisheye_px_to_dir(u, v, eye_w, eye_h)
    hu, hv = _dir_to_hequirect_px(x, y, z, eye_w, eye_h)

    lt = int(math.floor(float(hv.min()))) - int(margin)
    ll = int(math.floor(float(hu.min()))) - int(margin)
    lb = int(math.ceil(float(hv.max()))) + int(margin)
    lr = int(math.ceil(float(hu.max()))) + int(margin)
    lt = max(0, min(eye_h - 1, lt))
    ll = max(0, min(eye_w - 1, ll))
    lb = max(0, min(eye_h - 1, lb))
    lr = max(0, min(eye_w - 1, lr))
    return (lt, ll, lb, lr)


# ---------------------------------------------------------------------------
# VRProjection: cached grids + frame warp + projected composite
# ---------------------------------------------------------------------------

class VRProjection:
    """Per-eye hequirect<->fisheye warps for an SBS frame, grids cached on device.

    Both eyes share the same geometry, so exactly two grids exist regardless
    of layout: forward (fisheye output <- hequirect source) and inverse
    (hequirect output <- fisheye source), normalized for
    grid_sample(align_corners=False).
    """

    def __init__(self, eye_w: int, eye_h: int, device: torch.device):
        self.eye_w = int(eye_w)
        self.eye_h = int(eye_h)
        self.device = device
        self._grid_fwd: Optional[torch.Tensor] = None   # (1, eh, ew, 2)
        self._grid_inv: Optional[torch.Tensor] = None   # (1, eh, ew, 2)
        self._canvas_img: Optional[torch.Tensor] = None    # (1, 3, eh, ew) float
        self._canvas_alpha: Optional[torch.Tensor] = None  # (1, 1, eh, ew) float
        self._float_dtype = torch.float16 if device.type == "cuda" else torch.float32

    # -- grid construction ---------------------------------------------------

    def _pixel_center_mesh(self) -> Tuple[torch.Tensor, torch.Tensor]:
        ys = torch.arange(self.eye_h, dtype=torch.float32) + 0.5
        xs = torch.arange(self.eye_w, dtype=torch.float32) + 0.5
        v, u = torch.meshgrid(ys, xs, indexing="ij")
        return u, v

    def _normalize(self, u_px: torch.Tensor, v_px: torch.Tensor) -> torch.Tensor:
        gx = (2.0 * u_px) / float(self.eye_w) - 1.0
        gy = (2.0 * v_px) / float(self.eye_h) - 1.0
        return torch.stack([gx, gy], dim=-1).unsqueeze(0)  # (1, eh, ew, 2)

    def grid_fwd(self) -> torch.Tensor:
        """Grid producing the FISHEYE image by sampling the hequirect eye."""
        if self._grid_fwd is None:
            u, v = self._pixel_center_mesh()
            x, y, z = _fisheye_px_to_dir(u, v, self.eye_w, self.eye_h)
            hu, hv = _dir_to_hequirect_px(x, y, z, self.eye_w, self.eye_h)
            # Outside the 180-degree circle there is no valid direction; push
            # the sample far out of range so grid_sample zero-padding blanks it.
            uf = (2.0 * u) / float(self.eye_w) - 1.0
            vf = (2.0 * v) / float(self.eye_h) - 1.0
            invalid = torch.hypot(uf, vf) > 1.0
            hu = torch.where(invalid, torch.full_like(hu, -1e4), hu)
            hv = torch.where(invalid, torch.full_like(hv, -1e4), hv)
            self._grid_fwd = self._normalize(hu, hv).to(self.device)
        return self._grid_fwd

    def grid_inv(self) -> torch.Tensor:
        """Grid producing the HEQUIRECT image by sampling the fisheye eye."""
        if self._grid_inv is None:
            u, v = self._pixel_center_mesh()
            x, y, z = _hequirect_px_to_dir(u, v, self.eye_w, self.eye_h)
            fu, fv = _dir_to_fisheye_px(x, y, z, self.eye_w, self.eye_h)
            self._grid_inv = self._normalize(fu, fv).to(self.device)
        return self._grid_inv

    def vram_estimate_mb(self) -> float:
        return 2.0 * self.eye_h * self.eye_w * 2 * 4 / (1024.0 * 1024.0)

    # -- whole-frame forward warp (analysis side) ----------------------------

    def _warp_eye(self, eye_hwc_u8: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        x = eye_hwc_u8.permute(2, 0, 1).unsqueeze(0).to(self._float_dtype)
        g = grid.to(x.dtype) if x.dtype != torch.float32 else grid
        out = F.grid_sample(x, g, mode="bilinear",
                            padding_mode="zeros", align_corners=False)
        return out.round_().clamp_(0, 255).to(torch.uint8).squeeze(0).permute(1, 2, 0).contiguous()

    def warp_frame_to_fisheye(self, frame_bgr_u8: torch.Tensor) -> torch.Tensor:
        """Full SBS frame (H, W, 3 u8) -> both halves warped hequirect->fisheye.

        Positional halves: geometry is identical for either eye, so the
        lr/rl layout is irrelevant here.
        """
        h, w = int(frame_bgr_u8.shape[0]), int(frame_bgr_u8.shape[1])
        ew = w // 2
        if h != self.eye_h or ew != self.eye_w:
            raise ValueError(
                f"VRProjection built for eye {self.eye_w}x{self.eye_h}, "
                f"got frame {w}x{h}"
            )
        g = self.grid_fwd()
        left = self._warp_eye(frame_bgr_u8[:, :ew], g)
        right = self._warp_eye(frame_bgr_u8[:, w - ew:], g)
        return torch.cat([left, right], dim=1)

    # -- projected composite (restored regions only) -------------------------

    def _ensure_canvases(self) -> None:
        if self._canvas_img is None:
            self._canvas_img = torch.zeros(
                (1, 3, self.eye_h, self.eye_w),
                dtype=self._float_dtype, device=self.device)
            self._canvas_alpha = torch.zeros(
                (1, 1, self.eye_h, self.eye_w),
                dtype=self._float_dtype, device=self.device)

    def blend_projected(
        self,
        *,
        frame_bgr_u8: torch.Tensor,     # full SBS frame in the store (mutated)
        clip_img_u8: torch.Tensor,      # restored crop, fisheye space (h, w, 3)
        blend_alpha: torch.Tensor,      # float alpha 0..1, crop space (h, w)
        box: Box,                       # fisheye FULL-frame coords (t, l, b, r)
        model_dtype: torch.dtype,
    ) -> None:
        """Inverse-warp a restored fisheye-space region and blend it onto the
        original hequirect frame. Only the affected hequirect sub-rect is
        sampled; the rest of the frame is untouched."""
        self._ensure_canvases()
        ew, eh = self.eye_w, self.eye_h
        t, l, b, r = (int(v) for v in box)

        # Which positional half does this (seam-split) box live in?
        eye_off = 0 if l < ew else ew
        lt, ll = t, l - eye_off
        lb, lr = b, r - eye_off
        lt = max(0, min(eh - 1, lt)); lb = max(0, min(eh - 1, lb))
        ll = max(0, min(ew - 1, ll)); lr = max(0, min(ew - 1, lr))
        if lb < lt or lr < ll:
            return

        # Paste crop + alpha into the eye-space scratch canvases.
        ch = lb - lt + 1
        cw = lr - ll + 1
        img = clip_img_u8[:ch, :cw].permute(2, 0, 1).to(self._float_dtype)
        a = blend_alpha[:ch, :cw].to(self._float_dtype)
        self._canvas_img[0, :, lt:lb + 1, ll:lr + 1] = img
        self._canvas_alpha[0, 0, lt:lb + 1, ll:lr + 1] = a

        try:
            # Destination sub-rect in hequirect space.
            dt, dl, db, dr = fisheye_box_to_hequirect_bbox(
                (lt, ll, lb, lr), ew, eh)
            grid = self.grid_inv()[:, dt:db + 1, dl:dr + 1, :]
            grid = grid.to(self._canvas_img.dtype) if self._canvas_img.dtype != torch.float32 else grid

            w_img = F.grid_sample(self._canvas_img, grid, mode="bilinear",
                                  padding_mode="zeros", align_corners=False)
            w_a = F.grid_sample(self._canvas_alpha, grid, mode="bilinear",
                                padding_mode="zeros", align_corners=False)

            # LADA-style blend into the store frame (original projection).
            roi = frame_bgr_u8[dt:db + 1, eye_off + dl:eye_off + dr + 1]
            roi_f = roi.permute(2, 0, 1).unsqueeze(0).to(model_dtype)
            temp = w_img.to(model_dtype)
            am = w_a.to(model_dtype)
            temp.sub_(roi_f)
            temp.mul_(am)
            temp.add_(roi_f)
            temp.round_()
            temp.clamp_(0, 255)
            roi[:] = temp.squeeze(0).permute(1, 2, 0)
        finally:
            # Zero only the touched scratch region for reuse.
            self._canvas_img[0, :, lt:lb + 1, ll:lr + 1] = 0
            self._canvas_alpha[0, 0, lt:lb + 1, ll:lr + 1] = 0


# ---------------------------------------------------------------------------
# Clip-level projected composite (fisheye analogue of composite_clip_into_store)
# ---------------------------------------------------------------------------

def composite_clip_into_store_projected(
    *,
    clip,
    restored_frames_u8: List[torch.Tensor],
    store_bgr_u8: Dict[int, torch.Tensor],
    vrproj: VRProjection,
    model_dtype: torch.dtype,
    blendmask: str = "none",
    feather_radius: int = 0,
) -> None:
    """Projected variant of compositor.composite_clip_into_store: identical
    unpad/resize/alpha steps, but the paste-back inverse-warps each region
    from fisheye space onto the original hequirect frame."""
    from chitramaya.mosaic.restorer.compositor import (
        _resize_img_u8, _resize_mask_u8, _unpad_any,
    )
    from chitramaya.mosaic.utils import mask_utils

    n = min(len(restored_frames_u8), len(clip.frame_nums))
    for i in range(n):
        frame_num = int(clip.frame_nums[i])
        frame = store_bgr_u8.get(frame_num)
        if frame is None:
            continue

        clip_img = restored_frames_u8[i]
        clip_mask = clip.masks[i]
        box: Box = clip.boxes[i]
        orig_shape_hw = clip.crop_shapes[i]
        pad = clip.pad_after_resizes[i]

        clip_img = _unpad_any(clip_img, pad)
        clip_mask = _unpad_any(clip_mask, pad)
        clip_img = _resize_img_u8(clip_img, orig_shape_hw)
        clip_mask = _resize_mask_u8(clip_mask, orig_shape_hw)

        if str(blendmask).lower() == "facefusion":
            _feather = int(feather_radius) if int(feather_radius) > 0 else None
            alpha = mask_utils.create_support_blend_mask(
                clip_mask.float(), feather_px=_feather)
        else:
            alpha = mask_utils.create_blend_mask(clip_mask.float())
        if alpha.ndim == 3:
            alpha = alpha.squeeze(-1)

        vrproj.blend_projected(
            frame_bgr_u8=frame,
            clip_img_u8=clip_img,
            blend_alpha=alpha,
            box=box,
            model_dtype=model_dtype,
        )


__all__ = [
    "VR_PROJECTION_MODES",
    "VRProjection",
    "composite_clip_into_store_projected",
    "fisheye_box_to_hequirect_bbox",
]
