from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Dict

import torch

from chitramaya.mosaic.core.scene import Box, Scene, crop_box_to_target_v3
from chitramaya.mosaic.core.clip import Clip


@dataclass
class TrackerConfig:
    clip_size: int = 256
    max_clip_length: int = 30
    pad_mode: str = "reflect"
    border_size: float = 0.06
    max_box_expansion_factor: float = 1.0
    debug: bool = False

    # If True and the detector provides per-pixel masks, we'll use them for
    # clip masks (more LADA-faithful compositing). If False, clip masks will
    # be simple rectangle-box masks.
    use_seg_masks: bool = True

    # --- Crop stabilization (DISABLED by default — see analysis below) ---
    # These were inherited from gRestorer with non-zero defaults. Lada (the
    # reference implementation) does NOT do crop stabilization in original-
    # frame coordinates. Its strategy is to let each frame's crop follow its
    # detection precisely, then within the Clip apply a per-clip uniform
    # scale that registers content at a consistent position inside the
    # 256×256 model input. That uniform scale + BasicVSR++'s own optical-
    # flow propagation provide the temporal coherence — no need to lock
    # the original-frame crop box.
    #
    # When crop_sticky=True, the crop box is held fixed across frames,
    # which means the actual mosaic content (which is moving) drifts
    # *inside* the locked crop frame-to-frame. BasicVSR++ sees content
    # shifting around in its input, which is the *opposite* of what
    # helps it. Hence the new defaults of 0 / False.
    crop_quant_px: int = 0
    crop_sticky: bool = False
    crop_sticky_pad_px: int = 8

    # Expand the previous ROI box by this pad when deciding scene membership.
    # Marginally reduces scene fragmentation when a box jitters a few pixels.
    # Lada uses 0; we keep 8 because it's mildly forgiving without distorting
    # anything (only affects scene matching, not the crop fed to the model).
    match_pad_px: int = 8

    # --- TTL: scene linger after detection drops ---
    # Lada's behavior: scene completes immediately on any frame without
    # a detection. That's strictly worse than what we want — a 1-frame
    # YOLO miss breaks a scene in two, each half restored independently.
    #
    # With ttl_after_end > 0, scenes persist for N frames after their
    # last real detection. If detection returns within that window, the
    # gap is filled by interpolation (interpolated boxes + buffered
    # source frames + box-as-mask) and the clip remains one continuous
    # piece. If detection doesn't return, the scene is completed.
    #
    # Default 3 is a reasonable starting point. Set 0 to match Lada
    # exactly (no linger).
    ttl_after_end: int = 3


@dataclass
class TrackerStepResult:
    overlay_boxes: List[Box]
    new_clips: List[Clip]
    active_scenes: int
    t_track: float
    t_clip_build: float


def _union_box(a: Box, b: Box) -> Box:
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    )


def _box_overlap_strict(a: Box, b: Box) -> bool:
    """Strict overlap: touching edges is NOT overlap (matches LADA semantics)."""
    at, al, ab, ar = a
    bt, bl, bb, br = b
    if ar <= bl or br <= al:
        return False
    if ab <= bt or bb <= at:
        return False
    return True


def _box_overlap_pad(a: Box, b: Box, pad: int) -> bool:
    """Strict overlap after expanding both boxes by `pad` pixels."""
    if pad <= 0:
        return _box_overlap_strict(a, b)
    at, al, ab, ar = a
    bt, bl, bb, br = b
    a2 = (at - pad, al - pad, ab + pad, ar + pad)
    b2 = (bt - pad, bl - pad, bb + pad, br + pad)
    return _box_overlap_strict(a2, b2)


def _roi_inside_crop(roi: Box, crop: Box) -> bool:
    rt, rl, rb, rr = roi
    ct, cl, cb, cr = crop
    return (rt >= ct) and (rl >= cl) and (rb <= cb) and (rr <= cr)


def _quantize_crop_box(crop: Box, img_h: int, img_w: int, q: int) -> Box:
    """Quantize crop edges to a q-pixel grid.
    - top/left are floored
    - bottom/right are ceiled
    This tends to slightly expand crops, trading a bit of compute for stability.
    """
    if q <= 1:
        t, l, b, r = crop
        return (max(0, t), max(0, l), min(img_h - 1, b), min(img_w - 1, r))

    t, l, b, r = crop

    t2 = (t // q) * q
    l2 = (l // q) * q

    # Inclusive bottom/right: quantize (b+1) and subtract 1.
    b2 = (((b + 1 + q - 1) // q) * q) - 1
    r2 = (((r + 1 + q - 1) // q) * q) - 1

    # Clamp within bounds
    t2 = max(0, t2)
    l2 = max(0, l2)
    b2 = min(img_h - 1, b2)
    r2 = min(img_w - 1, r2)

    # Ensure valid box
    if b2 < t2:
        b2 = min(img_h - 1, t2)
    if r2 < l2:
        r2 = min(img_w - 1, l2)

    return (int(t2), int(l2), int(b2), int(r2))


class SceneTracker:
    """Track per-frame detections into LADA-style Scenes, then emit Clips."""

    def __init__(self, cfg: TrackerConfig, *, seg_mask_only: bool = False) -> None:
        self.cfg = cfg
        self._seg_mask_only = bool(seg_mask_only)

        self._scenes: List[Scene] = []
        self._scene_counter: int = 0
        self._clip_counter: int = 0

        # Ring buffer of recent source frames, used for gap-fill when a
        # scene is reactivated within its TTL window. Sized to TTL + 2
        # (safety margin). Keys are frame_num. Trimmed in step_frame.
        self._frame_buffer: dict = {}

    def _scene_to_clip(self, s: Scene):
        if len(s) == 0:
            return None

        return Clip.from_scene(
            scene=s,
            clip_id=self._clip_counter,
            clip_size=self.cfg.clip_size,
            pad_mode=self.cfg.pad_mode,
        )

    def _belongs_scene(self, s: Scene, roi_box: Box) -> bool:
        """Scene membership with optional padding to reduce jitter-induced fragmentation."""
        if not s.roi_boxes:
            return False
        pad = int(getattr(self.cfg, "match_pad_px", 0) or 0)
        if pad > 0:
            return _box_overlap_pad(s.roi_boxes[-1], roi_box, pad)
        return s.belongs(roi_box)

    def _stabilize_crop_box(
            self,
            *,
            roi_box: Box,
            base_crop_box: Box,
            prev_crop_box: Optional[Box],
            img_h: int,
            img_w: int,
    ) -> Box:
        """Apply quantization + sticky crop stabilization.

        Order:
          1) Quantize crop to a pixel grid (crop_quant_px).
          2) If sticky enabled and ROI still fits inside prev crop, keep prev crop when
             the new crop differs only slightly (crop_sticky_pad_px).
        """
        q = int(getattr(self.cfg, "crop_quant_px", 0) or 0)
        crop = _quantize_crop_box(base_crop_box, img_h=img_h, img_w=img_w, q=q) if q > 1 else base_crop_box

        if not bool(getattr(self.cfg, "crop_sticky", False)):
            return crop
        if prev_crop_box is None:
            return crop

        # Only keep the old crop if the current ROI is still fully inside it.
        if not _roi_inside_crop(roi_box, prev_crop_box):
            return crop

        pad = int(getattr(self.cfg, "crop_sticky_pad_px", 0) or 0)
        if pad <= 0:
            return crop

        dt = abs(int(crop[0]) - int(prev_crop_box[0]))
        dl = abs(int(crop[1]) - int(prev_crop_box[1]))
        db = abs(int(crop[2]) - int(prev_crop_box[2]))
        dr = abs(int(crop[3]) - int(prev_crop_box[3]))
        if max(dt, dl, db, dr) <= pad:
            return prev_crop_box

        return crop

    def reset(self) -> None:
        self._scenes.clear()
        self._scene_counter = 0
        self._clip_counter = 0

    @property
    def scenes_active(self) -> int:
        return len(self._scenes)

    # Back-compat alias used by the CLI pipeline.
    @property
    def active_scenes(self) -> int:
        return self.scenes_active

    def min_active_start(self) -> Optional[int]:
        """Earliest start-frame among active scenes (None if no active scenes)."""
        if not self._scenes:
            return None
        return min(s.frame_start for s in self._scenes)

    def diagnostic_snapshot(self) -> List[dict]:
        """Return a list of dicts describing every currently-active scene.

        Used by the pipeline's stall detector to print why backpressure isn't
        clearing. Each dict has:
          id           : scene id
          start_frame  : scene.frame_start
          end_frame    : scene.frame_end (last frame currently in the scene)
          length       : len(scene.frame_nums)
          last_detected: scene.last_detected_frame (-1 if never)
          mcl_reached  : True if length >= max_clip_length
        """
        snap: List[dict] = []
        for s in self._scenes:
            snap.append({
                "id": int(s.id),
                "start_frame": int(s.frame_start),
                "end_frame": int(s.frame_end),
                "length": int(len(s)),
                "last_detected": int(s.last_detected_frame),
                "mcl_reached": bool(len(s) >= self.cfg.max_clip_length),
            })
        return snap

    def clips_emitted(self) -> int:
        return self._clip_counter

    def _new_scene(self, frame_num: int) -> Scene:
        s = Scene(id=self._scene_counter, start_frame=frame_num)
        self._scene_counter += 1
        return s

    def _compute_crop(
            self,
            frame_bgr_u8: torch.Tensor,
            roi_box: Box,
            roi_mask: Optional[torch.Tensor] = None,
            *,
            force_crop_box: Optional[Box] = None,
    ) -> Tuple[Box, torch.Tensor, torch.Tensor]:
        """Compute LADA crop_to_box_v3 crop box and slice crop from the frame.

        We create a per-crop mask on the frame device.

        - If roi_mask is provided (per-pixel segmentation) and use_seg_masks=True, we use it
          as the clip mask (cropped to crop_box). If roi_mask is on CPU at full resolution,
          we slice *only the crop region* on CPU and transfer just that to the frame device
          (avoids copying full HxW masks each frame).
        - Otherwise we fall back to a rectangle mask derived from roi_box.
        """
        h, w = int(frame_bgr_u8.shape[0]), int(frame_bgr_u8.shape[1])

        if force_crop_box is None:
            crop_box, _scale = crop_box_to_target_v3(
                roi_box,
                img_h=h,
                img_w=w,
                target_hw=(self.cfg.clip_size, self.cfg.clip_size),
                max_box_expansion_factor=self.cfg.max_box_expansion_factor,
                border_size=float(self.cfg.border_size),
            )
        else:
            crop_box = force_crop_box

        t, l, b, r = crop_box
        crop_img = frame_bgr_u8[t: b + 1, l: r + 1, :].clone()

        # Mask generation (seam-sensitive)
        crop_h = int(b - t + 1)
        crop_w = int(r - l + 1)

        # 1) Rectangle base mask (always)
        crop_mask_out = torch.zeros((crop_h, crop_w), device=frame_bgr_u8.device, dtype=torch.uint8)

        # If we have a seg mask and seg-only is enabled (LADA parity), use seg only.
        if roi_mask is not None and self.cfg.use_seg_masks and self._seg_mask_only:
            seg = roi_mask[t: b + 1, l: r + 1]

            # seg may be CPU (lada-yolo returns CPU masks); crop_mask_out is on frame device (cuda)
            if isinstance(seg, torch.Tensor):
                if seg.ndim == 3 and seg.shape[-1] == 1:
                    seg = seg[:, :, 0]
                if seg.dtype != torch.uint8:
                    seg = seg.to(torch.uint8)
                if seg.device != crop_mask_out.device:
                    seg = seg.to(device=crop_mask_out.device, non_blocking=(crop_mask_out.device.type == "cuda"))
            else:
                # if somehow seg is numpy, convert then move
                seg = torch.as_tensor(seg, dtype=torch.uint8, device=crop_mask_out.device)

            crop_mask_out = torch.maximum(crop_mask_out, seg)

        else:
            # Rectangle base mask (gRestorer default)
            rt, rl, rb, rr = roi_box
            it = max(t, rt)
            il = max(l, rl)
            ib = min(b, rb)
            ir = min(r, rr)
            if ib >= it and ir >= il:
                crop_mask_out[it - t: ib - t + 1, il - l: ir - l + 1] = 255

            # Union-in segmentation mask if present
            if roi_mask is not None and self.cfg.use_seg_masks:
                seg = roi_mask[t: b + 1, l: r + 1]

                # If mask is HWC(1), squeeze to HW
                if isinstance(seg, torch.Tensor) and seg.ndim == 3 and seg.shape[-1] == 1:
                    seg = seg[:, :, 0]

                # seg may be CPU (lada-yolo returns CPU masks); crop_mask_out is on frame device (cuda)
                if isinstance(seg, torch.Tensor):
                    if seg.ndim == 3 and seg.shape[-1] == 1:
                        seg = seg[:, :, 0]
                    if seg.dtype != torch.uint8:
                        seg = seg.to(torch.uint8)
                    if seg.device != crop_mask_out.device:
                        seg = seg.to(device=crop_mask_out.device, non_blocking=(crop_mask_out.device.type == "cuda"))
                else:
                    # if somehow seg is numpy, convert then move
                    seg = torch.as_tensor(seg, dtype=torch.uint8, device=crop_mask_out.device)

                crop_mask_out = torch.maximum(crop_mask_out, seg)

        # 2) Optional seg mask: OR it in (never replace rectangle)
        if roi_mask is not None and self.cfg.use_seg_masks:
            try:
                def _mask_u8(m: torch.Tensor) -> torch.Tensor:
                    # Normalize to uint8 {0,255} without device-sync (no .item()).
                    if m.dtype == torch.bool:
                        return m.to(dtype=torch.uint8) * 255
                    if m.is_floating_point():
                        return torch.where(m > 0.5, 255, 0).to(dtype=torch.uint8)
                    return m.to(dtype=torch.uint8)

                seg_crop: Optional[torch.Tensor] = None
                if roi_mask.device == frame_bgr_u8.device:
                    if roi_mask.shape == (h, w):
                        seg_crop = roi_mask[t: b + 1, l: r + 1]
                    elif roi_mask.shape == (crop_h, crop_w):
                        seg_crop = roi_mask
                elif roi_mask.device.type == "cpu":
                    if roi_mask.shape == (h, w):
                        seg_crop = roi_mask[t: b + 1, l: r + 1].to(device=frame_bgr_u8.device)
                    elif roi_mask.shape == (crop_h, crop_w):
                        seg_crop = roi_mask.to(device=frame_bgr_u8.device)

                if seg_crop is not None:
                    crop_mask_out = torch.maximum(crop_mask_out, _mask_u8(seg_crop))

            except Exception:
                # Best-effort: don't crash the pipeline on mask oddities.
                pass

        return crop_box, crop_img, crop_mask_out

    def _interp_box(self, b1: Box, b2: Box, t: float) -> Box:
        """Linear interpolation between two boxes. t=0 -> b1, t=1 -> b2."""
        return (
            int(round(b1[0] + (b2[0] - b1[0]) * t)),
            int(round(b1[1] + (b2[1] - b1[1]) * t)),
            int(round(b1[2] + (b2[2] - b1[2]) * t)),
            int(round(b1[3] + (b2[3] - b1[3]) * t)),
        )

    def _fill_gap_frames(
            self,
            scene: Scene,
            prev_box: Box,
            new_box: Box,
            new_frame_num: int,
    ) -> bool:
        """Fill the gap between scene's last frame and ``new_frame_num``.

        For each frame F in (scene.frame_end + 1 ... new_frame_num - 1):
          - Look up the source frame from the ring buffer
          - Interpolate the box linearly between prev_box and new_box
          - Compute crop from the interpolated box; the resulting mask is
            a box-mask (rectangle), since we have no segmentation mask
            for gap frames
          - Append the gap frame with is_detection=False so the scene's
            last_detected_frame is NOT updated

        Returns True if all gap frames were filled successfully. Returns
        False if any source frame was missing from the buffer (in which
        case the caller should treat the scene as expired).
        """
        gap_start = scene.frame_end + 1
        gap_end = new_frame_num - 1
        if gap_start > gap_end:
            return True  # no gap

        total_steps = new_frame_num - scene.frame_end  # >= 2

        for gap_f in range(gap_start, gap_end + 1):
            src = self._frame_buffer.get(gap_f)
            if src is None:
                # Missing source — can't gap-fill safely. The caller
                # will treat this as a hard expiry.
                return False

            t = (gap_f - scene.frame_end) / total_steps
            interp_box = self._interp_box(prev_box, new_box, t)

            # _compute_crop with roi_mask=None gives us a box-mask
            # automatically (the rectangle-from-roi_box path).
            crop_box, crop_img, crop_mask = self._compute_crop(
                src,
                interp_box,
                None,
                force_crop_box=None,
            )
            scene.add_frame(
                frame_num=gap_f,
                roi_box=interp_box,
                crop_box=crop_box,
                crop_img=crop_img,
                crop_mask=crop_mask,
                is_detection=False,  # IMPORTANT: don't update last_detected_frame
            )
        return True

    def _update_frame_buffer(self, frame_num: int, frame_bgr_u8: torch.Tensor) -> None:
        """Cache current frame for potential gap-fill; trim entries older
        than the TTL window."""
        # We clone so the buffer entry survives even if the pipeline
        # mutates frame_bgr_u8 elsewhere. This is NOT theoretical: the same
        # tensor object lives in the pipeline's FrameStore, and
        # composite_clip_into_store() blends restored pixels into it IN
        # PLACE when a clip completes. Without the clone, a scene that
        # reactivates within its TTL window can gap-fill from frames
        # another clip already composited — feeding restored pixels back
        # through BasicVSR++ (double restoration, subtle smearing).
        # Cost: one device memcpy per frame (~25 MB at 4K), at most ttl+2
        # frames held.
        self._frame_buffer[int(frame_num)] = frame_bgr_u8.clone()
        ttl = int(getattr(self.cfg, "ttl_after_end", 0) or 0)
        keep_from = frame_num - (ttl + 2)
        if keep_from > 0:
            stale = [k for k in self._frame_buffer if k < keep_from]
            for k in stale:
                del self._frame_buffer[k]

    def step_frame(
            self,
            frame_num: int,
            frame_bgr_u8: torch.Tensor,
            roi_boxes: Sequence[Box],
            roi_masks: Optional[Sequence[Optional[torch.Tensor]]] = None,
    ) -> TrackerStepResult:
        """Ingest one frame's detections, update scenes, and flush completed scenes."""
        if roi_masks is not None and len(roi_masks) != len(roi_boxes):
            raise ValueError("roi_masks length must match roi_boxes length")

        t0 = time.perf_counter()

        # Cache current frame for gap-fill if a scene gets reactivated
        # within its TTL window. Must happen BEFORE the detection-matching
        # loop because the gap-fill helper reads from the buffer.
        self._update_frame_buffer(frame_num, frame_bgr_u8)

        # Update scenes with detections.
        for i, box in enumerate(roi_boxes):
            mask = None
            if self.cfg.use_seg_masks and roi_masks is not None:
                mask = roi_masks[i]

            matched: Optional[Scene] = None
            for s in self._scenes:
                if self._belongs_scene(s, box):
                    matched = s
                    break

            if matched is None:
                matched = self._new_scene(frame_num)
                self._scenes.append(matched)

            if matched.frame_end == frame_num:
                # Same-frame merge: union ROI and recompute crop from union.
                union_roi = _union_box(matched.roi_boxes[-1], box)

                h, w = int(frame_bgr_u8.shape[0]), int(frame_bgr_u8.shape[1])
                base_crop_box, _ = crop_box_to_target_v3(
                    union_roi,
                    img_h=h,
                    img_w=w,
                    target_hw=(self.cfg.clip_size, self.cfg.clip_size),
                    max_box_expansion_factor=self.cfg.max_box_expansion_factor,
                    border_size=float(self.cfg.border_size),
                )

                prev_crop_box = matched.crop_boxes[-1] if matched.crop_boxes else None
                stable_crop_box = self._stabilize_crop_box(
                    roi_box=union_roi,
                    base_crop_box=base_crop_box,
                    prev_crop_box=prev_crop_box,
                    img_h=h,
                    img_w=w,
                )

                crop_box, crop_img, cur_mask = self._compute_crop(
                    frame_bgr_u8,
                    union_roi,
                    mask,
                    force_crop_box=stable_crop_box,
                )

                # Re-crop from union ROI. IMPORTANT: pass current detection mask (if any),
                # then union it with the previous crop-mask by mapping old->new crop coords.
                prev_mask = matched.masks[-1] if matched.masks else None
                prev_crop_box = matched.crop_boxes[-1] if matched.crop_boxes else None

                if prev_mask is not None and prev_crop_box is not None:
                    nt, nl, nb, nr = crop_box
                    new_h = int(nb - nt + 1)
                    new_w = int(nr - nl + 1)
                    merged = torch.zeros((new_h, new_w), device=frame_bgr_u8.device, dtype=torch.uint8)

                    ot, ol, ob, or_ = prev_crop_box
                    it = max(nt, ot)
                    il = max(nl, ol)
                    ib = min(nb, ob)
                    ir = min(nr, or_)
                    if ib >= it and ir >= il:
                        h_int = int(ib - it + 1)
                        w_int = int(ir - il + 1)
                        oy0 = int(it - ot)
                        ox0 = int(il - ol)
                        ny0 = int(it - nt)
                        nx0 = int(il - nl)
                        merged[ny0: ny0 + h_int, nx0: nx0 + w_int] = torch.maximum(
                            merged[ny0: ny0 + h_int, nx0: nx0 + w_int],
                            prev_mask[oy0: oy0 + h_int, ox0: ox0 + w_int].to(dtype=torch.uint8),
                        )

                    crop_mask = torch.maximum(merged, cur_mask.to(dtype=torch.uint8))
                else:
                    crop_mask = cur_mask

                matched.merge_same_frame(
                    roi_box=union_roi,
                    crop_box=crop_box,
                    crop_img=crop_img,
                    crop_mask=crop_mask,
                )
            else:
                # Extension path. Two sub-cases:
                #   (a) Normal extend: matched.frame_end == frame_num - 1
                #   (b) Gap reactivation: matched.frame_end < frame_num - 1
                #       (the scene has been lingering after a detection drop).
                #       Fill the gap with interpolated boxes/sources first.
                if matched.frame_nums and matched.frame_end < frame_num - 1:
                    prev_box = matched.roi_boxes[-1]
                    filled = self._fill_gap_frames(matched, prev_box, box, frame_num)
                    if not filled:
                        # Source frame missing — treat as hard expiry. Let
                        # the scene complete naturally and start a brand-new
                        # one for this detection.
                        new_scene = self._new_scene(frame_num)
                        self._scenes.append(new_scene)
                        # Compute crop for the new scene
                        h, w = int(frame_bgr_u8.shape[0]), int(frame_bgr_u8.shape[1])
                        crop_box, crop_img, crop_mask = self._compute_crop(
                            frame_bgr_u8, box, mask, force_crop_box=None,
                        )
                        new_scene.add_frame(
                            frame_num=frame_num,
                            roi_box=box,
                            crop_box=crop_box,
                            crop_img=crop_img,
                            crop_mask=crop_mask,
                            is_detection=True,
                        )
                        continue

                h, w = int(frame_bgr_u8.shape[0]), int(frame_bgr_u8.shape[1])
                base_crop_box, _ = crop_box_to_target_v3(
                    box,
                    img_h=h,
                    img_w=w,
                    target_hw=(self.cfg.clip_size, self.cfg.clip_size),
                    max_box_expansion_factor=self.cfg.max_box_expansion_factor,
                    border_size=float(self.cfg.border_size),
                )

                prev_crop_box = matched.crop_boxes[-1] if matched.crop_boxes else None
                stable_crop_box = self._stabilize_crop_box(
                    roi_box=box,
                    base_crop_box=base_crop_box,
                    prev_crop_box=prev_crop_box,
                    img_h=h,
                    img_w=w,
                )

                crop_box, crop_img, crop_mask = self._compute_crop(
                    frame_bgr_u8,
                    box,
                    mask,
                    force_crop_box=stable_crop_box,
                )
                matched.add_frame(
                    frame_num=frame_num,
                    roi_box=box,
                    crop_box=crop_box,
                    crop_img=crop_img,
                    crop_mask=crop_mask,
                    is_detection=True,
                )

        # Decide which scenes complete THIS FRAME.
        #
        # Lada's rule: any scene not updated this frame → complete immediately.
        # We relax that to "any scene whose last *real* detection is older
        # than TTL frames → complete". Scenes within the TTL window stay
        # alive; if a future frame's detection matches them, the gap is
        # filled by interpolation (in the extension path above).
        ttl = int(getattr(self.cfg, "ttl_after_end", 0) or 0)

        completed_gap: List[Scene] = [
            s for s in self._scenes
            if s.last_detected_frame >= 0
            and (frame_num - s.last_detected_frame) > ttl
        ]

        # And any scenes that reached max length are completed.
        completed_maxlen: List[Scene] = [s for s in self._scenes if len(s) >= self.cfg.max_clip_length]

        # LADA-faithful rule: when a scene completes, also complete any scenes that started earlier.
        # This guarantees deterministic clip ordering and prevents early-started scenes from
        # blocking drain forever when later scenes end first.
        completed_scenes: List[Scene] = []
        reason_by_scene_id: Dict[int, str] = {}
        for s in completed_gap:
            reason_by_scene_id[s.id] = "ttl_expired"
        for s in completed_maxlen:
            reason_by_scene_id.setdefault(s.id, "max_len")

        for current_scene in list(self._scenes):
            ttl_expired = (
                current_scene.last_detected_frame >= 0
                and (frame_num - current_scene.last_detected_frame) > ttl
            )
            is_done = ttl_expired or (len(current_scene) >= self.cfg.max_clip_length)
            if not is_done:
                continue

            if current_scene not in completed_scenes:
                completed_scenes.append(current_scene)

            for other_scene in self._scenes:
                if other_scene is current_scene:
                    continue
                if other_scene.frame_start < current_scene.frame_start and other_scene not in completed_scenes:
                    completed_scenes.append(other_scene)
                    reason_by_scene_id.setdefault(other_scene.id, "cascade")

        # LADA: complete in ascending start-frame order.
        completed_unique: List[Scene] = sorted(completed_scenes, key=lambda s: s.frame_start)

        # Remove completed from active list.
        if completed_unique:
            completed_ids = {s.id for s in completed_unique}
            self._scenes = [s for s in self._scenes if s.id not in completed_ids]

        t1 = time.perf_counter()

        new_clips: List[Clip] = []
        t_clip_build = 0.0
        for s in completed_unique:
            tb0 = time.perf_counter()
            clip = self._scene_to_clip(s)
            if clip is not None:
                new_clips.append(clip)

                if self.cfg.debug:
                    why = reason_by_scene_id.get(s.id, "?")
                    roi_xyxy = s.roi_boxes[-1] if s.roi_boxes else (0, 0, 0, 0)
                    print(
                        f"[Clip] clip_id={self._clip_counter:5d} scene_id={s.id:4d} why={why:10s} "
                        f"frames={s.frame_start:5d}-{s.frame_end:5d} len={len(s):3d} roi_xyxy={roi_xyxy}"
                    )

                self._clip_counter += 1

            if self.cfg.debug:
                why = reason_by_scene_id.get(s.id, "?")
                roi_xyxy = s.roi_boxes[-1] if s.roi_boxes else (0, 0, 0, 0)
                print(
                    f"[Clip] clip_id={self._clip_counter:5d} scene_id={s.id:4d} why={why:10s} "
                    f"frames={s.frame_start:5d}-{s.frame_end:5d} len={len(s):3d} roi_xyxy={roi_xyxy}"
                )
            tb1 = time.perf_counter()
            t_clip_build += (tb1 - tb0)

        overlay_boxes: List[Box] = []
        # Include scenes that complete on this frame (e.g. max_len), otherwise you see missing overlays.
        for s in self._scenes:
            if s.frame_end == frame_num and s.roi_boxes:
                overlay_boxes.append(s.roi_boxes[-1])
        for s in completed_unique:
            if s.frame_end == frame_num and s.roi_boxes:
                overlay_boxes.append(s.roi_boxes[-1])

        return TrackerStepResult(
            overlay_boxes=overlay_boxes,
            new_clips=new_clips,
            active_scenes=len(self._scenes),
            t_track=(t1 - t0),
            t_clip_build=t_clip_build,
        )

    def flush_eof(self, *_: object) -> List[Clip]:
        """Flush all remaining scenes at end-of-file."""
        clips: List[Clip] = []
        for s in self._scenes:
            clip = self._scene_to_clip(s)
            if clip is None:
                continue
            clips.append(clip)
            self._clip_counter += 1

        self._scenes.clear()
        return clips

    # --- Compatibility helpers (older pipeline revisions) ---
    def ingest_frame(self, frame_num: int, frame_bgr_u8: torch.Tensor, roi_boxes: Sequence[Box]) -> List[Box]:
        """Compat: older pipelines expect an ingest_frame() that returns overlay boxes."""
        res = self.step_frame(frame_num, frame_bgr_u8, roi_boxes)
        return res.overlay_boxes

    def flush_completed(self, current_frame: int) -> List[Clip]:
        """Compat: older pipelines called flush_completed() explicitly.

        In the current implementation, completion is handled inside step_frame().
        This is therefore a no-op and returns an empty list.
        """
        return []


__all__ = [
    "TrackerConfig",
    "TrackerStepResult",
    "SceneTracker",
]
