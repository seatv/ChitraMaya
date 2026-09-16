# ChitraMaya/mosaic/redecode.py
"""CM-191 -- make frames free again: the re-decode frame source.

Until v1.70 every full frame from the oldest open clip's first frame to the
current frame lived in a FrameStore (VRAM, or system RAM on the CM-084 host
backend) so that the compositor could paste restored clips into it later.
At 4K that is 24.9 MB per frame; with a continuous mosaic the store sits at
~Max Clip Length frames for the whole run (5 GB at MCL 180), and the host
backend moved every frame GPU->host->GPU -- ~5 GB/s of PCIe traffic on the
Dell's 4-lane OCuLink link (measured 2026-09-12: 2.7 GB/s in + 2.4 GB/s
out while NVDEC sat at 15-35 %).

Here the frame is not stored. When a clip closes, the compositor's work
BEFORE the blend (secondary upscale, unpad, resize to the original crop
shape) runs at once and is kept as a small per-frame ``Patch``; the frame
itself is produced again by a second decoder (``LagDecoder``) on the same
source, running behind the primary, exactly when the frame is safe to
encode. The blend then runs on the re-decoded frame, in the same clip order
as before, so the output is bit-identical to the FrameStore path.

Alignment: frame numbers are the contract, PTS is the check. Both decoders
count frames from the same start of the same bitstream; the primary's PTS
for each frame is recorded in the ``FramePlan`` and the lagging decoder's
frame must carry the same PTS. Lag PTS below the expected value = a frame
the primary never delivered -> skipped and counted. Lag PTS above it = the
lagging decoder dropped a frame the primary had -> the previous output
frame is encoded again (the same honest freeze CM-120 uses) and counted.
Both counts print at the end of the run; the field expectation is zero.

CM-196 (2026-09-14): pending patches wait in pinned host RAM by default
(``RedecodeStore(patch_home="host")``). Frames never cross the bus; patches
do, at ~3 MB each -- an 8 GB card running the full stack has no VRAM to
spare for a gigabyte of them (three NVENC error-8 deaths on one 4K60 title
at its 2400-px regions, VRAM pinned at the ceiling from the first hour).

Weights travel; content never does. This module holds no content.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from chitramaya.mosaic.core.scene import Box, Pad
from chitramaya.mosaic.pipeline_utils import (
    FrameStore,
    PtsGapFiller,
    _maybe_dump_preencode_frame,
    bgr_u8_to_bgra_u8,
    sync_device,
)
from chitramaya.mosaic.restorer.compositor import blend_patch, finish_patch, prepare_patch


@dataclass
class Patch:
    """One restored region for one frame, kept at CLIP resolution (CM-193):
    the secondary upscale has run, unpad + resize to the frame happen at
    blend time. Memory per patch is bounded by clip_size x sec_scale, not
    by the region's size on the frame."""
    frame_num: int
    img_u8: torch.Tensor      # HWC uint8, clip_size*sec_scale square, still padded
    sec_scale: int
    mask_u8: torch.Tensor     # HW uint8 at clip resolution, still padded
    pad: Pad
    orig_shape_hw: Tuple[int, int]
    box: Box
    model_dtype: torch.dtype
    border_ratio: float
    blendmask: str
    feather_radius: int

    def nbytes(self) -> int:
        try:
            return int(self.img_u8.numel() * self.img_u8.element_size()
                       + self.mask_u8.numel() * self.mask_u8.element_size())
        except Exception:
            return 0


PATCH_HOMES = ("host", "device")


def normalize_patch_home(value) -> str:
    v = str(value or "host").strip().lower()
    return v if v in PATCH_HOMES else "host"


def _to_pinned_host(t: torch.Tensor) -> torch.Tensor:
    """CM-196: park a u8 tensor in page-locked host RAM. A byte copy both
    ways, so the blend arithmetic never changes. The copy is queued on the
    current stream; the tensor is only ever read back on that stream (the
    drain's ``.to(device)``), so no host-side sync is needed. Falls back to
    a pageable copy where pinning is unavailable."""
    if t.device.type == "cpu":
        return t.contiguous()
    try:
        h = torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=True)
        h.copy_(t, non_blocking=True)
        return h
    except Exception:
        return t.detach().to("cpu")


@dataclass
class FramePlan:
    """Per frame: the primary decoder's PTS and the patches to blend."""
    pts: Dict[int, Optional[int]] = field(default_factory=dict)
    patches: Dict[int, List[Patch]] = field(default_factory=dict)
    patch_bytes: int = 0
    peak_patch_bytes: int = 0

    def note_frame(self, frame_num: int, pts: Optional[int]) -> None:
        self.pts[int(frame_num)] = pts

    def add_patch(self, p: Patch) -> None:
        self.patches.setdefault(int(p.frame_num), []).append(p)
        self.patch_bytes += p.nbytes()
        if self.patch_bytes > self.peak_patch_bytes:
            self.peak_patch_bytes = self.patch_bytes

    def keys_before(self, safe_before: int) -> List[int]:
        return sorted(k for k in self.pts.keys() if k < int(safe_before))

    def pop(self, frame_num: int) -> Tuple[Optional[int], List[Patch]]:
        k = int(frame_num)
        pts = self.pts.pop(k, None)
        ps = self.patches.pop(k, [])
        for p in ps:
            self.patch_bytes -= p.nbytes()
        if self.patch_bytes < 0:
            self.patch_bytes = 0
        return pts, ps

    def __len__(self) -> int:
        return len(self.pts)


class RedecodeStore(FrameStore):
    """Drop-in for ``FrameStore`` that stores NOTHING.

    ``put`` records the frame's PTS in the plan and lets the frame go;
    ``frames_bgr_u8`` stays empty, so ``is_full()`` is never true, the
    backpressure loops never engage and the FOI snapshot guards see no
    frame. Everything the pipeline used to read off the store (gap_filler,
    max_frames, backend) is still there.
    """

    def __init__(self, patch_home: str = "host") -> None:
        super().__init__(max_frames=0, backend="device")
        self.backend = "redecode"
        self.plan = FramePlan()
        self.clips_planned: int = 0
        # CM-196: where pending patches wait. "host" (default) parks them in
        # pinned RAM -- ~3 MB per frame with a 4x secondary, so a 180-frame
        # clip is ~570 MB of RAM instead of VRAM. On an 8 GB card the full
        # stack (detector + restorer engines, RTX SS, temporal fix, two
        # NVDEC sessions, NVENC) already fills the card; 1.1 GB of patches
        # stacked by overlapping MCL-180 clips at the largest regions is
        # what pushed three runs of one 4K60 title into NVENC error 8.
        # "device" keeps the T10c behaviour for A/B.
        self.patch_home = normalize_patch_home(patch_home)

    def put(self, frame_num: int, frame_bgr_u8: torch.Tensor, pts: Optional[int] = None) -> None:
        # The frame is dropped here on purpose: it will be decoded again.
        self.plan.note_frame(frame_num, pts)
        if self._frame_bytes == 0 and frame_bgr_u8 is not None and frame_bgr_u8.numel() > 0:
            self._frame_bytes = int(frame_bgr_u8.nelement() * frame_bgr_u8.element_size())

    def add_clip(
        self,
        clip,
        restored_frames_u8: List[Optional[torch.Tensor]],
        *,
        model_dtype: torch.dtype,
        blendmask: str,
        feather_radius: int,
        secondary=None,
        border_ratio: float = 0.05,
    ) -> int:
        """Run the pre-blend half of the compositor for a closed clip and
        queue one Patch per frame. Returns the number of patches queued."""
        n = min(len(restored_frames_u8), len(clip.frame_nums))
        added = 0
        for i in range(n):
            prepared = prepare_patch(clip, i, restored_frames_u8[i], secondary)
            if prepared is None:
                continue
            clip_img, sec_scale, clip_mask, pad, orig_shape_hw, orig_box = prepared
            fn = int(clip.frame_nums[i])
            if fn not in self.plan.pts:
                # A clip frame the primary never handed us (cannot happen
                # in the sequential pipeline; guard anyway).
                continue
            if self.patch_home == "host":
                img_keep = _to_pinned_host(clip_img)
                msk_keep = _to_pinned_host(clip_mask)
            else:
                # .clone() rather than .contiguous(): a contiguous VIEW of a
                # batch output would keep the whole batch alive in VRAM.
                img_keep = clip_img.contiguous().clone()
                msk_keep = clip_mask.contiguous().clone()
            self.plan.add_patch(Patch(
                frame_num=fn, img_u8=img_keep, sec_scale=int(sec_scale),
                mask_u8=msk_keep, pad=pad, orig_shape_hw=orig_shape_hw,
                box=orig_box, model_dtype=model_dtype, border_ratio=float(border_ratio),
                blendmask=str(blendmask), feather_radius=int(feather_radius),
            ))
            added += 1
        self.clips_planned += 1
        return added

    def __len__(self) -> int:
        return len(self.plan)

    def planned_bytes_mb(self) -> float:
        return self.plan.patch_bytes / (1024.0 * 1024.0)


class LagDecoder:
    """The second decoder: same source, same backend, read one frame at a
    time, aligned to the primary's PTS."""

    def __init__(self, primary, *, batch_size: int = 8) -> None:
        from chitramaya.video.decoder import Decoder  # local: keeps import order light

        # Decode exactly what the primary decodes: for an MPEG-TS source that
        # is the CM-120 CFR remux temp, which the primary already made.
        decode_path = str(getattr(primary, "_decode_path", None) or primary.input_path)
        # Small batches, and a prefetch queue twice the batch (the same
        # ratio the primary runs with), so a surface handed to us is never
        # recycled by the background decoder before we have converted it.
        # CM-196: the pipeline asks for batch 2 -> a 4-frame queue, ~100 MB
        # of VRAM at 4K RGBP (was 8 frames / ~200 MB); the lag read is a
        # wait on an already-decoded frame, so the batch size costs nothing.
        self._dec = Decoder(
            input_path=decode_path,
            gpu_id=int(primary.gpu_id),
            batch_size=int(batch_size),
            output_format=str(getattr(primary, "output_format", "RGBP")),
            ffmpeg_input_args=str(getattr(primary, "ffmpeg_input_args", "") or ""),
            trim_negative_pts=False,
            threaded_buffer_size=max(4, 2 * int(batch_size)),
        )
        if self._dec.backend != primary.backend:
            # Different backends can deliver different frame sequences
            # (NVDEC vs software on a broken stream). Refuse: the caller
            # falls back to the FrameStore.
            try:
                self._dec.close()
            except Exception:
                pass
            raise RuntimeError(
                f"re-decode backend {self._dec.backend!r} differs from the "
                f"primary decoder's {primary.backend!r}")
        # Take over the TS remux temp so the primary's close() (which runs
        # before the final drain) does not delete the file we still read.
        remux = getattr(primary, "_ts_remux_path", None)
        if remux:
            self._dec._ts_remux_path = remux
            primary._ts_remux_path = None
        self.backend = self._dec.backend
        self._buf: List[Any] = []
        self._buf_pts: List[Optional[int]] = []
        self._eof = False
        self.frames_read = 0
        self.skipped = 0        # lag had a frame the primary never delivered
        self.missing = 0        # primary had a frame the lag never delivered
        self.count_only = 0     # frames aligned by count (no PTS on one side)
        self.t_decode = 0.0

    # -- raw sequential read ------------------------------------------------
    def _fill(self) -> bool:
        if self._eof:
            return False
        t0 = time.perf_counter()
        frames, pts = self._dec.read_batch_with_pts()
        self.t_decode += time.perf_counter() - t0
        if not frames:
            self._eof = True
            return False
        self._buf = list(frames)
        self._buf_pts = list(pts) if pts else [None] * len(frames)
        while len(self._buf_pts) < len(self._buf):
            self._buf_pts.append(None)
        return True

    def _peek(self) -> Optional[Tuple[Any, Optional[int]]]:
        if not self._buf and not self._fill():
            return None
        return self._buf[0], self._buf_pts[0]

    def _take(self) -> None:
        self._buf.pop(0)
        self._buf_pts.pop(0)
        self.frames_read += 1

    # -- aligned read -------------------------------------------------------
    def next_aligned(self, expected_pts: Optional[int]) -> Optional[Any]:
        """Return the decoder item for the frame whose PTS is expected_pts,
        or None when that frame cannot be produced (the caller freezes the
        previous output frame)."""
        while True:
            got = self._peek()
            if got is None:
                self.missing += 1
                return None
            item, pts = got
            if expected_pts is None or pts is None:
                self.count_only += 1
                self._take()
                return item
            if pts == expected_pts:
                self._take()
                return item
            if pts < expected_pts:
                # Primary never delivered this one; drop it and look again.
                self.skipped += 1
                self._take()
                continue
            # pts > expected: the lag decoder does not have this frame.
            self.missing += 1
            return None

    def close(self) -> None:
        try:
            self._dec.close()
        except Exception:
            pass
        self._buf = []
        self._buf_pts = []
        self._eof = True

    def summary(self) -> str:
        s = (f"[Redecode] frames re-decoded: {self.frames_read}"
             f" (t_redecode {self.t_decode:.1f}s)")
        if self.skipped or self.missing:
            s += (f"; WARNING alignment: {self.skipped} frame(s) skipped, "
                  f"{self.missing} missing (frozen)")
        if self.count_only:
            s += f"; {self.count_only} frame(s) aligned by count (no PTS)"
        return s


def drain_plan_to_encoder(
    *,
    store: RedecodeStore,
    lag: LagDecoder,
    to_bgr: Callable[[Any], torch.Tensor],
    safe_before: int,
    encoder,
    device: torch.device,
    sync_before_encode: bool = True,
    pts_log: Optional[List[Tuple[int, Optional[int]]]] = None,
    foi_target: Optional[int] = None,
    foi_capture: Optional[dict] = None,
) -> int:
    """Re-decode, blend and encode every planned frame below safe_before,
    in frame order. Mirrors ``drain_store_to_encoder`` step for step (sync
    once per drain, BGRA conversion, CM-120 gap filler, pts_log)."""
    keys = store.plan.keys_before(safe_before)
    if not keys:
        return 0
    if sync_before_encode:
        sync_device(device)

    filler: Optional[PtsGapFiller] = getattr(store, "gap_filler", None)
    last_bgra: Optional[torch.Tensor] = getattr(store, "_last_bgra", None)
    count = 0
    for k in keys:
        pts, patches = store.plan.pop(k)
        item = lag.next_aligned(pts)
        if item is None:
            # The lagging decoder cannot produce this frame: freeze the
            # previous output frame so the timeline keeps its length.
            if last_bgra is not None:
                if pts_log is not None:
                    pts_log.append((k, pts))
                encoder.encode_frame(last_bgra)
                count += 1
            continue
        frm_bgr = to_bgr(item)
        if foi_capture is not None and foi_target is not None and k == int(foi_target):
            foi_capture["original"] = frm_bgr.detach().clone()
        for p in patches:
            img = p.img_u8
            msk = p.mask_u8
            if img.device != frm_bgr.device:
                # CM-196: host-parked patch comes back on the current stream
                # (pinned -> async H2D; the blend below is stream-ordered).
                img = img.to(frm_bgr.device, non_blocking=bool(img.is_pinned()))
            if msk.device != frm_bgr.device:
                msk = msk.to(frm_bgr.device, non_blocking=bool(msk.is_pinned()))
            # CM-193: unpad + resize to the frame happen here, one frame at a
            # time, so nothing at frame resolution is ever held pending.
            img, msk, box = finish_patch(img, p.sec_scale, msk, p.pad, p.orig_shape_hw, p.box)
            blend_patch(
                frm_bgr, img, msk, box,
                model_dtype=p.model_dtype, border_ratio=p.border_ratio,
                blendmask=p.blendmask, feather_radius=p.feather_radius,
            )
        if foi_capture is not None and foi_target is not None and k == int(foi_target):
            foi_capture["composited"] = frm_bgr.detach().clone()
        _maybe_dump_preencode_frame(k, frm_bgr, pts)
        if pts_log is not None:
            pts_log.append((k, pts))
        bgra = bgr_u8_to_bgra_u8(frm_bgr).contiguous()
        if filler is not None:
            n_fill = filler.fills_before(k, pts)
            if n_fill and filler.last_bgra is not None:
                for _ in range(n_fill):
                    encoder.encode_frame(filler.last_bgra)
            filler.remember(pts, bgra)
        encoder.encode_frame(bgra)
        last_bgra = bgra
        count += 1
    store._last_bgra = last_bgra
    return count


__all__ = ["Patch", "FramePlan", "RedecodeStore", "LagDecoder", "drain_plan_to_encoder",
           "PATCH_HOMES", "normalize_patch_home"]
