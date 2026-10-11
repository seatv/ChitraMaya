# chitramaya/video/chroma_guard.py
"""T20d/T20e/T20f (CM-234): the decoded-frame guard for the ffmpeg NV12 lane.

Field case (09-24, 9060 XT, 20b): a 1080p60 clip restored with everything
else clean came back with single frames whose picture was intact but green
from one row down -- R and B at zero, the image in G alone. Measured: the
frame's luma plane was complete and the right frame; its UV plane was
correct for the first ~107 chroma rows and ZERO after. That is a frame whose
download from the GPU decoder left ffmpeg unfinished (d3d11va: a staging
texture read before the GPU had written all of it). The same clip through
software decode (CM_HW_DECODE=off) had no such frame; every bad frame sat on
the first frame of a decode batch, i.e. right after ffmpeg had been blocked
on the full pipe and resumed. NVIDIA never sees it (NVDEC, no ffmpeg pipe).

T20e: the 20d run on the 17 s cut showed the download can stop EARLIER, in
the luma plane (NV12 = all of Y, then all of UV): those frames were black
from a row down, with no chroma at all, and the detector -- which reads
luma -- missed them. So the guard covers both planes.

T20f: the unfinished part is not always zero. Two more frames from the same
box were magenta (U=V=255) over the top half with a white band (Y=255) full
of black dashes (Y=0) below it, and a frame with green/blue line PAIRS every
14 rows (one chroma row each, bytes at 0 with some at 255). Every colour in
those frames -- green, blue, magenta, white, black -- is what luma or chroma
bytes at exactly 0 or 255 produce, and nothing else: the staging buffer's
stale 0x00/0xFF pattern, landing as regions, as alternating row pairs, or as
dashes inside a row.

Why "a row with at least a quarter of its bytes at 0 or 255" is the
signature: in limited-range video (every title and stream seen so far) luma
lives in 16..235 and chroma in 16..240, so neither byte value occurs at all;
codecs may undershoot a little below 16 in crushed blacks, but not to
exactly 0 over a quarter of a row. In full-range video a zero luma row is a
black bar and a 255 row a blown-out flash -- and the donor row from the
previous frame is the same bar or the same flash, so the repair is a no-op
there; a zero or 255 chroma byte is an extreme saturated colour no camera
produces. Black frames (Y=16, U=V=128) are not hits.

The repair: unfinished rows take the previous frame's rows of the same
plane (one thirtieth of a second stale, invisible at play speed; the
detector then finds the mosaic where the previous frame had it). The very
first frame of a lane has no predecessor and is left as decoded, counted
(T20f: a flat fill could damage legitimate full-range content). Never silent: the pipeline prints the first hit, counts
the rest, puts the frames in the report (`decoder_frame_repair` event and
`frames.frame_repair`) and in run.warnings, so the run ends DONE WITH
WARNINGS and the Complete panel says so.

Cost: two row-wise reductions (3 MB at 1080p) and two `where` per frame,
all on the device and stream-ordered -- no host sync per frame. `flush()`
reads the accumulated flags with ONE sync per batch, where the pipeline has
synced anyway (the detector's boxes come back to the host).

One instance per decode lane (primary, re-decode, Add Mosaic): the
previous-frame reference must be the previous frame of THAT lane.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch


class Nv12ChromaGuard:
    """Check-and-repair for uploaded NV12 frames ([H*3/2, W] uint8 on the
    pipeline device). `apply()` per frame, `flush()` once per batch. (The
    name predates T20e; the guard covers luma and chroma.)"""

    def __init__(self, label: str = "decoder") -> None:
        self.label = str(label)
        self.frames_repaired: List[int] = []     # frame numbers (when the caller knows them)
        self.luma_rows: int = 0                  # luma rows taken from the previous frame / black
        self.chroma_rows: int = 0                # chroma rows taken from the previous frame / grey
        self.hits: int = 0                       # frames touched
        self.no_donor: int = 0                   # flagged frames left as decoded (no previous frame)
        self.checked: int = 0
        self._prev: Optional[torch.Tensor] = None   # the previous frame's (repaired) NV12
        # Per-batch pending flags: (frame_num, any_bad (0-d bool), n_luma (0-d), n_chroma (0-d), from_prev)
        self._pending: List[Tuple[Optional[int], torch.Tensor, torch.Tensor, torch.Tensor, bool]] = []

    @property
    def rows_repaired(self) -> int:
        return int(self.luma_rows + self.chroma_rows)

    # -- per frame -------------------------------------------------------------

    def apply(self, nv12: torch.Tensor, height: int, frame_num: Optional[int] = None) -> torch.Tensor:
        """Repair the unfinished rows of `nv12` in place (it is the lane's own
        upload) and remember it as the next frame's donor. Returns the same
        tensor. Frames whose shape does not match the expected NV12 layout
        pass through untouched."""
        h = int(height)
        if nv12.ndim != 2 or nv12.dtype != torch.uint8 or int(nv12.shape[0]) != h * 3 // 2:
            return nv12
        w = int(nv12.shape[1])
        y = nv12[:h, :]                                    # (h, w) luma
        uv = nv12[h:, :]                                   # (h/2, w) interleaved U,V bytes
        # A row the decoder finished has (almost) no bytes at 0 or 255 at all;
        # an unfinished one is the buffer's stale 0x00 / 0xFF pattern from some
        # column on, or dashes of it. ">= a quarter of the row" catches the
        # partly written boundary row and the dashed rows too (T20f).
        y_zero = ((y == 0) | (y == 255)).sum(dim=1) * 4 >= w       # (h,) bool, on device
        uv_zero = ((uv == 0) | (uv == 255)).sum(dim=1) * 4 >= w    # (h/2,) bool
        prev = self._prev
        from_prev = prev is not None and tuple(prev.shape) == tuple(nv12.shape)
        if from_prev:
            donor_y, donor_uv = prev[:h, :], prev[h:, :]
            y.copy_(torch.where(y_zero[:, None], donor_y, y))      # in place: the CSC reads the repaired planes
            uv.copy_(torch.where(uv_zero[:, None], donor_uv, uv))
        # T20f: with no previous frame (the lane's first frame) the row is left
        # as decoded and only counted -- a flat fill could damage legitimate
        # full-range content (a white flash, a black bar) and one first frame
        # is not worth the risk.
        self._prev = nv12                                  # the frame's own storage; lives as long as we hold it
        self._pending.append((frame_num, y_zero.any() | uv_zero.any(), y_zero.sum(), uv_zero.sum(), bool(from_prev)))
        self.checked += 1
        return nv12

    # -- per batch -------------------------------------------------------------

    def flush(self) -> List[Tuple[int, int, int, bool]]:
        """ONE host sync: returns [(position, luma_rows, chroma_rows, from_prev)]
        for the frames repaired since the last flush -- position = index of
        the apply() call in that span (= index in the batch when every item
        of the batch went through apply, which is the case on the CPU lane)
        -- and folds them into the counters. When apply() was given frame
        numbers they are recorded here; otherwise the caller adds them."""
        if not self._pending:
            return []
        pend = self._pending
        self._pending = []
        try:
            flags = torch.stack([p[1] for p in pend]).tolist()
            n_y = torch.stack([p[2] for p in pend]).tolist()
            n_uv = torch.stack([p[3] for p in pend]).tolist()
        except Exception:
            return []
        out: List[Tuple[int, int, int, bool]] = []
        for pos, ((fn, _a, _ny, _nuv, from_prev), f, ny, nuv) in enumerate(zip(pend, flags, n_y, n_uv)):
            if not f:
                continue
            ny, nuv = int(ny), int(nuv)
            self.hits += 1
            self.luma_rows += ny
            self.chroma_rows += nuv
            if not from_prev:
                self.no_donor += 1
            if fn is not None:
                self.frames_repaired.append(int(fn))
            out.append((pos, ny, nuv, from_prev))
        return out

    @staticmethod
    def describe_rows(luma_rows: int, chroma_rows: int) -> str:
        parts = []
        if luma_rows:
            parts.append(f"{int(luma_rows)} luma row(s)")
        if chroma_rows:
            parts.append(f"{int(chroma_rows)} chroma row(s)")
        return " + ".join(parts) if parts else "0 rows"

    def summary(self) -> str:
        """One line for the run summary; '' when nothing happened."""
        if not self.hits:
            return ""
        fr = self.frames_repaired
        where = (f" (frames {min(fr)}..{max(fr)})" if fr else "")
        nod = (f"; {self.no_donor} had no previous frame and was left as decoded"
               if self.no_donor else "")
        return (f"{self.hits} decoded frame(s) arrived unfinished from the hardware decoder "
                f"({self.describe_rows(self.luma_rows, self.chroma_rows)} at 0/255) and were completed "
                f"from the previous frame{where}{nod}")
