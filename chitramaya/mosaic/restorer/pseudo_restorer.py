# gRestorer/restorer/pseudo_restorer.py
from __future__ import annotations

"""
Pseudo Restorer - GPU-only visualization.

This restorer does NOT restore content. It only draws detection boxes / fills ROI
to verify detector alignment and measure pipeline overhead.

Contract:
- Input frames:  BGR uint8 [H,W,3] on GPU
- Output frames: BGR uint8 [H,W,3] on GPU (same tensors, modified in-place)
"""

from typing import List, Optional, Sequence, Tuple

import torch

from chitramaya.mosaic.detector.core import Detection
from chitramaya.mosaic.restorer.base_restorer import BaseRestorer
from chitramaya.mosaic.utils.visualization import draw_detections


class PseudoRestorer(BaseRestorer):
    """
    Pseudo restorer that draws boxes (and optional fill) around detected regions.

    Note: draw_detections currently implements boxes + optional fill on GPU.
          Text rendering is intentionally omitted.
    """

    def __init__(
        self,
        width: int,
        height: int,
        box_color: Tuple[int, int, int] = (0, 255, 0),  # BGR
        box_thickness: int = 2,
        show_confidence: bool = True,
        show_class: bool = True,
        fill_color: Optional[Tuple[int, int, int]] = None,  # BGR
        fill_opacity: float = 0.5,
        gpu_id: int = 0,
    ):
        super().__init__(width=width, height=height)
        self.box_color = box_color
        self.box_thickness = int(box_thickness)
        self.show_confidence = bool(show_confidence)
        self.show_class = bool(show_class)
        self.fill_color = fill_color
        self.fill_opacity = float(fill_opacity)
        self.gpu_id = int(gpu_id)

    @property
    def requires_detection(self) -> bool:
        return True

    def process_batch(
        self,
        bgr_frames: List[torch.Tensor],
        detections_per_frame: Optional[Sequence[Detection]] = None,
    ) -> List[torch.Tensor]:
        if detections_per_frame is None:
            return bgr_frames

        if len(detections_per_frame) != len(bgr_frames):
            raise ValueError(
                f"detections_per_frame length ({len(detections_per_frame)}) "
                f"!= frames length ({len(bgr_frames)})"
            )

        # Draw in-place for zero extra allocations.
        for i, det in enumerate(detections_per_frame):
            draw_detections(
                bgr_img=bgr_frames[i],
                detection=det,
                box_color=self.box_color,
                box_thickness=self.box_thickness,
                show_confidence=self.show_confidence,
                show_class=self.show_class,
                fill_color=self.fill_color,
                fill_opacity=self.fill_opacity,
            )

        return bgr_frames

    def __repr__(self) -> str:
        return f"PseudoRestorer({self.width}x{self.height}, GPU-only)"
