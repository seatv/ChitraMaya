from __future__ import annotations

from typing import List, Optional, Sequence
import torch

from .base_restorer import BaseRestorer
from chitramaya.mosaic.detector.core import Detection


class NoneRestorer(BaseRestorer):
    """
    No-op baseline restorer.
    - No detection required
    - Returns frames as-is (no copies).
    """

    def __init__(self, width: int, height: int, gpu_id: int = 0, **kwargs):
        super().__init__(width=width, height=height)
        self.gpu_id = int(gpu_id)

    @property
    def requires_detection(self) -> bool:
        return False

    def process_batch(
        self,
        bgr_frames: List[torch.Tensor],
        detections_per_frame: Optional[Sequence[Detection]] = None,
    ) -> List[torch.Tensor]:
        return bgr_frames
