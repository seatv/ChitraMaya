from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover
    from chitramaya.mosaic.detector.core import Detection


class BaseRestorer(ABC):
    """
    Base class for all restorers.

    Pipeline contract (what the pipeline feeds every restorer, now and later):
      - bgr_frames: List[torch.Tensor]
          Each tensor is BGR uint8, shape [H, W, 3], on the decode device (CUDA/XPU/CPU).

      - detections_per_frame: Optional[Sequence[Detection]]
          One Detection per frame (or None if detection not run).

    Must return:
      - List[torch.Tensor] in the SAME format (BGR uint8 [H,W,3]).
    """

    def __init__(self, width: int, height: int) -> None:
        self.width = int(width)
        self.height = int(height)

    @property
    def requires_detection(self) -> bool:
        return False

    @abstractmethod
    def process_batch(
        self,
        bgr_frames: List[torch.Tensor],
        detections_per_frame: Optional[Sequence["Detection"]] = None,
    ) -> List[torch.Tensor]:
        raise NotImplementedError
