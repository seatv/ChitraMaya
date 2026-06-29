from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import torch

from chitramaya.mosaic.core.clip import Clip


class BaseClipRestorer(ABC):
    """Base class for clip-based restorers (LADA-style).

    A clip restorer works on *clips* (cropped/resized temporal windows), not whole frames.

    Contract:
      - Input: Clip
          clip.frames: List[float32 HWC] in [0,1], each [clip_size, clip_size, 3]
          clip.masks:  List[uint8 HW] (0..255), each [clip_size, clip_size]

      - Output: List[float32 HWC] in [0,1], same length and shape as clip.frames.

    The compositor/pipeline is responsible for mapping the restored clip frames back
    onto the full-resolution frames using the clip's mapping metadata.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device

    @abstractmethod
    def restore_clip(self, clip: Clip) -> List[torch.Tensor]:
        raise NotImplementedError


__all__ = ["BaseClipRestorer"]
