"""gRestorer utils package.

IMPORTANT: keep this file lightweight.
Do NOT import submodules here (mask_utils, image_utils, etc.), because many
call sites import things like `gRestorer.utils.config_util`, which executes
this package __init__ first. Importing submodules here easily creates circular
imports during startup.
"""

from __future__ import annotations

from typing import Any, Tuple, TypeAlias
import numpy as np

# Simple type aliases used throughout the project (and by vendored LADA code).
Image: TypeAlias = np.ndarray                 # HWC uint8/float image
Mask: TypeAlias = np.ndarray                  # HW or HWC(1) uint8 mask
Box: TypeAlias = Tuple[int, int, int, int]    # (t, l, b, r)
Pad: TypeAlias = Tuple[int, int, int, int]    # (top, left, bottom, right)
ImageTensor: TypeAlias = Any                  # torch.Tensor at runtime (kept as Any to avoid torch import)

__all__ = ["Image", "Mask", "Box", "Pad", "ImageTensor"]
