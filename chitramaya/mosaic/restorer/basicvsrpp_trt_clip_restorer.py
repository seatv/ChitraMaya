# ChitraMaya/mosaic/restorer/basicvsrpp_trt_clip_restorer.py
"""TensorRT-backed clip restorer for BasicVSR++.

Mirrors ``BasicVSRPPClipRestorer`` (the PyTorch path) but uses pre-compiled
TensorRT sub-engines for inference. Public interface is identical:
``restore_clip(clip) -> List[uint8 HWC tensor]``.

The split wrapper (``BasicVSRPlusPlusNetSplit``) handles the recurrent
propagation in Python while delegating each loop body, the preprocess
stage, and the upsample stage to compiled TRT engines.

Engine files are expected to exist alongside the .pth checkpoint in a
``<stem>_sub_engines/`` directory; see
``ChitraMaya.mosaic.models.basicvsrpp.engine_paths`` for the convention.
Compile them once with ``tools/compile_basicvsrpp.py`` before using this
restorer.
"""
from __future__ import annotations

from typing import List

import torch

from chitramaya.mosaic.restorer.clip_restorer import BaseClipRestorer
from chitramaya.mosaic.models.basicvsrpp.inference import load_model
from chitramaya.mosaic.models.basicvsrpp.sub_engines import create_split_forward
from chitramaya.mosaic.models.basicvsrpp.engine_paths import (
    all_basicvsrpp_sub_engines_exist,
)


class BasicVSRPPTRTClipRestorer(BaseClipRestorer):
    """TensorRT-backed BasicVSR++ clip restorer.

    Constructor loads the PyTorch model shell (BasicVSRPlusPlusNet) for its
    backbone modules used in the split orchestrator's edge cases (first
    frame of each propagation direction), then loads the 6 compiled
    sub-engines and wires them into a ``BasicVSRPlusPlusNetSplit``.

    Inference call style matches the PyTorch restorer: the same chunking
    loop with the same uint8 input/output conversions.
    """

    def __init__(
            self,
            model_path: str,
            device: torch.device,
            *,
            fp16: bool,
            max_clip_size: int = 60,
            max_frames: int | None = None,
    ) -> None:
        super().__init__(device=device)

        self.fp16 = bool(fp16) and (self.device.type == "cuda")
        self.model_dtype = torch.float16 if self.fp16 else torch.float32
        self.max_clip_size = int(max_clip_size)
        # Chunk size for the inference loop. Default = max_clip_size so we
        # send whole compatible clips through one engine call.
        self.max_frames = int(max_frames) if max_frames is not None else self.max_clip_size

        if not all_basicvsrpp_sub_engines_exist(
            model_path, fp16=self.fp16, max_clip_size=self.max_clip_size,
        ):
            raise FileNotFoundError(
                f"BasicVSR++ TRT sub-engines not found for "
                f"max_clip_size={self.max_clip_size}, fp16={self.fp16} "
                f"alongside {model_path}. "
                f"Run `ChitraMaya -compile-rest --rest-model {model_path} "
                f"--rest-max-clip-length {self.max_clip_size}` first."
            )

        # Load PyTorch model first — the split orchestrator references
        # generator.backbone[direction] for the first-iteration edge case
        # (when feat_prop is still zero, we use the PyTorch backbone, not
        # the loop-body engine). This is a small amount of compute per
        # clip; the bulk runs through TRT.
        pt_model = load_model(
            config=None,
            checkpoint_path=model_path,
            device=self.device,
            fp16=self.fp16,
        )

        split = create_split_forward(
            model=pt_model,
            model_weights_path=model_path,
            device=self.device,
            fp16=self.fp16,
            max_clip_size=self.max_clip_size,
        )
        if split is None:
            raise RuntimeError(
                "create_split_forward returned None despite engine files "
                "being present — engine files may be corrupted or incompatible."
            )
        self.model = split

    @torch.inference_mode()
    def restore_clip(self, clip) -> List[torch.Tensor]:
        frames = clip.frames  # uint8 HWC
        if not frames:
            return []

        out_frames: List[torch.Tensor] = []
        n = len(frames)
        dtype = self.model_dtype

        for start in range(0, n, self.max_frames):
            chunk = frames[start : start + self.max_frames]

            # TCHW uint8 → 1,T,C,H,W float (model expects BTCHW)
            tchw_u8 = torch.stack(
                [f.permute(2, 0, 1).contiguous() for f in chunk], dim=0,
            )
            btchw = tchw_u8.to(device=self.device, dtype=dtype).div_(255.0).unsqueeze(0)

            # Split wrapper expects positional input (not inputs= kwarg)
            out = self.model(btchw)  # -> BTCHW
            out_tchw = out.squeeze(0)

            # Back to uint8 HWC, with LADA's rounding+clamp
            out_u8 = (
                out_tchw.mul(255.0)
                .round_()
                .clamp_(0, 255)
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
                .contiguous()
            )
            out_frames.extend(list(torch.unbind(out_u8, dim=0)))

        return out_frames


__all__ = ["BasicVSRPPTRTClipRestorer"]
