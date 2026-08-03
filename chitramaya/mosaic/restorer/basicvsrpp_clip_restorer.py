from __future__ import annotations

from typing import List

import torch

from chitramaya.mosaic.restorer.clip_restorer import BaseClipRestorer
from chitramaya.mosaic.models.basicvsrpp.inference import load_model


class BasicVSRPPClipRestorer(BaseClipRestorer):
    """
    Near-direct port of LADA's BasicvsrppMosaicRestorer behavior:
      - input: list of uint8 HWC frames
      - model input: float (0..1) via /255
      - output: uint8 HWC with round+clamp
      - optional chunking via max_frames
    """

    def __init__(
            self,
            model_path: str,
            device: torch.device,
            *,
            fp16: bool,
            max_frames: int = 32,
    ) -> None:
        super().__init__(device=device)

        # Define fp16 + dtype BEFORE loading. CM-093: XPU passes the fp16
        # op battery (incl. deform_conv2d), so Arc runs half too.
        self.fp16 = bool(fp16) and (self.device.type in ("cuda", "xpu"))
        self.model_dtype = torch.float16 if self.fp16 else torch.float32
        self.max_frames = int(max_frames)

        # gRestorer load_model signature: (config, checkpoint_path, device, fp16=...)
        self.model = load_model(config=None, checkpoint_path=model_path, device=self.device, fp16=self.fp16)

        # CM-093 channels_last experiment: NEGATIVE RESULT, default OFF.
        # The probe (tools/xpu_xmx_probe.py) measured 6.5x for channels_last
        # on a pure conv stack, but the FIELD test (Arc A580, 2026-07-31)
        # measured per-conv hook coercion at 27s -> 31-41s per chunk: real
        # BasicVSR++ interleaves convs with NCHW-forcing ops (deform_conv,
        # grid_sample, cat), so the hooks pay two transposes around nearly
        # every conv and the tax exceeds the XMX win at these tensor sizes.
        # Harvesting the layout win needs GRAPH-level planning (torch.compile),
        # not per-op hooks. Kept behind an env flag for future experiments.
        import os as _os
        if (self.device.type == "xpu"
                and _os.environ.get("CM_XPU_CHANNELS_LAST", "") == "1"):
            from chitramaya.device import enable_channels_last_convs
            _n = enable_channels_last_convs(self.model)
            print(f"[Restorer] XPU: channels_last EXPERIMENT enabled on {_n} "
                  f"conv layers (field-measured SLOWER in eager; see "
                  f"basicvsrpp_clip_restorer.py note)")

        # CM-093 torch.compile experiment (DEV-MACHINE ONLY, default OFF).
        # This is the graph-level layout planning the channels_last note
        # above points at: Inductor can plan NHWC across the whole graph
        # instead of paying per-op transpose tax -- the only known route
        # to Arc's XMX units for this model.
        #
        # WHY THIS IS NOT A USER-FACING FEATURE: Inductor is a runtime
        # JIT that generates C++ host code and invokes a SYSTEM compiler
        # (MSVC cl.exe on Windows) on whatever machine runs it -- unlike
        # TensorRT, whose builder ships as a library inside the distro.
        # Users cannot be asked to install Build Tools, and Dynamo's
        # source introspection is unreliable inside a frozen PyInstaller
        # app anyway. This flag exists for SOURCE-INSTALL experiments on
        # a dev box with MSVC (run from an "x64 Native Tools" prompt).
        # If the experiment wins, the shippable path is ahead-of-time
        # export (torch.export / AOTInductor artifacts, TRT-engine
        # style), which is a separate work item.
        #
        # Safety: suppress_errors=True makes ANY downstream Dynamo/
        # Inductor failure (which surfaces lazily at first forward, not
        # here) fall back to eager per-graph instead of crashing the
        # run -- so even a mistakenly-set flag on a machine without MSVC
        # degrades to a warning + eager, never a dead run. First chunk
        # of each distinct clip length compiles for minutes (the X1c
        # heartbeat makes that visible); steady state is the number that
        # matters. CM_XPU_COMPILE=1 -> default mode; any other value is
        # passed through as the Inductor mode (e.g. max-autotune).
        _cmp = _os.environ.get("CM_XPU_COMPILE", "").strip()
        if self.device.type == "xpu" and _cmp:
            try:
                import torch._dynamo as _dynamo
                _dynamo.config.suppress_errors = True
                _mode = None if _cmp == "1" else _cmp
                self.model = torch.compile(self.model, mode=_mode)
                print(f"[Restorer] XPU: torch.compile EXPERIMENT enabled "
                      f"(mode={_mode or 'default'}, dev-machine flag). "
                      f"First chunk per clip length will be SLOW (kernel "
                      f"compilation); steady state is the number that "
                      f"matters. Compile failures fall back to eager.")
            except Exception as _ce:
                print(f"[Restorer] WARNING: torch.compile unavailable "
                      f"({_ce}); continuing in eager mode.")

    @torch.inference_mode()
    def restore_clip(self, clip) -> List[torch.Tensor]:
        frames = clip.frames  # uint8 HWC
        if not frames:
            return []

        out_frames: List[torch.Tensor] = []
        n = len(frames)
        dtype = self.model_dtype

        # CM-093 X1c: heartbeat around each chunk forward on non-CUDA
        # devices. The first BasicVSR++ forward on a fresh XPU stack can
        # legitimately take minutes (kernel compilation), which reads as a
        # watchdog "stall" -- these prints distinguish slow-but-alive
        # (chunk lines keep appearing, times shrink) from truly wedged
        # (a "chunk start" with no matching "done"). Silent on CUDA.
        _hb = self.device.type != "cuda"
        import time as _time

        for start in range(0, n, self.max_frames):
            chunk = frames[start : start + self.max_frames]

            # TCHW uint8
            tchw_u8 = torch.stack([f.permute(2, 0, 1).contiguous() for f in chunk], dim=0)
            # 1,T,C,H,W float
            btchw = tchw_u8.to(device=self.device, dtype=dtype).div_(255.0).unsqueeze(0)

            if _hb:
                print(f"[Restorer] chunk start: frames {start}..{start + len(chunk) - 1} "
                      f"of {n} ({self.device.type}, "
                      f"{'fp16' if dtype == torch.float16 else 'fp32'})", flush=True)
                _t0 = _time.perf_counter()

            out = self.model(inputs=btchw)  # -> BTCHW
            out_tchw = out.squeeze(0)

            if _hb:
                # Force completion of the queued device work so the timing
                # is honest (async dispatch would otherwise report ~0s).
                from chitramaya.device import sync as _dev_sync
                _dev_sync(self.device)
                print(f"[Restorer] chunk done in {_time.perf_counter() - _t0:.1f}s "
                      f"({len(chunk)} frames)", flush=True)

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


__all__ = ["BasicVSRPPClipRestorer"]
