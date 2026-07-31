# chitramaya/device.py
"""CM-093 X1: accelerator helpers -- CUDA / Intel XPU / CPU.

Central, dependency-free device plumbing so the rest of the codebase never
needs `torch.cuda.*` guards inline. Resolution order is ALWAYS cuda ->
xpu -> cpu, so the NVIDIA fleet behaves byte-identically and an Intel Arc
machine lights up the xpu backend with the same build.

Phase-0 basis (2026-07-31, Arc A580, torch 2.13.0+xpu): full op battery
PASS on xpu in fp32 AND fp16 (incl. deform_conv2d, BasicVSR++'s alignment
op), ultralytics accepts device="xpu" natively, and the CM-078 stabilizer
runs end-to-end. See claude/ChitraMaya-CM093-XPU-Phase0.md (project) and
tools/xpu_probe.py.

What stays NVIDIA-only regardless of this module: NVDEC/NVENC
(PyNvVideoCodec), TensorRT engines, the Maxine RTX Super-Res secondary,
and the NVML PCIe canary. Their call sites gate on availability and
degrade with clear messages; XPU decode/encode arrive with CM-093 X2/X3.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def xpu_available() -> bool:
    return bool(getattr(getattr(torch, "xpu", None), "is_available",
                        lambda: False)())


def pick_device(gpu_id: int = 0) -> torch.device:
    """cuda -> xpu -> cpu, mirroring the pipelines' _pick_device."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    if xpu_available():
        return torch.device(f"xpu:{gpu_id}")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    """Wait for the device's queued work. Best-effort; never raises."""
    try:
        if device.type == "cuda":
            torch.cuda.synchronize(device=device)
        elif device.type == "xpu":
            torch.xpu.synchronize(device=device)  # type: ignore[attr-defined]
    except Exception:
        pass


def empty_cache(device: Optional[torch.device] = None) -> None:
    """Return the caching allocator's memory to the driver (see the Batch
    23c field story: hoarded cache starves out-of-torch allocators). With
    device=None, releases whatever accelerator is present."""
    try:
        t = device.type if device is not None else None
        if t in (None, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if t in (None, "xpu") and xpu_available():
            torch.xpu.empty_cache()  # type: ignore[attr-defined]
    except Exception:
        pass


def mem_get_info(device: torch.device) -> Tuple[Optional[int], Optional[int]]:
    """(free_bytes, total_bytes) at DRIVER level where the backend offers
    it, else (None, total) where only capacity is known, else (None, None).

    CUDA's number includes out-of-torch allocations (TRT, NVDEC/NVENC).
    XPU: torch.xpu.mem_get_info exists on current builds; older builds
    fall back to total capacity only -- callers already treat free=None
    as 'cannot measure' and skip the warning rather than guessing."""
    try:
        if device.type == "cuda":
            free, total = torch.cuda.mem_get_info(device)
            return int(free), int(total)
        if device.type == "xpu":
            mgi = getattr(torch.xpu, "mem_get_info", None)  # type: ignore[attr-defined]
            if mgi is not None:
                free, total = mgi(device)
                return int(free), int(total)
            props = torch.xpu.get_device_properties(  # type: ignore[attr-defined]
                getattr(device, "index", 0) or 0)
            total = int(getattr(props, "total_memory", 0))
            return None, (total if total > 0 else None)
    except Exception:
        pass
    return None, None


def fp16_supported(device: torch.device) -> bool:
    """fp16 inference eligibility. CUDA: needs Volta+ (cap major >= 7).
    XPU: phase-0 op battery passed fp16 across the board on Arc."""
    try:
        if device.type == "cuda":
            return torch.cuda.get_device_capability(device)[0] >= 7
        if device.type == "xpu":
            return True
    except Exception:
        pass
    return False


def enable_channels_last_convs(model: "torch.nn.Module") -> int:
    """CM-093 channels_last experiment. STATUS: NEGATIVE in eager mode.

    The probe (tools/xpu_xmx_probe.py) measured 6.5x for channels_last on
    a pure conv stack on Arc A580 -- eager oneDNN routes convs to XMX only
    for NHWC tensors. But the FIELD test on the real BasicVSR++ (Arc A580,
    2026-07-31) measured 27s -> 31-41s per chunk WITH this helper: the
    model interleaves convs with NCHW-forcing ops (deform_conv,
    grid_sample, cat), so per-conv hook coercion pays two transposes
    around nearly every conv, and the transpose tax exceeds the XMX win
    at the model's tensor sizes. Harvesting the layout win requires
    GRAPH-level layout planning (torch.compile / inductor), not per-op
    hooks. Callers gate this behind CM_XPU_CHANNELS_LAST=1 for future
    experiments; it is NOT part of the default xpu path.

    Mechanics (still the safe way to do it, when wanted): a blanket
    model.to(memory_format=channels_last) raises on non-4D buffers (e.g.
    the temporalfix grid); instead, per Conv2d, convert its 4D weight and
    add a forward-pre-hook coercing 4D NCHW input to channels_last.

    Returns the number of Conv2d modules treated."""
    import torch.nn as nn

    def _pre_hook(mod, args):
        x = args[0] if args else None
        if (isinstance(x, torch.Tensor) and x.dim() == 4
                and not x.is_contiguous(memory_format=torch.channels_last)):
            return (x.contiguous(memory_format=torch.channels_last),) + tuple(args[1:])
        return None

    count = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            try:
                m.to(memory_format=torch.channels_last)  # weight (4D) only
            except Exception:
                pass
            m.register_forward_pre_hook(_pre_hook)
            count += 1
    return count


def device_name(device: torch.device) -> str:
    try:
        if device.type == "cuda":
            return torch.cuda.get_device_name(device)
        if device.type == "xpu":
            return torch.xpu.get_device_name(  # type: ignore[attr-defined]
                getattr(device, "index", 0) or 0)
    except Exception:
        pass
    return device.type


__all__ = [
    "xpu_available", "pick_device", "sync", "empty_cache",
    "mem_get_info", "fp16_supported", "enable_channels_last_convs",
    "device_name",
]
