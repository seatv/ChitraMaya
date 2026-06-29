# gRestorer/gRestorer/models/basicvsrpp/lada/deformconv.py
from __future__ import annotations

import math
import torch
import torch.nn as nn

try:
    from torchvision.ops import deform_conv2d as _deform_conv2d
except Exception:  # torchvision missing or broken
    _deform_conv2d = None


class ModulatedDeformConv2d(nn.Module):
    """
    Modulated Deformable Conv2d (DCNv2) using torchvision's deform_conv2d.

    This is the missing piece in the LADA-vendored code: the original file was a stub.
    We keep the same constructor signature expected by BasicVSR++.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deform_groups=1,
        bias=True,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deform_groups = deform_groups

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.init_weights()

    def init_weights(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        with torch.no_grad():
            self.weight.uniform_(-stdv, stdv)
            if self.bias is not None:
                self.bias.zero_()

    def forward(self, x: torch.Tensor, offset: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if _deform_conv2d is None:
            raise RuntimeError("torchvision.ops.deform_conv2d is unavailable; install/repair torchvision.")

        # Try on the current device first (CUDA/CPU/XPU if supported)
        try:
            return _deform_conv2d(
                x,
                offset,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                mask=mask,
            )
        except TypeError:
            # older torchvision signature might not accept mask= kw
            return _deform_conv2d(
                x,
                offset,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                mask,
            )
        except RuntimeError as e:
            # If the op isn't implemented for the device (common on some accelerators),
            # fall back to CPU for correctness (slow but functional).
            dev = x.device
            dtype = x.dtype
            x_cpu = x.detach().to("cpu", dtype=torch.float32)
            off_cpu = offset.detach().to("cpu", dtype=torch.float32)
            mask_cpu = mask.detach().to("cpu", dtype=torch.float32)
            w_cpu = self.weight.detach().to("cpu", dtype=torch.float32)
            b_cpu = self.bias.detach().to("cpu", dtype=torch.float32) if self.bias is not None else None

            y_cpu = _deform_conv2d(
                x_cpu, off_cpu, w_cpu, b_cpu, self.stride, self.padding, self.dilation, mask=mask_cpu
            )
            return y_cpu.to(dev, dtype=dtype)
