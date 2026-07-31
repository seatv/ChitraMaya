# tools/xpu_xmx_probe.py
"""CM-093: does Arc's XMX fast path exist for OUR conv shapes, and what
unlocks it? Field data says fp16 == fp32 for BasicVSR++ chunks in eager
NCHW. This benchmarks a BasicVSR++-like residual conv stack (64ch, 3x3,
256px, 32-frame batch) across the candidate levers:

    fp32 NCHW          (current Arc default)
    fp16 NCHW          (what we tested -- expect: no gain)
    fp16 channels_last (oneDNN XMX kernel-selection lever)
    bf16 NCHW
    bf16 channels_last
    + optionally the same under torch.compile (--compile; takes minutes)

Usage:  python tools/xpu_xmx_probe.py [--device xpu] [--compile]
Read the table: if channels_last or bf16 rows are ~2x+ faster than fp32
NCHW, that's the pipeline change worth making. If everything is flat,
eager is the ceiling and torch.compile is the only remaining lever.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return x + self.c2(self.act(self.c1(x)))


def make_stack(depth=15, ch=64):
    return nn.Sequential(*[ResBlock(ch) for _ in range(depth)])


def sync(device):
    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def bench(model, x, device, iters=8, warmup=3):
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        sync(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        sync(device)
    return (time.perf_counter() - t0) / iters * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--compile", action="store_true",
                    help="also test torch.compile variants (slow first call)")
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--depth", type=int, default=15)
    ap.add_argument("--iters", type=int, default=8)
    args = ap.parse_args()
    device = torch.device(args.device)

    print(f"torch {torch.__version__}  device {device}")
    if device.type == "xpu":
        print(f"xpu: {torch.xpu.get_device_name(0)}")

    # 32 frames x 64ch x 256x256 features -- the BasicVSR++ working set shape
    variants = [
        ("fp32 NCHW",  torch.float32, False),
        ("fp16 NCHW",  torch.float16, False),
        ("fp16 chlast", torch.float16, True),
        ("bf16 NCHW",  torch.bfloat16, False),
        ("bf16 chlast", torch.bfloat16, True),
    ]
    results = []
    for name, dtype, chlast in variants:
        try:
            model = make_stack(depth=args.depth).to(device=device, dtype=dtype).eval()
            x = torch.randn(args.frames, 64, 256, 256, device=device, dtype=dtype)
            if chlast:
                model = model.to(memory_format=torch.channels_last)
                x = x.contiguous(memory_format=torch.channels_last)
            ms = bench(model, x, device, iters=args.iters)
            results.append((name, ms))
            print(f"  {name:<12} {ms:8.1f} ms / {args.frames}-frame stack pass")
        except Exception as e:
            results.append((name, None))
            print(f"  {name:<12} FAILED: {repr(e)[:100]}")
        finally:
            del model, x
            if device.type == "xpu":
                torch.xpu.empty_cache()

    if args.compile:
        print("-- torch.compile variants (first call = compilation, be patient) --")
        for name, dtype, chlast in [("fp16 chlast +compile", torch.float16, True),
                                    ("fp32 NCHW +compile", torch.float32, False)]:
            try:
                model = make_stack(depth=args.depth).to(device=device, dtype=dtype).eval()
                x = torch.randn(args.frames, 64, 256, 256, device=device, dtype=dtype)
                if chlast:
                    model = model.to(memory_format=torch.channels_last)
                    x = x.contiguous(memory_format=torch.channels_last)
                cmodel = torch.compile(model)
                t0 = time.perf_counter()
                with torch.inference_mode():
                    cmodel(x)
                sync(device)
                print(f"  {name}: first (compile) call {time.perf_counter() - t0:.0f}s")
                ms = bench(cmodel, x, device, iters=args.iters)
                results.append((name, ms))
                print(f"  {name:<22} {ms:8.1f} ms / stack pass (steady)")
            except Exception as e:
                print(f"  {name:<22} FAILED: {repr(e)[:100]}")
            finally:
                if device.type == "xpu":
                    torch.xpu.empty_cache()

    base = next((ms for n, ms in results if n == "fp32 NCHW" and ms), None)
    if base:
        print("-" * 46)
        print("speedup vs fp32 NCHW:")
        for n, ms in results:
            if ms:
                print(f"  {n:<22} {base / ms:5.2f}x")


if __name__ == "__main__":
    main()
