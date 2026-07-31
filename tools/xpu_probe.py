# tools/xpu_probe.py
"""CM-093 Phase 0: Intel Arc / XPU capability probe for ChitraMaya.

Answers, WITHOUT touching app code, the three questions that decide the
port plan:

  1. OP BATTERY -- do the ops ChitraMaya's models actually use run on xpu?
     conv2d, grid_sample (flow warp), torchvision.ops.deform_conv2d (the
     BasicVSR++ second-order alignment op -- the make-or-break one),
     pixel_shuffle (upsampler), bicubic interpolate. fp32 and fp16 each.
  2. DETECTION -- does ultralytics accept device="xpu" directly, or does
     YOLO need the raw-torch fallback? Plus ms/frame either way.
  3. STABILIZER -- a real ChitraMaya component end-to-end: the CM-078
     temporal stabilizer with the repo's bundled weights, on xpu.
     (Also smokes out any NVIDIA-only imports reachable from that module
     on a machine with no NVIDIA packages installed.)

Usage (from the repo root, in the xpu venv):
    python tools/xpu_probe.py
    python tools/xpu_probe.py --det-model models/<detector>.pt
    python tools/xpu_probe.py --device cpu     (sanity baseline)

All output is ASCII. Exit code 0 = probe ran (read the verdicts); the
summary table at the end is the deliverable.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback

import torch

RESULTS = []  # (section, name, status, note)


def record(section, name, status, note=""):
    RESULTS.append((section, name, status, note))
    print(f"  [{status}] {name}" + (f" -- {note}" if note else ""))


def sync(device):
    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def timed(fn, device, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(device)
    return (time.perf_counter() - t0) / iters * 1000.0  # ms


# ---------------------------------------------------------------------------
def section_env(device):
    print("=" * 64)
    print("SECTION 0: environment")
    print("=" * 64)
    print(f"  torch          : {torch.__version__}")
    print(f"  device request : {device}")
    if device.type == "xpu":
        avail = torch.xpu.is_available()
        print(f"  xpu available  : {avail}")
        if avail:
            print(f"  xpu device     : {torch.xpu.get_device_name(0)}")
            props = torch.xpu.get_device_properties(0)
            total = getattr(props, "total_memory", None)
            if total:
                print(f"  xpu memory     : {total / (1 << 30):.1f} GB")
        else:
            print("  FATAL: xpu requested but not available")
            sys.exit(1)
    try:
        import torchvision
        print(f"  torchvision    : {torchvision.__version__}")
    except Exception as e:
        print(f"  torchvision    : IMPORT FAILED ({e})")


# ---------------------------------------------------------------------------
def section_ops(device):
    print("=" * 64)
    print("SECTION 1: op battery (the ops ChitraMaya's models use)")
    print("=" * 64)
    for dtype in (torch.float32, torch.float16):
        dname = str(dtype).replace("torch.", "")
        print(f"-- dtype {dname} --")

        # conv2d (everything)
        try:
            x = torch.randn(1, 64, 128, 128, device=device, dtype=dtype)
            w = torch.randn(64, 64, 3, 3, device=device, dtype=dtype)
            ms = timed(lambda: torch.nn.functional.conv2d(x, w, padding=1), device)
            record("ops", f"conv2d {dname}", "PASS", f"{ms:.2f} ms")
        except Exception as e:
            record("ops", f"conv2d {dname}", "FAIL", repr(e)[:120])

        # grid_sample (BasicVSR++ flow warp; stabilizer confidence gating)
        try:
            x = torch.randn(1, 3, 256, 256, device=device, dtype=dtype)
            g = torch.rand(1, 256, 256, 2, device=device, dtype=dtype) * 2 - 1
            ms = timed(lambda: torch.nn.functional.grid_sample(
                x, g, align_corners=True), device)
            record("ops", f"grid_sample {dname}", "PASS", f"{ms:.2f} ms")
        except Exception as e:
            record("ops", f"grid_sample {dname}", "FAIL", repr(e)[:120])

        # deform_conv2d (BasicVSR++ second-order alignment -- MAKE OR BREAK)
        try:
            from torchvision.ops import deform_conv2d
            x = torch.randn(1, 64, 64, 64, device=device, dtype=dtype)
            off = torch.randn(1, 2 * 3 * 3, 64, 64, device=device, dtype=dtype)
            w = torch.randn(64, 64, 3, 3, device=device, dtype=dtype)
            ms = timed(lambda: deform_conv2d(x, off, w, padding=1), device)
            record("ops", f"deform_conv2d {dname}", "PASS", f"{ms:.2f} ms")
        except Exception as e:
            record("ops", f"deform_conv2d {dname}", "FAIL", repr(e)[:120])

        # pixel_shuffle (upsampler)
        try:
            x = torch.randn(1, 64 * 4, 64, 64, device=device, dtype=dtype)
            ms = timed(lambda: torch.nn.functional.pixel_shuffle(x, 2), device)
            record("ops", f"pixel_shuffle {dname}", "PASS", f"{ms:.2f} ms")
        except Exception as e:
            record("ops", f"pixel_shuffle {dname}", "FAIL", repr(e)[:120])

        # bicubic interpolate (InterpSecondary / resizes)
        try:
            x = torch.randn(1, 3, 256, 256, device=device, dtype=dtype)
            ms = timed(lambda: torch.nn.functional.interpolate(
                x, size=(512, 512), mode="bicubic", align_corners=False), device)
            record("ops", f"interpolate_bicubic {dname}", "PASS", f"{ms:.2f} ms")
        except Exception as e:
            record("ops", f"interpolate_bicubic {dname}", "FAIL", repr(e)[:120])


# ---------------------------------------------------------------------------
def section_detection(device, det_model):
    print("=" * 64)
    print("SECTION 2: YOLO detection")
    print("=" * 64)
    if not det_model:
        record("det", "ultralytics", "SKIP", "no --det-model given")
        return
    try:
        from ultralytics import YOLO
    except Exception as e:
        record("det", "ultralytics import", "FAIL", repr(e)[:120])
        return

    import numpy as np
    img = (np.random.rand(640, 640, 3) * 255).astype("uint8")

    # Path A: the polite way -- ultralytics handles the device itself.
    try:
        model = YOLO(det_model)
        model.predict(img, device=str(device), verbose=False)  # warmup + accept?
        t0 = time.perf_counter()
        n = 10
        for _ in range(n):
            model.predict(img, device=str(device), verbose=False)
        sync(device)
        ms = (time.perf_counter() - t0) / n * 1000.0
        record("det", f'ultralytics device="{device}"', "PASS",
               f"{ms:.1f} ms/frame (predict path)")
        return
    except Exception as e:
        record("det", f'ultralytics device="{device}"', "FAIL", repr(e)[:160])

    # Path B: raw torch fallback -- move the underlying module ourselves and
    # run the forward. Detections are meaningless on noise; this tests op
    # support + speed of the network itself.
    try:
        model = YOLO(det_model)
        net = model.model.to(device).eval()
        x = torch.rand(1, 3, 640, 640, device=device)
        with torch.inference_mode():
            ms = timed(lambda: net(x), device)
        record("det", "raw torch forward on device", "PASS",
               f"{ms:.1f} ms/frame (no pre/post-processing)")
    except Exception as e:
        record("det", "raw torch forward on device", "FAIL", repr(e)[:160])


# ---------------------------------------------------------------------------
def section_stabilizer(device):
    print("=" * 64)
    print("SECTION 3: temporal stabilizer (real component, bundled weights)")
    print("=" * 64)
    # Runnable straight from a repo checkout even without `pip install -e .`:
    # put the repo root (parent of tools/) on sys.path as a fallback.
    import os
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    try:
        from chitramaya.mosaic.restorer.temporal_stabilizer import (
            bundled_weights_dir, load_temporal_stabilizer)
    except Exception as e:
        record("stab", "import chitramaya stabilizer", "FAIL",
               "NVIDIA-only import reachable? " + repr(e)[:140])
        traceback.print_exc()
        return
    record("stab", "import chitramaya stabilizer", "PASS")
    try:
        stab, err = load_temporal_stabilizer(
            device=device, strength=2,
            search_dirs=["models", bundled_weights_dir()], clip_size=256)
        if stab is None:
            record("stab", "load weights (s2)", "FAIL", str(err))
            return
        record("stab", "load weights (s2)", "PASS")
        frames = [torch.randint(0, 255, (256, 256, 3), dtype=torch.uint8)
                  for _ in range(12)]
        sync(device)
        t0 = time.perf_counter()
        out = stab.stabilize_clip(frames)
        sync(device)
        ms = (time.perf_counter() - t0) / len(frames) * 1000.0
        ok = (len(out) == 12 and out[0].shape == (256, 256, 3)
              and out[0].dtype == torch.uint8)
        record("stab", "stabilize 12-frame clip", "PASS" if ok else "FAIL",
               f"{ms:.1f} ms/frame")
    except Exception as e:
        record("stab", "stabilize 12-frame clip", "FAIL", repr(e)[:160])
        traceback.print_exc()


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="xpu",
                    help="xpu (default) | cpu | cuda -- cpu is the sanity baseline")
    ap.add_argument("--det-model", default="",
                    help="path to a YOLO .pt to test detection (optional)")
    args = ap.parse_args()
    device = torch.device(args.device)

    section_env(device)
    section_ops(device)
    section_detection(device, args.det_model)
    section_stabilizer(device)

    print("=" * 64)
    print("SUMMARY")
    print("=" * 64)
    for sec, name, status, note in RESULTS:
        print(f"  {status:<4} [{sec}] {name}" + (f" -- {note}" if note else ""))
    fails = [r for r in RESULTS if r[2] == "FAIL"]
    print("-" * 64)
    if not fails:
        print("VERDICT: all probes passed -- the XPU port is device-string")
        print("plumbing, not op archaeology. Phase 1 (device abstraction +")
        print("ffmpeg decode + QSV encode) is GO.")
    else:
        print(f"VERDICT: {len(fails)} probe(s) failed -- see FAIL lines above.")
        print("deform_conv2d failures = BasicVSR++ blocker (restoration).")
        print("ultralytics-only failures = detection needs the raw-torch shim.")
        print("import failures = lazy-import work needed before app bring-up.")


if __name__ == "__main__":
    main()
