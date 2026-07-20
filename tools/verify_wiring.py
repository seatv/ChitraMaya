# tools/verify_wiring.py
"""Round-trip wiring self-test for the mosaic config plumbing.

Runs the UI config through the SAME path a real run uses —
MosaicPipelineConfig -> MosaicPipeline._build_base_config -> Pipeline.__post_init__
— with distinctive sentinel values, and asserts each one survives to the
Pipeline attribute that actually drives behavior. This is the executable
companion to the import-time field-classification guardrail in
chitramaya/mosaic/pipeline.py: the guardrail catches an *unclassified* new
field; this catches a field that is classified but wired to the wrong key
(the Detection-Batch / Feather / Blend-Mask failure mode).

Build no models and touch no GPU — Pipeline.__post_init__ only parses config.

Usage:
    python -m tools.verify_wiring
Exit code 0 = all wired; nonzero = a knob is not reaching the pipeline.
"""
from __future__ import annotations

import sys


def main() -> int:
    from chitramaya.mosaic.pipeline import (
        MosaicPipelineConfig,
        MosaicPipeline,
        Pipeline,
    )

    cfg = MosaicPipelineConfig(
        detection_model="det.pt",
        restoration_model="rest.pth",
        detection_batch_size=7,      # A1 — was silently stuck at 8
        max_clip_size=45,
        detection_score=0.42,
        det_iou=0.55,
        roi_dilate=9,
        feather_radius=13,           # A2 — never reached compositor
        blendmask="facefusion",      # A3 — never reached compositor
        use_seg_masks=False,
        sbs_enabled=True,
        sbs_det_split=True,
        codec="h264",
        preset="P3",
        qp=21,
        async_encoder=True,          # opt-in flag must reach the pipeline
        use_trt=False,               # -> restoration.backend = pytorch
        mask_preview=False,
        det_imgsz=736,               # Batch 17 — runtime Image Size dial
        vr_projection="fisheye",     # Batch 19 (CM-045) — VR Projection mode
    )

    # Drive _build_base_config without constructing detector/restorer.
    mp = MosaicPipeline.__new__(MosaicPipeline)
    mp.config = cfg
    mp.gpu_id = 0
    base = mp._build_base_config("in.mp4", "out.mp4")

    host = Pipeline(base)  # __post_init__ parses config; builds no models

    checks = {
        "batch_size (A1 Detection Batch)": (host.batch_size, 7),
        "feather_radius (A2)": (host.feather_radius, 13),
        "rest_blendmask (A3)": (host.rest_blendmask, "facefusion"),
        "det_conf": (host.det_conf, 0.42),
        "det_iou": (host.det_iou, 0.55),
        "roi_dilate": (host.roi_dilate, 9),
        "rest_max_clip_length": (host.rest_max_clip_length, 45),
        "use_seg_masks": (host.use_seg_masks, False),
        "sbs_enabled": (host.sbs_enabled, True),
        "sbs_det_split": (host.sbs_det_split, True),
        "enc_codec": (host.enc_codec, "h264"),
        "enc_preset": (host.enc_preset, "P3"),
        "enc_qp": (host.enc_qp, 21),
        "async_encoder (opt-in)": (host.async_encoder, True),
        "rest_backend (use_trt=False)": (host.rest_backend, "pytorch"),
        "det_imgsz (Image Size dial)": (host.det_imgsz, 736),
        "vr_projection (CM-045 fisheye)": (host.vr_projection, "fisheye"),
    }

    failed = []
    for name, (got, want) in checks.items():
        ok = (got == want)
        print(f"[{'OK ' if ok else 'BAD'}] {name}: got={got!r} want={want!r}")
        if not ok:
            failed.append(name)

    if failed:
        print(f"\nFAIL: {len(failed)} knob(s) not wired: {failed}")
        return 1
    print("\nPASS: all sampled knobs reach the pipeline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
