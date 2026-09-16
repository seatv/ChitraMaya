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
        secondary_restoration="rtx-2x",
        secondary_denoise="high",        # Batch 74 (CM-146) -- the hop Batch 70 missed  # Batch 20 (CM-077) — secondary upscale
        temporal_stability=2,            # Batch 26 (CM-078) — temporal stabilizer
        store_backend="redecode",        # CM-191 -- the value the dropdown lacked on 09-13
        run_report="temp",               # CM-202 -- the panel's Run Files switch
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
        "secondary_restoration (CM-077 rtx-2x)": (host.secondary_restoration, "rtx-2x"),
        "secondary_denoise (CM-146 high)": (host.secondary_denoise, "high"),
        "temporal_stability (CM-078 strength 2)": (host.temporal_stability, 2),
        "store_backend (CM-191 redecode)": (host.store_backend, "redecode"),
        "redecode_patches default (CM-196 host)": (host.redecode_patches, "host"),
        "run_report_mode (CM-202 panel 'temp' wins)": (host.run_report_mode, "temp"),
    }

    # CM-196: the pending-patch home has NO panel control on purpose (a
    # hand-edit key + CLI flag for A/B); check both channels reach the
    # pipeline so the report's effective.redecode_patches can be trusted.
    try:
        base.set("redecode_patches", value="device")
        checks["redecode_patches config key (CM-196 device)"] = (Pipeline(base).redecode_patches, "device")
    except Exception as _e:
        checks["redecode_patches config key (CM-196 device)"] = (f"raised {type(_e).__name__}: {_e}", "device")
    checks["vram_cache_release default (CM-196 T10f off)"] = (host.vram_cache_release, False)
    try:
        base.set("vram_cache_release", value=True)
        checks["vram_cache_release config key (CM-196 T10f)"] = (Pipeline(base).vram_cache_release, True)
    except Exception as _e:
        checks["vram_cache_release config key (CM-196 T10f)"] = (f"raised {type(_e).__name__}: {_e}", True)
    try:
        from chitramaya.mosaic.cli_config import create_parser
        _ns2 = create_parser().parse_args(["--input", "in.mp4", "--output", "out.mp4", "--vram-cache-release"])
        checks["--vram-cache-release CLI flag (CM-196 T10f)"] = (getattr(_ns2, "vram_cache_release", None), True)
    except SystemExit:
        checks["--vram-cache-release CLI flag (CM-196 T10f)"] = ("parser rejected the flag", True)
    try:
        from chitramaya.mosaic.cli_config import create_parser
        _ns = create_parser().parse_args(["--input", "in.mp4", "--output", "out.mp4",
                                          "--redecode-patches", "device"])
        checks["--redecode-patches CLI flag (CM-196)"] = (getattr(_ns, "redecode_patches", None), "device")
    except SystemExit:
        checks["--redecode-patches CLI flag (CM-196)"] = ("parser rejected the flag", "device")

    # CM-191 lesson (2026-09-14): a value the pipeline accepts but the UI
    # dropdown does not offer makes a preset load BLANK and save as "".
    # Every enumerated control must offer exactly the values the pipeline
    # accepts, and nothing else. Parsed from ui.html, compared to the
    # pipeline's own validation lists.
    import re
    from pathlib import Path as _P
    _html = (_P(__file__).resolve().parents[1] / "chitramaya" / "templates" / "ui.html").read_text(encoding="utf-8")

    def _options(select_id: str) -> set:
        m = re.search(r'<select[^>]*id="%s"[^>]*>(.*?)</select>' % re.escape(select_id), _html, re.S)
        if not m:
            return set()
        return set(re.findall(r'<option[^>]*value="([^"]*)"', m.group(1)))

    enum_checks = {
        "ctrlStoreBackend": {"auto", "redecode", "device", "host"},
        "ctrlMosaicBlendMask": {"none", "facefusion"},
        "ctrlRunReport": {"beside", "temp", "off"},          # CM-202
    }
    # CM-202: the two new panel controls must exist in the markup and ride in
    # the preset schema (MOSAIC_CONFIG_CONTROLS) -- the 09-13 lesson again.
    _js = (_P(__file__).resolve().parents[1] / "chitramaya" / "static" / "js" / "mosaic.js").read_text(encoding="utf-8")
    for _cid in ("ctrlOutputSuffix", "ctrlRunReport"):
        checks[f"control {_cid} in ui.html"] = (bool(re.search(r'id="%s"' % _cid, _html)), True)
        checks[f"control {_cid} in MOSAIC_CONFIG_CONTROLS"] = (("'%s'" % _cid) in _js, True)
    # and the models.py -> pipeline hop for the suffix (server-side field)
    try:
        from chitramaya.models import MosaicConfig as _MC
        _mc = _MC.from_dict({"mosaic_output_suffix": "-clean", "mosaic_run_report": "off"})
        checks["mosaic_output_suffix (models.py)"] = (_mc.mosaic_output_suffix, "-clean")
        checks["mosaic_run_report -> run_report (to_pipeline_config)"] = (
            _mc.to_pipeline_config(encoder={}).run_report, "off")
    except Exception as _e:
        checks["mosaic_output_suffix (models.py)"] = (f"raised {type(_e).__name__}: {_e}", "-clean")
    for sel_id, accepted in enum_checks.items():
        offered = _options(sel_id)
        ok = bool(offered) and offered == accepted
        print(f"[{'OK ' if ok else 'BAD'}] dropdown {sel_id}: offers={sorted(offered)} pipeline accepts={sorted(accepted)}")
        if not ok:
            failed_enum = checks.setdefault("_enum_failures", (None, None))
            checks[f"dropdown {sel_id}"] = (sorted(offered), sorted(accepted))
    checks.pop("_enum_failures", None)

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
