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
        crop_cap_px=768,             # CM-227: Crop Cap (T18)
        feather_radius=13,           # A2 — never reached compositor
        blendmask="facefusion",      # A3 — never reached compositor
        use_seg_masks=False,
        residual_scan=False,         # CM-205
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
        temporal_overlap=6,              # CM-213 -- Overlap dial (was inert until 1.80)
        crossfade=True,                  # CM-213
        blend_frames=-1,                 # CM-213 auto -> 6 // 3 = 2
        secondary_quality="ultra",       # CM-217 -- Scaler Quality
        output_upscale="2x",             # CM-219 -- Output Upscale
        hw_decode="off",                 # T20d (CM-234) -- Hardware decode
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
        "crop_cap_px (CM-227)": (host.crop_cap_px, 768),
        "rest_max_clip_length": (host.rest_max_clip_length, 45),
        "use_seg_masks": (host.use_seg_masks, False),
        "residual_scan": (host.residual_scan, False),
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
        "rest_temporal_overlap (CM-213 Overlap 6)": (host.rest_temporal_overlap, 6),
        "rest_blend_frames (CM-213 auto = 6 // 3)": (host.rest_blend_frames, 2),
        "rest_hold_frames (CM-213 45 + 2 * 6)": (host.rest_hold_frames, 57),
        "secondary_quality (CM-217 ultra)": (host.secondary_quality, "ultra"),
        "output_upscale (CM-219 2x)": (host.output_upscale, "2x"),
        "dec_hw_accel (T20d Hardware decode off)": (host.dec_hw_accel, "off"),
    }
    # CM-213: the clamp -- Overlap above Max Clip // 4 runs at the quarter.
    try:
        base.set("restoration", "temporal_overlap", value=16)
        checks["rest_temporal_overlap clamp (16 at MCL 45 -> 11)"] = (Pipeline(base).rest_temporal_overlap, 11)
        base.set("restoration", "crossfade", value=False)
        checks["rest_blend_frames (crossfade off -> 0)"] = (Pipeline(base).rest_blend_frames, 0)
    except Exception as e:  # pragma: no cover
        checks["CM-213 clamp"] = (f"error: {e}", "no error")

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
        "ctrlOutputUpscale": {"none", "2x", "4x", "1080p", "1440p", "2160p"},   # CM-219 / CM-228
        "ctrlUpscaleScaler": {"esrgan", "rtx"},               # CM-228
        "ctrlMosaicSecondaryQuality": {"low", "medium", "high", "ultra"},   # CM-217
        "ctrlMosaicCropCap": {"0", "512", "768", "1024"},     # CM-227 (T18)
        "ctrlHwDecode": {"auto", "off"},                      # T20d (CM-234)
    }
    # CM-202: the two new panel controls must exist in the markup and ride in
    # the preset schema (MOSAIC_CONFIG_CONTROLS) -- the 09-13 lesson again.
    _js = (_P(__file__).resolve().parents[1] / "chitramaya" / "static" / "js" / "mosaic.js").read_text(encoding="utf-8")
    for _cid in ("ctrlOutputSuffix", "ctrlRunReport",
                 "ctrlMosaicOverlap", "ctrlMosaicCrossfade",          # CM-213
                 "ctrlMosaicSecondaryQuality", "ctrlOutputUpscale",   # CM-217 / CM-219
                 "ctrlUpscaleScaler", "ctrlMosaicUpscale",            # CM-228 / T17
                 "ctrlMosaicCropCap",                                  # CM-227 (T18)
                 "ctrlHwDecode",                                       # T20d (CM-234)
                 "mmStdRun", "mmStdList", "mmUpList", "mmUpAdd", "mmUpRescan", "mmUpOpen", "mmUpRemove"):   # CM-232 (T20)
        # (CM-229 custom scaler options are added at runtime from /api/upscale-models)
        checks[f"control {_cid} in ui.html"] = (bool(re.search(r'id="%s"' % _cid, _html)), True)
        if _cid.startswith("ctrl"):   # modal buttons (mm*) are not preset keys
            checks[f"control {_cid} in MOSAIC_CONFIG_CONTROLS"] = (("'%s'" % _cid) in _js, True)
    # and the models.py -> pipeline hop for the suffix (server-side field)
    try:
        from chitramaya.models import MosaicConfig as _MC
        _mc = _MC.from_dict({"mosaic_output_suffix": "-clean", "mosaic_run_report": "off"})
        checks["mosaic_output_suffix (models.py)"] = (_mc.mosaic_output_suffix, "-clean")
        checks["mosaic_run_report -> run_report (to_pipeline_config)"] = (
            _mc.to_pipeline_config(encoder={}).run_report, "off")
        _mc2 = _MC.from_dict({"mosaic_temporal_overlap": 6, "mosaic_crossfade": False,
                              "mosaic_secondary_quality": "low", "mosaic_output_upscale": "4x"})
        _pc2 = _mc2.to_pipeline_config(encoder={})
        checks["mosaic_temporal_overlap -> temporal_overlap (CM-213)"] = (_pc2.temporal_overlap, 6)
        checks["mosaic_crossfade -> crossfade (CM-213)"] = (_pc2.crossfade, False)
        checks["mosaic_secondary_quality -> secondary_quality (CM-217)"] = (_pc2.secondary_quality, "low")
        checks["mosaic_output_upscale -> output_upscale (CM-219)"] = (_pc2.output_upscale, "4x")
        _mc3 = _MC.from_dict({"mosaic_output_upscale": "2160p", "mosaic_upscale_scaler": "rtx", "mosaic_upscale_only": True})
        _pc3 = _mc3.to_pipeline_config(encoder={})
        checks["mosaic_output_upscale target -> output_upscale (CM-228)"] = (_pc3.output_upscale, "2160p")
        checks["mosaic_upscale_scaler -> upscale_scaler (CM-228)"] = (_pc3.upscale_scaler, "rtx")
        checks["mosaic_upscale_only -> upscale_only (CM-228)"] = (_pc3.upscale_only, True)
        # CM-227 (T18): Crop Cap default OFF end to end; a set value reaches the tracker config
        _mc4 = _MC.from_dict({})
        checks["mosaic_crop_cap default 0 = Off (CM-227)"] = (_mc4.mosaic_crop_cap, 0)
        checks["crop_cap_px default 0 in pipeline config (CM-227)"] = (_mc4.to_pipeline_config(encoder={}).crop_cap_px, 0)
        _mc5 = _MC.from_dict({"mosaic_crop_cap": 512})
        checks["mosaic_crop_cap -> crop_cap_px (CM-227)"] = (_mc5.to_pipeline_config(encoder={}).crop_cap_px, 512)
        from chitramaya.mosaic.core.scene import SCENE_CROP_CAP_PX as _SCC, cap_crop_storage as _ccs, Scene as _Scene
        checks["SCENE_CROP_CAP_PX default 0 = native (CM-227)"] = (int(_SCC), 0)
        import torch as _t
        _img = _t.zeros((1530, 900, 3), dtype=_t.uint8); _img[::2, :, :] = 255   # 1-px lines
        _small, _ = _ccs(_img, None, 512)
        checks["cap 1530 -> 512 long side (CM-227)"] = (tuple(_small.shape[:2]), (512, 301))
        # C resample: bilinear without antialias keeps line contrast an area average would flatten
        _var = float(_small.to(_t.float32).var())
        checks["cap resample keeps texture contrast (C arm, CM-227)"] = (_var > 1000.0, True)
        _same, _ = _ccs(_img, None, 0)
        checks["cap 0 stores native (CM-227)"] = (_same is _img, True)
        from chitramaya.mosaic.core.scene_tracker import TrackerConfig as _TC, SceneTracker as _ST
        _trk = _ST(cfg=_TC(crop_cap_px=768), seg_mask_only=True)
        checks["tracker hands crop_cap_px to new scenes (CM-227)"] = (_trk._new_scene(0).crop_cap_px, 768)
        # CM-227 T26: the capped path's clip input keeps the native path's fine detail.
        # A 4K skin-like texture (shading + 1-2 px line network + grain) through a 3000-px
        # box: native = one bilinear 3000 -> 256; capped = 3000 -> 512 at capture, then
        # 512 -> 256 at clip build. Before T26 that second step was bilinear at 2x (a box
        # average) and the clip carried ~40 % of the native high-frequency energy; with
        # nearest it is within the native path's range. The uncapped path must be
        # bit-identical to a plain bilinear resize (unchanged behaviour).
        from chitramaya.mosaic.core.clip import Clip as _Clip
        from chitramaya.mosaic.utils import image_utils
        import cv2
        import torch.nn.functional as _F
        _g = _t.Generator().manual_seed(0)
        _H, _W = 1700, 3000
        _yy, _xx = _t.meshgrid(_t.linspace(0, 1, _H), _t.linspace(0, 1, _W), indexing="ij")
        _tex = (0.55 + 0.15 * _t.sin(6 * _xx) * _t.cos(4 * _yy)
                + (_t.sin(_xx * _W / 3.0) * _t.sin(_yy * _H / 2.7)).abs().pow(8) * 0.25
                + _t.randn(_H, _W, generator=_g) * 0.03).clamp(0, 1)
        _tex_u8 = (_tex * 255).round().to(_t.uint8).unsqueeze(-1).expand(_H, _W, 3).contiguous()
        _box = (0, 0, _H - 1, _W - 1)
        def _clip_frame(cap):
            _sc = _Scene(id=1, start_frame=0)
            _sc.crop_cap_px = cap
            _sc.add_frame(frame_num=0, roi_box=_box, crop_box=_box, crop_img=_tex_u8, crop_mask=None)
            _cl = _Clip.from_scene(_sc, clip_id=1, clip_size=256, pad_mode="reflect")
            _pt, _pb, _pl, _pr = _cl.pad_after_resizes[0]
            _fr = _cl.frames[0]
            return _fr[_pt: _fr.shape[0] - _pb, _pl: _fr.shape[1] - _pr]      # the unpadded resized crop
        _lap = _t.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=_t.float32)[None, None]
        def _hf(fr):
            _x = fr[..., 0].to(_t.float32)[None, None]
            return float(_F.conv2d(_x, _lap).std())
        _nat = _clip_frame(0)
        _cap = _clip_frame(512)
        _ratio = _hf(_cap) / max(_hf(_nat), 1e-6)
        checks["capped clip keeps native fine detail: HF ratio in [0.8, 1.4] (CM-227 T26)"] = (0.8 <= _ratio <= 1.4, True)
        _oh, _ow = int(_H * (256.0 / _H)), int(_W * (256.0 / _W))     # from_scene's per-axis scale
        _ref = image_utils.resize(_tex_u8, size=(_oh, _ow), interpolation=cv2.INTER_LINEAR)
        checks["uncapped clip == plain bilinear resize, bit-identical (CM-227 T26)"] = (
            bool(tuple(_nat.shape) == tuple(_ref.shape) and (_nat == _ref).all()), True)
        # T26 alignment: the capped chain samples the frame WHERE the native chain does. For
        # a square crop (both axes an integer ratio to the clip) the capped clip must equal
        # the native clip within the stored uint8's rounding (|diff| <= 1) at every cap
        # (the first cut's floor-nearest was 3.75-5.6 px off at 4K: the window grill moved).
        _sq = (_t.rand(3000, 3000, generator=_g) * 255).round().to(_t.uint8).unsqueeze(-1).expand(3000, 3000, 3).contiguous()
        _sqbox = (0, 0, 2999, 2999)
        def _sq_clip(cap):
            _sc = _Scene(id=2, start_frame=0)
            _sc.crop_cap_px = cap
            _sc.add_frame(frame_num=0, roi_box=_sqbox, crop_box=_sqbox, crop_img=_sq, crop_mask=None)
            _cl = _Clip.from_scene(_sc, clip_id=2, clip_size=256, pad_mode="reflect")
            _pt, _pb, _pl, _pr = _cl.pad_after_resizes[0]
            _fr = _cl.frames[0]
            return _fr[_pt: _fr.shape[0] - _pb, _pl: _fr.shape[1] - _pr]
        _sq_nat = _sq_clip(0)
        for _cap, _tol in ((512, 1), (768, 1), (1024, 1)):     # 1 = the stored uint8's own rounding
            _d = (_sq_clip(_cap).to(_t.int16) - _sq_nat.to(_t.int16)).abs().max().item()
            checks[f"capped clip (cap {_cap}) samples where native does: max|diff| <= {_tol} (CM-227 T26)"] = (_d <= _tol, True)
        # CM-245 (T28): Upscale-mode naming and folder listing. A restored output brought to 4K
        # must not get the restoration suffix twice, and the folder planner must list "-restored"
        # files as INPUT in Upscale mode while still excluding that mode's own "-upscaled" outputs.
        from chitramaya.mosaic import batch as _bm
        import tempfile as _tf
        checks["output stem: restore mode appends the suffix (CM-245)"] = (
            _bm.output_stem_for("Title", "-restored-18RC2"), "Title-restored-18RC2")
        checks["output stem: upscale of a restored output -> -upscaled, no double suffix (CM-245)"] = (
            _bm.output_stem_for("Title-restored-18RC2", "-restored-18RC2", upscale_only=True), "Title-restored-18RC2-upscaled")
        checks["output stem: upscale of a plain source keeps the panel suffix (CM-245)"] = (
            _bm.output_stem_for("Title", "-restored-18RC2", upscale_only=True), "Title-restored-18RC2")
        checks["output stem: restore mode never rewrites to -upscaled (CM-245)"] = (
            _bm.output_stem_for("Title-restored-18RC2", "-restored-18RC2"), "Title-restored-18RC2-restored-18RC2")
        with _tf.TemporaryDirectory() as _fd:
            for _n in ("A.mp4", "B-restored.mp4", "C-restored-18RC2.mp4", "D-restored-18RC2-upscaled.mp4", "E-upscaled.mp4"):
                (_P(_fd) / _n).write_bytes(b"\x00")
            _bm.register_output_suffix("-restored-18RC2")
            _restore_list = sorted(q.name for q in _bm.enumerate_videos(_fd))
            _upscale_list = sorted(q.name for q in _bm.enumerate_videos(_fd, own_suffixes=["-upscaled", "-restored-18RC2-upscaled"]))
        checks["folder listing, restore mode: own outputs excluded (CM-245)"] = (_restore_list, ["A.mp4"])
        checks["folder listing, upscale mode: restored outputs ARE inputs, -upscaled excluded (CM-245)"] = (
            _upscale_list, ["A.mp4", "B-restored.mp4", "C-restored-18RC2.mp4"])
        # the server captures the source once and refuses loads while an op runs
        import inspect as _insp
        from chitramaya import server as _srv
        _mf_src = _insp.getsource(_srv.SwapServer.mosaic_full)
        checks["mosaic_full passes the captured source to process_file, not self.video_path (CM-245)"] = (
            "src_video, output_path" in _mf_src and "self.video_path, output_path" not in _mf_src, True)
        checks["load-video refuses while the operation lock is held (CM-245)"] = (
            "_op_lock.locked()" in _insp.getsource(_srv.api_load_video), True)
        # CM-246: a picture header glued in front of a transport stream (the 10-09 STZY-016
        # field case: ffprobe said png_pipe, the audio probe found nothing, two runs finalized
        # video-only). The demuxer is forced from the extension and every source call gets it.
        import shutil as _shu, struct as _st, zlib as _zl, types as _ty, subprocess as _sp
        if _shu.which("ffmpeg") and _shu.which("ffprobe"):
            from chitramaya.video import source_format as _sf
            from chitramaya.video.finalize import source_has_audio as _sha
            from chitramaya.video.decoder import Decoder as _Dec
            with _tf.TemporaryDirectory() as _sd:
                _clean = _P(_sd) / "clean.ts"
                _sp.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=25",
                         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "0.6",
                         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-f", "mpegts", str(_clean)],
                        check=True, capture_output=True)
                def _chunk(t, d):
                    return _st.pack(">I", len(d)) + t + d + _st.pack(">I", _zl.crc32(t + d) & 0xffffffff)
                _png = (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", _st.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
                        + _chunk(b"IDAT", _zl.compress(b"\x00" * 5)) + _chunk(b"IEND", b""))
                _dis = _P(_sd) / "disguised.ts"
                _dis.write_bytes(_png + _clean.read_bytes())
                _auto = _sp.run(["ffprobe", "-v", "error", "-show_entries", "format=format_name", "-of", "csv=p=0",
                                 str(_dis)], capture_output=True, text=True).stdout.strip()
                checks["disguised TS: ffprobe auto-detection really says png_pipe (CM-246 premise)"] = (_auto, "png_pipe")
                checks["disguised TS: demuxer forced to mpegts from the extension (CM-246)"] = (
                    _sf.forced_demuxer(str(_dis), say=lambda m: None), "mpegts")
                checks["clean TS: nothing forced, input args empty (CM-246)"] = (
                    (_sf.forced_demuxer(str(_clean), say=lambda m: None), _sf.input_args(str(_clean), say=lambda m: None)), ("", []))
                checks["disguised TS: audio probe finds the audio stream (CM-246)"] = (_sha("ffprobe", str(_dis)), True)
                _dm = _ty.SimpleNamespace(input_path=str(_dis))
                _md = _Dec._ffprobe(_dm)
                checks["disguised TS: decoder probe sees the real picture and the mpegts container (CM-246)"] = (
                    (int(_md.width), int(_md.height), "mpegts" in _dm._container_format), (160, 120, True))
                checks["disguised TS: run report names the forced demuxer (CM-246)"] = (
                    (_sf.describe(str(_dis)), _sf.describe(str(_clean))), ("mpegts", None))
            # every ffmpeg/ffprobe call on the SOURCE carries the forced args
            import re as _re
            for _mod, _name in ((_srv, "server"),):
                _src_txt = _insp.getsource(_mod)
                _bare = [m.start() for m in _re.finditer(r'(?<!_src_args\(self\.video_path\), )"-i", self\.video_path', _src_txt)]
                checks[f"{_name}: every '-i self.video_path' carries the forced demuxer (CM-246)"] = (len(_bare), 0)
            from chitramaya.video import decoder as _decm, finalize as _finm, encoder as _encm
            checks["decoder/finalize/encoder import the forced-demuxer helper (CM-246)"] = (
                all("source_format import input_args" in _insp.getsource(m) for m in (_decm, _finm, _encm)), True)
        # CM-229 (T19): custom scalers -- ids keep their case through models.py, the pipeline
        # accepts them for both scalers, the loader identifies our own bundled weights, and the
        # custom output path is bit-identical to the bundled Real-ESRGAN path on the same file.
        _mc6 = _MC.from_dict({"mosaic_secondary": "custom:MyModel_x2.pth", "mosaic_upscale_scaler": "custom:Big.safetensors"})
        _pc6 = _mc6.to_pipeline_config(encoder={})
        checks["mosaic_secondary custom id keeps case (CM-229)"] = (_pc6.secondary_restoration, "custom:MyModel_x2.pth")
        checks["mosaic_upscale_scaler custom id keeps case (CM-229)"] = (_pc6.upscale_scaler, "custom:Big.safetensors")
        _mp6 = MosaicPipeline.__new__(MosaicPipeline)
        _mp6.config = _pc6
        _mp6.gpu_id = 0
        _base6 = _mp6._build_base_config("in.mp4", "out.mp4")
        _h6 = Pipeline(_base6)
        checks["custom secondary id accepted by the pipeline (CM-229)"] = (_h6.secondary_restoration, "custom:MyModel_x2.pth")
        checks["custom upscale scaler id accepted by the pipeline (CM-229)"] = (_h6.upscale_scaler, "custom:Big.safetensors")
        import tempfile as _tf, shutil as _sh
        from chitramaya.mosaic.restorer.custom_sr import (scan_custom_models as _scm, CustomOutputUpscaler as _COU,
                                                         CustomSecondaryRestorer as _CSR, passes_for as _pf, LoadedModel as _LM)
        from chitramaya.mosaic.restorer.output_upscale import OutputUpscaler as _OU
        _wdir = _P(__file__).resolve().parents[1] / "chitramaya" / "mosaic" / "restorer" / "weights" / "realesr-general-x4v3.pth"
        with _tf.TemporaryDirectory() as _td:
            _up = _P(_td) / "models" / "upscalers"
            _up.mkdir(parents=True)
            _sh.copy(_wdir, _up / "compact-test.pth")
            (_up / "junk.pth").write_bytes(b"not a model")
            _found = _scm(dirs=[_up], log=lambda m: None)
            checks["scan finds the usable model and skips junk (CM-229)"] = ([m["id"] for m in _found], ["custom:compact-test.pth"])
            checks["scan labels file (arch Nx) (CM-229)"] = (_found[0]["label"] if _found else "", "compact-test (RealESRGAN Compact 4x)")
            checks["scan caches an index (CM-229)"] = ((_up / ".index.json").is_file(), True)
            _x = _t.randint(0, 256, (72, 96, 4), dtype=_t.uint8, generator=_t.Generator().manual_seed(7))
            _c = _COU(device=_t.device("cpu"), in_hw=(72, 96), out_hw=(288, 384), model_path=_up / "compact-test.pth", tile=64,
                      color_match=False)   # parity check: the CM-232 guard off, the network alone
            _r = _OU(device=_t.device("cpu"), in_hw=(72, 96), out_hw=(288, 384), tile=64)
            checks["custom output path == bundled path, same weights (CM-229)"] = (bool(_t.equal(_c.upscale_bgra_u8(_x), _r.upscale_bgra_u8(_x))), True)
            # T25c: the run report names the backend the frames went through, read
            # after close() has released the engines (the 10-07 STZY-016 report said
            # "torch" for a 131 ms/frame engine run)
            checks["custom output upscaler: no engine -> report says torch (T25c)"] = (_c.summary()["backend"], "torch")
            _c2 = _COU(device=_t.device("cpu"), in_hw=(72, 96), out_hw=(288, 384), model_path=_up / "compact-test.pth", tile=64)
            _c2.model.engines = {704: None}; _c2.engine_tile = 704; _c2._backend_used = "tensorrt"   # as a loaded engine would set it
            _c2.close()
            checks["custom output upscaler: engine run still says tensorrt after close (T25c)"] = (
                (_c2.summary()["backend"], bool(_c2.model.engines)), ("tensorrt", False))
            _s = _CSR(device=_t.device("cpu"), model_path=_up / "compact-test.pth")
            checks["custom secondary: 256 -> 256*scale (CM-229)"] = (tuple(_s.upscale_frame_bgr_u8(_t.zeros((256, 256, 3), dtype=_t.uint8)).shape), (1024, 1024, 3))
            checks["custom secondary exposes the compositor interface (CM-229)"] = (
                all(hasattr(_s, a) for a in ("should_apply", "min_apply_size", "scale", "stats", "gate_margin_px", "upscale_frame_bgr_u8")), True)
            _lm = _LM(_up / "compact-test.pth", _t.device("cpu")); _lm.multiple_of = 16
            checks["multiple-of padding keeps the output size (CM-229)"] = (tuple(_lm(_t.zeros((1, 3, 37, 53))).shape[-2:]), (148, 212))
        checks["passes: 2x model 720p -> 2160p = 2 (CM-229)"] = (_pf(2, (720, 1280), (2160, 3840)), 2)
        checks["passes: 2x model 1080p -> 2160p = 1 (CM-229)"] = (_pf(2, (1080, 1920), (2160, 3840)), 1)
        checks["passes: 4x model 1080p -> 1440p = 1 (CM-229)"] = (_pf(4, (1080, 1920), (1440, 2560)), 1)
        # CM-232 (T20): colour-match guard -- tint comes back to the input's, identity when stats match,
        # contrast follows only gently; the standard set is four files from two repos
        from chitramaya.mosaic.restorer.custom_sr import color_match as _cm
        _ref = _t.rand(1, 3, 64, 64) * 0.5 + 0.25
        _shift = (_ref.clone() + 0.12).clamp(0, 1)
        _m = _cm(_shift, _ref)
        checks["color_match brings the mean back (CM-232)"] = (float((_m.mean((2, 3)) - _ref.mean((2, 3))).abs().max()) < 0.01, True)
        checks["color_match is identity on matching stats (CM-232)"] = (float((_cm(_ref, _ref) - _ref).abs().max()) < 1e-5, True)
        _flat = (_ref - _ref.mean((2, 3), keepdim=True)) * 0.2 + _ref.mean((2, 3), keepdim=True)   # heavily smoothed
        _g = float((_cm(_flat, _ref).std((2, 3)) / _flat.std((2, 3))).mean())
        checks["color_match contrast gain clamped at 1.25 (CM-232)"] = (1.2 < _g <= 1.26, True)
        # T20b: SPAN's Conv3XC is fused ONCE at load (the reference code refused the weights on
        # every forward: 2,596 tensor ops per crop instead of 256); output bit-identical
        try:
            from spandrel.architectures.SPAN.__arch.span import SPAN as _SPAN
            from chitramaya.mosaic.restorer.custom_sr import _apply_upstream_quirks as _auq
            _t.manual_seed(0)
            _sp = _SPAN(num_in_ch=3, num_out_ch=3, feature_channels=48, upscale=4).eval()
            _sx = _t.rand(1, 3, 32, 32)
            with _t.no_grad():
                _sy0 = _sp(_sx).clone()
            from torch.profiler import profile as _prof, ProfilerActivity as _PA
            def _nops():
                with _t.no_grad(), _prof(activities=[_PA.CPU]) as _pp:
                    _sp(_sx)
                return sum(1 for e in _pp.events() if e.name.startswith("aten::"))
            _n0 = _nops()
            _note = _auq("SPAN", _sp)
            _n1 = _nops()
            with _t.no_grad():
                _sy1 = _sp(_sx)
            checks["SPAN quirk: 20 Conv3XC fused once at load (T20b)"] = (_note.startswith("20 Conv3XC"), True)
            checks["SPAN quirk: per-forward op count drops >5x (T20b)"] = (_n0 > 5 * _n1, True)
            checks["SPAN quirk: output bit-identical after fusing (T20b)"] = (float((_sy0 - _sy1).abs().max()), 0.0)
            # T25 (CM-242): the fused net must export (torch.export is the TensorRT
            # front door) to a graph of native TRT layers only, bit-identical to eager.
            _ep = _t.export.export(_sp, (_sx,))
            _ops = {str(n.target) for n in _ep.graph.nodes if n.op == "call_function"}
            _allowed = {"aten.conv2d.default", "aten.silu_.default", "aten.silu.default", "aten.sigmoid.default",
                        "aten.sub.Tensor", "aten.add.Tensor", "aten.mul.Tensor", "aten.cat.default",
                        "aten.pixel_shuffle.default", "aten.leaky_relu.default", "aten.leaky_relu_.default",
                        # norm=True SPAN checkpoints: (x - mean.type_as(x)) -- a same-dtype cast, a TRT identity
                        "aten.type_as.default", "aten._to_copy.default", "aten.div.Tensor"}
            with _t.no_grad():
                _sy2 = _ep.module()(_sx)
            checks["custom SR engine: fused SPAN exports to TRT-native ops only (T25)"] = (sorted(_ops - _allowed), [])
            checks["custom SR engine: exported graph bit-identical to eager (T25)"] = (float((_sy1 - _sy2).abs().max()), 0.0)
            from chitramaya.mosaic.restorer.custom_sr_trt import engine_paths_for as _epf
            _e, _j = _epf(_P("models/upscalers/x.pth"), 704, True)
            checks["custom SR engine: file naming <stem>.trt_t704_fp16.<os>.engine/.json (T25)"] = (
                _e.name.startswith("x.trt_t704_fp16.") and _e.suffix == ".engine" and _j.suffix == ".json", True)
        except ImportError:
            pass
        # T20b: an unidentified file says what it looks like (the official-ATD case of 09-23)
        import tempfile as _tf, os as _os
        from pathlib import Path
        from chitramaya.mosaic.restorer.custom_sr import identify_file as _idf
        _td = _tf.mkdtemp()
        _fa = _os.path.join(_td, "003_ATD_SRx4_finetune.pth")
        _t.save({"params": {"conv_first.weight": _t.zeros(1), "layers.0.residual_group.layers.0.fc_td.weight": _t.zeros(1)}}, _fa)
        checks["unidentified file: official-ATD hint (T20b)"] = ("REVISED architecture (has fc_td" in _idf(Path(_fa))["reason"], True)
        _fb = _os.path.join(_td, "odd.pth")
        _t.save({"foo.weight": _t.zeros(1), "bar.bias": _t.zeros(1)}, _fb)
        checks["unidentified file: first keys listed (T20b)"] = ("first keys: foo.weight, bar.bias" in _idf(Path(_fb))["reason"], True)
        # T20c: the temporal stabilizer guards its own windows -- a bad window keeps the restored
        # frame; a bad fp16 window switches the model to fp32 and is redone (nothing lost)
        try:
            import os as _os2
            from chitramaya.mosaic.restorer.temporal_stabilizer import TemporalStabilizer as _TS, bundled_weights_dir as _bwd
            _w1 = _os2.path.join(_bwd(), "temporalfix_s1_v1.1.pth")
            _st = _TS(device=_t.device("cpu"), strength=1, model_path=_w1, clip_size=32)
            checks["TemporalFix: fp32 on CPU / never fp16 on HIP (T20c)"] = (_st.precision() in ("fp32", "fp16") and (not _st._is_hip or _st.precision() == "fp32"), True)
            _fr = [(_t.rand(32, 32, 3) * 255).to(_t.uint8) for _ in range(6)]
            _base = _st.stabilize_clip(_fr)
            _real = _st._model
            class _Flaky(_t.nn.Module):
                def __init__(self, inner, bad_at, once):
                    super().__init__(); self.inner = inner; self.bad_at = bad_at; self.once = once; self.k = 0; self.fired = False
                def forward(self, x):
                    y = self.inner(x); i = self.k; self.k += 1
                    if i == self.bad_at and not (self.once and self.fired):
                        self.fired = True
                        return _t.full_like(y, float("nan"))
                    return y
                def float(self): return self
            _st._model = _Flaky(_real, 3, False); _st.bad_windows = 0; _st.fp32_fallbacks = 0; _st._warned = True
            _o = _st.stabilize_clip(_fr)
            checks["TemporalFix guard: bad window keeps the restored frame (T20c)"] = ((_st.bad_windows, _t.equal(_o[3], _fr[3]), _t.equal(_o[2], _base[2])), (1, True, True))
            _st._model = _Flaky(_real, 2, True); _st._fp16 = True; _st._dtype = _t.float32; _st.bad_windows = 0; _st.fp32_fallbacks = 0; _st._warned = True
            _o2 = _st.stabilize_clip(_fr)
            checks["TemporalFix guard: fp16 failure -> fp32 redo, nothing lost (T20c)"] = ((_st.fp32_fallbacks, _st.bad_windows, _st._fp16, all(_t.equal(a, b) for a, b in zip(_o2, _base))), (1, 0, False, True))
        except FileNotFoundError:
            pass
        from chitramaya.server import _STANDARD_MODELS as _SM
        checks["standard set: four files, two repos (CM-232)"] = (
            (len(_SM), sorted({(m["owner"], m["name"]) for m in _SM}), sum(m["kind"] == "restoration" for m in _SM)),
            (4, [("ladaapp", "lada"), ("zelefans", "vrmr")], 1))
        _mp3 = MosaicPipeline.__new__(MosaicPipeline)
        _mp3.config = _pc3
        _mp3.gpu_id = 0
        _base3 = _mp3._build_base_config("in.mp4", "out.mp4")
        checks["upscale_only -> pipeline mode none (CM-228)"] = (_base3.get("mode"), "none")
        _h3 = Pipeline(_base3)
        checks["upscale_scaler reaches the pipeline (CM-228)"] = (_h3.upscale_scaler, "rtx")
        checks["output_upscale target reaches the pipeline (CM-228)"] = (_h3.output_upscale, "2160p")
        from chitramaya.mosaic.restorer.output_upscale import resolve_output_size as _ros, rtx_factor_for as _rff, output_size_check as _osc
        checks["resolve 720p -> 2160p"] = (_ros(1280, 720, "2160p")[0], (3840, 2160))
        checks["resolve 1080p -> 1440p keeps aspect"] = (_ros(1920, 1080, "1440p")[0], (2560, 1440))
        checks["resolve 4K -> 2160p = off"] = (_ros(3840, 2160, "2160p")[0], None)
        checks["resolve 2x"] = (_ros(1920, 1080, "2x")[0], (3840, 2160))
        checks["rtx factor 720p -> 2160p is native 3x"] = (round(_rff((720, 1280), (2160, 3840)), 3), 3.0)
        checks["rtx factor 1080p -> 1440p is native 4/3"] = (round(_rff((1080, 1920), (1440, 2560)), 3), 1.333)
        checks["size check 7680x4320 H264 refused"] = (_osc(7680, 4320, "h264")[0], False)
    except Exception as _e:
        checks["mosaic_output_suffix (models.py)"] = (f"raised {type(_e).__name__}: {_e}", "-clean")
    # T20d (CM-234): Hardware decode reaches the Decoder; the chroma guard repairs the
    # 09-24 frame shape (chroma complete to a row, zero after) from the previous frame,
    # leaves luma and clean frames alone, ignores black frames, greys a first frame.
    try:
        import os as _os4
        import torch as _t
        from chitramaya.models import MosaicConfig as _MC4
        _mc5 = _MC4.from_dict({})
        checks["mosaic_hw_decode default auto (T20d)"] = (_mc5.mosaic_hw_decode, "auto")
        checks["mosaic_hw_decode -> hw_decode (to_pipeline_config, T20d)"] = (
            _MC4.from_dict({"mosaic_hw_decode": "off"}).to_pipeline_config(encoder={}).hw_decode, "off")
        from chitramaya.mosaic.cli_config import create_parser as _cp4
        try:
            _ns4 = _cp4().parse_args(["--input", "in.mp4", "--output", "out.mp4", "--hw-decode", "off"])
            checks["--hw-decode CLI flag (T20d)"] = (getattr(_ns4, "hw_decode", None), "off")
        except SystemExit:
            checks["--hw-decode CLI flag (T20d)"] = ("parser rejected the flag", "off")
        from chitramaya.video.decoder import Decoder as _Dec
        _d = _Dec.__new__(_Dec); _d.hw_accel = "off"; _d.input_path = "no-such-file.mp4"; _d.ffmpeg_pix_fmt = "nv12"
        _d.hw_decode = "nvdec"; _d.hw_decode_reason = ""
        _saved = _os4.environ.pop("CM_HW_DECODE", None)
        try:
            checks["Decoder: panel Off -> CPU decode, hw_decode none (T20d)"] = ((_d._probe_hwaccel(), _d.hw_decode), ([], "none"))
            _os4.environ["CM_HW_DECODE"] = "off"; _d.hw_accel = "auto"
            checks["Decoder: CM_HW_DECODE=off overrides panel Auto (T20d)"] = ((_d._probe_hwaccel(), "environment" in _d.hw_decode_reason), ([], True))
        finally:
            if _saved is None:
                _os4.environ.pop("CM_HW_DECODE", None)
            else:
                _os4.environ["CM_HW_DECODE"] = _saved
        from chitramaya.video.chroma_guard import Nv12ChromaGuard as _CG
        from chitramaya.mosaic.pipeline_utils import nv12_to_rgb_hwc_u8 as _csc
        _h, _w = 144, 256
        def _nv(seed):
            _g = _t.Generator().manual_seed(seed)
            return _t.cat([_t.randint(16, 236, (_h, _w), generator=_g, dtype=_t.uint8),
                           _t.randint(16, 241, (_h // 2, _w), generator=_g, dtype=_t.uint8)], 0)
        _f0, _f1, _f2 = _nv(1), _nv(2), _nv(3)
        # the 09-24 green frame: luma complete, chroma zero from row 20 + a partly written row 18
        _bad = _f1.clone(); _bad[_h + 20:, :] = 0; _bad[_h + 18, _w // 2:] = 0
        _cg = _CG("test"); _a0, _a1, _a2 = _f0.clone(), _bad.clone(), _f2.clone()
        for _x in (_a0, _a1, _a2):
            _cg.apply(_x, _h)
        _hits = _cg.flush()
        checks["decode guard: flags the frame, counts the rows, uses the previous frame (T20d)"] = (_hits, [(1, 0, (_h // 2 - 20) + 1, True)])
        checks["decode guard: zero chroma rows take the previous frame's chroma (T20d)"] = (
            (bool(_t.equal(_a1[_h + 20:], _f0[_h + 20:])), bool(_t.equal(_a1[_h + 18], _f0[_h + 18]))), (True, True))
        checks["decode guard: finished rows, luma and clean frames untouched (T20d)"] = (
            (bool(_t.equal(_a1[_h:_h + 18], _bad[_h:_h + 18])), bool(_t.equal(_a1[:_h], _f1[:_h])),
             bool(_t.equal(_a0, _f0)), bool(_t.equal(_a2, _f2))), (True, True, True, True))
        _rgb_bad = _csc(_bad, width=_w, height=_h).float()[40:].mean((0, 1))
        _rgb_fix = _csc(_a1, width=_w, height=_h).float()[40:].mean((0, 1))
        checks["decode guard: the green cast is gone after repair (T20d)"] = (
            (bool(_rgb_bad[0] < 40 and _rgb_bad[2] < 10 and _rgb_bad[1] > 150), bool((_rgb_fix.max() - _rgb_fix.min()) < 8)), (True, True))   # bt601 at this size
        # T20e: the 17 s cut's frames -- the download stopped INSIDE the luma: black from row 60, no chroma at all
        _cut = _f1.clone(); _cut[60:, :] = 0
        _cg5 = _CG("luma"); _b0, _b1 = _f0.clone(), _cut.clone()
        _cg5.apply(_b0, _h); _cg5.apply(_b1, _h)
        _h5 = _cg5.flush()
        checks["decode guard: a luma cut is flagged with both counts (T20e)"] = (_h5, [(1, _h - 60, _h // 2, True)])
        checks["decode guard: cut luma + chroma take the previous frame's rows (T20e)"] = (
            (bool(_t.equal(_b1[60:_h], _f0[60:_h])), bool(_t.equal(_b1[_h:], _f0[_h:])), bool(_t.equal(_b1[:60], _f1[:60]))), (True, True, True))
        checks["decode guard: summary names both planes (T20e)"] = (
            ("luma row(s)" in _cg5.summary() and "chroma row(s)" in _cg5.summary() and "completed from the previous frame" in _cg5.summary()), True)
        _blk = _t.cat([_t.full((_h, _w), 16, dtype=_t.uint8), _t.full((_h // 2, _w), 128, dtype=_t.uint8)], 0)
        _cg2 = _CG("black"); _cg2.apply(_blk.clone(), _h)
        checks["decode guard: a black frame is not a hit (T20d)"] = (_cg2.flush(), [])
        _cg3 = _CG("first"); _z = _f1.clone(); _z[_h + 30:] = 0; _cg3.apply(_z, _h); _hz = _cg3.flush()
        checks["decode guard: a first frame with no donor is counted, left as decoded (T20d/f)"] = (
            (_hz, int(_z[_h + 30:].max()), "no previous frame" in _cg3.summary()), ([(0, 0, _h // 2 - 30, False)], 0, True))
        _cg6 = _CG("first-luma"); _zz = _f1.clone(); _zz[50:, :] = 0; _cg6.apply(_zz, _h); _cg6.flush()
        checks["decode guard: a first frame's cut luma is left as decoded (T20e/f)"] = ((int(_zz[50:_h].max()), _cg6.no_donor), (0, 1))
        # T20f: the 0xFF flavour -- magenta chroma (U=V=255) over the top, a white luma band (255) with
        # black dashes (0) below it; and the green/blue line PAIRS (one chroma row every 14 rows, 0 with some 255)
        _mg = _f1.clone()
        _mg[_h:_h + 30, :] = 255                                   # chroma 255 -> magenta
        _mg[80:100, :] = 255                                       # luma white band ...
        _mg[80:100, ::3] = 0                                       # ... with black dashes (a third of the bytes)
        _st = _f1.clone()
        for _r in range(7, _h // 2, 7):                            # every 7th chroma row = a luma pair every 14 rows
            _st[_h + _r, :] = 0; _st[_h + _r, ::5] = 255
        _cg7 = _CG("ff"); _c0, _c1, _c2 = _f0.clone(), _mg.clone(), _st.clone()
        for _x in (_c0, _c1, _c2):
            _cg7.apply(_x, _h)
        _h7 = _cg7.flush()
        checks["decode guard: 0xFF fill and dashed rows are hits (T20f)"] = ([(_p, _ny, _nuv) for (_p, _ny, _nuv, _fp) in _h7], [(1, 20, 30), (2, 0, len(range(7, _h // 2, 7)))])
        checks["decode guard: magenta / white rows take the previous frame's rows (T20f)"] = (
            (bool(_t.equal(_c1[_h:_h + 30], _f0[_h:_h + 30])), bool(_t.equal(_c1[80:100], _f0[80:100])),
             bool(_t.equal(_c1[:80], _f1[:80])), bool(_t.equal(_c1[100:_h], _f1[100:_h]))), (True, True, True, True))
        checks["decode guard: stripe pairs repaired row by row, good rows untouched (T20f)"] = (
            (bool(_t.equal(_c2[_h + 7], _c1[_h + 7])), bool(_t.equal(_c2[_h + 8], _f1[_h + 8])), bool(_t.equal(_c2[:_h], _f1[:_h]))), (True, True, True))
        # a full-range white flash frame following a white frame: the repair is a no-op (donor == row)
        _wf = _t.cat([_t.full((_h, _w), 255, dtype=_t.uint8), _t.full((_h // 2, _w), 128, dtype=_t.uint8)], 0)
        _cg8 = _CG("flash"); _w0, _w1 = _wf.clone(), _wf.clone(); _cg8.apply(_w0, _h); _cg8.apply(_w1, _h); _cg8.flush()
        checks["decode guard: full-range white on white is a no-op repair (T20f)"] = (bool(_t.equal(_w1, _wf)), True)
        checks["decode guard: summary says 0/255 (T20f)"] = ("at 0/255" in _cg7.summary(), True)
    except Exception as _e:
        checks["T20d/e hardware decode / decode guard checks"] = (f"raised {type(_e).__name__}: {_e}", "ok")
    # T15: CM-227 bounded crop storage, CM-224 per-frame letterbox parity
    try:
        import torch as _t
        from chitramaya.mosaic.core.scene import Scene as _Sc, SCENE_CROP_CAP_PX as _cap
        from chitramaya.mosaic.core.clip import Clip as _Cl
        _s = _Sc(id=1, start_frame=0)
        _s.crop_cap_px = 512     # T18: the cap is per scene (default 0 = native); this exercises it ON
        _s.add_frame(frame_num=0, roi_box=(0, 0, 1999, 2199), crop_box=(0, 0, 1999, 2199),
                     crop_img=_t.zeros((2000, 2200, 3), dtype=_t.uint8), crop_mask=_t.ones((2000, 2200), dtype=_t.uint8))
        checks["CM-227 stored crop long side <= cap"] = (max(_s.crops[0].shape[:2]) <= 512, True)
        checks["CM-227 native mask back at box size"] = (tuple(_s.last_mask_native().shape), (2000, 2200))
        _c = _Cl.from_scene(_s, 1, 256, "zero")
        checks["CM-227 clip crop_shapes stay native"] = (_c.crop_shapes[0], (2000, 2200))
        checks["CM-227 clip stored_shapes carry the capped size (T18)"] = (_c.stored_shapes[0], (465, 512))
        _s2 = _Sc(id=2, start_frame=0)
        _img = _t.randint(0, 256, (300, 400, 3), dtype=_t.uint8)
        _s2.add_frame(frame_num=0, roi_box=(0, 0, 299, 399), crop_box=(0, 0, 299, 399), crop_img=_img, crop_mask=None)
        checks["CM-227 small crop stored as is (no copy)"] = (_s2.crops[0].data_ptr() == _img.data_ptr(), True)
        from chitramaya.mosaic.detector.yolo import _letterbox_bchw as _lb, _compute_letterbox_meta as _lm
        _fr = [_t.randint(0, 256, (3, 216, 384), dtype=_t.uint8) for _ in range(4)]
        _meta = _lm((216, 384), (640, 640), stride=32)
        _old = _lb(_t.stack(_fr).to(_t.float32) / 255.0, _meta)
        _new = _t.cat([_lb(f.unsqueeze(0).to(_t.float32) / 255.0, _meta) for f in _fr], 0)
        checks["CM-224 per-frame letterbox == batched (bit-exact)"] = (bool(_t.equal(_old, _new)), True)
    except Exception as _e:
        checks["T15 CM-227/CM-224 checks"] = (f"raised {type(_e).__name__}: {_e}", "ok")
    # CM-235 (T21): the gap-fill output rate + the finalize commands.
    try:
        from fractions import Fraction as _F
        from chitramaya.video.encoder import gap_fill_output_rate as _gr
        from chitramaya.video.finalize import build_onepass_container_cmd as _bc, build_onepass_raw_cmd as _br
        from chitramaya.mosaic.pipeline_utils import PtsGapFiller as _PG
        # (b) source holes: the 09-25 title -- 194,773 frames stamped at 30.000
        # with a hole every 990 -> 197 fills; the rate must become 30 exactly.
        _f, _fs, _r = _gr(194773 / 6499, "194773/6499", 194970, 197, 194773, False)
        checks["CM-235 holes: rate lifts to the base cadence"] = (_fs, "30")
        checks["CM-235 holes: ratio"] = (_r, _F(194970, 194773))
        # (a) decoder drops (CM-120 field case): count restored, rate unchanged.
        _f, _fs, _r = _gr(29.97, "30000/1001", 127498, 150, 127498, False)
        checks["CM-235 drops: rate unchanged"] = ((_fs, _r), ("30000/1001", _F(1)))
        # discontinuity / nothing filled / out of band -> unchanged
        checks["CM-235 discontinuous -> unchanged"] = (_gr(29.97, "30000/1001", 1000, 5, 995, True)[1], "30000/1001")
        checks["CM-235 no fill -> unchanged"] = (_gr(29.97, "30000/1001", 1000, 0, 1000, False)[1], "30000/1001")
        _f, _fs, _r = _gr(29.97, "30000/1001", 1200, 200, 1000, False)
        checks["CM-235 out of band -> unchanged, ratio reported"] = ((_fs, _r), ("30000/1001", _F(6, 5)))
        # the filler itself on the 512/1024-tick pattern: 1 fill per hole
        _pg = _PG()
        _pts = []
        _t_ = 507
        for _i in range(3000):
            _pts.append(_t_)
            _t_ += 1024 if (_i % 990 == 989) else 512
        _n = 0
        for _i, _p in enumerate(_pts):
            _n += _pg.fills_before(_i, _p)
            _pg.remember(_p, None)
        checks["CM-235 filler: one duplicate per hole (3 holes in 3000)"] = ((_n, len(_pg.gaps), _pg.skipped_gaps), (3, 3, []))
        # finalize commands carry the rate / the scale
        _cmd = _br("ffmpeg", fps_str="30", input_fmt_args=["-f", "hevc"], raw_path="r.hevc", sidecar="a.mka",
                   audio_itsoffset=-0.033, color_args=[], tag_args=[], timescale_args=[], extra_args=[],
                   moov_bytes=0, out_path="o.mp4", duration_s=6499.0)
        checks["CM-235 raw finalize: -r carries the lifted rate"] = (_cmd[_cmd.index("-r") + 1], "30")
        _cmd = _bc("ffmpeg", video_path="v.venc", video_itsoffset=0.0, sidecar="a.mka", audio_itsoffset=0.0,
                   tag_args=[], timescale_args=[], extra_args=[], duration_s=6499.0, moov_bytes=0,
                   out_path="o.mp4", video_itsscale=(194773 / 6499) / 30.0)
        _i_ = _cmd.index("-itsscale")
        checks["CM-235 container finalize: -itsscale before the video input"] = (
            (_cmd[_i_ + 1][:8], _cmd[_i_ + 2], _cmd[_i_ + 3]), ("0.998989", "-i", "v.venc"))
        _cmd = _bc("ffmpeg", video_path="v.venc", video_itsoffset=0.0, sidecar=None, audio_itsoffset=0.0,
                   tag_args=[], timescale_args=[], extra_args=[], duration_s=10.0, moov_bytes=0, out_path="o.mp4")
        checks["CM-235 container finalize: no scale when ratio 1"] = ("-itsscale" in _cmd, False)
        # both encoder classes expose the hop
        from chitramaya.video.encoder import Encoder as _E, FfmpegEncoder as _FE
        checks["CM-235 encoders expose note_gap_fill"] = (
            (hasattr(_E, "note_gap_fill"), hasattr(_FE, "note_gap_fill"), hasattr(_E, "_output_rate")), (True, True, True))
    except Exception as _e:
        checks["CM-235 checks"] = (f"raised {type(_e).__name__}: {_e}", "ok")
    # CM-236 (T22): the pre-run VRAM preflight -- the shared estimate, the ledger, the endpoint.
    try:
        import os as _os, tempfile as _tf, json as _json
        from chitramaya import vram_preflight as _vp
        # the 09-26 field case: 4K60, Crop Cap off, MCL 180 + 2x8 overlap, batch 8, imgsz 800, async on, secondary on
        e = _vp.estimate_working_set(3840, 2160, crop_cap_px=0, hold_frames=196, batch_size=8, det_imgsz=800,
                                     async_on=True, async_queue=16, secondary_on=True, max_clip=180, free_mb=4291, total_mb=8123)
        checks["CM-236 estimate: the 09-26 numbers (need 8080, scenes 5512, async 506)"] = (
            (e["need_mb"], e["scenes_mb"], e["async_mb"], e["fits"]), (8080, 5512, 506, False))
        _kinds = [lv["kind"] for lv in e["levers"]]
        checks["CM-236 estimate: levers in the console's order"] = (_kinds, ["crop_cap", "async_off", "det_batch", "max_clip", "secondary_off"])
        checks["CM-236 estimate: Crop Cap lever = 1024, run ~3544"] = ((e["levers"][0]["value"], e["levers"][0]["need_mb"]), (1024, 3544))
        e2 = _vp.estimate_working_set(1920, 1080, crop_cap_px=0, hold_frames=196, batch_size=8, det_imgsz=640,
                                      async_on=False, async_queue=16, secondary_on=True, max_clip=180, free_mb=4000)
        checks["CM-236 estimate: 1080p fits, no levers"] = ((e2["fits"], e2["levers"]), (True, []))
        checks["CM-236 estimate: unknown free -> fits None"] = (_vp.estimate_working_set(1920, 1080, crop_cap_px=0, hold_frames=10, batch_size=8, det_imgsz=640, async_on=False, async_queue=16, secondary_on=False, max_clip=10, free_mb=None)["fits"], None)
        # the ledger round-trips, keyed by the signature; the table stands in before the first run
        _sig = _vp.signature(edition="cuda", gpu_name="NVIDIA GeForce RTX 5060 Ti", det_imgsz=800, det_batch=8, det_fp16=True,
                             det_trt=True, rest_trt=True, secondary="rtx-4x", temporal_fix=True, output_upscale="none",
                             upscale_scaler="esrgan", width=3840, height=2160)
        checks["CM-236 signature: coarse and stable"] = (_sig, "cuda|NVIDIA GeForce RTX 5060 Ti|det:800/b8/fp16/trt|rest:trt|sec:rtx|tf:on|ou:none/-|res:2160p")
        _cwd = _os.getcwd()
        with _tf.TemporaryDirectory() as _td:
            _os.chdir(_td)
            try:
                checks["CM-236 ledger: empty before any run"] = (_vp.read_ledger(), {})
                _vp.record_models_footprint(_sig, 2722, 8123, "NVIDIA GeForce RTX 5060 Ti")
                checks["CM-236 ledger: footprint remembered"] = (_vp.read_ledger()[_sig]["footprint_mb"], 2722)
                _t = _vp.table_footprint(edition="cuda", det_imgsz=800, det_trt=True, rest_trt=True, secondary="rtx-4x",
                                         temporal_fix=True, output_upscale="none", upscale_scaler="esrgan", width=3840, height=2160)
                checks["CM-236 table: 4K NVIDIA full stack in the field's range (2.5-3.5 GB)"] = (2500 <= _t <= 3500, True)
                # preflight(): free from the driver (None here on CPU -> skipped, honest)
                _pf = _vp.preflight(edition="cuda", gpu_name="x", gpu_id=0, width=3840, height=2160, crop_cap_px=0, max_clip=180,
                                    overlap=8, batch_size=8, det_imgsz=800, det_fp16=True, det_trt=True, rest_trt=True,
                                    secondary="rtx-4x", temporal_fix=True, output_upscale="none", upscale_scaler="esrgan",
                                    async_on=True, async_queue=16)
                import torch as _t_
                if not _t_.cuda.is_available():
                    checks["CM-236 preflight: no VRAM query -> skipped with a reason, fits None"] = ((_pf["fits"], bool(_pf["skipped"])), (None, True))
            finally:
                _os.chdir(_cwd)
        # the endpoint exists and answers without a video (skipped, not an error)
        from chitramaya.server import app as _app
        _c = _app.test_client()
        _r = _c.post("/api/vram-preflight", json={"params": {"mosaic": {}, "encoder": {}}})
        checks["CM-236 endpoint: answers 200 and says why it skipped without a video"] = (
            (_r.status_code, bool((_r.get_json() or {}).get("skipped"))), (200, True))
    except Exception as _e:
        checks["CM-236 checks"] = (f"raised {type(_e).__name__}: {_e}", "ok")
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
