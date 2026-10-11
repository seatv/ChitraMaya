# chitramaya/vram_preflight.py -- CM-236 (T22): the VRAM preflight, before the run.
"""The working-set estimate the run-time preflight (CM-221) prints after the
models are loaded, made available BEFORE a run starts -- so Restore & Save
and Process Folder can show it in the app with a choice, instead of a
console line nobody reads at the start of an overnight batch (field
2026-09-26: a 4K60 title on a 7.9 GB card, Crop Cap off, died at its first
close-up exactly as the console had predicted 25,000 frames earlier).

Three parts, all explicit:

  estimate_working_set()  -- the pure arithmetic, shared with the pipeline's
                             _vram_preflight so the two can never disagree.
  the models-footprint ledger -- what the engines / Maxine / decoders /
                             encoder took on THIS card for THIS shape of run,
                             measured by every real run (after models minus
                             before models) and remembered per signature in
                             <app>/vram-ledger.json; a first-time signature
                             falls back to a per-edition table of the
                             footprints seen in the field, and says so.
  preflight()             -- free VRAM now, minus the footprint, against the
                             estimate; the levers as values the UI can apply
                             to the panel controls (Crop Cap, Async Encode)
                             in plain view.

Nothing here changes a setting by itself. ASCII-only stdout.
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

MB = 1024.0 * 1024.0
LEDGER_NAME = "vram-ledger.json"
LEVER_CAPS = (1024, 768, 512)


# -- the pure estimate --------------------------------------------------------
def estimate_working_set(
    w: int, h: int, *, crop_cap_px: int, hold_frames: int, batch_size: int,
    det_imgsz: int, async_on: bool, async_queue: int, secondary_on: bool,
    max_clip: int, free_mb: Optional[int], total_mb: Optional[int] = None,
) -> Dict[str, Any]:
    """CM-221's terms, in bytes, then MB (see Pipeline._vram_preflight for
    the derivation of each): scenes (2 open scenes x hold x min(cap, frame)^2
    x 4), detector letterbox transients, the decoder's in-flight frames, the
    paste intermediates, the Async Encode queue, 15 % allocator slack.
    ``free_mb`` None = unknown -> ``fits`` None (the caller says why).
    Returns the parts, the need, the fit, and the levers (text + the values
    a UI can apply)."""
    w = max(1, int(w))
    h = max(1, int(h))
    frame_px = w * h
    cap_cfg = int(crop_cap_px or 0)
    long_side = max(w, h)
    cap = min(cap_cfg if cap_cfg > 0 else max(1, long_side // 2), long_side)
    hold = max(1, int(hold_frames))
    scenes_b = 2 * hold * cap * cap * 4
    det_b = 2 * frame_px * 3 * 4 + int(batch_size) * 3 * int(det_imgsz) ** 2 * 4
    decode_b = 16 * frame_px * 3
    paste_b = 4 * frame_px * 3 * 4
    async_b = (int(async_queue) * frame_px * 4) if async_on else 0
    subtotal = scenes_b + det_b + decode_b + paste_b + async_b
    slack_b = int(0.15 * subtotal)
    need_mb = int((subtotal + slack_b) / MB)
    parts = (f"scenes {int(scenes_b / MB)}"
             + (f" (crops up to {cap} px)" if cap_cfg <= 0 else f" (Crop Cap {cap})")
             + f" + detector {int(det_b / MB)} + decode {int(decode_b / MB)} + paste {int(paste_b / MB)}"
             + (f" + async {int(async_b / MB)}" if async_b else "")
             + f" + slack {int(slack_b / MB)}")
    fits: Optional[bool] = None if free_mb is None else bool(need_mb <= int(free_mb))
    levers: List[Dict[str, Any]] = []
    if fits is False:
        free = int(free_mb)
        if cap_cfg <= 0 or cap_cfg > 512:
            for _try in LEVER_CAPS:
                if cap_cfg > 0 and _try >= cap_cfg:
                    continue
                _try_b = 2 * hold * min(_try, long_side) ** 2 * 4
                _try_need = int((subtotal - scenes_b + _try_b) * 1.15 / MB)
                if _try_need <= free:
                    levers.append({"kind": "crop_cap", "value": int(_try), "need_mb": _try_need,
                                   "text": (f"Crop Cap {_try} px (scene buffers {int(scenes_b / MB)} -> "
                                            f"{int(_try_b / MB)} MB, run ~{_try_need} MB; costs some fine "
                                            f"texture on crops above {_try} px)")})
                    break
        if async_on:
            levers.append({"kind": "async_off", "value": False, "need_mb": None,
                           "text": f"Async Encode off (-{int(async_b / MB)} MB)"})
        if int(batch_size) > 4:
            levers.append({"kind": "det_batch", "value": 4, "need_mb": None, "text": "Detection Batch 4"})
        if int(max_clip) > 90:
            levers.append({"kind": "max_clip", "value": 90, "need_mb": None,
                           "text": f"Max Clip 90 (scene buffers -{int(scenes_b * (1 - 90.0 / hold) / MB)} MB)"})
        if secondary_on:
            levers.append({"kind": "secondary_off", "value": "none", "need_mb": None, "text": "Secondary off"})
    return {
        "need_mb": need_mb, "free_mb": (None if free_mb is None else int(free_mb)),
        "total_mb": (None if total_mb is None else int(total_mb)), "fits": fits,
        "scenes_mb": int(scenes_b / MB), "crop_cap_px": cap_cfg, "scene_crop_px": int(cap),
        "detector_mb": int(det_b / MB), "decode_mb": int(decode_b / MB), "paste_mb": int(paste_b / MB),
        "async_mb": int(async_b / MB), "slack_mb": int(slack_b / MB), "parts": parts,
        "levers": levers, "width": w, "height": h,
    }


# -- the models-footprint ledger ---------------------------------------------
def _app_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path.cwd()


def ledger_path() -> Path:
    return _app_base_dir() / LEDGER_NAME


def _resolution_class(w: int, h: int) -> str:
    px = int(w) * int(h)
    if px >= 3840 * 2160 * 0.9:
        return "2160p"
    if px >= 2560 * 1440 * 0.9:
        return "1440p"
    if px >= 1920 * 1080 * 0.9:
        return "1080p"
    return "720p"


def signature(*, edition: str, gpu_name: str, det_imgsz: int, det_batch: int, det_fp16: bool,
              det_trt: bool, rest_trt: bool, secondary: str, temporal_fix: bool,
              output_upscale: str, upscale_scaler: str, width: int, height: int) -> str:
    """What the resident models' VRAM depends on -- coarse on purpose, so a
    signature repeats across titles of the same shape."""
    sec = str(secondary or "none")
    sec = "rtx" if sec.startswith("rtx") else ("custom" if sec.startswith("custom") else sec)
    ou = str(output_upscale or "none")
    us = str(upscale_scaler or "esrgan")
    us = "custom" if us.startswith("custom") else us
    return "|".join([
        str(edition), str(gpu_name).strip(), f"det:{int(det_imgsz)}/b{int(det_batch)}/"
        f"{'fp16' if det_fp16 else 'fp32'}/{'trt' if det_trt else 'torch'}",
        f"rest:{'trt' if rest_trt else 'torch'}", f"sec:{sec}", f"tf:{'on' if temporal_fix else 'off'}",
        f"ou:{ou}/{us if ou != 'none' else '-'}", f"res:{_resolution_class(width, height)}",
    ])


def read_ledger() -> Dict[str, Any]:
    try:
        p = ledger_path()
        if p.is_file():
            d = json.loads(p.read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else {}
    except Exception:
        pass
    return {}


def record_models_footprint(sig: str, footprint_mb: int, total_mb: int, gpu_name: str) -> None:
    """Called by the pipeline once every real run's models are resident:
    footprint = used after models - used before models (this process's
    engines, Maxine, decoders, encoder; other processes cancel out)."""
    try:
        if int(footprint_mb) <= 0:
            return
        d = read_ledger()
        d[str(sig)] = {"footprint_mb": int(footprint_mb), "total_mb": int(total_mb),
                       "gpu": str(gpu_name), "when": _dt.datetime.now().strftime("%Y-%m-%d %H:%M")}
        ledger_path().write_text(json.dumps(d, indent=1, ensure_ascii=True), encoding="utf-8")
    except Exception as e:
        print(f"[VRAM] ledger not written ({e})")


def table_footprint(*, edition: str, det_imgsz: int, det_trt: bool, rest_trt: bool, secondary: str,
                    temporal_fix: bool, output_upscale: str, upscale_scaler: str,
                    width: int, height: int) -> int:
    """Per-edition footprints seen in the field (MB), for a signature with no
    ledger entry yet. Conservative on purpose."""
    res = _resolution_class(width, height)
    big = res in ("2160p", "1440p")
    ed = str(edition).lower()
    if ed == "cuda":
        det = {640: 450, 800: 750, 1024: 1050}.get(int(det_imgsz), 1050) if det_trt else 900
        rest = 1250 if rest_trt else 1600
        sec = str(secondary or "none")
        secondary_mb = (700 if big else 260) if sec.startswith("rtx") else (0 if sec == "none" else 250)
        ou = str(output_upscale or "none")
        us = str(upscale_scaler or "esrgan")
        if ou == "none":
            upscale_mb = 0
        elif us == "rtx":
            upscale_mb = 3600 if big or ou in ("2160p", "4x") else 1500
        else:
            upscale_mb = 900 if (big or ou in ("2160p", "4x")) else 400
        return det + rest + secondary_mb + (120 if temporal_fix else 0) + 260 + 200 + upscale_mb
    # AMD / Intel: PyTorch everywhere; kernels and the encoder session.
    det = 900
    rest = 1700
    secondary_mb = 0 if str(secondary or "none") == "none" else 300
    ou = str(output_upscale or "none")
    upscale_mb = 0 if ou == "none" else (900 if (big or ou in ("2160p", "4x")) else 400)
    return det + rest + secondary_mb + (150 if temporal_fix else 0) + 300 + 200 + upscale_mb


# -- free VRAM now ------------------------------------------------------------
def free_vram_now(gpu_id: int = 0) -> Optional[Dict[str, int]]:
    """(free, total) MB from the driver for the run device, None when the
    edition offers no such query (CPU) -- the check then says so."""
    try:
        import torch
        if torch.cuda.is_available():
            f, t = torch.cuda.mem_get_info(int(gpu_id))
            return {"free_mb": int(f // MB), "total_mb": int(t // MB)}
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available() and hasattr(xpu, "mem_get_info"):
            f, t = xpu.mem_get_info(int(gpu_id))
            return {"free_mb": int(f // MB), "total_mb": int(t // MB)}
    except Exception:
        pass
    return None


def preflight(*, edition: str, gpu_name: str, gpu_id: int, width: int, height: int,
              crop_cap_px: int, max_clip: int, overlap: int, batch_size: int, det_imgsz: int,
              det_fp16: bool, det_trt: bool, rest_trt: bool, secondary: str, temporal_fix: bool,
              output_upscale: str, upscale_scaler: str, async_on: bool, async_queue: int,
              upscale_only: bool = False) -> Dict[str, Any]:
    """The pre-run check. Returns the estimate plus where the free number
    came from: ``footprint_source`` = 'measured' (this card, this run shape,
    from the ledger) | 'estimated' (the table) and ``skipped`` (reason) when
    the device offers no VRAM query."""
    now = free_vram_now(gpu_id)
    if now is None:
        est = estimate_working_set(width, height, crop_cap_px=crop_cap_px,
                                   hold_frames=int(max_clip) + 2 * int(overlap), batch_size=batch_size,
                                   det_imgsz=det_imgsz, async_on=async_on, async_queue=async_queue,
                                   secondary_on=(str(secondary or "none") != "none"), max_clip=max_clip,
                                   free_mb=None)
        est.update({"skipped": "no VRAM query on this device", "footprint_source": None,
                    "footprint_mb": None, "free_now_mb": None})
        return est
    if upscale_only:
        # No detector, no restorer, no scenes: the scaler alone. Not modelled
        # beyond the table's upscale term; report it as fitting with the note.
        est = estimate_working_set(width, height, crop_cap_px=512, hold_frames=1, batch_size=1,
                                   det_imgsz=64, async_on=async_on, async_queue=async_queue,
                                   secondary_on=False, max_clip=1, free_mb=now["free_mb"],
                                   total_mb=now["total_mb"])
        est.update({"skipped": "upscale-only mode is not modelled", "fits": None, "levers": [],
                    "footprint_source": None, "footprint_mb": None, "free_now_mb": now["free_mb"]})
        return est
    sig = signature(edition=edition, gpu_name=gpu_name, det_imgsz=det_imgsz, det_batch=batch_size,
                    det_fp16=det_fp16, det_trt=det_trt, rest_trt=rest_trt, secondary=secondary,
                    temporal_fix=temporal_fix, output_upscale=output_upscale,
                    upscale_scaler=upscale_scaler, width=width, height=height)
    entry = read_ledger().get(sig)
    if entry and int(entry.get("footprint_mb", 0)) > 0:
        footprint = int(entry["footprint_mb"])
        source = "measured"
    else:
        footprint = table_footprint(edition=edition, det_imgsz=det_imgsz, det_trt=det_trt, rest_trt=rest_trt,
                                    secondary=secondary, temporal_fix=temporal_fix,
                                    output_upscale=output_upscale, upscale_scaler=upscale_scaler,
                                    width=width, height=height)
        source = "estimated"
    free_after = max(0, int(now["free_mb"]) - footprint)
    est = estimate_working_set(width, height, crop_cap_px=crop_cap_px,
                               hold_frames=int(max_clip) + 2 * int(overlap), batch_size=batch_size,
                               det_imgsz=det_imgsz, async_on=async_on, async_queue=async_queue,
                               secondary_on=(str(secondary or "none") != "none"), max_clip=max_clip,
                               free_mb=free_after, total_mb=now["total_mb"])
    est.update({"skipped": None, "footprint_source": source, "footprint_mb": footprint,
                "free_now_mb": int(now["free_mb"]), "signature": sig,
                "ledger_when": (entry or {}).get("when")})
    return est
