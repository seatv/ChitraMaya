# chitramaya/mosaic/restorer/custom_sr.py
"""CM-229: custom super-resolution models -- the user's own weights as the
Secondary (crop) scaler and as the whole-frame Output Upscale scaler.

Weights travel; content never does. We ship no model beyond the bundled
Real-ESRGAN compact; the user drops a ``.pth`` / ``.safetensors`` into
``models\\upscalers\\`` (OpenModelDB is the zoo; each file carries its own
licence) and it appears in the two dropdowns as ``file (RealPLKSR 2x)``.

Loading is spandrel's (chaiNNer's model loader, MIT): it reads the state
dict, identifies the architecture (ESRGAN/RRDB, Compact, RealPLKSR/PLKSR,
SPAN, ATD, DRCT, HAT, SwinIR, DAT, OmniSR, ... 42 families in 0.4.2) and
hands back a plain ``nn.Module`` plus what we need to drive it: ``scale``,
``supports_half``, ``size_requirements`` (a multiple-of constraint some
transformers have), ``tiling`` advice, ``purpose``. Only SR models with 3
input channels and an integer scale of 2..4 are offered; a file spandrel
cannot identify is skipped with one console line naming it.

Two consumers, one loader:
  CustomSecondaryRestorer -- the compositor's secondary interface (same as
      EsrganSecondaryRestorer / RtxSecondaryRestorer): a 256-px restored
      crop in, 256 * scale out. Crops are small, so no tiling; the input is
      reflect-padded to the model's multiple-of when it has one.
  CustomOutputUpscaler   -- the whole-frame stage (output_upscale.py's
      _OutputUpscalerBase): tiled with a halo, run the fewest passes that
      reach or pass the target (a 2x model does 720p -> 2160p in two passes),
      then the exact/area/bicubic ``_resample`` of output_upscale.py. On an
      out-of-memory the tile halves and the frame is retried (transformers
      want small tiles on 8 GB); a tile can never go below 128.

Precision: fp16 on cuda/xpu when the model says it supports half (ROCm torch
reports cuda), else fp32. Every model is evaluated once at construction on a
small tensor so a broken file fails at build time, never at frame 0; the
pipeline then falls back to the bundled scaler and says so.

The scan (``scan_custom_models``) is cached per (path, size, mtime) in
``.index.json`` inside the folder, so a 40-file zoo costs one pass, not one
pass per app start; identification loads the state dict on CPU only.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from chitramaya.mosaic.restorer.output_upscale import _OutputUpscalerBase, _resample

CUSTOM_PREFIX = "custom:"
CUSTOM_DIR = os.path.join("models", "upscalers")
_INDEX_FILE = ".index.json"
_EXTS = (".pth", ".pt", ".safetensors", ".ckpt")
_MAX_SCALE = 4
# architectures whose memory grows fast with the tile (attention); they
# start on a smaller tile than the CNNs
_TRANSFORMER_IDS = {"ATD", "HAT", "DRCT", "SwinIR", "Swin2SR", "DAT", "RGT", "GRL", "CRAFT", "OmniSR",
                    "FDAT", "IPT", "DITN", "SeemoRe", "Uformer", "RetinexFormer", "FFTformer"}


def is_custom(value: Optional[str]) -> bool:
    return str(value or "").startswith(CUSTOM_PREFIX)


def custom_file(value: str) -> str:
    """'custom:name.pth' -> 'name.pth' (the file name inside the folder)."""
    return str(value)[len(CUSTOM_PREFIX):] if is_custom(value) else str(value)


def custom_dirs(base_dir: Optional[str] = None) -> List[Path]:
    """The folders scanned, in order: <base>/models/upscalers, then ./models/upscalers."""
    out: List[Path] = []
    if base_dir:
        out.append(Path(base_dir) / "models" / "upscalers")
    out.append(Path(CUSTOM_DIR))
    seen = set()
    uniq = []
    for d in out:
        try:
            k = str(d.resolve())
        except Exception:
            k = str(d)
        if k not in seen:
            seen.add(k)
            uniq.append(d)
    return uniq


_IMPORT_LOCK = __import__("threading").Lock()


_NO_SPANDREL_SAID = False   # T20c: the "not installed" line prints once per process


def _spandrel():
    # One thread at a time through the heavy imports (spandrel pulls
    # torchvision); a concurrent import of torchvision from another request
    # thread produced a circular-import error in the field (09-21).
    try:
        with _IMPORT_LOCK:
            import torchvision  # noqa: F401
            import spandrel  # noqa: F401
            from spandrel import ModelLoader
        return ModelLoader
    except Exception as e:  # pragma: no cover - depends on the install
        raise RuntimeError(f"the spandrel package is not installed ({e}); custom scalers need it "
                           f"(pip install spandrel)") from e


def _describe(desc) -> Dict[str, Any]:
    sr = getattr(desc, "size_requirements", None)
    return {
        "arch": str(getattr(getattr(desc, "architecture", None), "id", "") or ""),
        "name": str(getattr(getattr(desc, "architecture", None), "name", "") or ""),
        "purpose": str(getattr(desc, "purpose", "") or ""),
        "scale": int(getattr(desc, "scale", 0) or 0),
        "in_ch": int(getattr(desc, "input_channels", 0) or 0),
        "out_ch": int(getattr(desc, "output_channels", 0) or 0),
        "half": bool(getattr(desc, "supports_half", False)),
        "multiple_of": int(getattr(sr, "multiple_of", 1) or 1) if sr is not None else 1,
        "min_size": int(getattr(sr, "minimum", 0) or 0) if sr is not None else 0,
        "tiling": str(getattr(getattr(desc, "tiling", None), "name", "SUPPORTED") or "SUPPORTED"),
        "tags": [str(t) for t in (getattr(desc, "tags", None) or [])],
    }


def _usable(info: Dict[str, Any]) -> Tuple[bool, str]:
    if info.get("purpose") != "SR":
        return False, f"purpose {info.get('purpose') or '?'} (only SR models scale a frame)"
    if info.get("in_ch") != 3 or info.get("out_ch") != 3:
        return False, f"{info.get('in_ch')}->{info.get('out_ch')} channels (needs RGB in, RGB out)"
    s = int(info.get("scale") or 0)
    if s < 2 or s > _MAX_SCALE:
        return False, f"scale {s}x (2x..{_MAX_SCALE}x are offered)"
    return True, ""


def identify_file(path: Path) -> Dict[str, Any]:
    """spandrel identification on CPU. Returns the info dict with 'ok' and
    'reason'; never raises for a bad file."""
    ModelLoader = _spandrel()
    try:
        desc = ModelLoader().load_from_file(str(path))
        info = _describe(desc)
        ok, reason = _usable(info)
        info["ok"], info["reason"] = ok, reason
        del desc
        return info
    except Exception as e:
        # T20b: "UnsupportedModelError: " with nothing after the colon sent the
        # 09-23 question; say what the file looks like instead of nothing.
        return {"ok": False, "reason": f"not identified ({type(e).__name__}: {str(e)[:120]}){_unidentified_hint(path)}",
                "arch": "", "name": "", "scale": 0}


def _peek_state_dict_keys(path: Path, limit: int = 4000) -> list:
    """Top-level parameter names of a checkpoint (BasicSR params/params_ema
    wrappers unwrapped), on CPU, weights only. [] when the file cannot be
    read as one."""
    try:
        p = str(path)
        if p.lower().endswith(".safetensors"):
            from safetensors import safe_open
            with safe_open(p, framework="pt", device="cpu") as f:
                return list(f.keys())[:limit]
        sd = torch.load(p, map_location="cpu", weights_only=True)
        if not isinstance(sd, dict):
            return []
        for k in ("params_ema", "params-ema", "params", "state_dict", "model_state_dict", "model", "net"):
            if isinstance(sd.get(k), dict):
                sd = sd[k]
                break
        return [str(k) for k in list(sd.keys())[:limit]]
    except Exception:
        return []


# T20b: known families spandrel does NOT load, told apart by their keys, and
# what to say about them. Checked in order; the first match wins.
_UNIDENTIFIED_HINTS = (
    # the official ATD repo revised the block (fc_td / ConvFFN_td, no sigma /
    # norm3 / logit_scale); spandrel + the community trainers carry the earlier
    # layout (09-23 finding)
    (lambda ks: any(".fc_td." in k for k in ks) and not any(k.endswith(".sigma") for k in ks),
     "looks like ATD from the official repo's REVISED architecture (has fc_td, no sigma); "
     "spandrel knows the earlier layout that community-trained ATD models use"),
    (lambda ks: any("optimizer" in k.lower() for k in ks) and len(ks) < 8,
     "looks like a training checkpoint / state file, not model weights"),
    (lambda ks: any(k.startswith(("model.diffusion_model", "first_stage_model", "cond_stage_model")) for k in ks),
     "a diffusion / Stable Diffusion checkpoint, not a super-resolution model"),
    (lambda ks: any(k.startswith("model.") and ".cv" in k for k in ks) or any(k.startswith("model.0.conv") for k in ks),
     "looks like a YOLO detection model -- it belongs in models\\, not models\\upscalers"),
)


def _unidentified_hint(path: Path) -> str:
    """' -- <hint>' for a file spandrel rejected, or the first few key names
    so the reader can tell what it is."""
    keys = _peek_state_dict_keys(path)
    if not keys:
        return ""
    for test, text in _UNIDENTIFIED_HINTS:
        try:
            if test(keys):
                return " -- " + text
        except Exception:
            pass
    shown = ", ".join(keys[:4])
    return f" -- {len(keys)} tensors; first keys: {shown}"


def _label(file: str, info: Dict[str, Any]) -> str:
    stem = Path(file).stem
    nm = info.get("name") or info.get("arch") or "?"
    return f"{stem} ({nm} {int(info.get('scale') or 0)}x)"


def scan_custom_models(base_dir: Optional[str] = None, *, force: bool = False,
                       log=print, dirs: Optional[List[Path]] = None) -> List[Dict[str, Any]]:
    """Every usable model in the upscalers folder(s): [{id, file, path, label,
    arch, name, scale, half, multiple_of, tiling, tags}], sorted by label.
    ``dirs`` names the folders outright (the server's models/upscalers);
    else ``custom_dirs(base_dir)``. Cached in .index.json per (size, mtime)."""
    out: List[Dict[str, Any]] = []
    have_spandrel = True
    try:
        _spandrel()
    except Exception as e:
        have_spandrel = False
        # T20c: said once per process -- every dropdown fill and every Rescan
        # re-scans, and the AMD start-up printed it ten times (09-23)
        global _NO_SPANDREL_SAID
        if not _NO_SPANDREL_SAID:
            _NO_SPANDREL_SAID = True
            log(f"[Scalers] {e}")
    for d in ([Path(x) for x in dirs] if dirs else custom_dirs(base_dir)):
        try:
            if not d.is_dir():
                continue
        except Exception:
            continue
        idx_path = d / _INDEX_FILE
        idx: Dict[str, Any] = {}
        if not force:
            try:
                idx = json.loads(idx_path.read_text(encoding="utf-8")) or {}
            except Exception:
                idx = {}
        changed = False
        for p in sorted(d.iterdir(), key=lambda x: x.name.lower()):
            try:
                if not p.is_file() or p.suffix.lower() not in _EXTS:
                    continue
                st = p.stat()
            except Exception:
                continue
            key = p.name
            ent = idx.get(key)
            if not (isinstance(ent, dict) and ent.get("size") == st.st_size and ent.get("mtime") == int(st.st_mtime)):
                if not have_spandrel:
                    continue
                t0 = time.perf_counter()
                info = identify_file(p)
                ent = {"size": st.st_size, "mtime": int(st.st_mtime), "info": info}
                idx[key] = ent
                changed = True
                if info.get("ok"):
                    log(f"[Scalers] {p.name}: {info.get('name') or info.get('arch')} {info.get('scale')}x"
                        f"{' fp16' if info.get('half') else ' fp32'}"
                        f"{' (multiple of %d)' % info['multiple_of'] if info.get('multiple_of', 1) > 1 else ''}"
                        f" -- identified in {time.perf_counter() - t0:.1f}s")
                else:
                    log(f"[Scalers] {p.name}: skipped -- {info.get('reason')}")
            info = ent.get("info") or {}
            if not info.get("ok"):
                continue
            out.append({"id": CUSTOM_PREFIX + p.name, "file": p.name, "path": str(p), "label": _label(p.name, info),
                        "arch": info.get("arch", ""), "name": info.get("name", ""), "scale": int(info.get("scale", 0)),
                        "half": bool(info.get("half")), "multiple_of": int(info.get("multiple_of", 1) or 1),
                        "tiling": info.get("tiling", "SUPPORTED"), "tags": list(info.get("tags") or []),
                        "size_mb": int(round(st.st_size / 1048576.0)),
                        # attention architectures: seconds per 256-px crop in the field (09-22); say so up front
                        "slow": str(info.get("arch", "")) in _TRANSFORMER_IDS})
        # drop index entries for files that are gone
        present = {p.name for p in d.iterdir() if p.is_file()} if d.is_dir() else set()
        for k in list(idx):
            if k not in present:
                idx.pop(k, None)
                changed = True
        if changed:
            try:
                idx_path.write_text(json.dumps(idx, indent=1), encoding="utf-8")
            except Exception:
                pass
    # one entry per file name (first folder wins), sorted by label
    seen = set()
    uniq = []
    for m in out:
        if m["file"].lower() in seen:
            continue
        seen.add(m["file"].lower())
        uniq.append(m)
    uniq.sort(key=lambda m: m["label"].lower())
    return uniq


def find_custom_model(value: str, base_dir: Optional[str] = None) -> Optional[Path]:
    """'custom:name.pth' -> the file, searching the upscalers folders."""
    name = custom_file(value)
    if not name or os.sep in name or "/" in name:
        return None
    for d in custom_dirs(base_dir):
        p = d / name
        try:
            if p.is_file():
                return p
        except Exception:
            pass
    return None


# -- T20b: upstream quirks fixed at load time ---------------------------------
# spandrel ports the reference implementations faithfully -- quirks included.
# SPAN's Conv3XC is a reparameterised conv (three convs + a skip folded into
# one 3x3 for inference) and the reference code REFUSES THE WEIGHTS ON EVERY
# FORWARD in eval mode: six clones, two weight-space conv2ds, pads, adds and a
# .data assignment, per layer, per crop. 20 layers x 584 crops on the 4060
# (09-23) = 2,596 tensor ops per forward instead of 256, +15 ms per crop, GPU
# at 38 %. Fusing once here gives a bit-identical output (verified) at the
# model's real cost. Keyed by spandrel architecture id; nothing else touched.

def _fuse_span_conv3xc(model) -> int:
    """Fold every Conv3XC once and replace its forward with the plain eval
    conv (+ its LeakyReLU). Returns the number of layers fused."""
    import types
    n = 0
    for mod in model.modules():
        if type(mod).__name__ != "Conv3XC" or not hasattr(mod, "update_params") or not hasattr(mod, "eval_conv"):
            continue
        try:
            mod.update_params()
        except Exception:
            continue

        def _fused_forward(self, x):
            out = self.eval_conv(x)
            if getattr(self, "has_relu", False):
                out = F.leaky_relu(out, negative_slope=0.05)
            return out

        mod.forward = types.MethodType(_fused_forward, mod)
        n += 1
    return n


_UPSTREAM_QUIRKS = {
    "SPAN": _fuse_span_conv3xc,
}


def _apply_upstream_quirks(arch: str, model) -> str:
    """Run the fix for this architecture, if any. Returns a short note for
    the console ('' when nothing applied)."""
    fn = _UPSTREAM_QUIRKS.get(str(arch or ""))
    if fn is None:
        return ""
    try:
        n = int(fn(model) or 0)
    except Exception:
        return ""
    return f"{n} Conv3XC layer(s) fused once at load (reference code refused them per frame)" if n else ""


class LoadedModel:
    """A spandrel model on a device with the facts the two consumers need."""

    def __init__(self, path: Path, device: torch.device, *, prefer_half: bool = True,
                 use_engines: bool = True) -> None:
        ModelLoader = _spandrel()
        self.path = Path(path)
        self.device = torch.device(device)
        desc = ModelLoader().load_from_file(str(self.path))
        self.info = _describe(desc)
        ok, reason = _usable(self.info)
        if not ok:
            raise RuntimeError(f"{self.path.name}: {reason}")
        self.scale = int(self.info["scale"])
        self.multiple_of = max(1, int(self.info.get("multiple_of", 1) or 1))
        self.arch = str(self.info.get("arch") or "")
        self.name = str(self.info.get("name") or self.arch or "custom")
        use_half = bool(prefer_half and self.info.get("half") and self.device.type in ("cuda", "xpu"))
        self.dtype = torch.float16 if use_half else torch.float32
        desc = desc.to(self.device)
        desc = desc.half() if use_half else desc.float()
        desc.eval()
        self._desc = desc
        self.quirks = _apply_upstream_quirks(self.arch, desc.model)
        if self.quirks:
            print(f"[Scalers] {self.path.name}: {self.quirks}")
        self.engines: Dict[int, Any] = {}      # T25: filled after the probe below
        # a broken file fails here, at build, not at frame 0
        probe = 64 if self.multiple_of <= 1 else max(64, self.multiple_of * ((64 + self.multiple_of - 1) // self.multiple_of))
        y = self(torch.zeros((1, 3, probe, probe), device=self.device, dtype=self.dtype))
        if tuple(y.shape[-2:]) != (probe * self.scale, probe * self.scale):
            raise RuntimeError(f"{self.path.name}: reported {self.scale}x but produced {tuple(y.shape[-2:])} "
                               f"from {probe}x{probe}")
        # T25 (CM-242): TensorRT engines beside the weights, one per tile size.
        # Used only for an input of exactly that shape; everything else takes
        # the torch path above. CUDA only; a load failure says why and is not fatal.
        if use_engines and self.device.type == "cuda":
            try:
                from chitramaya.mosaic.restorer.custom_sr_trt import load_engines
                self.engines = load_engines(self.path, self.device, fp16=(self.dtype == torch.float16),
                                            scale=self.scale)
            except Exception as _e:
                print(f"[Scalers] {self.path.name}: engine lookup skipped ({type(_e).__name__}: {str(_e)[:120]})")
                self.engines = {}
            for _tile in sorted(self.engines):
                print(f"[Scalers] {self.path.name}: {self.engines[_tile].describe()} ACTIVE for {_tile}x{_tile} inputs")

    def module(self) -> torch.nn.Module:
        """The fused nn.Module itself (export / compile)."""
        return self._desc.model

    def engine_tile(self) -> int:
        """The tile of the whole-frame engine (largest loaded), 0 when none."""
        return max((t for t in self.engines if t > 256), default=0) or (256 if 256 in self.engines else 0)

    def drop_engines(self, why: str = "") -> None:
        if not self.engines:
            return
        print(f"[Scalers] {self.path.name}: TensorRT engine(s) released{(' -- ' + why) if why else ''}; "
              f"continuing in torch")
        for e in self.engines.values():
            try:
                e.close()
            except Exception:
                pass
        self.engines = {}

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """1,3,h,w float in [0,1] -> 1,3,s*h,s*w float. Pads to the model's
        multiple-of (reflect) and crops the result back. no_grad, not
        inference_mode: the callers post-process the result in place.
        T25: an input of exactly an engine's tile goes through that engine."""
        h, w = int(x.shape[-2]), int(x.shape[-1])
        if self.engines and h == w and h in self.engines:
            eng = self.engines[h]
            if eng.accepts(x):
                return eng(x)
        m = self.multiple_of
        ph = (-h) % m
        pw = (-w) % m
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect" if (pw < w and ph < h) else "replicate")
        y = self._desc(x)
        if ph or pw:
            y = y[:, :, :h * self.scale, :w * self.scale]
        return y

    def describe(self) -> str:
        s = (f"{self.path.name} ({self.name} {self.scale}x, "
             f"{'fp16' if self.dtype == torch.float16 else 'fp32'}"
             f"{', multiple of %d' % self.multiple_of if self.multiple_of > 1 else ''}")
        if self.engines:
            s += "; TensorRT " + "/".join(f"{t}px" for t in sorted(self.engines))
        return s + ")"

    def backend(self) -> str:
        return "tensorrt" if self.engines else "torch"

    def is_transformer(self) -> bool:
        return self.arch in _TRANSFORMER_IDS

    def close(self) -> None:
        for e in list(self.engines.values()):
            try:
                e.close()
            except Exception:
                pass
        self.engines = {}
        self._desc = None


def color_match(out: torch.Tensor, ref: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """CM-232: pull ``out``'s per-channel mean / std to ``ref``'s (both
    1,3,H,W float in [0,1]; sizes may differ). A guard for custom scalers: a
    model trained to correct colour (a VHS restorer, 09-22) shifts the whole
    patch a few units, and the paste-back feather then shows as a band
    between shifted patch and untouched frame. Texture is untouched -- this
    is an affine per-channel remap -- only the tint and contrast come back
    to what the model was given. Built-ins (Maxine, the bundled compact) are
    colour-neutral and skip it."""
    if strength <= 0:
        return out
    o = out.to(torch.float32)
    r = ref.to(torch.float32)
    o_mean = o.mean(dim=(2, 3), keepdim=True)
    o_std = o.std(dim=(2, 3), keepdim=True).clamp_min(1e-4)
    r_mean = r.mean(dim=(2, 3), keepdim=True)
    r_std = r.std(dim=(2, 3), keepdim=True).clamp_min(1e-4)
    # the tint is the complaint; contrast follows only gently, so a model that
    # deliberately smooths or sharpens keeps most of its character
    gain = (r_std / o_std).clamp(0.8, 1.25)
    matched = (o - o_mean) * gain + r_mean
    if strength < 1.0:
        matched = o + (matched - o) * float(strength)
    return matched.clamp_(0.0, 1.0).to(out.dtype)


def _is_oom(e: BaseException) -> bool:
    s = str(e).lower()
    return ("out of memory" in s) or ("allocation" in s and "failed" in s) or isinstance(e, getattr(torch, "OutOfMemoryError", ()))


def _free_cache(device: torch.device) -> None:
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "xpu" and hasattr(torch, "xpu"):
            torch.xpu.empty_cache()
    except Exception:
        pass


class CustomSecondaryRestorer:
    """A custom model as the crop secondary (compositor interface: should_apply,
    min_apply_size, scale, stats, gate_margin_px, upscale_frame_bgr_u8)."""

    def __init__(self, *, device: torch.device, model_path: Path, input_size: int = 256,
                 color_match: bool = True) -> None:
        from chitramaya.mosaic.restorer.rtx_secondary import SecondaryStats
        if int(input_size) != 256:
            raise RuntimeError(f"custom secondary expects clip size 256, got {input_size}")
        self.device = torch.device(device)
        self.color_match = bool(color_match)
        self.model = LoadedModel(model_path, self.device)
        self.scale = int(self.model.scale)
        self.min_apply_size = int(input_size)
        self.out_size = self.min_apply_size * self.scale
        self.dtype = self.model.dtype
        self.weights_path = str(model_path)
        self.stats = SecondaryStats()
        self.gate_margin_px = 0

    def should_apply(self, orig_shape_hw: Tuple[int, int]) -> bool:
        return max(int(orig_shape_hw[0]), int(orig_shape_hw[1])) > self.min_apply_size

    @torch.inference_mode()
    def upscale_frame_bgr_u8(self, img_hwc_u8: torch.Tensor) -> torch.Tensor:
        src_device = img_hwc_u8.device
        x = img_hwc_u8.to(device=self.device, non_blocking=True)
        rgb = x[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).to(self.dtype).div_(255.0)
        try:
            out = self.model(rgb)
        except Exception as e:
            if not _is_oom(e):
                raise
            _free_cache(self.device)
            out = self.model(rgb)      # once more after a cache release; a second OOM propagates
        if out.is_inference():
            out = out.clone()
        if self.color_match:
            out = color_match(out, rgb)
        out_u8 = out.squeeze(0).clamp_(0, 1).mul_(255.0).round_().to(torch.uint8).permute(1, 2, 0)
        return out_u8[..., [2, 1, 0]].contiguous().to(device=src_device, non_blocking=True)

    def describe(self) -> str:
        return self.model.describe() + (", colour-matched to its input" if self.color_match else "")

    def close(self) -> None:
        try:
            self.model.close()
        except Exception:
            pass


def passes_for(scale: int, in_hw: Tuple[int, int], out_hw: Tuple[int, int], max_passes: int = 2) -> int:
    """The fewest passes of an s-x model whose output covers the target
    (else max_passes); the remainder is a resample."""
    s = max(2, int(scale))
    for n in range(1, max_passes + 1):
        f = s ** n
        if in_hw[0] * f >= out_hw[0] and in_hw[1] * f >= out_hw[1]:
            return n
    return max_passes


class CustomOutputUpscaler(_OutputUpscalerBase):
    """A custom model as the whole-frame Output Upscale scaler."""

    def __init__(self, *, device: torch.device, in_hw: Tuple[int, int], out_hw: Tuple[int, int],
                 model_path: Path, tile: int = 0, halo: int = 32, color_match: bool = True) -> None:
        super().__init__(device=device, in_hw=in_hw, out_hw=out_hw)
        self.color_match = bool(color_match)
        self.model = LoadedModel(model_path, self.device)
        self.work_dtype = self.model.dtype
        self.name = f"Custom {self.model.path.name} ({self.model.name} {self.model.scale}x)"
        self.passes = passes_for(self.model.scale, self.in_hw, self.out_hw)
        self.net_scale = self.model.scale ** self.passes
        self.mid_hw = (self.in_hw[0] * self.net_scale, self.in_hw[1] * self.net_scale)
        # tile: CNNs 640, attention models 320; halved on OOM down to 128
        self.tile = int(tile) if int(tile) > 0 else (320 if self.model.is_transformer() else 640)
        self.tile = max(128, self.tile)
        self.halo = max(0, int(halo))
        self._oom_halvings = 0
        # T25 (CM-242): with a whole-frame engine the window fed to the net is
        # FIXED at the engine's tile (halo included); the inner tile follows.
        self.engine_tile = int(self.model.engine_tile() or 0)
        if self.engine_tile and self.engine_tile - 2 * self.halo >= 128:
            self.tile = self.engine_tile - 2 * self.halo
        elif self.engine_tile:
            self.halo = max(0, (self.engine_tile - 128) // 2)
            self.tile = self.engine_tile - 2 * self.halo
        # T25c: the backend the frames actually went through, kept for the run
        # report. ``summary()`` is read after the encoder (and with it this
        # upscaler) is closed, when the engines are already released -- reading
        # ``model.engines`` there said "torch" for an engine run.
        self._backend_used = "tensorrt" if (self.engine_tile and self.model.engines) else "torch"
        self._engine_dropped_at = -1

    @torch.no_grad()
    def _net_once(self, rgb: torch.Tensor) -> torch.Tensor:
        """1,3,h,w -> 1,3,s*h,s*w, tiled with a reflect halo; the whole frame
        in one call when it fits the tile. With an engine every window is
        padded to exactly (engine_tile x engine_tile) so it takes the engine."""
        s = self.model.scale
        _, _, h, w = rgb.shape
        t, p = self.tile, self.halo
        pad_h = min(p, h - 1)
        pad_w = min(p, w - 1)
        # the frame border sees reflected picture instead of conv zeros,
        # tiled or not (same rule as the bundled scaler)
        src = F.pad(rgb, (pad_w, pad_w, pad_h, pad_h), mode="reflect") if (pad_h or pad_w) else rgb
        T = self.engine_tile if (self.engine_tile and self.model.engines) else 0
        if not T and h <= t and w <= t:
            o = self.model(src)
            return o[:, :, s * pad_h:s * pad_h + s * h, s * pad_w:s * pad_w + s * w]
        out = torch.empty((1, 3, s * h, s * w), device=rgb.device, dtype=rgb.dtype)
        for y0 in range(0, h, t):
            y1 = min(h, y0 + t)
            for x0 in range(0, w, t):
                x1 = min(w, x0 + t)
                win = src[:, :, y0:y1 + 2 * pad_h, x0:x1 + 2 * pad_w]
                if T:
                    wh, ww = int(win.shape[-2]), int(win.shape[-1])
                    eh, ew = T - wh, T - ww
                    if eh or ew:
                        # an edge window short of the engine tile: extend it
                        # (reflect when the picture is wide enough, else replicate)
                        mode = "reflect" if (eh < wh and ew < ww) else "replicate"
                        win = F.pad(win, (0, ew, 0, eh), mode=mode)
                o = self.model(win)
                out[:, :, s * y0:s * y1, s * x0:s * x1] = o[:, :, s * pad_h:s * pad_h + s * (y1 - y0),
                                                            s * pad_w:s * pad_w + s * (x1 - x0)]
        return out

    def _net(self, rgb: torch.Tensor) -> torch.Tensor:
        while True:
            try:
                x = rgb
                for _ in range(self.passes):
                    x = self._net_once(x)
                return x
            except Exception as e:
                if self.engine_tile and self.model.engines:
                    # T25: an engine failing mid-run (OOM or a TRT error) is
                    # released; the frame is redone on the torch path, which
                    # can halve its tile below.
                    self.model.drop_engines(f"{type(e).__name__}: {str(e)[:120]}")
                    _free_cache(self.device)
                    self.engine_tile = 0
                    self._backend_used = "torch (engine released mid-run)"
                    self._engine_dropped_at = int(self.frames)
                    self.tile = max(128, self.tile)
                    continue
                if not _is_oom(e) or self.tile <= 128:
                    raise
                _free_cache(self.device)
                self.tile = max(128, self.tile // 2)
                self._oom_halvings += 1
                print(f"[Upscale] {self.model.path.name}: out of memory at the current tile; "
                      f"retrying with {self.tile}-px tiles")

    def _run(self, rgb: torch.Tensor) -> torch.Tensor:
        y = self._net(rgb)
        if y.is_inference():
            # spandrel runs the module under inference_mode; the base class
            # post-processes in place, which an inference tensor refuses
            y = y.clone()
        if self.color_match:
            y = color_match(y, rgb)        # whole-frame statistics: the frame's own tint, kept
        return _resample(y, self.out_hw)

    def describe(self) -> str:
        s = super().describe()
        s += f" ({self.model.describe()}; {self.passes} pass{'es' if self.passes > 1 else ''} = {self.net_scale}x"
        if self.mid_hw != self.out_hw:
            s += f" to {self.mid_hw[1]}x{self.mid_hw[0]}, resampled"
        s += f"; tiles {self.tile} px, halo {self.halo}"
        if self.engine_tile and self.model.engines:
            s += f"; TensorRT {self.engine_tile}-px engine"
        return s + ("; colour-matched" if self.color_match else "") + ")"

    def summary(self) -> dict:
        d = super().summary()
        d["backend"] = self._backend_used
        d["tile"] = int(self.tile)
        if self._engine_dropped_at >= 0:
            d["engine_dropped_at_frame"] = int(self._engine_dropped_at)
        return d

    def close(self) -> None:
        try:
            self.model.close()
        except Exception:
            pass


__all__ = ["CUSTOM_PREFIX", "CUSTOM_DIR", "is_custom", "custom_file", "custom_dirs", "scan_custom_models",
           "find_custom_model", "identify_file", "LoadedModel", "CustomSecondaryRestorer",
           "CustomOutputUpscaler", "passes_for", "color_match"]
