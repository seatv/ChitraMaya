# chitramaya/mosaic/restorer/custom_sr_trt.py
"""T25 (CM-242): TensorRT engines for custom super-resolution models.

A custom scaler (custom_sr.py: the user's .pth in models\\upscalers, loaded
through spandrel) runs in plain torch fp16. For the whole-frame Output
Upscale that network IS the run's bottleneck -- SPAN-48 4x costs 1.76 TFLOP
per 1080p input frame, more than the restoration itself -- so this module
compiles it once to a TensorRT engine with the same helper the BasicVSR++
sub-engines use (torch.export -> torch_tensorrt dynamo -> saved program),
and custom_sr's LoadedModel routes exact-shape calls through it.

Why it is possible at all: our T20b fusion of SPAN's Conv3XC (the reference
code re-folded the reparam weights inside forward) leaves a graph of 60
nodes in 8 op types -- conv2d, SiLU, sigmoid, add/sub/mul, cat,
pixel_shuffle -- every one a native TensorRT layer. torch.export of the
fused net is bit-identical to eager (sandbox check, 10-01). Other CNN
families (Compact, RealPLKSR, ESRGAN/RRDB, OmniSR) should export the same
way; attention models (HAT, DAT, ATD...) are not attempted here.

Shapes are STATIC. An engine is built for one tile -- (1, 3, T, T) -- and
LoadedModel uses it only when the input is exactly that; anything else
takes the torch path. The whole-frame tiler feeds T x T windows (tile 640 +
a 32-px halo each side = 704 by default, padding the edge windows), the
Secondary feeds 256 x 256 crops, so one engine per use: ``--tile 704`` for
Output Upscale, ``--tile 256`` for the Secondary. Dynamic shapes export
fine too, but a static profile is the fastest kernel set and the simplest
thing to reason about on 8 GB.

Files, beside the weights (same OS / precision tags as the sub-engines):
    <stem>.trt_t704_fp16.win.engine     the saved TRT program
    <stem>.trt_t704_fp16.win.json       sidecar: weights sha256, tile, dtype,
                                        scale, arch, GPU, torch/TRT versions,
                                        parity (max abs diff / PSNR vs torch)
The sidecar's sha256 must match the weights file or the engine is ignored
(the user replaced the .pth; the engine would be someone else's weights).
A TRT engine is tied to the GPU family it was built on: a load failure on a
different card prints one line naming the card it was built on and the
run continues in torch.

NVIDIA edition only (torch_tensorrt). Nothing here is imported unless a
custom scaler is actually in use on a CUDA device.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

TILE_DEFAULT = 704          # whole-frame tiler: 640-px tile + 32-px halo each side
TILE_SECONDARY = 256        # the Secondary's crop size
_WORKSPACE_MB_DEFAULT = 2048


def _os_suffix() -> str:
    return ".win" if os.name == "nt" else ".linux"


def _prec(fp16: bool) -> str:
    return "fp16" if fp16 else "fp32"


def engine_paths_for(model_path: Path, tile: int, fp16: bool) -> Tuple[Path, Path]:
    """(engine, sidecar) for a weights file, tile and precision."""
    model_path = Path(model_path)
    stem = model_path.stem
    base = model_path.with_name(f"{stem}.trt_t{int(tile)}_{_prec(fp16)}{_os_suffix()}")
    return base.with_suffix(base.suffix + ".engine"), base.with_suffix(base.suffix + ".json")


def file_digest(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def list_engines(model_path: Path, fp16: bool) -> List[Dict[str, Any]]:
    """Every engine beside the weights whose sidecar matches them (by sha256),
    for this OS and precision. Each: {tile, engine, sidecar, meta}."""
    model_path = Path(model_path)
    out: List[Dict[str, Any]] = []
    pat = f"{model_path.stem}.trt_t*_{_prec(fp16)}{_os_suffix()}.json"
    digest: Optional[str] = None
    for side in sorted(model_path.parent.glob(pat)):
        eng = side.with_suffix(".engine")
        if not eng.is_file():
            continue
        try:
            meta = json.loads(side.read_text(encoding="utf-8"))
        except Exception:
            continue
        try:
            tile = int(meta.get("tile") or 0)
        except (TypeError, ValueError):
            tile = 0
        if tile <= 0:
            continue
        if digest is None:
            try:
                digest = file_digest(model_path)
            except Exception:
                digest = ""
        if str(meta.get("weights_sha256") or "") != digest:
            meta["_stale"] = True   # the .pth changed under the engine
        out.append({"tile": tile, "engine": eng, "sidecar": side, "meta": meta})
    return out


def _gpu_name(device: torch.device) -> str:
    try:
        return torch.cuda.get_device_name(device)
    except Exception:
        return "?"


class EngineRunner:
    """One loaded engine: (1, 3, T, T) in -> (1, 3, s*T, s*T) out."""

    def __init__(self, module: torch.nn.Module, *, tile: int, scale: int, dtype: torch.dtype,
                 meta: Dict[str, Any], path: Path) -> None:
        self.module = module
        self.tile = int(tile)
        self.scale = int(scale)
        self.dtype = dtype
        self.meta = dict(meta)
        self.path = Path(path)
        self.calls = 0

    def accepts(self, x: torch.Tensor) -> bool:
        return (x.dim() == 4 and int(x.shape[0]) == 1 and int(x.shape[1]) == 3
                and int(x.shape[2]) == self.tile and int(x.shape[3]) == self.tile
                and x.dtype == self.dtype)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.module(x.contiguous())

    def describe(self) -> str:
        m = self.meta
        return (f"TensorRT {self.tile}-px engine ({_prec(self.dtype == torch.float16)}; "
                f"built on {m.get('gpu', '?')} {str(m.get('built', ''))[:10]})")

    def close(self) -> None:
        try:
            from chitramaya.mosaic.models.basicvsrpp.sub_engines import _release_torchtrt_module
            _release_torchtrt_module(self.module)
        except Exception:
            pass
        self.module = None


def load_engines(model_path: Path, device: torch.device, *, fp16: bool, scale: int,
                 say=print) -> Dict[int, EngineRunner]:
    """Load every matching engine beside the weights. A stale or unloadable
    engine prints one line and is skipped; never raises."""
    found: Dict[int, EngineRunner] = {}
    if torch.device(device).type != "cuda":
        return found
    for ent in list_engines(model_path, fp16):
        meta = ent["meta"]
        if meta.get("_stale"):
            say(f"[Scalers] {ent['engine'].name}: ignored -- the weights file changed since it was "
                f"built (sha256 differs); recompile with tools/compile_custom_sr.py")
            continue
        try:
            import torch_tensorrt  # noqa: F401  (registers the TRT runtime ops)
            from chitramaya.mosaic.models.basicvsrpp.trt_export import load_torchtrt_export
            mod = load_torchtrt_export(checkpoint_path=str(ent["engine"]), device=device)
            # no .eval(): the module torch.export hands back is inference-only and
            # raises NotImplementedError on train()/eval() (the 10-02 3060 Ti field
            # run: "not loadable (NotImplementedError: Calling eval() is not supported yet.)")
            dtype = torch.float16 if fp16 else torch.float32
            runner = EngineRunner(mod, tile=ent["tile"], scale=int(meta.get("scale") or scale), dtype=dtype,
                                  meta=meta, path=ent["engine"])
            # one call at load: a mismatched GPU or TRT version fails HERE, not at frame 0
            x = torch.zeros((1, 3, runner.tile, runner.tile), device=device, dtype=dtype)
            y = runner(x)
            if tuple(y.shape[-2:]) != (runner.tile * scale, runner.tile * scale):
                raise RuntimeError(f"engine produced {tuple(y.shape[-2:])} from {runner.tile}x{runner.tile}")
            found[runner.tile] = runner
        except Exception as e:
            built_on = str(meta.get("gpu") or "?")
            here = _gpu_name(device)
            hint = (f"; it was built on {built_on}, this is {here}" if built_on != here else "")
            say(f"[Scalers] {ent['engine'].name}: not loadable ({type(e).__name__}: {str(e)[:160]}){hint}; "
                f"the torch path is used")
    return found


# ---------------------------------------------------------------------------
# compile
# ---------------------------------------------------------------------------

def _parity(engine_out: torch.Tensor, ref_out: torch.Tensor) -> Tuple[float, float, int, int]:
    """(max |diff|, PSNR dB, non-finite in engine out, non-finite in torch out),
    the diff taken over the elements finite in BOTH. fp16 can overflow on an
    unnatural input (the 10-02 field run: 'max |diff| nan (PSNR inf dB) on
    noise'); the counts say so instead of poisoning the numbers."""
    a = engine_out.float()
    b = ref_out.float()
    fa = torch.isfinite(a)
    fb = torch.isfinite(b)
    both = fa & fb
    n_bad_a = int((~fa).sum().item())
    n_bad_b = int((~fb).sum().item())
    if not bool(both.any()):
        return float("nan"), float("nan"), n_bad_a, n_bad_b
    diff = (a[both] - b[both]).abs()
    mse = float((diff * diff).mean())
    psnr = (10.0 * torch.log10(torch.tensor(1.0 / mse)).item()) if mse > 0 else float("inf")
    return float(diff.max()), psnr, n_bad_a, n_bad_b


def compile_engine(model_path: Path, device: torch.device, *, tile: int = TILE_DEFAULT,
                   fp16: bool = True, workspace_mb: int = _WORKSPACE_MB_DEFAULT,
                   optimization_level: int = 3, say=print) -> Dict[str, Any]:
    """Compile one engine for a custom scaler and write its sidecar.
    Returns the sidecar dict (with 'engine' and 'seconds' added)."""
    from chitramaya.mosaic.restorer.custom_sr import LoadedModel
    from chitramaya.mosaic.models.basicvsrpp.trt_export import compile_and_save_torchtrt_dynamo, load_torchtrt_export

    model_path = Path(model_path)
    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError("TensorRT engines need a CUDA device")
    tile = int(tile)
    if tile < 64 or tile % 8:
        raise ValueError("tile must be a multiple of 8 and at least 64")

    lm = LoadedModel(model_path, device, prefer_half=fp16, use_engines=False)
    if fp16 and lm.dtype != torch.float16:
        say(f"[Scalers] {model_path.name}: the model does not support fp16; compiling fp32")
        fp16 = False
    if lm.multiple_of > 1 and tile % lm.multiple_of:
        raise ValueError(f"{model_path.name} needs a multiple of {lm.multiple_of}; tile {tile} is not")
    dtype = lm.dtype
    net = lm.module()           # the fused nn.Module, eval, on device
    eng_path, side_path = engine_paths_for(model_path, tile, fp16)
    eng_path.parent.mkdir(parents=True, exist_ok=True)

    # The compile/parity input: a smooth field with mild texture, i.e. the kind
    # of picture the model will see -- not white noise, which can push an fp16
    # SPAN past the half-precision range in both torch and TensorRT.
    _g = torch.Generator(device="cpu").manual_seed(0)
    _ramp = torch.linspace(0.2, 0.8, tile)
    _base = torch.stack([_ramp[None, :].expand(tile, tile), _ramp[:, None].expand(tile, tile),
                         0.5 * (_ramp[None, :] + _ramp[:, None])], 0)
    x = (_base + 0.05 * torch.randn((3, tile, tile), generator=_g)).clamp(0, 1)[None].to(device=device, dtype=dtype)
    with torch.no_grad():
        ref = net(x).clone()
    t0 = time.perf_counter()
    compile_and_save_torchtrt_dynamo(
        module=net, inputs=[x], output_path=str(eng_path), dtype=dtype,
        workspace_size_bytes=int(workspace_mb) * 1024 * 1024,
        message=(f"[Scalers] compiling TensorRT engine for {model_path.name}: {lm.name} {lm.scale}x, "
                 f"{tile}x{tile} {_prec(fp16)} (one-time; a minute or two)"),
        device=device, optimization_level=int(optimization_level),
    )
    seconds = time.perf_counter() - t0

    # read it back and compare with torch on the SAME tensor + a smooth one
    import torch_tensorrt  # noqa: F401
    mod = load_torchtrt_export(checkpoint_path=str(eng_path), device=device)
    with torch.no_grad():
        out = mod(x.contiguous())
        max_diff, psnr, bad_e, bad_t = _parity(out, ref)
        g = torch.linspace(0, 1, tile, device=device, dtype=dtype)
        smooth = torch.stack([g[None, :].expand(tile, tile), g[:, None].expand(tile, tile),
                              (g[None, :] * g[:, None])], 0)[None]
        max_diff2, psnr2, bad_e2, bad_t2 = _parity(mod(smooth.contiguous()), net(smooth))
    if (bad_e or bad_e2) and not (bad_t or bad_t2):
        say(f"[Scalers] WARNING: the engine produced {bad_e + bad_e2} non-finite value(s) where torch did not; "
            f"the engine is written but check a real frame before trusting it")
    elif bad_t or bad_t2:
        say(f"[Scalers] note: torch {_prec(fp16)} itself produced {bad_t + bad_t2} non-finite value(s) on the test "
            f"inputs (engine: {bad_e + bad_e2}); parity is measured over the finite elements")
    try:
        from chitramaya.mosaic.models.basicvsrpp.sub_engines import _release_torchtrt_module
        _release_torchtrt_module(mod)
    except Exception:
        pass
    del mod

    trt_ver = "?"
    try:
        import tensorrt as _trt
        trt_ver = str(_trt.__version__)
    except Exception:
        pass
    meta: Dict[str, Any] = {
        "schema": 1,
        "weights": model_path.name,
        "weights_sha256": file_digest(model_path),
        "arch": lm.arch, "name": lm.name, "scale": int(lm.scale), "multiple_of": int(lm.multiple_of),
        "tile": tile, "dtype": _prec(fp16),
        "gpu": _gpu_name(device), "os": os.name, "torch": torch.__version__, "tensorrt": trt_ver,
        "built": _dt.datetime.now().isoformat(timespec="seconds"),
        "compile_seconds": round(seconds, 1), "workspace_mb": int(workspace_mb),
        "parity": {"texture_max_abs_diff": round(max_diff, 6), "texture_psnr_db": round(psnr, 2),
                   "smooth_max_abs_diff": round(max_diff2, 6), "smooth_psnr_db": round(psnr2, 2),
                   "non_finite_engine": int(bad_e + bad_e2), "non_finite_torch": int(bad_t + bad_t2)},
        "quirks": lm.quirks,
    }
    side_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    say(f"[Scalers] engine written: {eng_path.name} ({eng_path.stat().st_size / 1e6:.1f} MB) in {seconds:.0f} s; "
        f"parity vs torch: max |diff| {max_diff:.4f} (PSNR {psnr:.1f} dB) on a textured field, "
        f"{max_diff2:.4f} ({psnr2:.1f} dB) on a gradient")
    lm.close()
    meta["engine"] = str(eng_path)
    meta["seconds"] = seconds
    return meta


def bench(model_path: Path, device: torch.device, *, tile: int, fp16: bool = True, iters: int = 20,
          frame_hw: Tuple[int, int] = (1080, 1920), say=print) -> Dict[str, Any]:
    """Time one tile through torch and through the engine; project a frame."""
    from chitramaya.mosaic.restorer.custom_sr import LoadedModel
    device = torch.device(device)
    lm = LoadedModel(model_path, device, prefer_half=fp16, use_engines=True)
    dtype = lm.dtype
    x = torch.rand((1, 3, tile, tile), device=device, dtype=dtype)
    net = lm.module()

    def _time(fn) -> float:
        with torch.no_grad():
            for _ in range(3):
                fn(x)
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            for _ in range(iters):
                fn(x)
            torch.cuda.synchronize(device)
        return (time.perf_counter() - t0) * 1000.0 / iters

    ms_torch = _time(net)
    eng = lm.engines.get(tile)
    ms_eng = _time(eng) if eng is not None else None
    halo = 32
    inner = max(1, tile - 2 * halo)
    tiles = -(-frame_hw[0] // inner) * -(-frame_hw[1] // inner)
    res = {"tile": tile, "dtype": _prec(dtype == torch.float16), "ms_tile_torch": round(ms_torch, 2),
           "ms_tile_engine": (round(ms_eng, 2) if ms_eng is not None else None),
           "tiles_per_frame": tiles, "frame": f"{frame_hw[1]}x{frame_hw[0]}",
           "ms_frame_torch": round(ms_torch * tiles, 1),
           "ms_frame_engine": (round(ms_eng * tiles, 1) if ms_eng is not None else None),
           "speedup": (round(ms_torch / ms_eng, 2) if ms_eng else None)}
    say(f"[Scalers] bench {model_path.name} @ {tile}px {res['dtype']}: torch {ms_torch:.2f} ms/tile"
        + (f", engine {ms_eng:.2f} ms/tile ({res['speedup']}x)" if ms_eng is not None else ", no engine for this tile")
        + f"; {tiles} tiles per {res['frame']} frame -> torch {res['ms_frame_torch']} ms"
        + (f", engine {res['ms_frame_engine']} ms" if ms_eng is not None else ""))
    lm.close()
    return res


__all__ = ["TILE_DEFAULT", "TILE_SECONDARY", "engine_paths_for", "file_digest", "list_engines",
           "load_engines", "compile_engine", "bench", "EngineRunner"]
