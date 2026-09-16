# ChitraMaya/mosaic/models/basicvsrpp/inference.py
# Ported from gRestorer; mmengine-free, uses the standalone BasicVSR++ adapter
# under ChitraMaya.mosaic.models.basicvsrpp.grestorer (the dir was named .lada in
# gRestorer; we kept it named .grestorer here).
from __future__ import annotations

import logging
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)

# IMPORTANT:
# This inference path is mmengine-free. The standalone BasicVSR++ adapter
# is at ChitraMaya.mosaic.models.basicvsrpp.grestorer (renamed from `.lada`).
from chitramaya.mosaic.models.basicvsrpp.grestorer.basicvsr_plusplus_net import BasicVSRPlusPlusNet


def get_default_gan_inference_config() -> dict:
    """
    LADA-ish default: only the generator matters for inference here.
    Keep this so callers can pass `config=None` like LADA.
    """
    return dict(
        generator=dict(
            mid_channels=64,
            num_blocks=15,
            max_residue_magnitude=10,
            spynet_pretrained=None,
        )
    )


def _load_permissive(checkpoint_path: str):
    """torch.load with an Unpickler that substitutes an inert stub for any
    class whose module is not installed (mmengine, mmagic, mmcv ...). Only
    tensors are used from the result; the stubs are discarded."""
    import pickle
    import types

    class _StubMeta(type):
        # torch.save writes pickle protocol 2. At that protocol a nested
        # qualified name (mmengine's HistoryBuffer.min, stored in the
        # buffer's statistics table) is written as getattr(HistoryBuffer,
        # "min") -- so the stub CLASS must answer any attribute lookup with
        # another stub, or the field _full.pth fails with "type object
        # 'HistoryBuffer' has no attribute 'min'" (#9, 09-15 field test).
        def __getattr__(cls, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return _StubMeta(f"{cls.__name__}.{name}", (cls,), {"__module__": cls.__module__})

    class _Stub(metaclass=_StubMeta):
        def __new__(cls, *a, **k):
            return object.__new__(cls)

        def __init__(self, *a, **k):
            pass

        def __setstate__(self, state):
            self.__dict__["_state"] = state

        def __call__(self, *a, **k):
            return self

        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return self

        # dict / list subclasses (mmengine ConfigDict, addict Dict) are
        # rebuilt by pickle with obj[key] = value / obj.append(...): swallow.
        def __setitem__(self, key, value):
            pass

        def append(self, item):
            pass

        def extend(self, items):
            pass

    class _PermissiveUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except (ModuleNotFoundError, AttributeError, ImportError):
                return _StubMeta(str(name), (_Stub,), {"__module__": str(module)})

    def _load(f, **kw):
        return _PermissiveUnpickler(f, **kw).load()

    pm = types.SimpleNamespace(Unpickler=_PermissiveUnpickler, load=_load,
                               __name__="chitramaya_permissive_pickle")
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False, pickle_module=pm)


def _load_checkpoint_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    # CM-095 (v1.50.00): a wrong or corrupt file here used to surface as a
    # cryptic torch/pickle traceback deep inside compile or load. Catch the
    # common cases and say, in one message, WHICH file failed, WHY it is not
    # usable, and WHAT to do (Manage Models downloads known-good weights).
    import os as _os
    if not _os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Restoration checkpoint not found: {checkpoint_path}. "
            f"Download a restoration model in Manage Models, or point "
            f"--rest-model at an existing .pth file."
        )
    # torch.load(weights_only=...) exists on newer torch; fall back if needed.
    try:
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
        except (ModuleNotFoundError, AttributeError) as _mnf:
            # GitHub #9: lada's `_full.pth` files are mmengine TRAINING
            # checkpoints -- the pickle references mmengine/mmagic classes
            # (message hub, config objects, optimizer state) that are not
            # part of ChitraMaya. The generator weights inside are the same
            # ones the runtime .pth carries, so read them with a permissive
            # unpickler that stands in an inert stub for any class it cannot
            # import; everything we use is plain tensors under state_dict.
            print(f"[Restorer] {_os.path.basename(checkpoint_path)}: full training checkpoint "
                  f"({_mnf}); reading its weights with a permissive loader (#9).")
            ckpt = _load_permissive(checkpoint_path)
    except Exception as e:
        _sz = _os.path.getsize(checkpoint_path)
        raise RuntimeError(
            f"Could not read {checkpoint_path} as a PyTorch checkpoint "
            f"({type(e).__name__}: {e}). The file is {_sz} bytes -- if that "
            f"is much smaller than expected, the download was likely "
            f"interrupted: delete it and re-download in Manage Models. If it "
            f"is a TensorRT .engine or an ONNX file, select the matching "
            f".pth checkpoint instead."
        ) from e

    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        # sometimes checkpoints are already a state_dict
        return ckpt
    raise TypeError(
        f"{checkpoint_path} loaded, but it is not a usable checkpoint "
        f"(got {type(ckpt).__name__}, expected a state_dict). This usually "
        f"means the file is a full training checkpoint or a different "
        f"model family -- download a BasicVSR++ restoration model in "
        f"Manage Models."
    )


def _strip_known_prefixes(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    LADA/mmengine checkpoints often store generator weights under prefixes like:
      - 'generator.'
      - 'module.generator.'
      - 'ema_model.generator.'
      - 'net_g.'
    We strip whichever matches.
    """
    prefixes = (
        "generator.",
        "module.generator.",
        "ema_model.generator.",
        "net_g.",
        "module.net_g.",
        "model.generator.",
        "module.model.generator.",
        "module.",  # last resort (DataParallel)
    )

    for p in prefixes:
        if any(k.startswith(p) for k in sd.keys()):
            stripped = {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}
            if stripped:
                return stripped
    return sd


class _BasicVSRPPWrapper(torch.nn.Module):
    """
    Minimal, inference-only wrapper that matches the call style used by LADA:

        out = model(inputs=btchw)

    where btchw is [B, T, C, H, W].
    """
    def __init__(self, generator: torch.nn.Module):
        super().__init__()
        self.generator = generator

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.generator(inputs)


def load_model(
    config: str | dict | None,
    checkpoint_path: str,
    device: torch.device | str,
    fp16: bool = False,
) -> torch.nn.Module:
    """
    Build generator + load checkpoint WITHOUT mmengine.
    Returns a callable module that accepts `inputs=BTCHW` and returns `BTCHW`.
    """

    if isinstance(device, str):
        device = torch.device(device)

    # Config parsing (keep LADA-like signature)
    if config is None:
        config = get_default_gan_inference_config()
    if isinstance(config, str):
        # Lightweight "config.py" support (optional):
        # exec the file and read `model` or `config` dict from it.
        scope: Dict[str, Any] = {}
        with open(config, "r", encoding="utf-8") as f:
            code = f.read()
        exec(compile(code, config, "exec"), scope, scope)
        if "model" in scope and isinstance(scope["model"], dict):
            config = scope["model"]
        elif "config" in scope and isinstance(scope["config"], dict):
            config = scope["config"]
        else:
            raise ValueError(f"Config file {config!r} did not define dict `model` or `config`.")
    if not isinstance(config, dict):
        raise TypeError("config must be a dict, a config.py path, or None")

    # Accept both:
    #   { generator: {...} }
    # or LADA-style:
    #   { type: '...', generator: {...}, ... }
    gen_cfg = config.get("generator")
    if not isinstance(gen_cfg, dict):
        raise ValueError("config must contain dict `generator`")

    generator = BasicVSRPlusPlusNet(
        mid_channels=int(gen_cfg.get("mid_channels", 64)),
        num_blocks=int(gen_cfg.get("num_blocks", 7)),
        max_residue_magnitude=int(gen_cfg.get("max_residue_magnitude", 10)),
        spynet_pretrained=gen_cfg.get("spynet_pretrained", None),
    )

    sd = _load_checkpoint_state_dict(checkpoint_path)
    sd = _strip_known_prefixes(sd)

    missing, unexpected = generator.load_state_dict(sd, strict=False)
    if missing or unexpected:
        logger.warning(
            "[BasicVSR++] load_state_dict: missing=%d unexpected=%d",
            len(missing),
            len(unexpected),
        )

    model = _BasicVSRPPWrapper(generator=generator).to(device).eval()

    # fp16 on CUDA and XPU (CM-093 phase-0 validated); CPU stays fp32.
    use_fp16 = bool(fp16) and device.type in ("cuda", "xpu")
    if use_fp16:
        model = model.half()

    return model
