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


def _load_checkpoint_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    # torch.load(weights_only=...) exists on newer torch; fall back if needed.
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        # sometimes checkpoints are already a state_dict
        return ckpt
    raise TypeError(f"Unsupported checkpoint format: {type(ckpt)}")


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

    # fp16 only makes sense on CUDA; keep it safe
    use_fp16 = bool(fp16) and device.type == "cuda"
    if use_fp16:
        model = model.half()

    return model
