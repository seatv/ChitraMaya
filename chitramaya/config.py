"""Configuration file support for ChitraMaya.

Follows the Tilester pattern:
    built-in defaults → config file → CLI arguments

Built-in defaults are derived directly from ``MosaicConfig`` dataclass fields
(single source of truth — no manual sync required).

Usage::

    ChitraMaya --init-config                  # create default config
    ChitraMaya --config my.json input.mp4     # use custom config
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

from chitramaya.models import MosaicConfig

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_NAME = "ChitraMaya-config.json"


def _derive_defaults() -> dict[str, Any]:
    """Return defaults derived directly from ``MosaicConfig`` fields.

    This is the single source of truth.  Any change to a MosaicConfig field
    default is automatically reflected here.
    """
    out: dict[str, Any] = {}
    for f in dataclasses.fields(MosaicConfig):
        if f.default is not dataclasses.MISSING:
            val = f.default
        elif f.default_factory is not dataclasses.MISSING:
            val = f.default_factory()
        else:
            continue

        # Normalize tuples → lists for JSON round-trip
        if isinstance(val, tuple):
            val = list(val)

        out[f.name] = val
    return out


BUILTIN_DEFAULTS: dict[str, Any] = _derive_defaults()


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load ChitraMaya-config.json as a flat dict.

    Returns {} on missing file or parse error (with warning).
    """
    if path is None:
        path = Path.cwd() / DEFAULT_CONFIG_NAME
    else:
        path = Path(path)

    if not path.exists():
        return {}

    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        if not isinstance(data, dict):
            logger.warning("Config file %s is not a JSON object; ignoring", path)
            return {}
        logger.info("Loaded config from %s", path)
        return data
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse config %s: %s", path, e)
        return {}
    except OSError as e:
        logger.warning("Failed to read config %s: %s", path, e)
        return {}


def merge_config(
    config: dict[str, Any],
    cli_args: dict[str, Any],
) -> dict[str, Any]:
    """Merge config file values with CLI arguments.

    Precedence: built-in defaults < config file < CLI args.

    CLI args that are ``None`` fall through to config, which falls through
    to built-in defaults.
    """
    merged: dict[str, Any] = {}

    for key, builtin_default in BUILTIN_DEFAULTS.items():
        cli_val = cli_args.get(key)
        cfg_val = config.get(key)

        if cli_val is not None:
            merged[key] = cli_val
        elif cfg_val is not None:
            merged[key] = cfg_val
        else:
            merged[key] = builtin_default

    # Pass through CLI-only keys not in BUILTIN_DEFAULTS
    for key, val in cli_args.items():
        if key not in merged:
            merged[key] = val

    return merged


def generate_default_config(path: Path | str | None = None) -> Path:
    """Write a default config file with all knobs documented.

    Args:
        path: Destination.  Defaults to ``./ChitraMaya-config.json``.

    Returns:
        The path that was written.
    """
    if path is None:
        path = Path.cwd() / DEFAULT_CONFIG_NAME
    path = Path(path)

    # Write clean JSON with all defaults
    path.write_text(
        json.dumps(BUILTIN_DEFAULTS, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    logger.info("Default config written to %s", path)
    return path
