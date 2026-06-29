# gRestorer/utils/app_config.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge src into dst (in place), returning dst."""
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


@dataclass
class Config:
    """
    Unified configuration object.

    - data: backing dictionary (may contain nested dicts, e.g. data["visualization"]["fill_color"])
    - get(): nested getter
    - set(): nested setter
    - merge_dict(): recursive merge of a dict into config
    - attribute access: cfg.input_path works for top-level keys
    """
    data: Dict[str, Any]

    @staticmethod
    def load_json(path: str | Path) -> Dict[str, Any]:
        p = Path(path)
        if not p.exists():
            return {}
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            raise ValueError(f"Config JSON root must be an object/dict, got {type(obj)}")
        return obj

    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Nested access:
          cfg.get("visualization", "fill_color", default=None)
          cfg.get("batch_size", default=8)
        """
        cur: Any = self.data
        for k in keys:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return default
        return cur

    def set(self, *keys: str, value: Any) -> None:
        """Nested set, creating dicts as needed."""
        if not keys:
            raise ValueError("set() requires at least one key")
        cur = self.data
        for k in keys[:-1]:
            nxt = cur.get(k)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[k] = nxt
            cur = nxt
        cur[keys[-1]] = value

    def merge_dict(self, extra: Dict[str, Any]) -> None:
        _deep_update(self.data, extra)

    def __getattr__(self, name: str) -> Any:
        if name in self.data:
            return self.data[name]
        raise AttributeError(name)
