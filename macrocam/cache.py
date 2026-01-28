from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _cache_dir(base_dir: str | Path | None = None) -> Path:
    if base_dir is not None:
        return Path(base_dir).expanduser()
    override = os.environ.get("MACROCAM_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    return Path("~/.macrocam/cache").expanduser()


def _cache_path(image_hash: str, base_dir: str | Path | None = None) -> Path:
    cache_root = _cache_dir(base_dir)
    return cache_root / f"{image_hash}.json"


def get_vision_cache(image_hash: str, base_dir: str | Path | None = None) -> dict[str, Any] | None:
    path = _cache_path(image_hash, base_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        try:
            path.unlink()
        except OSError:
            pass
        return None


def set_vision_cache(image_hash: str, payload: dict[str, Any], base_dir: str | Path | None = None) -> None:
    path = _cache_path(image_hash, base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload))
    tmp_path.replace(path)
