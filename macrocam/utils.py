from __future__ import annotations

import hashlib
import re
from pathlib import Path


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    file_path = Path(path)
    hasher = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def is_supported_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def require_existing_file(path: str | Path) -> Path:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Not a file: {file_path}")
    return file_path


def parse_grams(raw: str) -> float:
    if raw is None:
        raise ValueError("grams input is required")
    text = raw.strip().lower()
    if not text:
        raise ValueError("grams input is required")

    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([a-z]*)", text)
    if not match:
        raise ValueError("invalid grams input")

    value = float(match.group(1))
    unit = match.group(2) or "g"
    if unit not in {"g", "gram", "grams"}:
        raise ValueError("unsupported unit")
    if value <= 0:
        raise ValueError("grams must be greater than 0")
    return value
