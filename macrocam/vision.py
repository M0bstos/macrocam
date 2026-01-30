from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Any

import httpx
from dotenv import load_dotenv

from macrocam.models import Candidate

_ENV_LOADED = False


def _ensure_env_loaded() -> None:
    global _ENV_LOADED
    if not _ENV_LOADED:
        load_dotenv()
        _ENV_LOADED = True


@dataclass
class VisionConfig:
    api_key: str
    model: str
    api_base: str = "https://generativelanguage.googleapis.com/v1beta"


@dataclass
class VisionResult:
    items: list[Candidate]
    overall_notes: str = ""


def get_vision_config() -> VisionConfig:
    _ensure_env_loaded()
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    model = os.environ.get("LLM_MODEL", "").strip()
    api_base = os.environ.get("LLM_API_BASE", "").strip()
    if not api_key:
        raise ValueError("LLM_API_KEY is required")
    if not model:
        raise ValueError("LLM_MODEL is required")
    if not api_base:
        return VisionConfig(api_key=api_key, model=model)
    return VisionConfig(api_key=api_key, model=model, api_base=api_base)


def build_prompt(retry: bool = False) -> str:
    base = (
        "You are a food detector. Return JSON only.\n"
        "Identify 3-5 foods in the image. Each item has label, confidence (0-1), notes.\n"
        "Do not compute nutrition. Use broad categories if unsure.\n"
        "Output schema:\n"
        "{\n"
        '  "items": [\n'
        '    {"label": "string", "confidence": 0.0, "notes": "string"}\n'
        "  ],\n"
        '  "overall_notes": "string"\n'
        "}"
    )
    if not retry:
        return base
    return base + "\nJSON only. No prose. No markdown."


def parse_vision_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("vision response was not valid JSON") from exc


def parse_vision_json_with_retry(primary_text: str, retry_text: str | None) -> dict[str, Any]:
    try:
        return parse_vision_json(primary_text)
    except ValueError:
        if retry_text is None:
            raise
        return parse_vision_json(retry_text)


def normalize_candidates(raw: dict[str, Any]) -> VisionResult:
    items = raw.get("items", [])
    if not isinstance(items, list):
        raise ValueError("vision JSON must include items list")

    candidates: list[Candidate] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip() or "Unknown"
        confidence_raw = item.get("confidence", 0.0)
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = min(max(confidence, 0.0), 1.0)
        notes = str(item.get("notes", "")).strip()
        candidates.append(Candidate(label=label, confidence=confidence, notes=notes))

    if not candidates:
        candidates = [Candidate(label="Unknown", confidence=0.01, notes="")]

    candidates.sort(key=lambda candidate: candidate.confidence, reverse=True)
    if len(candidates) > 5:
        candidates = candidates[:5]
    if len(candidates) < 3:
        for _ in range(3 - len(candidates)):
            candidates.append(Candidate(label="Unknown", confidence=0.01, notes=""))

    overall_notes = str(raw.get("overall_notes", "")).strip()
    return VisionResult(items=candidates, overall_notes=overall_notes)


def _extract_text_from_gemini_response(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("vision response missing candidates")
    content = candidates[0].get("content", {})
    parts = content.get("parts", [])
    if not isinstance(parts, list) or not parts:
        raise ValueError("vision response missing content parts")
    texts: list[str] = []
    for part in parts:
        if isinstance(part, dict) and "text" in part:
            texts.append(str(part["text"]))
    if not texts:
        raise ValueError("vision response contained no text")
    return "\n".join(texts).strip()


def analyze_image(
    image_bytes: bytes,
    mime_type: str,
    config: VisionConfig | None = None,
) -> VisionResult:
    if not mime_type:
        raise ValueError("mime_type is required")

    if config is None:
        config = get_vision_config()

    encoded = base64.b64encode(image_bytes).decode("ascii")
    url = f"{config.api_base}/models/{config.model}:generateContent"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": build_prompt(retry=False)},
                    {"inlineData": {"mimeType": mime_type, "data": encoded}},
                ],
            }
        ]
    }
    retry_payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": build_prompt(retry=True)},
                    {"inlineData": {"mimeType": mime_type, "data": encoded}},
                ],
            }
        ]
    }

    headers = {"x-goog-api-key": config.api_key}
    timeout = httpx.Timeout(30.0)
    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        primary_text = _extract_text_from_gemini_response(response.json())
        try:
            raw = parse_vision_json(primary_text)
        except ValueError:
            retry = client.post(url, json=retry_payload, headers=headers)
            retry.raise_for_status()
            retry_text = _extract_text_from_gemini_response(retry.json())
            raw = parse_vision_json_with_retry(primary_text, retry_text)

    return normalize_candidates(raw)
