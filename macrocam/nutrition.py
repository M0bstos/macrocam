from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import httpx

from macrocam.models import DbMatch, NutritionFacts
from macrocam.vision import VisionConfig, get_vision_config


DEFAULT_DATA_DIR = Path("data/usda_fdc")
DATA_DIR_ENV = "MACROCAM_USDA_DIR"

REQUIRED_FILES = ("food.csv", "food_nutrient.csv", "nutrient.csv")

NUTRIENT_NAME_PREFERENCES: dict[str, list[str]] = {
    "calories_kcal": [
        "Energy",
        "Energy (Atwater General Factors)",
        "Energy (Atwater Specific Factors)",
    ],
    "protein_g": ["Protein", "Adjusted Protein"],
    "carbs_g": [
        "Carbohydrate, by difference",
        "Carbohydrate, by summation",
        "Carbohydrates",
    ],
    "fat_g": ["Total lipid (fat)"],
    "sugars_g": ["Total Sugars", "Sugars, Total"],
    "fiber_g": ["Fiber, total dietary"],
    "sodium_mg": ["Sodium, Na"],
    "cholesterol_mg": ["Cholesterol"],
}

PREFERRED_DATA_TYPES = (
    "foundation_food",
    "survey_fndds_food",
    "sr_legacy_food",
    "sub_sample_food",
    "sample_food",
    "experimental_food",
    "market_acquistion",
    "market_acquisition",
    "agricultural_acquisition",
    "branded_food",
)

DATA_TYPE_BONUS = {
    data_type: (len(PREFERRED_DATA_TYPES) - idx) * 0.02
    for idx, data_type in enumerate(PREFERRED_DATA_TYPES)
}


@dataclass(frozen=True)
class FoodCandidate:
    fdc_id: str
    description: str
    data_type: str
    score: float


def _resolve_data_dir(data_dir: str | Path | None) -> Path:
    if data_dir is None:
        override = os.environ.get(DATA_DIR_ENV)
        base = Path(override) if override else DEFAULT_DATA_DIR
    else:
        base = Path(data_dir)
    base = base.expanduser()
    for filename in REQUIRED_FILES:
        if not (base / filename).exists():
            raise FileNotFoundError(f"Missing USDA file: {base / filename}")
    return base


def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    return re.sub(r"\s+", " ", cleaned)


def _extract_text_from_gemini_response(payload: dict[str, object]) -> str:
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("LLM response missing candidates")
    content = candidates[0].get("content", {})
    parts = content.get("parts", [])
    if not isinstance(parts, list) or not parts:
        raise ValueError("LLM response missing content parts")
    texts: list[str] = []
    for part in parts:
        if isinstance(part, dict) and "text" in part:
            texts.append(str(part["text"]))
    if not texts:
        raise ValueError("LLM response contained no text")
    return "\n".join(texts).strip()


def suggest_fallback_queries(
    query: str,
    *,
    config: VisionConfig | None = None,
) -> list[str]:
    if not query or not query.strip():
        return []
    if config is None:
        config = get_vision_config()

    prompt = (
        "You help map food labels to USDA FoodData Central entries.\n"
        "Return JSON only: a list of 3-5 short alternative search phrases.\n"
        "Rules: no nutrition values, no explanations, no markdown.\n"
        "Prefer generic food names over brands when possible.\n"
        f"Food label: {query}\n"
        "Output example: [\"grilled chicken breast\", \"chicken breast\", \"roasted chicken\"]"
    )

    url = f"{config.api_base}/models/{config.model}:generateContent"
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    headers = {"x-goog-api-key": config.api_key}

    timeout = httpx.Timeout(30.0)
    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        text = _extract_text_from_gemini_response(response.json())

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []

    if not isinstance(data, list):
        return []

    cleaned: list[str] = []
    for item in data:
        if not isinstance(item, str):
            continue
        label = item.strip()
        if label:
            cleaned.append(label)
    return cleaned[:5]


def _score_description(
    query_norm: str,
    query_tokens: set[str],
    description: str,
    data_type: str,
) -> float:
    if not description:
        return 0.0
    desc_norm = _normalize_text(description)
    if not query_norm or not desc_norm:
        return 0.0

    bonus = DATA_TYPE_BONUS.get(data_type, 0.0)
    if query_norm == desc_norm:
        return 2.0 + bonus
    if query_norm in desc_norm:
        return 1.2 + bonus

    desc_tokens = set(desc_norm.split())
    if not desc_tokens:
        return 0.0
    overlap = len(query_tokens.intersection(desc_tokens))
    if overlap == 0:
        return 0.0
    base = overlap / max(len(query_tokens), 1)
    return base + bonus


def _iter_food_rows(food_path: Path) -> Iterable[dict[str, str]]:
    with food_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def search_usda_foods(
    query: str,
    data_dir: str | Path | None = None,
    *,
    limit: int = 5,
    prefer_types: Iterable[str] | None = None,
    max_rows: int | None = None,
) -> list[FoodCandidate]:
    if not query or not query.strip():
        raise ValueError("query must be non-empty")

    base_dir = _resolve_data_dir(data_dir)
    food_path = base_dir / "food.csv"
    prefer = set(prefer_types) if prefer_types is not None else set(PREFERRED_DATA_TYPES)

    query_norm = _normalize_text(query)
    if not query_norm:
        raise ValueError("query must be non-empty")
    query_tokens = set(query_norm.split())

    candidates: list[FoodCandidate] = []
    for idx, row in enumerate(_iter_food_rows(food_path)):
        if max_rows is not None and idx >= max_rows:
            break
        description = (row.get("description") or "").strip()
        if not description:
            continue
        data_type = (row.get("data_type") or "").strip()
        if prefer and data_type and data_type not in prefer:
            continue
        score = _score_description(query_norm, query_tokens, description, data_type)
        if score <= 0:
            continue
        fdc_id = str(row.get("fdc_id") or "").strip()
        if not fdc_id:
            continue

        candidate = FoodCandidate(
            fdc_id=fdc_id,
            description=description,
            data_type=data_type,
            score=score,
        )
        if len(candidates) < limit:
            candidates.append(candidate)
        else:
            lowest_index = min(
                range(len(candidates)), key=lambda idx: candidates[idx].score
            )
            if candidate.score > candidates[lowest_index].score:
                candidates[lowest_index] = candidate

    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    return candidates[:limit]


def _load_nutrient_index(base_dir: Path) -> dict[str, dict[str, str]]:
    nutrient_path = base_dir / "nutrient.csv"
    lookup: dict[str, dict[str, str]] = {}
    with nutrient_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = (row.get("name") or "").strip()
            if not name:
                continue
            entry = {
                "id": str(row.get("id") or "").strip(),
                "unit": (row.get("unit_name") or "").strip().upper(),
            }
            key = name.lower()
            if key in lookup:
                existing = lookup[key]
                if existing.get("unit") == "KCAL":
                    continue
                if entry.get("unit") != "KCAL":
                    continue
            lookup[key] = entry
    return lookup


def _resolve_nutrient_ids(base_dir: Path) -> dict[str, list[str]]:
    nutrient_lookup = _load_nutrient_index(base_dir)
    resolved: dict[str, list[str]] = {}
    for field, names in NUTRIENT_NAME_PREFERENCES.items():
        ids: list[str] = []
        for name in names:
            entry = nutrient_lookup.get(name.lower())
            if entry and entry.get("id"):
                ids.append(entry["id"])
        if ids:
            resolved[field] = ids
    return resolved


def _load_nutrient_amounts(
    fdc_id: str,
    base_dir: Path,
    required_ids: set[str],
) -> dict[str, float]:
    nutrient_path = base_dir / "food_nutrient.csv"
    amounts: dict[str, float] = {}
    with nutrient_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("fdc_id") or "").strip() != fdc_id:
                continue
            nutrient_id = str(row.get("nutrient_id") or "").strip()
            if nutrient_id not in required_ids:
                continue
            if nutrient_id in amounts:
                continue
            raw_amount = (row.get("amount") or "").strip()
            if raw_amount == "":
                continue
            try:
                amounts[nutrient_id] = float(raw_amount)
            except ValueError:
                continue
    return amounts


def _build_nutrition_facts(
    nutrient_ids: dict[str, list[str]],
    nutrient_amounts: dict[str, float],
) -> NutritionFacts:
    def pick(field: str) -> float | None:
        ids = nutrient_ids.get(field, [])
        for nutrient_id in ids:
            if nutrient_id in nutrient_amounts:
                return nutrient_amounts[nutrient_id]
        return None

    calories = pick("calories_kcal")
    protein = pick("protein_g")
    carbs = pick("carbs_g")
    fat = pick("fat_g")

    if calories is None or protein is None or carbs is None or fat is None:
        raise ValueError("Missing required nutrient values for selected food")

    sugars = pick("sugars_g")
    fiber = pick("fiber_g")
    sodium = pick("sodium_mg")
    cholesterol = pick("cholesterol_mg")

    return NutritionFacts(
        calories_kcal=calories,
        protein_g=protein,
        carbs_g=carbs,
        fat_g=fat,
        sugars_g=sugars,
        fiber_g=fiber,
        sodium_mg=sodium,
        cholesterol_mg=cholesterol,
    )


def lookup_usda_food(
    query: str,
    data_dir: str | Path | None = None,
    *,
    prefer_types: Iterable[str] | None = None,
    fallback_queries: Iterable[str] | None = None,
    use_llm_fallback: bool = True,
) -> DbMatch:
    base_dir = _resolve_data_dir(data_dir)
    matches = search_usda_foods(query, base_dir, limit=20, prefer_types=prefer_types)
    if not matches:
        if fallback_queries is None and use_llm_fallback:
            fallback_queries = suggest_fallback_queries(query)
        for alt_query in fallback_queries or []:
            matches.extend(
                search_usda_foods(
                    alt_query, base_dir, limit=20, prefer_types=prefer_types
                )
            )
    if not matches:
        raise ValueError(f"No USDA matches found for query: {query}")
    seen: set[str] = set()
    unique_matches: list[FoodCandidate] = []
    for candidate in matches:
        if candidate.fdc_id in seen:
            continue
        seen.add(candidate.fdc_id)
        unique_matches.append(candidate)

    nutrient_ids = _resolve_nutrient_ids(base_dir)
    required_ids: set[str] = set()
    for ids in nutrient_ids.values():
        required_ids.update(ids)
    last_error: Exception | None = None
    for candidate in unique_matches:
        try:
            nutrient_amounts = _load_nutrient_amounts(
                candidate.fdc_id, base_dir, required_ids
            )
            facts = _build_nutrition_facts(nutrient_ids, nutrient_amounts)
        except ValueError as exc:
            last_error = exc
            continue
        return DbMatch(
            fdc_id=candidate.fdc_id,
            description=candidate.description,
            per_100g=facts,
        )

    if last_error is not None:
        raise last_error
    raise ValueError(f"No USDA matches found for query: {query}")
