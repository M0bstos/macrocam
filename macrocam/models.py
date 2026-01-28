from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _require_non_empty(value: str, field_name: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_non_negative(value: float, field_name: str) -> None:
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")


@dataclass
class Candidate:
    label: str
    confidence: float
    notes: str = ""

    def __post_init__(self) -> None:
        _require_non_empty(self.label, "label")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")


@dataclass
class NutritionFacts:
    calories_kcal: float
    protein_g: float
    carbs_g: float
    fat_g: float
    sugars_g: float | None = None
    fiber_g: float | None = None
    sodium_mg: float | None = None
    cholesterol_mg: float | None = None

    def __post_init__(self) -> None:
        _require_non_negative(self.calories_kcal, "calories_kcal")
        _require_non_negative(self.protein_g, "protein_g")
        _require_non_negative(self.carbs_g, "carbs_g")
        _require_non_negative(self.fat_g, "fat_g")
        if self.sugars_g is not None:
            _require_non_negative(self.sugars_g, "sugars_g")
        if self.fiber_g is not None:
            _require_non_negative(self.fiber_g, "fiber_g")
        if self.sodium_mg is not None:
            _require_non_negative(self.sodium_mg, "sodium_mg")
        if self.cholesterol_mg is not None:
            _require_non_negative(self.cholesterol_mg, "cholesterol_mg")


@dataclass
class DbMatch:
    fdc_id: str
    description: str
    per_100g: NutritionFacts
    db_name: str = "USDA FoodData Central"

    def __post_init__(self) -> None:
        _require_non_empty(self.fdc_id, "fdc_id")
        _require_non_empty(self.description, "description")
        _require_non_empty(self.db_name, "db_name")


@dataclass
class MealItem:
    label: str
    grams: float
    db_match: DbMatch

    def __post_init__(self) -> None:
        _require_non_empty(self.label, "label")
        if self.grams <= 0:
            raise ValueError("grams must be greater than 0")


@dataclass
class CacheEntry:
    image_hash: str
    vision_response_json: dict[str, Any]
    created_at: str = field(default="")

    def __post_init__(self) -> None:
        _require_non_empty(self.image_hash, "image_hash")
        if not self.created_at or not self.created_at.strip():
            self.created_at = datetime.now(timezone.utc).isoformat()
        else:
            try:
                datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("created_at must be ISO-8601") from exc
