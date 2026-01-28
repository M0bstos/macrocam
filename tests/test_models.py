from datetime import datetime

import pytest

from macrocam.models import CacheEntry, Candidate, DbMatch, MealItem, NutritionFacts


def test_candidate_confidence_range() -> None:
    with pytest.raises(ValueError):
        Candidate(label="Salad", confidence=1.5)


def test_nutrition_facts_non_negative() -> None:
    with pytest.raises(ValueError):
        NutritionFacts(
            calories_kcal=100,
            protein_g=-1,
            carbs_g=10,
            fat_g=5,
        )


def test_meal_item_grams_positive() -> None:
    facts = NutritionFacts(
        calories_kcal=100,
        protein_g=5,
        carbs_g=10,
        fat_g=3,
    )
    match = DbMatch(
        fdc_id="123",
        description="Test Food",
        per_100g=facts,
    )
    with pytest.raises(ValueError):
        MealItem(label="Test", grams=0, db_match=match)


def test_candidate_requires_label() -> None:
    with pytest.raises(ValueError):
        Candidate(label="   ", confidence=0.5)


def test_dbmatch_requires_fields() -> None:
    facts = NutritionFacts(
        calories_kcal=10,
        protein_g=1,
        carbs_g=2,
        fat_g=3,
    )
    with pytest.raises(ValueError):
        DbMatch(fdc_id="", description="Test", per_100g=facts)


def test_cache_entry_requires_image_hash() -> None:
    with pytest.raises(ValueError):
        CacheEntry(image_hash=" ", vision_response_json={})


def test_cache_entry_sets_created_at() -> None:
    entry = CacheEntry(image_hash="abc", vision_response_json={})
    assert entry.created_at
    datetime.fromisoformat(entry.created_at.replace("Z", "+00:00"))


def test_cache_entry_rejects_invalid_created_at() -> None:
    with pytest.raises(ValueError):
        CacheEntry(image_hash="abc", vision_response_json={}, created_at="not-a-date")
