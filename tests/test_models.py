import pytest

from macrocam.models import Candidate, DbMatch, MealItem, NutritionFacts


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
