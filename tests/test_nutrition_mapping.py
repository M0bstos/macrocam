import csv
from pathlib import Path

from macrocam.nutrition import lookup_usda_food, search_usda_foods


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _setup_usda_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "usda_fdc"
    data_dir.mkdir()

    _write_csv(
        data_dir / "nutrient.csv",
        ["id", "name", "unit_name", "nutrient_nbr", "rank"],
        [
            ["1008", "Energy", "KCAL", "208", "300.0"],
            ["1003", "Protein", "G", "203", "600.0"],
            ["1005", "Carbohydrate, by difference", "G", "205", "1110.0"],
            ["1004", "Total lipid (fat)", "G", "204", "800.0"],
            ["2000", "Total Sugars", "G", "269", "1510.0"],
            ["1079", "Fiber, total dietary", "G", "291", "1200.0"],
            ["1093", "Sodium, Na", "MG", "307", "5800.0"],
            ["1253", "Cholesterol", "MG", "601", "15700.0"],
        ],
    )

    _write_csv(
        data_dir / "food.csv",
        ["fdc_id", "data_type", "description", "food_category_id", "publication_date"],
        [
            ["1", "foundation_food", "Test Food", "Test Category", "2024-01-01"],
            ["2", "branded_food", "Other Item", "Other Category", "2024-01-01"],
        ],
    )

    _write_csv(
        data_dir / "food_nutrient.csv",
        [
            "id",
            "fdc_id",
            "nutrient_id",
            "amount",
            "data_points",
            "derivation_id",
            "min",
            "max",
            "median",
            "loq",
            "footnote",
            "min_year_acquired",
            "percent_daily_value",
        ],
        [
            ["1", "1", "1008", "250", "", "", "", "", "", "", "", "", ""],
            ["2", "1", "1003", "10", "", "", "", "", "", "", "", "", ""],
            ["3", "1", "1005", "20", "", "", "", "", "", "", "", "", ""],
            ["4", "1", "1004", "5", "", "", "", "", "", "", "", "", ""],
            ["5", "1", "2000", "3", "", "", "", "", "", "", "", "", ""],
            ["6", "1", "1079", "2", "", "", "", "", "", "", "", "", ""],
            ["7", "1", "1093", "100", "", "", "", "", "", "", "", "", ""],
            ["8", "1", "1253", "30", "", "", "", "", "", "", "", "", ""],
        ],
    )

    return data_dir


def test_search_and_lookup_usda_food(tmp_path: Path) -> None:
    data_dir = _setup_usda_dir(tmp_path)

    matches = search_usda_foods("test food", data_dir)
    assert matches
    assert matches[0].fdc_id == "1"

    match = lookup_usda_food("test food", data_dir)
    facts = match.per_100g

    assert match.fdc_id == "1"
    assert match.description == "Test Food"
    assert facts.calories_kcal == 250
    assert facts.protein_g == 10
    assert facts.carbs_g == 20
    assert facts.fat_g == 5
    assert facts.sugars_g == 3
    assert facts.fiber_g == 2
    assert facts.sodium_mg == 100
    assert facts.cholesterol_mg == 30


def test_lookup_usda_food_with_fallback_queries(tmp_path: Path) -> None:
    data_dir = _setup_usda_dir(tmp_path)

    match = lookup_usda_food(
        "no match here",
        data_dir,
        fallback_queries=["test food"],
        use_llm_fallback=False,
    )

    assert match.fdc_id == "1"
    assert match.description == "Test Food"
