import re
from pathlib import Path

from typer.testing import CliRunner

from macrocam.cli import app
from macrocam.models import DbMatch, NutritionFacts


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[mK]", "", text)


def _fake_match() -> DbMatch:
    facts = NutritionFacts(
        calories_kcal=200,
        protein_g=10,
        carbs_g=20,
        fat_g=5,
        sugars_g=3,
        fiber_g=2,
        sodium_mg=100,
        cholesterol_mg=30,
    )
    return DbMatch(
        fdc_id="12345",
        description="Test Food",
        per_100g=facts,
    )


def test_cli_noninteractive_success(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "meal.jpg"
    image_path.write_bytes(b"fake-image")

    monkeypatch.setattr("macrocam.cli.lookup_usda_food", lambda *args, **kwargs: _fake_match())

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            str(image_path),
            "--no-interactive",
            "--food",
            "Test Food",
            "--grams",
            "150",
        ],
    )

    assert result.exit_code == 0
    output = _strip_ansi(result.stdout)
    assert "Matched database item" in output
    assert "USDA FDC ID: 12345" in output
    assert "Macros (150g)" in output
    assert "Calories" in output
    assert "Protein" in output


def test_cli_noninteractive_requires_food_and_grams(tmp_path: Path) -> None:
    image_path = tmp_path / "meal.jpg"
    image_path.write_bytes(b"fake-image")

    runner = CliRunner()
    result = runner.invoke(app, [str(image_path), "--no-interactive", "--grams", "100"])
    assert result.exit_code == 1

    result = runner.invoke(app, [str(image_path), "--no-interactive", "--food", "Test Food"])
    assert result.exit_code == 1


def test_cli_noninteractive_invalid_grams(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "meal.jpg"
    image_path.write_bytes(b"fake-image")

    monkeypatch.setattr("macrocam.cli.lookup_usda_food", lambda *args, **kwargs: _fake_match())

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            str(image_path),
            "--no-interactive",
            "--food",
            "Test Food",
            "--grams",
            "10oz",
        ],
    )
    assert result.exit_code == 1
