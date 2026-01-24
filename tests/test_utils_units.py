import pytest

from macrocam.utils import parse_grams


@pytest.mark.parametrize("raw, expected", [("250", 250.0), ("250g", 250.0), (" 250 g ", 250.0)])
def test_parse_grams_valid(raw: str, expected: float) -> None:
    assert parse_grams(raw) == expected


@pytest.mark.parametrize("raw", ["0", "-1", "10oz", "abc", ""])
def test_parse_grams_invalid(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_grams(raw)
