import pytest

from macrocam.vision import (
    _extract_text_from_gemini_response,
    normalize_candidates,
    parse_vision_json_with_retry,
)


def test_parse_vision_json_with_retry() -> None:
    primary = "{bad json"
    retry = '{"items":[{"label":"Salad","confidence":0.9,"notes":""}],"overall_notes":""}'
    parsed = parse_vision_json_with_retry(primary, retry)
    assert parsed["items"][0]["label"] == "Salad"


def test_normalize_candidates_clamps_and_pads() -> None:
    raw = {"items": [{"label": "  ", "confidence": 1.5, "notes": "test"}]}
    result = normalize_candidates(raw)
    assert len(result.items) == 3
    assert result.items[0].label == "Unknown"
    assert result.items[0].confidence == 1.0


def test_extract_text_from_gemini_response() -> None:
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": '{"items":[{"label":"Soup","confidence":0.7,"notes":""}],'
                            '"overall_notes":"warm"}'
                        }
                    ]
                }
            }
        ]
    }
    assert _extract_text_from_gemini_response(payload).startswith("{")


def test_extract_text_from_gemini_response_missing_text() -> None:
    payload = {"candidates": [{"content": {"parts": [{"foo": "bar"}]}}]}
    with pytest.raises(ValueError):
        _extract_text_from_gemini_response(payload)
