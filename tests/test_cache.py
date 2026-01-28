from macrocam.cache import get_vision_cache, set_vision_cache


def test_cache_roundtrip(tmp_path) -> None:
    image_hash = "abc123"
    payload = {"items": [{"label": "Test", "confidence": 0.9, "notes": ""}]}

    set_vision_cache(image_hash, payload, base_dir=tmp_path)
    loaded = get_vision_cache(image_hash, base_dir=tmp_path)

    assert loaded == payload


def test_cache_missing_returns_none(tmp_path) -> None:
    assert get_vision_cache("missing", base_dir=tmp_path) is None


def test_cache_corrupt_json_returns_none(tmp_path) -> None:
    image_hash = "broken"
    cache_path = tmp_path / f"{image_hash}.json"
    cache_path.write_text("{not-json")

    assert get_vision_cache(image_hash, base_dir=tmp_path) is None
    assert not cache_path.exists()
