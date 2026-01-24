from pathlib import Path

from macrocam.utils import sha256_bytes, sha256_file


def test_sha256_file_matches_bytes(tmp_path: Path) -> None:
    data = b"macrocam-test"
    path = tmp_path / "sample.bin"
    path.write_bytes(data)

    assert sha256_file(path) == sha256_bytes(data)
