"""Ensure replay cycling cannot conceal malformed or changed physical samples."""

import hashlib
import json
from pathlib import Path
import sys

import pytest

from replay_sampler import Sampler


@pytest.mark.skipif(
    sys.platform != "linux", reason="Allocator trimming is Linux/glibc-specific"
)
def test_heap_trimming_preserves_samples_and_records_every_call(tmp_path: Path):
    import ctypes

    if not hasattr(ctypes.CDLL(None), "malloc_trim"):
        pytest.skip("glibc malloc_trim is unavailable")
    path = tmp_path / "shot.b8"
    path.write_bytes(b"\x13")
    log = tmp_path / "allocator.jsonl"
    config = {"files": [str(path)], "num_measurements": 8}
    with pytest.raises(ValueError, match="memory_log"):
        Sampler("", config | {"trim_heap": True})
    with pytest.raises(ValueError, match="boolean"):
        Sampler("", config | {"trim_heap": "yes"})
    sampler = Sampler("", config | {"trim_heap": True, "memory_log": str(log)})
    assert [sampler.sample(), sampler.sample()] == [b"\x13", b"\x13"]
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert [row["sample_call"] for row in records] == [0, 1]
    assert all(
        row["rss_before_bytes"] > 0 and row["rss_after_bytes"] > 0 for row in records
    )
    assert all(row["trim_seconds"] >= 0 for row in records)


def test_warmups_cycle_into_ten_measured_shots(tmp_path: Path):
    files = []
    values = [bytes([index]) for index in range(10)]
    for index, value in enumerate(values):
        path = tmp_path / f"{index}.b8"
        path.write_bytes(value)
        files.append(str(path))
    sampler = Sampler(
        "",
        {
            "files": files,
            "num_measurements": 8,
            "sha256": [hashlib.sha256(value).hexdigest() for value in values],
        },
    )
    assert [sampler.sample() for _ in range(12)] == values + values[:2]
    shifted = Sampler(
        "", {"files": files, "num_measurements": 8, "skip_shots": 2, "offset": 1}
    )
    assert shifted.sample() == values[3]


@pytest.mark.parametrize("data", [b"", b"\x00\x00\x00", b"\x00\x04"])
def test_changed_size_or_padding_is_rejected(tmp_path: Path, data: bytes):
    path = tmp_path / "shot.b8"
    path.write_bytes(b"\x00\x00")
    sampler = Sampler("", {"files": [str(path)], "num_measurements": 10})
    path.write_bytes(data)
    with pytest.raises(ValueError):
        sampler.sample()
    assert sampler.index == 0


def test_digest_mismatch_is_not_replayed(tmp_path: Path):
    path = tmp_path / "shot.b8"
    path.write_bytes(b"\x01")
    sampler = Sampler(
        "",
        {
            "files": [str(path)],
            "num_measurements": 8,
            "sha256": [hashlib.sha256(b"\x00").hexdigest()],
        },
    )
    with pytest.raises(ValueError, match="SHA-256"):
        sampler.sample()
