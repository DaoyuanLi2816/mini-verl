"""Conversion refuses to publish a report describing bytes it never read.

Conversion streams the source over a long period and then publishes a report
claiming a `source_sha256`. If the file is replaced in between, that claim
describes a file the conversion never converted -- and the output rows come
from the old one. The source identity is captured up front and re-checked
before anything is published.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import miniverl.bridge.dataset as dataset_module
from miniverl.bridge.dataset import convert_dataset
from miniverl.errors import ConfigError


def _row(index: int) -> dict[str, Any]:
    return {
        "data_source": "calculator",
        "prompt": [{"role": "user", "content": f"{index}+{index}"}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": str(index * 2)},
        "extra_info": {"split": "train"},
    }


def _source(path: Path, *, rows: int) -> Path:
    pq.write_table(pa.Table.from_pylist([_row(index) for index in range(rows)]), path)
    return path


def _outputs(tmp_path: Path) -> list[str]:
    return sorted(item.name for item in tmp_path.iterdir() if item.name.startswith("out."))


def test_an_unchanged_source_converts(tmp_path: Path) -> None:
    source = _source(tmp_path / "train.parquet", rows=4)

    report = convert_dataset(source, out=tmp_path / "out.parquet", direction="from-verl-parquet")

    assert report["accepted_rows"] == 4
    assert report["source_sha256"]


def test_a_replaced_source_fails_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path / "train.parquet", rows=4)
    original = dataset_module._sha256

    def replace_then_hash(path: Path) -> str:
        # Fires while the conversion is computing the source digest, standing
        # in for another process rewriting the file mid-run.
        if path == source and not getattr(replace_then_hash, "done", False):
            replace_then_hash.done = True  # type: ignore[attr-defined]
            _source(source, rows=9)
        return original(path)

    monkeypatch.setattr(dataset_module, "_sha256", replace_then_hash)

    with pytest.raises(ConfigError) as caught:
        convert_dataset(source, out=tmp_path / "out.parquet", direction="from-verl-parquet")

    assert "changed during conversion" in str(caught.value)
    # Nothing from the aborted run survives: no Parquet, no sidecar, no report.
    assert _outputs(tmp_path) == []


def test_a_same_size_content_change_is_still_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """stat alone is not enough; the digest is the authoritative check."""
    source = _source(tmp_path / "train.parquet", rows=4)
    identity = dataset_module._source_identity(source)

    monkeypatch.setattr(dataset_module, "_source_identity", lambda path: identity)

    original = dataset_module._sha256
    calls: list[Path] = []

    def drifting_hash(path: Path) -> str:
        calls.append(path)
        if path == source and len(calls) > 1:
            return "f" * 64
        return original(path)

    monkeypatch.setattr(dataset_module, "_sha256", drifting_hash)

    with pytest.raises(ConfigError) as caught:
        convert_dataset(source, out=tmp_path / "out.parquet", direction="from-verl-parquet")

    assert "content digest" in str(caught.value)
    assert _outputs(tmp_path) == []


def test_an_unstattable_source_at_verification_time_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the source cannot be stat'd on the way out, nothing is published.

    The file is not actually deleted here: Windows keeps the Parquet handle
    open, so `unlink` would raise a platform error rather than exercise the
    code under test. Failing `stat` directly hits the same path everywhere.
    """
    source = _source(tmp_path / "train.parquet", rows=4)
    real_stat = Path.stat
    calls: list[int] = []

    def failing_stat(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self == source:
            calls.append(1)
            if len(calls) > 1:
                raise OSError(5, "simulated I/O error")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", failing_stat)

    with pytest.raises(ConfigError) as caught:
        convert_dataset(source, out=tmp_path / "out.parquet", direction="from-verl-parquet")

    assert "cannot stat source dataset" in str(caught.value)
    assert _outputs(tmp_path) == []


def test_the_reported_digest_is_the_verified_one(tmp_path: Path) -> None:
    """The report's source_sha256 is the digest that was checked, not a re-read."""
    import hashlib

    source = _source(tmp_path / "train.parquet", rows=3)
    expected = hashlib.sha256(source.read_bytes()).hexdigest()

    report = convert_dataset(source, out=tmp_path / "out.parquet", direction="from-verl-parquet")

    assert report["source_sha256"] == expected
