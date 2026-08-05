"""Transactional, collision-safe Parquet/sidecar/report publication."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from miniverl.errors import ConfigError


def _rows(*, with_extension: bool = True) -> list[dict[str, Any]]:
    row: dict[str, Any] = {
        "data_source": "calculator",
        "prompt": [
            {"role": "system", "content": "Use the calculator."},
            {"role": "user", "content": "What is 2 + 2?"},
        ],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": "4"},
        "extra_info": {"split": "train", "index": 0},
    }
    if with_extension:
        row["miniverl_extensions"] = {"token_provenance": {"schema": 1}}
    return [row]


def _source(tmp_path: Path, *, with_extension: bool = True) -> Path:
    source = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist(_rows(with_extension=with_extension)), source)
    return source


def _convert(source: Path, out: Path, **kwargs: Any) -> dict[str, Any]:
    from miniverl.bridge.dataset import convert_dataset

    kwargs.setdefault("direction", "from-verl-parquet")
    return convert_dataset(source, out=out, **kwargs)


# ------------------------------------------------------------------ naming


def test_conversion_publishes_one_coherent_output_family(tmp_path: Path) -> None:
    source = _source(tmp_path)
    out = tmp_path / "converted" / "train.parquet"
    report = _convert(source, out)

    assert out.is_file()
    assert (out.parent / "train.parquet.report.json").is_file()
    assert (out.parent / "train.parquet.miniverl.json").is_file()
    assert report["report_path"] == "train.parquet.report.json"
    assert sorted(p.name for p in out.parent.iterdir() if p.is_file()) == [
        "train.parquet",
        "train.parquet.miniverl.json",
        "train.parquet.report.json",
    ]


def test_sidecar_is_absent_when_not_required(tmp_path: Path) -> None:
    source = _source(tmp_path, with_extension=False)
    out = tmp_path / "converted" / "train.parquet"
    _convert(source, out)

    assert out.is_file()
    assert not (out.parent / "train.parquet.miniverl.json").exists()
    assert (out.parent / "train.parquet.report.json").is_file()


# -------------------------------------------------------------- collisions


def test_conversion_collision_fails_before_modifying_anything(tmp_path: Path) -> None:
    source = _source(tmp_path)
    out = tmp_path / "converted" / "train.parquet"
    _convert(source, out)
    parquet_before = out.read_bytes()
    report_before = (out.parent / "train.parquet.report.json").read_bytes()

    with pytest.raises(ConfigError) as excinfo:
        _convert(source, out)

    message = str(excinfo.value)
    assert "train.parquet" in message
    assert "--overwrite" in message
    assert out.read_bytes() == parquet_before
    assert (out.parent / "train.parquet.report.json").read_bytes() == report_before


def test_conversion_overwrite_replaces_the_whole_family(tmp_path: Path) -> None:
    source = _source(tmp_path)
    out = tmp_path / "converted" / "train.parquet"
    _convert(source, out)
    report = _convert(source, out, overwrite=True)

    on_disk = json.loads((out.parent / "train.parquet.report.json").read_text(encoding="utf-8"))
    assert on_disk == report
    assert on_disk["output_sha256"] == report["output_sha256"]


def test_overwrite_removes_a_stale_sidecar_only_on_success(tmp_path: Path) -> None:
    """A previous run's sidecar must not survive a run that produces none."""
    source_with = _source(tmp_path)
    out = tmp_path / "converted" / "train.parquet"
    _convert(source_with, out)
    assert (out.parent / "train.parquet.miniverl.json").is_file()

    plain = tmp_path / "plain.parquet"
    pq.write_table(pa.Table.from_pylist(_rows(with_extension=False)), plain)
    _convert(plain, out, overwrite=True)
    assert not (out.parent / "train.parquet.miniverl.json").exists()


# --------------------------------------------------------- fault injection


def test_report_write_failure_leaves_the_previous_parquet_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from miniverl.bridge import publish

    source = _source(tmp_path)
    out = tmp_path / "converted" / "train.parquet"
    _convert(source, out)
    parquet_before = out.read_bytes()
    sidecar_before = (out.parent / "train.parquet.miniverl.json").read_bytes()
    report_before = (out.parent / "train.parquet.report.json").read_bytes()

    def explode(self: Any, name: str, payload: Any) -> None:
        raise OSError("simulated sidecar/report failure")

    monkeypatch.setattr(publish.OutputTransaction, "write_json", explode)
    with pytest.raises(OSError, match="simulated sidecar/report failure"):
        _convert(source, out, overwrite=True)

    assert out.read_bytes() == parquet_before
    assert (out.parent / "train.parquet.miniverl.json").read_bytes() == sidecar_before
    assert (out.parent / "train.parquet.report.json").read_bytes() == report_before


def test_parquet_write_failure_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from miniverl.bridge import dataset

    source = _source(tmp_path)
    out = tmp_path / "converted" / "train.parquet"

    def explode(*args: Any, **kwargs: Any) -> None:
        raise OSError("simulated parquet write failure")

    monkeypatch.setattr(dataset, "_write_parquet", explode)
    with pytest.raises(ConfigError, match="cannot write Parquet"):
        _convert(source, out)

    assert not out.exists()
    assert not (out.parent / "train.parquet.report.json").exists()
    assert not (out.parent / "train.parquet.miniverl.json").exists()


def test_final_rename_failure_restores_the_previous_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from miniverl.bridge import publish

    source = _source(tmp_path)
    out = tmp_path / "converted" / "train.parquet"
    _convert(source, out)
    parquet_before = out.read_bytes()
    report_before = (out.parent / "train.parquet.report.json").read_bytes()

    calls = {"count": 0}
    real_replace = publish._replace

    def flaky(src: Path, dst: Path) -> None:
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("simulated rename failure")
        real_replace(src, dst)

    monkeypatch.setattr(publish, "_replace", flaky)
    with pytest.raises(OSError, match="simulated rename failure"):
        _convert(source, out, overwrite=True)

    assert out.read_bytes() == parquet_before
    assert (out.parent / "train.parquet.report.json").read_bytes() == report_before


def test_no_staging_directory_survives_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from miniverl.bridge import publish

    source = _source(tmp_path)
    out = tmp_path / "converted" / "train.parquet"

    def explode(self: Any, name: str, payload: Any) -> None:
        raise OSError("boom")

    monkeypatch.setattr(publish.OutputTransaction, "write_json", explode)
    with pytest.raises(OSError):
        _convert(source, out)

    leftovers = [p.name for p in out.parent.iterdir() if p.name != ".miniverl-locks"]
    assert leftovers == []


# ------------------------------------------------------------ concurrency


def test_concurrent_conversions_to_one_stem_stay_coherent(tmp_path: Path) -> None:
    source_a = _source(tmp_path)
    source_b = tmp_path / "other.parquet"
    rows = _rows(with_extension=False)
    rows[0]["reward_model"] = {"style": "rule", "ground_truth": "5"}
    pq.write_table(pa.Table.from_pylist(rows), source_b)

    out = tmp_path / "converted" / "train.parquet"
    start = threading.Barrier(2, timeout=30)
    errors: list[BaseException] = []

    def worker(source: Path) -> None:
        try:
            start.wait()
            _convert(source, out, overwrite=True, lock_timeout=30.0)
        except BaseException as exc:  # pragma: no cover - surfaced by assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(src,)) for src in (source_a, source_b)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert not errors, errors
    import hashlib

    report = json.loads((out.parent / "train.parquet.report.json").read_text(encoding="utf-8"))
    assert report["output_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    sidecar = out.parent / "train.parquet.miniverl.json"
    assert sidecar.is_file() == (report["extension_sidecar"] is not None)
