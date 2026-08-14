"""Parquet bounds must be enforced while reading, not after.

Up to v0.6.2 the "bounded" dataset scan called ``pq.read_table(path)`` and
``table.to_pylist()`` first and applied ``max_rows``/``max_bytes`` to the
already-materialized result, and the schema check read every row to learn its
column names. Both would stall or OOM on a real verl-scale dataset.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from miniverl.bridge.doctor import _check_parquet, _check_privacy, _scan_dataset_text

SECRET = "AKIAIOSFODNN7EXAMPLE"


# ------------------------------------------------------------------ fixtures


def _row(index: int, *, content: str | None = None) -> dict[str, Any]:
    return {
        "data_source": "calc",
        "prompt": [{"role": "user", "content": content or f"question {index}"}],
        "ability": "math",
        "reward_model": {"ground_truth": str(index)},
        "extra_info": None,
    }


def _write_many_row_groups(
    path: Path, *, groups: int = 8, per_group: int = 4, secret_in_group: int | None = None
) -> Path:
    """One Parquet file with a known, verifiable row-group layout."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(path, pa.Table.from_pylist([_row(0)]).schema)
    try:
        for group in range(groups):
            rows = []
            for offset in range(per_group):
                index = group * per_group + offset
                content = SECRET if group == secret_in_group and offset == 0 else None
                rows.append(_row(index, content=content))
            writer.write_table(pa.Table.from_pylist(rows))
    finally:
        writer.close()
    return path


def _bundle(tmp_path: Path, **kwargs: Any) -> Path:
    data = tmp_path / "data"
    _write_many_row_groups(data / "train.parquet", **kwargs)
    return tmp_path


class _RowGroupRecorder:
    """Records exactly which row groups the scanner asks pyarrow to decode."""

    def __init__(self) -> None:
        self.requested: list[int] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        original = pq.ParquetFile.iter_batches
        recorder = self

        def _iter_batches(self_: Any, *args: Any, **kwargs: Any) -> Any:
            recorder.requested.extend(kwargs.get("row_groups") or [])
            return original(self_, *args, **kwargs)

        monkeypatch.setattr(pq.ParquetFile, "iter_batches", _iter_batches)


# ------------------------------------------------------- schema check bounds


def test_schema_check_reads_no_row_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Column names and row counts live in the footer."""
    root = _bundle(tmp_path)
    _write_many_row_groups(root / "data" / "val.parquet")

    def _forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the schema check must not decode row data")

    monkeypatch.setattr(pq, "read_table", _forbidden)
    monkeypatch.setattr(pq.ParquetFile, "read_row_group", _forbidden)
    monkeypatch.setattr(pq.ParquetFile, "iter_batches", _forbidden)

    check = _check_parquet(root)

    assert check["status"] == "ok"
    assert check["rows"]["train"] == 32
    assert "footer metadata only" in check["read_scope"]


def test_reward_free_opd_schema_does_not_require_reward_model(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    row = _row(0)
    row.pop("reward_model")
    pq.write_table(pa.Table.from_pylist([row]), data / "train.parquet")

    assert _check_parquet(tmp_path)["status"] == "fail"
    assert _check_parquet(tmp_path, require_reward_model=False)["status"] == "ok"


# --------------------------------------------------------- streaming bounds


def test_row_bound_stops_after_the_first_row_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The instrumented proof: later row groups are never requested."""
    root = _bundle(tmp_path, groups=8, per_group=4)
    recorder = _RowGroupRecorder()
    recorder.install(monkeypatch)

    report = _scan_dataset_text(root, sentinels=(), max_rows=2, max_bytes=1 << 30)

    assert recorder.requested == [0], "the scanner decoded row groups past its bound"
    assert report["row_groups_read"] == 1
    assert report["rows_scanned"] == 2
    assert report["rows_total"] == 32
    assert report["scan_scope"] == "sampled"


def test_byte_bound_stops_mid_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _bundle(tmp_path, groups=8, per_group=4)
    recorder = _RowGroupRecorder()
    recorder.install(monkeypatch)

    report = _scan_dataset_text(root, sentinels=(), max_rows=10_000, max_bytes=40)

    assert len(recorder.requested) < 8
    assert report["scan_scope"] == "sampled"
    assert report["bytes_scanned"] >= 40
    assert report["rows_total"] == 32


def test_full_scan_reads_every_row_group_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _bundle(tmp_path, groups=4, per_group=2)
    recorder = _RowGroupRecorder()
    recorder.install(monkeypatch)

    report = _scan_dataset_text(root, sentinels=(), max_rows=1000, max_bytes=1 << 30)

    assert recorder.requested == [0, 1, 2, 3]
    assert report["scan_scope"] == "full"
    assert report["rows_scanned"] == 8
    assert report["row_groups_read"] == 4


def test_second_file_is_not_decoded_once_bounds_are_reached(tmp_path: Path) -> None:
    """Totals still come from the footer, but nothing more is read."""
    root = _bundle(tmp_path, groups=4, per_group=4)
    _write_many_row_groups(root / "data" / "val.parquet", groups=4, per_group=4)

    report = _scan_dataset_text(root, sentinels=(), max_rows=1, max_bytes=1 << 30)

    assert report["files_total"] == 2
    assert report["files_inspected"] == 1
    assert report["rows_total"] == 32
    assert report["scan_scope"] == "sampled"


# ------------------------------------------------------------- detection


def test_nested_prompt_values_are_still_scanned(tmp_path: Path) -> None:
    """Secrets hide inside the nested chat structure, not only flat columns."""
    root = _bundle(tmp_path, groups=2, per_group=2, secret_in_group=0)

    report = _scan_dataset_text(root, sentinels=(), max_rows=1000, max_bytes=1 << 30)

    assert report["status"] == "failed"
    categories = {finding["category"] for finding in report["findings"]}
    assert "aws_access_key_id" in categories
    assert {finding["column"] for finding in report["findings"]} == {"prompt"}


def test_findings_never_contain_the_matched_text(tmp_path: Path) -> None:
    root = _bundle(tmp_path, groups=2, per_group=2, secret_in_group=1)

    report = _scan_dataset_text(root, sentinels=(SECRET,), max_rows=1000, max_bytes=1 << 30)

    serialized = json.dumps(report)
    assert SECRET not in serialized
    assert report["status"] == "failed"


def test_row_index_is_absolute_within_the_file(tmp_path: Path) -> None:
    root = _bundle(tmp_path, groups=4, per_group=4, secret_in_group=2)

    report = _scan_dataset_text(root, sentinels=(), max_rows=1000, max_bytes=1 << 30)

    # Group 2, offset 0 -> absolute row 8.
    assert {finding["row"] for finding in report["findings"]} == {8}


# ------------------------------------------------------------ failure modes


def test_unreadable_row_group_fails_without_partial_findings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _bundle(tmp_path, groups=4, per_group=2, secret_in_group=0)

    def _explode(self_: Any, *args: Any, **kwargs: Any) -> Any:
        raise OSError("simulated corrupt row group")

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", _explode)

    report = _scan_dataset_text(root, sentinels=(), max_rows=1000, max_bytes=1 << 30)

    assert report["status"] == "failed"
    assert "row group 0" in report["reason"]
    assert report["findings"] == []


def test_empty_parquet_is_handled(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([_row(0)]).schema.empty_table(), data / "train.parquet")

    report = _scan_dataset_text(tmp_path, sentinels=(), max_rows=1000, max_bytes=1 << 30)

    assert report["status"] == "passed"
    assert report["rows_total"] == 0
    assert report["scan_scope"] == "full"


def test_missing_pyarrow_reports_not_inspected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
    report = _scan_dataset_text(tmp_path, sentinels=(), max_rows=10, max_bytes=10)
    assert report["status"] == "not_inspected"
    assert report["findings"] == []


def test_no_parquet_file_reports_not_inspected(tmp_path: Path) -> None:
    report = _scan_dataset_text(tmp_path, sentinels=(), max_rows=10, max_bytes=10)
    assert report["status"] == "not_inspected"


# --------------------------------------------------------- privacy wiring


def test_privacy_scan_surfaces_streaming_counters(tmp_path: Path) -> None:
    root = _bundle(tmp_path, groups=4, per_group=4)

    privacy = _check_privacy(root, scan_dataset_text=True, max_rows=4, max_bytes=1 << 30)

    scan = privacy["dataset_scan"]
    assert scan["scan_scope"] == "sampled"
    assert scan["rows_scanned"] == 4
    assert scan["rows_total"] == 16
    assert scan["row_groups_read"] == 1
    assert privacy["dataset_content_privacy"] == "passed"
    # An uninspected scope is never promoted.
    assert privacy["model_weight_privacy"] == "not_inspected"
