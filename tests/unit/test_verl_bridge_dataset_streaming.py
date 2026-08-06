"""Strict sidecar input contract and streaming Parquet conversion.

Two defects survived into the v0.6.3 release candidate:

* an existing ``.miniverl.json`` sidecar was accepted whenever it parsed as a
  JSON object, so ``{}``, ``{"rows": []}`` and a wrong-namespace file were all
  silently treated as "this dataset has no extensions" -- losing provenance
  rather than failing closed;
* conversion called ``pq.read_table(source).to_pylist()``, materializing the
  entire dataset and every Python object before converting a single row.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from miniverl.bridge.dataset import convert_dataset
from miniverl.errors import ConfigError


def _row(index: int, *, extension: Any = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "data_source": "calculator",
        "prompt": [{"role": "user", "content": f"{index}+{index}"}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": str(index * 2)},
        "extra_info": {"split": "train"},
    }
    if extension is not None:
        row["extra_info"] = {"split": "train", "miniverl": extension}
    return row


def _write(path: Path, rows: list[dict[str, Any]], *, row_group_size: int | None = None) -> Path:
    table = pa.Table.from_pylist(rows)
    if row_group_size is None:
        pq.write_table(table, path)
    else:
        pq.write_table(table, path, row_group_size=row_group_size)
    return path


def _sidecar(path: Path, payload: Any) -> None:
    path.with_suffix(path.suffix + ".miniverl.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


# ------------------------------------------------------------- sidecar contract


@pytest.mark.parametrize(
    ("name", "payload"),
    [
        ("empty_object", {}),
        ("rows_is_a_list", {"schema_version": 1, "namespace": "extra_info.miniverl", "rows": []}),
        ("missing_rows", {"schema_version": 1, "namespace": "extra_info.miniverl"}),
        (
            "unsupported_version",
            {"schema_version": 999, "namespace": "extra_info.miniverl", "rows": {}},
        ),
        ("wrong_namespace", {"schema_version": 1, "namespace": "wrong", "rows": {}}),
        (
            "non_canonical_key",
            {"schema_version": 1, "namespace": "extra_info.miniverl", "rows": {"01": {"a": 1}}},
        ),
        (
            "negative_key",
            {"schema_version": 1, "namespace": "extra_info.miniverl", "rows": {"-1": {"a": 1}}},
        ),
        (
            "non_integer_key",
            {"schema_version": 1, "namespace": "extra_info.miniverl", "rows": {"x": {"a": 1}}},
        ),
        (
            "out_of_bounds_key",
            {"schema_version": 1, "namespace": "extra_info.miniverl", "rows": {"99": {"a": 1}}},
        ),
        (
            "unknown_critical_field",
            {
                "schema_version": 1,
                "namespace": "extra_info.miniverl",
                "rows": {},
                "encryption": "aes",
            },
        ),
        ("not_an_object", [1, 2, 3]),
    ],
)
def test_malformed_sidecar_fails_closed(tmp_path: Path, name: str, payload: Any) -> None:
    source = _write(tmp_path / "train.parquet", [_row(0), _row(1)])
    _sidecar(source, payload)

    with pytest.raises(ConfigError) as error:
        convert_dataset(source, out=tmp_path / "out" / "train.parquet", direction="to-verl-parquet")

    assert "sidecar" in str(error.value).lower(), name
    assert not (tmp_path / "out" / "train.parquet").exists(), name


def test_valid_v1_sidecar_from_a_public_release_still_reads(tmp_path: Path) -> None:
    """Exactly the shape miniVERL 0.6.0-0.6.2 published."""
    source = _write(tmp_path / "train.parquet", [_row(0), _row(1)])
    _sidecar(
        source,
        {
            "schema_version": 1,
            "namespace": "extra_info.miniverl",
            "semantics": "miniVERL token provenance and teacher targets",
            "rows": {"0": {"teacher": "t"}},
        },
    )

    report = convert_dataset(
        source, out=tmp_path / "out" / "train.parquet", direction="to-verl-parquet"
    )

    assert report["accepted_rows"] == 2
    written = pq.read_table(tmp_path / "out" / "train.parquet").to_pylist()
    assert written[0]["extra_info"]["miniverl"] == {"teacher": "t"}


def test_optional_v2_binding_fields_are_accepted_and_checked(tmp_path: Path) -> None:
    from miniverl.bridge.dataset import _sha256

    source = _write(tmp_path / "train.parquet", [_row(0)])
    _sidecar(
        source,
        {
            "schema_version": 1,
            "namespace": "extra_info.miniverl",
            "rows": {"0": {"teacher": "t"}},
            "source_sha256": _sha256(source),
            "source_rows": 1,
        },
    )

    report = convert_dataset(
        source, out=tmp_path / "out" / "train.parquet", direction="to-verl-parquet"
    )

    assert report["accepted_rows"] == 1


def test_sidecar_bound_to_a_different_source_fails_closed(tmp_path: Path) -> None:
    source = _write(tmp_path / "train.parquet", [_row(0)])
    _sidecar(
        source,
        {
            "schema_version": 1,
            "namespace": "extra_info.miniverl",
            "rows": {"0": {"teacher": "t"}},
            "source_sha256": "0" * 64,
        },
    )

    with pytest.raises(ConfigError) as error:
        convert_dataset(source, out=tmp_path / "out" / "train.parquet", direction="to-verl-parquet")

    assert "source_sha256" in str(error.value)


def test_sidecar_values_are_never_printed_in_diagnostics(tmp_path: Path) -> None:
    secret = "TEACHER_LOGIT_SECRET_123456"
    source = _write(tmp_path / "train.parquet", [_row(0)])
    _sidecar(
        source,
        {"schema_version": 1, "namespace": "extra_info.miniverl", "rows": {"5": {"t": secret}}},
    )

    with pytest.raises(ConfigError) as error:
        convert_dataset(source, out=tmp_path / "out" / "train.parquet", direction="to-verl-parquet")

    assert secret not in str(error.value)


# ---------------------------------------------------------------- streaming


def test_conversion_never_materializes_the_whole_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`read_table` reads every row group at once; conversion must not call it."""
    source = _write(tmp_path / "train.parquet", [_row(i) for i in range(64)], row_group_size=8)

    def _forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("conversion materialized the whole table via read_table")

    monkeypatch.setattr(pq, "read_table", _forbidden)

    report = convert_dataset(
        source, out=tmp_path / "out" / "train.parquet", direction="to-verl-parquet"
    )

    assert report["accepted_rows"] == 64


def test_strict_failure_stops_before_reading_later_row_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [_row(index) for index in range(64)]
    rows[9]["data_source"] = ""  # invalid, lands in the second row group
    source = _write(tmp_path / "train.parquet", rows, row_group_size=8)

    seen: list[int] = []
    original = pq.ParquetFile.iter_batches

    def _counting(self: Any, *args: Any, **kwargs: Any) -> Any:
        for index, batch in enumerate(original(self, *args, **kwargs)):
            seen.append(index)
            yield batch

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", _counting)

    with pytest.raises(ConfigError):
        convert_dataset(source, out=tmp_path / "out" / "train.parquet", direction="to-verl-parquet")

    # The whole file is 8 batches; the failure is in the second one.
    assert len(seen) <= 2
    assert not (tmp_path / "out" / "train.parquet").exists()


def test_many_row_groups_convert_deterministically(tmp_path: Path) -> None:
    source = _write(tmp_path / "train.parquet", [_row(i) for i in range(200)], row_group_size=7)

    first = convert_dataset(
        source, out=tmp_path / "a" / "train.parquet", direction="to-verl-parquet"
    )
    second = convert_dataset(
        source, out=tmp_path / "b" / "train.parquet", direction="to-verl-parquet"
    )

    assert first["accepted_rows"] == 200
    assert first["output_sha256"] == second["output_sha256"]


def test_optional_nested_field_appearing_only_in_a_later_row_group(tmp_path: Path) -> None:
    """Per-batch type inference would produce incompatible schemas here."""
    rows = [_row(index) for index in range(16)]
    for row in rows:
        row["extra_info"] = {"split": "train", "note": None}
    rows[15]["extra_info"] = {"split": "train", "note": "only the last row has this"}
    source = _write(tmp_path / "train.parquet", rows, row_group_size=4)

    report = convert_dataset(
        source, out=tmp_path / "out" / "train.parquet", direction="to-verl-parquet"
    )

    assert report["accepted_rows"] == 16
    written = pq.read_table(tmp_path / "out" / "train.parquet").to_pylist()
    assert written[15]["extra_info"]["note"] == "only the last row has this"


def test_partial_conversion_still_requires_the_explicit_option(tmp_path: Path) -> None:
    rows = [_row(index) for index in range(10)]
    rows[3]["reward_model"] = {"style": "rule"}
    source = _write(tmp_path / "train.parquet", rows, row_group_size=4)

    with pytest.raises(ConfigError):
        convert_dataset(source, out=tmp_path / "out" / "train.parquet", direction="to-verl-parquet")

    report = convert_dataset(
        source,
        out=tmp_path / "out" / "train.parquet",
        direction="to-verl-parquet",
        allow_rejected_rows=True,
    )
    assert report["complete_dataset_conversion"] is False
    assert report["accepted_rows"] == 9
    assert report["rejected_rows"] == 1
    assert report["source_row_indices"]["3"] == 4


def test_rejection_detail_is_bounded(tmp_path: Path) -> None:
    from miniverl.bridge.dataset import MAX_REPORTED_REJECTIONS

    rows = [_row(index) for index in range(MAX_REPORTED_REJECTIONS + 40)]
    for row in rows[:-1]:
        row["data_source"] = ""  # valid Arrow type, invalid per the prompt schema
    source = _write(tmp_path / "train.parquet", rows, row_group_size=16)

    report = convert_dataset(
        source,
        out=tmp_path / "out" / "train.parquet",
        direction="to-verl-parquet",
        allow_rejected_rows=True,
    )

    assert report["rejected_rows"] == MAX_REPORTED_REJECTIONS + 39
    assert len(report["rejections"]) == MAX_REPORTED_REJECTIONS
    assert report["rejections_truncated"] is True
