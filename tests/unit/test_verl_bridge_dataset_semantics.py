"""Conversion fails closed on ambiguity and never calls a partial run lossless.

A row can carry miniVERL extension data in three independent places. Up to
v0.6.2 the converter silently preferred one and dropped the others, and it
published the accepted rows of a partially invalid dataset while the module
described itself as lossless.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from miniverl.errors import ConfigError

SECRET_TARGET = "teacher-logits-do-not-print-me"


# ------------------------------------------------------------------ fixtures


def _extension(marker: str = SECRET_TARGET) -> dict[str, Any]:
    return {"teacher_targets": {"representation": "top_k_plus_tail", "note": marker}}


def _row(
    *,
    content: str = "2+2",
    ground_truth: str = "4",
    nested: dict[str, Any] | None = None,
    top_level: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "data_source": "calc",
        "prompt": [{"role": "user", "content": content}],
        "ability": "math",
        "reward_model": {"ground_truth": ground_truth},
        "extra_info": {"miniverl": nested} if nested is not None else None,
    }
    if top_level is not None:
        row["miniverl_extensions"] = top_level
    return row


def _invalid_row() -> dict[str, Any]:
    row = _row()
    row["reward_model"] = {"ground_truth": None}
    return row


def _write(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # pyarrow drops a key that is absent from the first row, so give every row
    # the same key set and let the value be null.
    columns = {key for row in rows for key in row}
    padded = [{key: row.get(key) for key in sorted(columns)} for row in rows]
    pq.write_table(pa.Table.from_pylist(padded), path)
    return path


def _write_sidecar(source: Path, rows: dict[str, Any]) -> Path:
    sidecar = source.with_suffix(source.suffix + ".miniverl.json")
    sidecar.write_text(
        json.dumps({"schema_version": 1, "namespace": "extra_info.miniverl", "rows": rows}),
        encoding="utf-8",
    )
    return sidecar


def _convert(source: Path, out: Path, **kwargs: Any) -> dict[str, Any]:
    from miniverl.bridge.dataset import convert_dataset

    kwargs.setdefault("direction", "from-verl-parquet")
    return convert_dataset(source, out=out, **kwargs)


# ------------------------------------------------- extension source conflicts


def test_single_source_is_accepted(tmp_path: Path) -> None:
    source = _write(tmp_path / "in.parquet", [_row(nested=_extension())])
    report = _convert(source, tmp_path / "out.parquet")
    assert report["accepted_rows"] == 1
    assert report["extension_deduplication"] == []


def test_equal_top_level_and_sidecar_deduplicate(tmp_path: Path) -> None:
    source = _write(tmp_path / "in.parquet", [_row(top_level=_extension())])
    _write_sidecar(source, {"0": _extension()})

    report = _convert(source, tmp_path / "out.parquet")

    assert report["accepted_rows"] == 1
    assert report["extension_deduplication"] == [
        {"row": 0, "sources": ["miniverl_extensions", "sidecar"]}
    ]


def test_three_equal_sources_deduplicate(tmp_path: Path) -> None:
    source = _write(tmp_path / "in.parquet", [_row(nested=_extension(), top_level=_extension())])
    _write_sidecar(source, {"0": _extension()})

    report = _convert(source, tmp_path / "out.parquet")

    assert report["extension_deduplication"] == [
        {"row": 0, "sources": ["extra_info.miniverl", "miniverl_extensions", "sidecar"]}
    ]


def test_top_level_versus_sidecar_conflict_fails_closed(tmp_path: Path) -> None:
    source = _write(tmp_path / "in.parquet", [_row(top_level=_extension("A"))])
    _write_sidecar(source, {"0": _extension("B")})

    with pytest.raises(ConfigError) as excinfo:
        _convert(source, tmp_path / "out.parquet")

    message = str(excinfo.value)
    assert "row 0" in message
    assert "miniverl_extensions" in message and "sidecar" in message
    assert not (tmp_path / "out.parquet").exists()


def test_sidecar_versus_nested_conflict_fails_closed(tmp_path: Path) -> None:
    source = _write(tmp_path / "in.parquet", [_row(nested=_extension("A"))])
    _write_sidecar(source, {"0": _extension("B")})

    with pytest.raises(ConfigError) as excinfo:
        _convert(source, tmp_path / "out.parquet")

    assert "extra_info.miniverl" in str(excinfo.value)


def test_three_conflicting_sources_fail_closed(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "in.parquet", [_row(nested=_extension("A"), top_level=_extension("B"))]
    )
    _write_sidecar(source, {"0": _extension("C")})

    with pytest.raises(ConfigError) as excinfo:
        _convert(source, tmp_path / "out.parquet")

    message = str(excinfo.value)
    for name in ("extra_info.miniverl", "miniverl_extensions", "sidecar"):
        assert name in message


def test_conflict_diagnostics_never_print_extension_values(tmp_path: Path) -> None:
    """Extension payloads carry teacher targets; only names and indices leak."""
    source = _write(tmp_path / "in.parquet", [_row(top_level=_extension(SECRET_TARGET))])
    _write_sidecar(source, {"0": _extension("different")})

    with pytest.raises(ConfigError) as excinfo:
        _convert(source, tmp_path / "out.parquet")

    message = str(excinfo.value)
    assert SECRET_TARGET not in message
    assert "top_k_plus_tail" not in message


def test_conflict_on_a_late_row_rolls_back_and_keeps_the_previous_family(
    tmp_path: Path,
) -> None:
    """A conflict found after valid rows must publish nothing at all."""
    out = tmp_path / "out.parquet"
    clean = _write(tmp_path / "clean.parquet", [_row()])
    _convert(clean, out)
    previous_parquet = out.read_bytes()
    previous_report = (tmp_path / "out.parquet.report.json").read_bytes()

    source = _write(tmp_path / "in.parquet", [_row(), _row(), _row(top_level=_extension("A"))])
    _write_sidecar(source, {"2": _extension("B")})

    with pytest.raises(ConfigError):
        _convert(source, out, overwrite=True)

    assert out.read_bytes() == previous_parquet
    assert (tmp_path / "out.parquet.report.json").read_bytes() == previous_report
    assert not list(tmp_path.glob("*.staging"))


# ------------------------------------------------------------ row-loss policy


def test_invalid_row_fails_conversion_by_default(tmp_path: Path) -> None:
    source = _write(tmp_path / "in.parquet", [_row(), _invalid_row()])

    with pytest.raises(ConfigError) as excinfo:
        _convert(source, tmp_path / "out.parquet")

    message = str(excinfo.value)
    assert "1 of 2 source row(s) failed validation" in message
    assert "--allow-rejected-rows" in message
    assert not (tmp_path / "out.parquet").exists()


def test_explicit_partial_mode_is_labelled_incomplete(tmp_path: Path) -> None:
    source = _write(tmp_path / "in.parquet", [_row(), _invalid_row(), _row()])

    report = _convert(source, tmp_path / "out.parquet", allow_rejected_rows=True)

    assert report["source_rows"] == 3
    assert report["accepted_rows"] == 2
    assert report["rejected_rows"] == 1
    assert report["partial_conversion"] is True
    assert report["partial_conversion_authorized"] is True
    assert report["complete_dataset_conversion"] is False
    assert report["lossless_for_accepted_rows"] is True
    # Output row -> original source row, so the dropped row stays traceable.
    assert report["source_row_indices"] == {"0": 0, "1": 2}
    assert report["rejections"][0]["row"] == 1


def test_complete_conversion_is_labelled_complete(tmp_path: Path) -> None:
    source = _write(tmp_path / "in.parquet", [_row(), _row()])

    report = _convert(source, tmp_path / "out.parquet")

    assert report["partial_conversion"] is False
    assert report["complete_dataset_conversion"] is True
    assert report["partial_conversion_authorized"] is None
    assert report["source_row_indices"] is None


def test_zero_accepted_rows_always_fails(tmp_path: Path) -> None:
    source = _write(tmp_path / "in.parquet", [_invalid_row()])

    with pytest.raises(ConfigError):
        _convert(source, tmp_path / "out.parquet", allow_rejected_rows=True)

    assert not (tmp_path / "out.parquet").exists()


def test_partial_report_is_serializable(tmp_path: Path) -> None:
    source = _write(tmp_path / "in.parquet", [_row(), _invalid_row()])
    _convert(source, tmp_path / "out.parquet", allow_rejected_rows=True)

    payload = json.loads((tmp_path / "out.parquet.report.json").read_text(encoding="utf-8"))

    assert payload["complete_dataset_conversion"] is False
    assert payload["schema_version"] == 3
