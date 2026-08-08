"""A new extension sidecar names the dataset it belongs to.

A sidecar is keyed by row index. Copied beside a different Parquet file its
keys silently mean different rows, so v2 binds the digest and row count of the
dataset it was published with and refuses to be read without them. Sidecars
published by 0.6.0-0.6.3 declare schema_version 1 and stay readable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from miniverl import __version__
from miniverl.bridge.dataset import convert_dataset
from miniverl.errors import ConfigError


def _row(index: int, extension: Any = None) -> dict[str, Any]:
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


def _source(path: Path, *, rows: int = 3, with_extensions: bool = True) -> Path:
    table = pa.Table.from_pylist(
        [_row(index, {"selected": [index]} if with_extensions else None) for index in range(rows)]
    )
    pq.write_table(table, path)
    return path


def _convert(tmp_path: Path, source: Path, name: str = "out.parquet") -> dict[str, Any]:
    return convert_dataset(source, out=tmp_path / name, direction="from-verl-parquet")


def _sidecar(parquet: Path) -> dict[str, Any]:
    return json.loads(parquet.with_name(parquet.name + ".miniverl.json").read_text("utf-8"))


# ------------------------------------------------------------------- writing


def test_a_new_sidecar_is_v2_and_binds_its_dataset(tmp_path: Path) -> None:
    source = _source(tmp_path / "train.parquet")

    _convert(tmp_path, source)

    sidecar = _sidecar(tmp_path / "out.parquet")
    assert sidecar["schema_version"] == 2
    assert sidecar["namespace"] == "extra_info.miniverl"
    assert sidecar["source_rows"] == 3
    assert sidecar["generator"] == {"name": "miniverl", "version": __version__}
    # The digest is of the Parquet the sidecar sits beside, not of the input.
    import hashlib

    published = (tmp_path / "out.parquet").read_bytes()
    assert sidecar["source_sha256"] == hashlib.sha256(published).hexdigest()


def test_the_binding_survives_a_round_trip(tmp_path: Path) -> None:
    """The sidecar this version writes must be one this version accepts."""
    source = _source(tmp_path / "train.parquet")
    _convert(tmp_path, source)

    # Feed the published pair straight back in.
    report = convert_dataset(
        tmp_path / "out.parquet", out=tmp_path / "again.parquet", direction="to-verl-parquet"
    )

    assert report["accepted_rows"] == 3


# ------------------------------------------------------------------- reading


def test_a_v2_sidecar_without_binding_fails_closed(tmp_path: Path) -> None:
    source = _source(tmp_path / "train.parquet", with_extensions=False)
    (tmp_path / "train.parquet.miniverl.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "namespace": "extra_info.miniverl",
                "rows": {"0": {"selected": [0]}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError) as caught:
        _convert(tmp_path, source)

    message = str(caught.value)
    assert "source_sha256" in message
    assert "source_rows" in message


def test_a_v2_sidecar_bound_to_another_dataset_fails_closed(tmp_path: Path) -> None:
    source = _source(tmp_path / "train.parquet", with_extensions=False)
    (tmp_path / "train.parquet.miniverl.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "namespace": "extra_info.miniverl",
                "source_sha256": "0" * 64,
                "source_rows": 3,
                "rows": {"0": {"selected": [0]}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError) as caught:
        _convert(tmp_path, source)

    assert "does not match the dataset being converted" in str(caught.value)


def test_a_public_v1_sidecar_without_binding_still_reads(tmp_path: Path) -> None:
    """Backward compatibility: 0.6.0-0.6.3 published exactly this shape."""
    source = _source(tmp_path / "train.parquet", with_extensions=False)
    (tmp_path / "train.parquet.miniverl.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "namespace": "extra_info.miniverl",
                "semantics": "miniVERL token provenance and teacher targets",
                "rows": {"0": {"selected": [0]}, "2": {"selected": [2]}},
            }
        ),
        encoding="utf-8",
    )

    report = convert_dataset(source, out=tmp_path / "out.parquet", direction="to-verl-parquet")

    assert report["accepted_rows"] == 3


def test_an_unknown_schema_version_still_fails_closed(tmp_path: Path) -> None:
    source = _source(tmp_path / "train.parquet", with_extensions=False)
    (tmp_path / "train.parquet.miniverl.json").write_text(
        json.dumps({"schema_version": 999, "namespace": "extra_info.miniverl", "rows": {}}),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError) as caught:
        _convert(tmp_path, source)

    assert "schema_version" in str(caught.value)


def test_sidecar_diagnostics_never_show_extension_values(tmp_path: Path) -> None:
    secret = "TEACHER_TARGET_NOT_FOR_DIAGNOSTICS"
    source = _source(tmp_path / "train.parquet", with_extensions=False)
    (tmp_path / "train.parquet.miniverl.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "namespace": "extra_info.miniverl",
                "rows": {"0": {"selected": secret}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError) as caught:
        _convert(tmp_path, source)

    assert secret not in str(caught.value)
