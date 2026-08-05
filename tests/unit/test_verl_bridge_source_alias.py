"""An input file may never also be an output file.

Up to v0.6.2 ``import-verl --out`` could name its own source config. With
``--overwrite`` a successful import replaced the user's input; a *rejected*
import deleted it and kept only the rejection report. miniVERL has no in-place
mode, so every overlap is a hard error raised before anything is reserved.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from miniverl.bridge.publish import SourceOutputAliasError

PROFILE = "single-gpu-online-distillation-v1"
ALIAS_MESSAGE = "source and output families must be distinct; choose a new --out path"


# ------------------------------------------------------------------ fixtures


def _source_payload(*, unsupported: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "data": {
            "train_files": ["train.parquet"],
            "val_files": ["val.parquet"],
            "max_prompt_length": 512,
            "max_response_length": 128,
            "seed": 77,
        },
        "actor_rollout_ref": {
            "model": {"path": "Qwen/Qwen3-0.6B", "enable_gradient_checkpointing": True},
            "actor": {"optim": {"lr": 2.0e-5}},
        },
        "trainer": {
            "save_freq": 2,
            "test_freq": 1,
            "project_name": "bridge-smoke",
            "experiment_name": "strict-profile",
            "total_epochs": 3,
        },
    }
    if unsupported:
        payload["algorithm"] = {"adv_estimator": "gae"}
    return payload


def _write_source(path: Path, *, unsupported: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(_source_payload(unsupported=unsupported), sort_keys=False), encoding="utf-8"
    )
    return path


def _import(source: Path, out: Path, **kwargs: Any) -> dict[str, Any]:
    from miniverl.bridge.config import import_verl_config
    from miniverl.bridge.contract import VERL_TAG

    payload: dict[str, Any] = {
        "environment": "calculator",
        "teacher_model": "Qwen/Qwen3-1.7B",
        "loss_profile": "topk-tail-reverse-kl",
        "schedule_mapping": "epochs-as-cycles",
    }
    payload.update(kwargs)
    return import_verl_config(source, profile=PROFILE, target_verl=VERL_TAG, out=out, **payload)


def _sibling_files(directory: Path) -> dict[Path, bytes]:
    return {path: path.read_bytes() for path in sorted(directory.rglob("*")) if path.is_file()}


# ---------------------------------------------------------------- import-verl


def test_exact_same_path_is_rejected(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "recipes" / "foo.yaml")
    before = source.read_bytes()

    with pytest.raises(SourceOutputAliasError) as excinfo:
        _import(source, source)

    assert ALIAS_MESSAGE in str(excinfo.value)
    assert source.read_bytes() == before


def test_relative_and_absolute_alias_is_rejected(tmp_path: Path, monkeypatch: Any) -> None:
    _write_source(tmp_path / "recipes" / "foo.yaml")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SourceOutputAliasError):
        _import(Path("recipes/foo.yaml"), tmp_path / "recipes" / "foo.yaml")


def test_dot_segment_alias_is_rejected(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "recipes" / "foo.yaml")
    noisy = tmp_path / "recipes" / ".." / "recipes" / "foo.yaml"

    with pytest.raises(SourceOutputAliasError):
        _import(source, noisy)


def test_source_matching_the_report_path_is_rejected(tmp_path: Path) -> None:
    """``foo.import-report.json`` as the input must not be overwritten either."""
    source = tmp_path / "recipes" / "foo.import-report.json"
    _write_source(source)
    before = source.read_bytes()

    with pytest.raises(SourceOutputAliasError):
        _import(source, tmp_path / "recipes" / "foo.yaml")

    assert source.read_bytes() == before


def test_source_matching_the_template_path_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "recipes" / "foo.template.yaml"
    _write_source(source)

    with pytest.raises(SourceOutputAliasError):
        _import(source, tmp_path / "recipes" / "foo.yaml")


def test_overwrite_does_not_bypass_the_alias_check(tmp_path: Path) -> None:
    """--overwrite replaces a previous output family, never an input."""
    source = _write_source(tmp_path / "recipes" / "foo.yaml")
    before = source.read_bytes()

    with pytest.raises(SourceOutputAliasError):
        _import(source, source, overwrite=True)

    assert source.read_bytes() == before


def test_rejected_import_no_longer_deletes_its_source(tmp_path: Path) -> None:
    """The worst pre-fix path: rejection published a report and removed the input."""
    source = _write_source(tmp_path / "recipes" / "foo.yaml", unsupported=True)
    before = _sibling_files(tmp_path)

    with pytest.raises(SourceOutputAliasError):
        _import(source, source, overwrite=True)

    assert source.exists(), "a rejected import deleted its own source config"
    assert _sibling_files(tmp_path) == before


def test_alias_rejection_creates_no_output_file(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "recipes" / "foo.yaml")

    with pytest.raises(SourceOutputAliasError):
        _import(source, source, overwrite=True)

    assert sorted(path.name for path in (tmp_path / "recipes").iterdir()) == ["foo.yaml"]


def test_distinct_stem_in_the_same_directory_still_works(tmp_path: Path) -> None:
    """The guard must not block the ordinary case."""
    source = _write_source(tmp_path / "recipes" / "source.yaml")
    report = _import(source, tmp_path / "recipes" / "generated.yaml")

    assert report["status"] == "accepted"
    assert source.exists()
    assert (tmp_path / "recipes" / "generated.yaml").is_file()


# ------------------------------------------------------------- link aliases


@pytest.mark.skipif(sys.platform == "win32", reason="symlink creation needs privileges")
def test_symlink_alias_is_rejected(tmp_path: Path) -> None:
    real = _write_source(tmp_path / "recipes" / "real.yaml")
    link = tmp_path / "recipes" / "link.yaml"
    link.symlink_to(real)

    with pytest.raises(SourceOutputAliasError):
        _import(link, real)


def test_hard_link_alias_is_rejected(tmp_path: Path) -> None:
    real = _write_source(tmp_path / "recipes" / "real.yaml")
    link = tmp_path / "recipes" / "hard.yaml"
    try:
        os.link(real, link)
    except (OSError, NotImplementedError, AttributeError):
        pytest.skip("hard links are unavailable on this filesystem")

    with pytest.raises(SourceOutputAliasError):
        _import(real, link)

    assert real.read_bytes() == link.read_bytes()


@pytest.mark.skipif(sys.platform != "win32", reason="case-insensitive alias is Windows-specific")
def test_case_only_difference_is_rejected_on_windows(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "recipes" / "Foo.yaml")
    before = source.read_bytes()

    with pytest.raises(SourceOutputAliasError):
        _import(source, tmp_path / "recipes" / "FOO.YAML", overwrite=True)

    assert source.read_bytes() == before


# ------------------------------------------------------------ convert-dataset


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> Path:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def _rows() -> list[dict[str, Any]]:
    return [
        {
            "data_source": "calc",
            "prompt": [{"role": "user", "content": "1+1"}],
            "ability": "math",
            "reward_model": {"ground_truth": "2"},
            "extra_info": None,
        }
    ]


def _convert(source: Path, out: Path, **kwargs: Any) -> dict[str, Any]:
    from miniverl.bridge.dataset import convert_dataset

    return convert_dataset(source, out=out, direction="from-verl-parquet", **kwargs)


def test_dataset_exact_alias_is_rejected(tmp_path: Path) -> None:
    source = _write_parquet(tmp_path / "data" / "train.parquet", _rows())
    before = source.read_bytes()

    with pytest.raises(SourceOutputAliasError) as excinfo:
        _convert(source, source, overwrite=True)

    assert ALIAS_MESSAGE in str(excinfo.value)
    assert source.read_bytes() == before


def test_dataset_source_matching_output_sidecar_is_rejected(tmp_path: Path) -> None:
    """``train.parquet.miniverl.json`` is an output of ``train.parquet``."""
    source = _write_parquet(tmp_path / "data" / "train.parquet.miniverl.json", _rows())

    with pytest.raises(SourceOutputAliasError):
        _convert(source, tmp_path / "data" / "train.parquet", overwrite=True)

    assert source.exists()


def test_dataset_source_sidecar_matching_an_output_is_rejected(tmp_path: Path) -> None:
    """The *source's own* sidecar is an input and may not be republished."""
    source = _write_parquet(tmp_path / "data" / "in.parquet", _rows())
    sidecar = tmp_path / "data" / "in.parquet.miniverl.json"
    sidecar.write_text('{"schema_version": 1, "rows": {}}', encoding="utf-8")
    before = sidecar.read_bytes()

    # Writing the converted Parquet to ``in.parquet.miniverl.json`` would land it
    # on the extension sidecar this very conversion reads.
    with pytest.raises(SourceOutputAliasError):
        _convert(source, sidecar, overwrite=True)

    assert sidecar.read_bytes() == before


def test_dataset_distinct_output_still_works(tmp_path: Path) -> None:
    source = _write_parquet(tmp_path / "data" / "in.parquet", _rows())
    report = _convert(source, tmp_path / "data" / "out.parquet")

    assert report["accepted_rows"] == 1
    assert source.exists()
    assert (tmp_path / "data" / "out.parquet").is_file()
