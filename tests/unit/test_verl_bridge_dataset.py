from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _rows() -> list[dict[str, object]]:
    return [
        {
            "data_source": "calculator",
            "prompt": [
                {"role": "system", "content": "Use the calculator."},
                {"role": "user", "content": "What is 2 + 2?"},
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": "4"},
            "extra_info": {"split": "train", "index": 0},
            "miniverl_extensions": {
                "token_provenance": {"schema": 1},
                "teacher_targets": {"representation": "top_k_plus_tail"},
            },
        },
        {
            "data_source": "broken",
            "prompt": [{"role": "user"}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": "0"},
            "extra_info": {"split": "train", "index": 1},
        },
    ]


def test_verl_parquet_round_trip_preserves_chat_and_uses_an_extension_sidecar(
    tmp_path: Path,
) -> None:
    from miniverl.bridge.dataset import convert_dataset

    source = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist(_rows()), source)
    canonical = tmp_path / "canonical.parquet"

    first = convert_dataset(source, out=canonical, direction="from-verl-parquet")
    assert first["accepted_rows"] == 1
    assert first["rejected_rows"] == 1
    assert first["truncation_risk"]["status"] == "not_evaluated_no_tokenizer"
    assert first["source_sha256"]
    assert first["output_sha256"]
    assert Path(first["extension_sidecar"]).is_file()
    assert not any("reference_log" in name for name in pq.read_schema(canonical).names)

    exported = tmp_path / "export.parquet"
    second = convert_dataset(canonical, out=exported, direction="to-verl-parquet")
    assert second["accepted_rows"] == 1
    row = pq.read_table(exported).to_pylist()[0]
    assert row["prompt"] == _rows()[0]["prompt"]
    assert row["reward_model"]["ground_truth"] == "4"
    assert row["extra_info"]["miniverl"]["teacher_targets"]["representation"] == ("top_k_plus_tail")
    assert "reference_log_probs" not in json.dumps(row)


def test_dataset_report_counts_prompt_truncation_risk_without_truncating(tmp_path: Path) -> None:
    from miniverl.bridge.dataset import convert_dataset

    row = _rows()[0]
    source = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist([row]), source)
    out = tmp_path / "out.parquet"
    report = convert_dataset(
        source,
        out=out,
        direction="from-verl-parquet",
        max_prompt_characters=10,
    )
    assert report["truncation_risk"] == {
        "status": "character_bound_only",
        "max_prompt_characters": 10,
        "rows_over_bound": 1,
        "rows_truncated": 0,
    }
    assert pq.read_table(out).to_pylist()[0]["prompt"] == row["prompt"]
