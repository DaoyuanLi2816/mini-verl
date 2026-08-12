from __future__ import annotations

import json
from typing import ClassVar

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from miniverl.config.models import PromptTruncation, VerlParquetSourceConfig
from miniverl.data.verl_parquet import VerlParquetDataset, render_prompt
from miniverl.errors import ConfigError


class TinyTokenizer:
    fingerprint = "f" * 64
    identity: ClassVar[dict[str, str]] = {
        "behavioral_fingerprint_v1": fingerprint,
        "structural_digest_v2": "s" * 64,
    }
    eos_token_id = 0
    pad_token_id = 0
    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int]) -> str:
        return bytes(token_ids).decode("utf-8")

    def apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        return (
            "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages) + "<assistant>"
        )


def _write(path, rows: list[dict[str, object]], *, row_group_size: int = 2):
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=row_group_size)
    return path


def _row(index: int) -> dict[str, object]:
    return {
        "prompt": [{"role": "user", "content": f"question {index}"}],
        "data_source": "unit",
        "ability": "reasoning",
        "reward_model": {"style": "rule"},
        "extra_info": {"index": index},
    }


def test_streaming_loader_preserves_metadata_and_is_deterministic(tmp_path) -> None:
    train = _write(tmp_path / "train.parquet", [_row(i) for i in range(9)])
    source = VerlParquetSourceConfig(
        train_files=[str(train)], row_batch_size=2, shuffle=True, seed=71
    )
    dataset = VerlParquetDataset(source)

    first = list(dataset.iter_split("train", epoch=3))
    second = list(dataset.iter_split("train", epoch=3))

    assert [row.row_digest for row in first] == [row.row_digest for row in second]
    assert sorted(row.source_row_index for row in first) == list(range(9))
    assert first[0].data_source == "unit"
    assert first[0].ability == "reasoning"
    assert first[0].reward_model == {"style": "rule"}
    assert isinstance(first[0].extra_info, dict)
    assert first[0].source_file == str(train.resolve())
    manifest = dataset.inspect()
    assert manifest.rows == {"train": 9, "val": 0}
    assert len(manifest.content_digest) == len(manifest.schema_digest) == 64


def test_rejects_every_invalid_row_instead_of_silently_filtering(tmp_path) -> None:
    train = _write(tmp_path / "train.parquet", [{"prompt": "plain text"}])
    dataset = VerlParquetDataset(VerlParquetSourceConfig(train_files=[str(train)]))

    with pytest.raises(ConfigError, match="plain-string prompt"):
        list(dataset.iter_split("train"))


def test_task_rewards_require_reward_model_but_pure_opd_does_not(tmp_path) -> None:
    train = _write(
        tmp_path / "train.parquet",
        [{"prompt": [{"role": "user", "content": "hello"}]}],
    )
    pure = VerlParquetDataset(VerlParquetSourceConfig(train_files=[str(train)]))
    assert next(iter(pure.iter_split("train"))).reward_model is None

    rewarded = VerlParquetDataset(
        VerlParquetSourceConfig(train_files=[str(train)], use_task_rewards=True)
    )
    with pytest.raises(ConfigError, match="reward_model"):
        list(rewarded.iter_split("train"))


def test_chat_template_is_applied_once_and_records_provenance(tmp_path) -> None:
    train = _write(tmp_path / "train.parquet", [_row(1)])
    source = VerlParquetSourceConfig(train_files=[str(train)], max_prompt_length=200)
    record = next(VerlParquetDataset(source).iter_split("train"))
    rendered = render_prompt(record, TinyTokenizer(), source)

    assert rendered.text == "<user>question 1</user><assistant>"
    assert rendered.prompt_token_count == len(rendered.token_ids)
    assert rendered.truncation_decision == "not_needed"
    assert rendered.tokenizer_identity["structural_digest_v2"] == "s" * 64
    assert len(rendered.rendered_prompt_digest) == 64


@pytest.mark.parametrize(
    ("mode", "expected"),
    [(PromptTruncation.LEFT, "def"), (PromptTruncation.RIGHT, "abc")],
)
def test_truncation_is_exact(mode, expected, tmp_path) -> None:
    train = _write(tmp_path / "train.parquet", [{"prompt": "abcdef"}])
    source = VerlParquetSourceConfig(
        train_files=[str(train)],
        allow_plain_string_prompts=True,
        max_prompt_length=3,
        truncation=mode,
    )
    record = next(VerlParquetDataset(source).iter_split("train"))
    rendered = render_prompt(record, TinyTokenizer(), source)

    assert rendered.text == expected
    assert rendered.truncation_decision == f"truncated_{mode.value}"


def test_overlength_error_is_actionable(tmp_path) -> None:
    train = _write(tmp_path / "train.parquet", [{"prompt": "abcdef"}])
    source = VerlParquetSourceConfig(
        train_files=[str(train)], allow_plain_string_prompts=True, max_prompt_length=3
    )
    record = next(VerlParquetDataset(source).iter_split("train"))

    with pytest.raises(ConfigError, match=r"6 tokens.*max_prompt_length=3"):
        render_prompt(record, TinyTokenizer(), source)


def test_row_digest_binds_preserved_payload(tmp_path) -> None:
    a = _write(tmp_path / "a.parquet", [_row(1)])
    changed = _row(1)
    changed["extra_info"] = {"index": 2}
    b = _write(tmp_path / "b.parquet", [changed])
    first = next(
        VerlParquetDataset(VerlParquetSourceConfig(train_files=[str(a)])).iter_split("train")
    )
    second = next(
        VerlParquetDataset(VerlParquetSourceConfig(train_files=[str(b)])).iter_split("train")
    )

    assert first.row_digest != second.row_digest
    assert json.loads(first.canonical_payload)["extra_info"] == {"index": 1}
