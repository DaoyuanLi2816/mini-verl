"""Overlong rows are removed before batching, never silently truncated."""

import pytest

from miniverl.config.models import VerlParquetSourceConfig
from miniverl.data.verl_parquet import VerlParquetDataset


class Tokenizer:
    def encode(self, text):
        return list(range(len(text)))


def test_filter_before_shuffle_and_count(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "train.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"prompt": "ok"}, {"prompt": "too long"}, {"prompt": "yes"}]), path
    )
    config = VerlParquetSourceConfig(
        train_files=[str(path)],
        allow_plain_string_prompts=True,
        filter_overlong_prompts=True,
        max_prompt_length=3,
        shuffle=False,
    )
    dataset = VerlParquetDataset(config, tokenizer=Tokenizer())
    assert [row.source_row_index for row in dataset.iter_split("train")] == [0, 2]
    assert dataset.inspect().rows["train"] == 2
    with pytest.raises(Exception, match="tokenizer"):
        VerlParquetDataset(config).inspect()


def test_drop_last_never_crosses_an_epoch(tmp_path):
    from types import SimpleNamespace

    from miniverl.trainer import OPDTrainer

    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pylist([{"prompt": str(index)} for index in range(5)]), path)
    source = VerlParquetSourceConfig(
        train_files=[str(path)], allow_plain_string_prompts=True, drop_last=True, shuffle=False
    )
    dataset = VerlParquetDataset(source)
    host = SimpleNamespace(
        prompt_dataset=dataset,
        prompt_dataset_manifest=dataset.inspect(),
        config=SimpleNamespace(source=source),
        tokenizer=Tokenizer(),
        task_cursor=0,
        _prompt_train_iterator=None,
    )
    batches = [OPDTrainer._next_tasks(host, 2) for _ in range(3)]
    assert [[item.record.source_row_index for item in batch] for batch in batches] == [
        [0, 1],
        [2, 3],
        [0, 1],
    ]


def test_direct_input_cannot_drop_multimodal_payload(tmp_path):
    from miniverl.errors import ConfigError

    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "train.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [{"prompt": [{"role": "user", "content": "see image"}], "images": ["photo.png"]}]
        ),
        path,
    )
    source = VerlParquetSourceConfig(train_files=[str(path)], unsupported_input_columns=["images"])
    with pytest.raises(ConfigError, match="multimodal"):
        VerlParquetDataset(source).inspect()
