"""Versioned tokenizer identity must cover structure, not one probe string."""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import pytest

from miniverl.errors import TokenizerMismatchError
from miniverl.models.tokenizers import HFTokenizerAdapter, assert_same_tokenizer


class _BackendTokenizer:
    def __init__(self, normalizer: str) -> None:
        self.normalizer = normalizer

    def to_str(self) -> str:
        return f'{{"normalizer":{self.normalizer!r}}}'


class _FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    bos_token_id = 1
    additional_special_tokens: ClassVar[list[str]] = ["<tool>"]
    special_tokens_map: ClassVar[dict[str, object]] = {
        "eos_token": "</s>",
        "additional_special_tokens": ["<tool>"],
    }
    init_kwargs: ClassVar[dict[str, object]] = {"clean_up_tokenization_spaces": False}

    def __init__(
        self,
        *,
        vocab: dict[str, int] | None = None,
        added: dict[str, int] | None = None,
        normalizer: str = "NFC",
    ) -> None:
        self._vocab = vocab or {
            "<pad>": 0,
            "<s>": 1,
            "</s>": 2,
            "same": 3,
            "unprobed-a": 4,
        }
        self._added = added or {"<tool>": max(self._vocab.values()) + 1}
        self.backend_tokenizer = _BackendTokenizer(normalizer)

    def __len__(self) -> int:
        return max([*self._vocab.values(), *self._added.values()]) + 1

    def __call__(self, _text: str, *, add_special_tokens: bool):
        assert add_special_tokens is False
        return {"input_ids": [3, 2]}

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def get_added_vocab(self) -> dict[str, int]:
        return dict(self._added)

    def decode(self, token_ids, *, skip_special_tokens: bool):
        return " ".join(str(item) for item in token_ids)

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return str(token_id)


def _adapter(**kwargs) -> HFTokenizerAdapter:
    return HFTokenizerAdapter(_FakeTokenizer(**kwargs), "same/repo", revision="abc")


@pytest.mark.parametrize(
    "changed",
    [
        {"vocab": {"<pad>": 0, "<s>": 1, "</s>": 2, "same": 3, "unprobed-b": 4}},
        {"added": {"<structurally-different>": 5}},
        {"normalizer": "NFKC"},
    ],
)
def test_same_probe_but_different_structure_has_a_different_v2_digest(changed) -> None:
    baseline = _adapter()
    candidate = _adapter(**changed)

    assert baseline.fingerprint == candidate.fingerprint
    assert baseline.identity["structural_digest_v2"] != candidate.identity["structural_digest_v2"]
    with pytest.raises(TokenizerMismatchError):
        assert_same_tokenizer(baseline, candidate)


def test_identical_tokenizer_structure_has_identical_v2_identity() -> None:
    assert _adapter().identity == _adapter().identity


def test_same_repo_with_different_revisions_is_loaded_and_compared(monkeypatch) -> None:
    from miniverl.config import RunConfig
    from miniverl.models import factory

    config = RunConfig.from_mapping(
        {
            "models": {
                "backend": "hf",
                "student": {"model_id": "same/repo", "revision": "student-rev"},
                "teacher": {"model_id": "same/repo", "revision": "teacher-rev"},
            },
            "environment": {
                "name": "calculator",
                "train_tasks": 1,
                "eval_tasks": 1,
                "test_tasks": 1,
            },
        }
    )
    loaded: list[tuple[str, str | None]] = []

    def load(model_id: str, *, revision: str | None, **_kwargs):
        loaded.append((model_id, revision))
        return SimpleNamespace(
            fingerprint="same",
            identity={"structural_digest_v2": "same"},
        )

    monkeypatch.setattr(factory.HFTokenizerAdapter, "load", load)
    factory.build_tokenizer(config)

    assert loaded == [
        ("same/repo", "student-rev"),
        ("same/repo", "teacher-rev"),
    ]
