"""Batching is a throughput knob, never a scientific setting.

The generation path is exercised with a fake model so these run on CPU in
milliseconds. The real batch-equivalence check against Qwen3-0.6B is a GPU test.
"""

from __future__ import annotations

from typing import Any

import pytest

from miniverl.alignment_external.generation import (
    BatchedGenerator,
    GenerationConfig,
    _is_out_of_memory,
)

# `generate()` wraps the forward pass in `torch.no_grad()`, so driving it needs
# torch even with a fake model. The config and OOM-detection tests below do not,
# and stay in the torch-free set.
requires_torch = pytest.mark.torch

# ------------------------------------------------------------------- config


def test_sampling_is_refused_for_the_final_test() -> None:
    with pytest.raises(ValueError, match="deterministic"):
        GenerationConfig(do_sample=True)

    with pytest.raises(ValueError, match="deterministic"):
        GenerationConfig(temperature=0.7)


def test_the_config_digest_ignores_batch_size() -> None:
    """An OOM backoff must not change the recorded decoding rule."""
    big = GenerationConfig(batch_size=16).as_dict()
    small = GenerationConfig(batch_size=1).as_dict()

    assert big == small
    assert "batch_size" not in big
    assert big["decoding"] == "greedy"


def test_the_config_digest_records_dtype() -> None:
    """dtype decides whether batch size can change the output, so it is recorded."""
    assert GenerationConfig().as_dict()["dtype"] == "float32"
    assert GenerationConfig(dtype="bfloat16", batch_size=1).as_dict()["dtype"] == "bfloat16"


def test_batched_generation_outside_float32_is_refused() -> None:
    """Measured: in bf16, batch size changes the decoded text on real weights.

    Batch 1 and batches 2-12 agree byte for byte in float32 and diverge in
    bfloat16 on the pinned Qwen3-0.6B student, so a bf16 batched run would let
    a throughput knob move a benchmark number.
    """
    with pytest.raises(ValueError, match="not reproducible"):
        GenerationConfig(dtype="bfloat16", batch_size=8)

    # Batch size 1 in bf16 is still allowed: nothing is being batched.
    assert GenerationConfig(dtype="bfloat16", batch_size=1).batch_size == 1


def test_token_budgets_must_be_positive() -> None:
    with pytest.raises(ValueError, match="token budgets"):
        GenerationConfig(max_new_tokens=0)


def test_min_batch_size_is_bounded_by_batch_size() -> None:
    with pytest.raises(ValueError, match="min_batch_size"):
        GenerationConfig(batch_size=4, min_batch_size=8)


# -------------------------------------------------------------- fake runtime


class _FakeTokenizer:
    """Records how it was called so padding side can be asserted."""

    pad_token_id = 0
    eos_token_id = 1

    def __init__(self) -> None:
        self.padding_side = "right"
        self.seen_padding_sides: list[str] = []

    def __call__(self, prompts: list[str], **_: Any) -> dict[str, Any]:
        self.seen_padding_sides.append(self.padding_side)
        width = max(len(p) for p in prompts)
        return _Encoded(
            {
                "input_ids": _FakeTensor([[ord(c) for c in p.rjust(width, "\0")] for p in prompts]),
                "attention_mask": _FakeTensor(
                    [[0] * (width - len(p)) + [1] * len(p) for p in prompts]
                ),
            }
        )

    def decode(self, row: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(value) for value in row if value)


class _FakeTensor(list):
    @property
    def shape(self) -> tuple[int, int]:
        return (len(self), len(self[0]) if self else 0)

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, tuple):
            _rows, columns = item
            return _FakeTensor([row[columns] for row in self])
        return list.__getitem__(self, item)


class _Encoded(dict):
    def to(self, _device: Any) -> _Encoded:
        return self


class _EchoModel:
    """Appends a fixed marker, so the output still depends on each input row.

    ``fail_until_batch`` makes it raise a CUDA OOM for any batch larger than
    the given size, which is how the backoff path is exercised without a GPU.
    """

    device = "cpu"

    def __init__(self, fail_until_batch: int | None = None) -> None:
        self.fail_until_batch = fail_until_batch
        self.batch_sizes: list[int] = []

    def generate(self, **kwargs: Any) -> _FakeTensor:
        input_ids = kwargs["input_ids"]
        size = len(input_ids)
        self.batch_sizes.append(size)
        if self.fail_until_batch is not None and size > self.fail_until_batch:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
        return _FakeTensor([[*list(row), ord("<"), ord(">")] for row in input_ids])


def _generator(model: Any, **config: Any) -> BatchedGenerator:
    return BatchedGenerator(model, _FakeTokenizer(), GenerationConfig(**config))


# ------------------------------------------------------------------ behaviour


@requires_torch
def test_generation_pads_on_the_left() -> None:
    """Right padding makes a decoder-only model continue from pad tokens."""
    generator = _generator(_EchoModel(), batch_size=2)

    generator.generate(["ab", "cdef"])

    assert generator.tokenizer.seen_padding_sides == ["left"]
    # And the tokenizer is left as it was found.
    assert generator.tokenizer.padding_side == "right"


@requires_torch
def test_results_stay_aligned_to_the_prompt_order() -> None:
    generator = _generator(_EchoModel(), batch_size=2)

    results = generator.generate(["aa", "bb", "cc", "dd", "ee"])

    assert len(results) == 5
    assert [r[:2] for r in results] == ["<>", "<>", "<>", "<>", "<>"]


@requires_torch
def test_an_oom_halves_the_batch_and_retries_without_changing_settings() -> None:
    model = _EchoModel(fail_until_batch=2)
    generator = _generator(model, batch_size=8, min_batch_size=1)
    before = generator.config.as_dict()

    results = generator.generate(["a", "b", "c", "d"])

    assert len(results) == 4
    assert generator.oom_backoffs, "the backoff should have been recorded"
    assert generator.oom_backoffs[0]["from"] == 8
    # 8 -> 4 -> 2, and 2 succeeds.
    assert max(size for size in model.batch_sizes if size <= 2) == 2
    # Nothing scientific moved.
    assert generator.config.as_dict() == before


@requires_torch
def test_an_oom_at_the_minimum_batch_size_is_raised() -> None:
    """Backoff has a floor; below it the run fails loudly rather than silently."""
    generator = _generator(_EchoModel(fail_until_batch=0), batch_size=1, min_batch_size=1)

    with pytest.raises(RuntimeError, match="out of memory"):
        generator.generate(["a"])


@requires_torch
def test_a_non_oom_error_is_not_swallowed() -> None:
    class _Broken(_EchoModel):
        def generate(self, **kwargs: Any) -> Any:
            raise ValueError("a real bug")

    generator = _generator(_Broken(), batch_size=4)

    with pytest.raises(ValueError, match="a real bug"):
        generator.generate(["a", "b"])


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (RuntimeError("CUDA out of memory"), True),
        (RuntimeError("cuda OutOfMemoryError"), True),
        (ValueError("shape mismatch"), False),
    ],
)
def test_out_of_memory_detection(exc: BaseException, expected: bool) -> None:
    assert _is_out_of_memory(exc) is expected
