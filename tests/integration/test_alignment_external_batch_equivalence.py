"""Batched decoding must produce the same text as batch size 1, on real weights.

The fake-model tests cover the control flow. This covers the thing that
actually goes wrong in practice: padding side, attention masking and position
ids interacting with a real tokenizer and real weights. If this fails, batched
benchmark results are not usable, because the batch size would be changing the
measurement.

Runs on the pinned student model, so it needs the local snapshot and a GPU.
"""

from __future__ import annotations

import pytest

from miniverl.alignment_external.generation import (
    BatchedGenerator,
    GenerationConfig,
    assert_batch_equivalence,
)

pytestmark = [pytest.mark.gpu, pytest.mark.torch, pytest.mark.slow]

STUDENT = "Qwen/Qwen3-0.6B"
STUDENT_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"

# Deliberately uneven lengths: equal-length prompts would pad to nothing and
# the test would pass without exercising the padding path at all.
PROMPTS = [
    "Name one primary colour.",
    "In one short sentence, explain what a compiler does and why it matters to a beginner.",
    "List two fruits.",
    "Write a single sentence about the sea.",
    "Summarise, in no more than two sentences, why sorting a list can be useful in software.",
    "Say hello.",
]


@pytest.fixture(scope="module")
def runtime():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            STUDENT, revision=STUDENT_REVISION, local_files_only=True
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            STUDENT, revision=STUDENT_REVISION, local_files_only=True, dtype=torch.float32
        )
    except Exception as exc:
        pytest.skip(f"pinned student snapshot unavailable locally: {exc}")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    model.to("cuda")
    return model, tokenizer


def test_batched_decoding_matches_batch_size_one(runtime) -> None:
    model, tokenizer = runtime
    generator = BatchedGenerator(
        model, tokenizer, GenerationConfig(max_new_tokens=24, max_prompt_tokens=128, batch_size=3)
    )

    # Raises with the first divergence if batching changed anything.
    results = assert_batch_equivalence(generator, PROMPTS)

    assert len(results) == len(PROMPTS)


def test_uneven_batch_sizes_all_agree(runtime) -> None:
    """Whatever the batch size, the same prompt gets the same answer."""
    model, tokenizer = runtime

    def run(batch_size: int) -> list[str]:
        return BatchedGenerator(
            model,
            tokenizer,
            GenerationConfig(max_new_tokens=24, max_prompt_tokens=128, batch_size=batch_size),
        ).generate(PROMPTS)

    baseline = run(1)
    for batch_size in (2, 3, 4, 6):
        assert run(batch_size) == baseline, f"batch size {batch_size} changed the output"


def test_generation_is_repeatable(runtime) -> None:
    """Greedy decoding twice in a row must be byte-identical."""
    model, tokenizer = runtime
    config = GenerationConfig(max_new_tokens=24, max_prompt_tokens=128, batch_size=3)

    first = BatchedGenerator(model, tokenizer, config).generate(PROMPTS)
    second = BatchedGenerator(model, tokenizer, config).generate(PROMPTS)

    assert first == second
