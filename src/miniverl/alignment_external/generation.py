"""Deterministic batched generation for the external benchmark endpoints.

External text benchmarks do not need the tool-agent loop: they are one prompt
in, one response out. Batching them is what makes 16 checkpoints across four
endpoints fit a 48 GPU-hour budget.

Batching must not change what is generated. That is the whole contract here:

* decoding is greedy for the final test -- no sampling, no temperature;
* padding is left-side, because a decoder-only model continues from the last
  position and right padding would make it continue from pad tokens;
* the attention mask is explicit, so a padded position can never attend into a
  neighbouring example;
* an OOM reduces the batch size and retries. It never changes a scientific
  setting -- not the token budget, not the decoding rule.

``assert_batch_equivalence`` is the test-facing proof that batch size 1 and
batched decoding produce byte-identical text.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "GenerationConfig",
    "BatchedGenerator",
    "assert_batch_equivalence",
]


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Everything that decides what text comes out.

    Frozen and hashed into every task record, so a result can name the exact
    decoding rule that produced it.
    """

    max_new_tokens: int = 256
    max_prompt_tokens: int = 512
    do_sample: bool = False
    temperature: float = 0.0
    batch_size: int = 8
    #: Minimum batch size an OOM backoff may reach before giving up.
    min_batch_size: int = 1

    def __post_init__(self) -> None:
        if self.do_sample or self.temperature != 0.0:
            raise ValueError(
                "final-test generation must be deterministic: do_sample=False and temperature=0.0"
            )
        if self.max_new_tokens < 1 or self.max_prompt_tokens < 1:
            raise ValueError("token budgets must be positive")
        if not 1 <= self.min_batch_size <= self.batch_size:
            raise ValueError("min_batch_size must be between 1 and batch_size")

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "max_prompt_tokens": self.max_prompt_tokens,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            # batch_size is deliberately excluded: it is a throughput knob, and
            # including it would make the same decoding rule hash differently
            # after an OOM backoff.
            "decoding": "greedy",
        }


class BatchedGenerator:
    """Generate responses for a fixed prompt list, in order, deterministically."""

    def __init__(self, model: Any, tokenizer: Any, config: GenerationConfig) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.oom_backoffs: list[dict[str, int]] = []

    def _encode(self, prompts: Sequence[str]) -> Any:
        # Left padding: a decoder-only model continues from the final position.
        previous_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            return self.tokenizer(
                list(prompts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_prompt_tokens,
            )
        finally:
            self.tokenizer.padding_side = previous_side

    def _generate_chunk(self, prompts: Sequence[str]) -> list[str]:
        import torch

        encoded = self._encode(prompts).to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **encoded,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        prompt_length = encoded["input_ids"].shape[1]
        completions = output[:, prompt_length:]
        return [self.tokenizer.decode(row, skip_special_tokens=True) for row in completions]

    def generate(self, prompts: Sequence[str]) -> list[str]:
        """Responses aligned to ``prompts``, in the same order."""
        results: list[str] = []
        index = 0
        batch_size = self.config.batch_size
        while index < len(prompts):
            chunk = prompts[index : index + batch_size]
            try:
                results.extend(self._generate_chunk(chunk))
            except Exception as exc:
                if not _is_out_of_memory(exc) or batch_size <= self.config.min_batch_size:
                    raise
                # Halve and retry the same chunk. Nothing scientific changes:
                # the decoding rule and token budgets are untouched.
                new_size = max(self.config.min_batch_size, batch_size // 2)
                self.oom_backoffs.append({"at_index": index, "from": batch_size, "to": new_size})
                batch_size = new_size
                _empty_cache()
                continue
            index += len(chunk)
        return results


def _is_out_of_memory(exc: BaseException) -> bool:
    """Whether this is an allocator failure worth retrying with a smaller batch.

    torch raises both `torch.cuda.OutOfMemoryError` and, historically,
    `RuntimeError: CUDA out of memory`, and a wrapper may stringify the former
    into a message. All three spellings count; anything else is a real bug and
    must propagate.
    """
    haystack = f"{type(exc).__name__} {exc}".lower().replace(" ", "")
    return "outofmemory" in haystack


def _empty_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover - torch-free path never OOMs
        pass


def assert_batch_equivalence(generator: BatchedGenerator, prompts: Sequence[str]) -> list[str]:
    """Prove batched decoding matches batch size 1, or raise with the first diff.

    Batching exists to fit the compute budget, not to change the outputs. If
    this ever fails, the batched results are not usable as a benchmark result.
    """
    single = BatchedGenerator(
        generator.model,
        generator.tokenizer,
        GenerationConfig(
            max_new_tokens=generator.config.max_new_tokens,
            max_prompt_tokens=generator.config.max_prompt_tokens,
            batch_size=1,
            min_batch_size=1,
        ),
    ).generate(prompts)
    batched = generator.generate(prompts)

    for index, (one, many) in enumerate(zip(single, batched, strict=True)):
        if one != many:
            raise AssertionError(
                f"batched decoding diverged at prompt {index}: batch-size-1 produced "
                f"{one[:120]!r} but batched produced {many[:120]!r}"
            )
    return batched
