"""Deterministic sampling and the shared generation loop.

Both backends share this loop so stop-string handling, seeding and token
accounting are implemented -- and tested -- exactly once.  A backend only has
to supply a ``step`` callable that advances its own KV cache.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch

from miniverl.models.base import GenerationOutput

__all__ = [
    "sample_from_logits",
    "run_generation",
    "run_greedy_padded_generation",
    "run_cached_padded_generation",
    "StepFn",
]

#: ``step(new_token_ids, state) -> (next_token_logits [V], new_state)``
StepFn = Callable[[list[int], Any], "tuple[torch.Tensor, Any]"]


def sample_from_logits(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    generator: torch.Generator | None = None,
) -> int:
    """Sample one token id from ``[V]`` logits.

    ``temperature == 0`` is exact greedy decoding (argmax), which is what the
    evaluator uses so that a run is reproducible token for token.

    Filtering happens on whatever device the logits live on, but the final
    ``multinomial`` draw is always taken on the CPU with a CPU generator.  That
    costs one small device-to-host copy per token and buys two things: a CUDA
    tensor cannot be drawn with a CPU generator at all, and a given seed now
    produces the *same* token sequence on CPU and on GPU.
    """
    logits = logits.detach().to(torch.float32).flatten()
    if temperature <= 0.0:
        return int(torch.argmax(logits).item())

    scaled = logits / temperature
    if top_k and 0 < top_k < scaled.numel():
        threshold = torch.topk(scaled, top_k).values[-1]
        scaled = torch.where(scaled < threshold, torch.full_like(scaled, float("-inf")), scaled)
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        keep = cumulative - probs < top_p
        keep[0] = True
        filtered = torch.full_like(scaled, float("-inf"))
        filtered[sorted_idx[keep]] = scaled[sorted_idx[keep]]
        scaled = filtered
    probs = torch.softmax(scaled, dim=-1).to("cpu")
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


def run_generation(
    *,
    step: StepFn,
    prefix_token_ids: Sequence[int],
    decode: Callable[[Sequence[int]], str],
    eos_token_id: int,
    max_new_tokens: int,
    stop_sequences: Sequence[str] = (),
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    generator: torch.Generator | None = None,
    record_logprobs: bool = False,
) -> GenerationOutput:
    """Autoregressive loop with stop strings and a hard token budget.

    Every generated token is kept, including one that overshoots a stop string
    (a single token may decode to ``"</tool_call>\\n"``).  Nothing is silently
    dropped; the parser works on characters and the extra characters simply
    belong to the same model span.
    """
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")

    state: Any = None
    pending: list[int] = list(prefix_token_ids)
    generated: list[int] = []
    logprobs: list[float] = []
    stop_reason = "max_new_tokens"
    matched_stop: str | None = None

    for _ in range(max_new_tokens):
        logits, state = step(pending, state)
        token = sample_from_logits(
            logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            generator=generator,
        )
        if record_logprobs:
            log_probs = torch.log_softmax(logits.detach().to(torch.float32).flatten(), dim=-1)
            logprobs.append(float(log_probs[token].item()))
        generated.append(token)
        pending = [token]

        if token == eos_token_id:
            stop_reason = "eos"
            break

        text_so_far = decode(generated)
        hit = next((s for s in stop_sequences if s and s in text_so_far), None)
        if hit is not None:
            stop_reason = "stop_sequence"
            matched_stop = hit
            break

    text = decode(generated) if generated else ""
    return GenerationOutput(
        token_ids=generated,
        text=text,
        stop_reason=stop_reason,
        matched_stop=matched_stop,
        logprobs=logprobs,
    )


def run_greedy_padded_generation(
    *,
    step: Callable[[list[list[int]]], torch.Tensor],
    prefix_token_ids: Sequence[Sequence[int]],
    decode: Callable[[Sequence[int]], str],
    eos_token_id: int,
    max_new_tokens: int,
    stop_sequences: Sequence[str] = (),
    record_logprobs: bool = False,
) -> list[GenerationOutput]:
    """Greedy decode a real padded batch while retaining per-row stop state.

    ``step`` receives compact unpadded rows and performs the one padded model
    forward.  Re-padding on every step keeps each row's next-token state at its
    true sequence end; padding can never become context or a selected token.
    """
    if not prefix_token_ids:
        return []
    if any(not row for row in prefix_token_ids):
        raise ValueError("padded generation cannot contain an empty prompt")
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
    sequences = [list(row) for row in prefix_token_ids]
    generated: list[list[int]] = [[] for _ in sequences]
    reasons = ["max_new_tokens"] * len(sequences)
    matched: list[str | None] = [None] * len(sequences)
    logprobs: list[list[float]] = [[] for _ in sequences]
    active = [True] * len(sequences)
    for _ in range(max_new_tokens):
        logits = step(sequences)
        if tuple(logits.shape[:1]) != (len(sequences),):
            raise ValueError(
                f"padded generation step returned {tuple(logits.shape)}, expected [batch, vocab]"
            )
        for index in range(len(sequences)):
            if not active[index]:
                continue
            token = int(torch.argmax(logits[index].detach().to(torch.float32)).item())
            if record_logprobs:
                row_log_probs = torch.log_softmax(logits[index].detach().to(torch.float32), dim=-1)
                logprobs[index].append(float(row_log_probs[token].item()))
            generated[index].append(token)
            sequences[index].append(token)
            if token == eos_token_id:
                reasons[index] = "eos"
                active[index] = False
                continue
            text = decode(generated[index])
            hit = next((value for value in stop_sequences if value and value in text), None)
            if hit is not None:
                reasons[index] = "stop_sequence"
                matched[index] = hit
                active[index] = False
        if not any(active):
            break
    return [
        GenerationOutput(
            token_ids=row,
            text=decode(row),
            stop_reason=reasons[index],
            matched_stop=matched[index],
            logprobs=logprobs[index],
        )
        for index, row in enumerate(generated)
    ]


def run_cached_padded_generation(
    *,
    prefill: Callable[[list[list[int]]], tuple[torch.Tensor, Any]],
    decode_step: Callable[
        [Sequence[int] | torch.Tensor, list[bool], Any], tuple[torch.Tensor, Any]
    ],
    prefix_token_ids: Sequence[Sequence[int]],
    decode: Callable[[Sequence[int]], str],
    eos_token_id: int,
    max_new_tokens: int,
    stop_sequences: Sequence[str] = (),
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    generators: Sequence[torch.Generator] | None = None,
    record_logprobs: bool = False,
) -> list[GenerationOutput]:
    """Run one prefill and cache-aware incremental decode for a padded batch.

    Randomness belongs to logical rows, not the physical batch. A caller can
    therefore bisect or repartition a batch without changing sampled tokens.
    ``decode_step`` advances only rows marked active; inactive rows remain as
    masked placeholders so a standard batched KV cache can retain its shape.
    """

    if not prefix_token_ids:
        return []
    if any(not row for row in prefix_token_ids):
        raise ValueError("cached generation cannot contain an empty prompt")
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
    row_generators = list(generators or [torch.Generator(device="cpu") for _ in prefix_token_ids])
    if len(row_generators) != len(prefix_token_ids):
        raise ValueError("cached generation needs exactly one generator per prompt")

    rows = [list(row) for row in prefix_token_ids]
    logits, state = prefill(rows)
    if tuple(logits.shape[:1]) != (len(rows),):
        raise ValueError(f"cached prefill returned {tuple(logits.shape)}, expected [batch, vocab]")
    generated: list[list[int]] = [[] for _ in rows]
    logprobs: list[list[float]] = [[] for _ in rows]
    reasons = ["max_new_tokens"] * len(rows)
    matched: list[str | None] = [None] * len(rows)
    active = [True] * len(rows)

    for step_index in range(max_new_tokens):
        chosen = [0] * len(rows)
        device_tokens: torch.Tensor | None = None
        greedy_tokens: list[int] | None = None
        greedy_logprobs: list[float] | None = None
        if temperature <= 0.0:
            device_tokens = torch.argmax(logits.detach(), dim=-1)
            greedy_tokens = [int(value) for value in device_tokens.cpu().tolist()]
            if record_logprobs:
                row_log_probs = torch.log_softmax(logits.detach().to(torch.float32), dim=-1)
                selected = row_log_probs.gather(1, device_tokens.unsqueeze(1)).squeeze(1)
                greedy_logprobs = [float(value) for value in selected.cpu().tolist()]
        for index, is_active in enumerate(active):
            if not is_active:
                continue
            row_logits = logits[index]
            token = (
                greedy_tokens[index]
                if greedy_tokens is not None
                else sample_from_logits(
                    row_logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    generator=row_generators[index],
                )
            )
            chosen[index] = token
            if record_logprobs:
                if greedy_logprobs is not None:
                    logprobs[index].append(greedy_logprobs[index])
                else:
                    row_log_probs = torch.log_softmax(
                        row_logits.detach().to(torch.float32).flatten(), dim=-1
                    )
                    logprobs[index].append(float(row_log_probs[token].item()))
            generated[index].append(token)
            if token == eos_token_id:
                reasons[index] = "eos"
                active[index] = False
                continue
            text = decode(generated[index])
            hit = next((value for value in stop_sequences if value and value in text), None)
            if hit is not None:
                reasons[index] = "stop_sequence"
                matched[index] = hit
                active[index] = False

        if not any(active) or step_index + 1 == max_new_tokens:
            break
        logits, state = decode_step(
            device_tokens if device_tokens is not None else chosen,
            active,
            state,
        )
        if tuple(logits.shape[:1]) != (len(rows),):
            raise ValueError(
                f"cached decode returned {tuple(logits.shape)}, expected [batch, vocab]"
            )

    return [
        GenerationOutput(
            token_ids=row,
            text=decode(row),
            stop_reason=reasons[index],
            matched_stop=matched[index],
            logprobs=logprobs[index],
        )
        for index, row in enumerate(generated)
    ]
