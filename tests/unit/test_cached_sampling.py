# ruff: noqa: E402

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

from miniverl.models.sampling import run_cached_padded_generation


def _logits(total: int, length: int):  # type: ignore[no-untyped-def]
    center = (total + length * 3) % 8
    return -torch.abs(torch.arange(8, dtype=torch.float32) - center)


def _run(rows, seeds, temperature):  # type: ignore[no-untyped-def]
    prefill_calls = 0
    decode_calls = 0

    def prefill(prefixes):  # type: ignore[no-untyped-def]
        nonlocal prefill_calls
        prefill_calls += 1
        state = [(sum(prefix), len(prefix)) for prefix in prefixes]
        return torch.stack([_logits(*item) for item in state]), state

    def decode(tokens, active, state):  # type: ignore[no-untyped-def]
        nonlocal decode_calls
        decode_calls += 1
        next_state = []
        output = []
        for token, is_active, (total, length) in zip(tokens, active, state, strict=True):
            if is_active:
                total += token
                length += 1
            next_state.append((total, length))
            output.append(_logits(total, length))
        return torch.stack(output), next_state

    outputs = run_cached_padded_generation(
        prefill=prefill,
        decode_step=decode,
        prefix_token_ids=rows,
        decode=lambda ids: ":".join(str(value) for value in ids),
        eos_token_id=99,
        max_new_tokens=5,
        temperature=temperature,
        top_p=0.93,
        top_k=6,
        generators=[torch.Generator(device="cpu").manual_seed(seed) for seed in seeds],
        record_logprobs=True,
    )
    return outputs, prefill_calls, decode_calls


@pytest.mark.parametrize("temperature", [0.0, 0.8])
def test_cached_sampling_is_batch_partition_invariant(temperature: float) -> None:
    rows = [[1, 2], [3, 4, 5], [6], [7, 1, 2, 3]]
    seeds = [101, 102, 103, 104]

    whole, prefill_calls, decode_calls = _run(rows, seeds, temperature)
    left, left_prefill, _ = _run(rows[:2], seeds[:2], temperature)
    right, right_prefill, _ = _run(rows[2:], seeds[2:], temperature)

    assert [output.token_ids for output in whole] == [
        output.token_ids for output in (*left, *right)
    ]
    assert [output.logprobs for output in whole] == [output.logprobs for output in (*left, *right)]
    assert prefill_calls == 1
    assert left_prefill == right_prefill == 1
    assert decode_calls == 4


def test_cached_sampling_tracks_eos_and_text_stop_per_row() -> None:
    calls = 0

    def prefill(rows):  # type: ignore[no-untyped-def]
        logits = torch.full((2, 5), -20.0)
        logits[0, 4] = 20.0
        logits[1, 2] = 20.0
        return logits, None

    def decode(tokens, active, state):  # type: ignore[no-untyped-def]
        nonlocal calls
        del tokens, active
        calls += 1
        logits = torch.full((2, 5), -20.0)
        logits[:, 1] = 20.0
        return logits, state

    outputs = run_cached_padded_generation(
        prefill=prefill,
        decode_step=decode,
        prefix_token_ids=[[1], [2]],
        decode=lambda ids: "".join(str(value) for value in ids),
        eos_token_id=4,
        max_new_tokens=3,
        stop_sequences=("21",),
        temperature=0.0,
        generators=[torch.Generator(), torch.Generator()],
    )

    assert outputs[0].stop_reason == "eos"
    assert outputs[0].token_ids == [4]
    assert outputs[1].stop_reason == "stop_sequence"
    assert outputs[1].matched_stop == "21"
    assert outputs[1].token_ids == [2, 1]
    assert calls == 1
