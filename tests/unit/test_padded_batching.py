"""Typed padded batches and selected-state projection invariants."""

from __future__ import annotations

import pytest

from tests.conftest import requires_torch

pytestmark = [requires_torch, pytest.mark.torch]

torch = pytest.importorskip("torch")


def test_padded_batch_records_masks_lengths_and_selected_positions() -> None:
    from miniverl.training.batching import build_padded_trajectory_batch

    batch = build_padded_trajectory_batch(
        token_ids=[[11, 12, 13, 14], [21, 22]],
        selected_positions=[[0, 2], [1]],
        pad_token_id=0,
        device="cpu",
    )

    assert batch.input_ids.tolist() == [[11, 12, 13, 14], [21, 22, 0, 0]]
    assert batch.attention_mask.tolist() == [[True, True, True, True], [True, True, False, False]]
    assert batch.lengths == (4, 2)
    assert batch.selected_batch_indices.tolist() == [0, 0, 1]
    assert batch.selected_positions.tolist() == [0, 2, 1]
    assert batch.selected_offsets == (0, 2, 3)


def test_padded_batch_rejects_a_selected_padding_position() -> None:
    from miniverl.training.batching import build_padded_trajectory_batch

    with pytest.raises(ValueError, match="outside trajectory 1"):
        build_padded_trajectory_batch(
            token_ids=[[1, 2, 3], [4]],
            selected_positions=[[0], [1]],
            pad_token_id=0,
            device="cpu",
        )


def test_length_bucketing_is_deterministic_and_stable_on_ties() -> None:
    from miniverl.training.batching import deterministic_length_batches

    lengths = [8, 3, 8, 5, 3, 9]
    assert deterministic_length_batches(lengths, batch_size=2) == ((1, 4), (3, 0), (2, 5))
    assert deterministic_length_batches(lengths, batch_size=4) == ((1, 4, 3, 0), (2, 5))


def test_physical_update_batches_respect_count_and_padded_token_limits() -> None:
    from miniverl.training.batching import deterministic_padded_token_batches

    lengths = [8, 2, 10, 5, 2, 9]
    batches = deterministic_padded_token_batches(
        lengths,
        batch_size=4,
        max_padded_tokens=18,
    )

    assert batches == ((1, 4, 3), (0, 5), (2,))
    assert all(len(batch) <= 4 for batch in batches)
    assert all(max(lengths[index] for index in batch) * len(batch) <= 18 for batch in batches)


def test_physical_update_token_limit_rejects_one_oversized_trajectory() -> None:
    from miniverl.training.batching import deterministic_padded_token_batches

    with pytest.raises(ValueError, match="trajectory 1 has 20 tokens"):
        deterministic_padded_token_batches([4, 20], batch_size=2, max_padded_tokens=16)


def test_toy_padded_hidden_states_match_sequential_for_variable_lengths() -> None:
    from miniverl.models.tokenizers import ToyTokenizer
    from miniverl.models.toy import ToyBackend
    from miniverl.training.batching import build_padded_trajectory_batch

    tokenizer = ToyTokenizer()
    backend = ToyBackend(tokenizer=tokenizer, model_id="toy", seed=17, trainable=True)
    sequences = [tokenizer.encode("short"), tokenizer.encode("a longer sequence")]
    positions = [[0, len(sequences[0]) - 1], [1, len(sequences[1]) - 1]]
    batch = build_padded_trajectory_batch(
        token_ids=sequences,
        selected_positions=positions,
        pad_token_id=tokenizer.pad_token_id,
        device="cpu",
    )

    actual = backend.hidden_states_at_batch(batch, with_grad=False)
    expected = torch.cat(
        [
            backend.hidden_states_at(sequence, selected, with_grad=False)
            for sequence, selected in zip(sequences, positions, strict=True)
        ]
    )
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_padding_token_values_cannot_change_valid_hidden_states() -> None:
    from miniverl.models.tokenizers import ToyTokenizer
    from miniverl.models.toy import ToyBackend
    from miniverl.training.batching import build_padded_trajectory_batch

    tokenizer = ToyTokenizer()
    backend = ToyBackend(tokenizer=tokenizer, model_id="toy", seed=23, trainable=True)
    sequences = [tokenizer.encode("x"), tokenizer.encode("padding isolation")]
    positions = [[0], [len(sequences[1]) - 1]]
    batch = build_padded_trajectory_batch(
        token_ids=sequences,
        selected_positions=positions,
        pad_token_id=tokenizer.pad_token_id,
        device="cpu",
    )
    changed = batch.with_input_ids(batch.input_ids.clone())
    changed.input_ids[0, 1:] = (changed.input_ids[0, 1:] + 7) % tokenizer.vocab_size

    expected = backend.hidden_states_at_batch(batch, with_grad=False)
    actual = backend.hidden_states_at_batch(changed, with_grad=False)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("target_shape", ["exact", "bucketed"])
def test_padded_loss_and_gradients_match_sequential_per_trajectory_reduction(
    target_shape: str,
) -> None:
    from miniverl.losses.bucketed import teacher_topk_targets
    from miniverl.losses.chunked import (
        BucketedTargetProvider,
        ExactTargetProvider,
        chunked_selected_position_loss,
    )
    from miniverl.models.tokenizers import ToyTokenizer
    from miniverl.models.toy import ToyBackend
    from miniverl.training.batching import (
        build_padded_trajectory_batch,
        concatenate_target_providers,
        normalize_trajectory_weights,
    )

    tokenizer = ToyTokenizer()
    sequential = ToyBackend(tokenizer=tokenizer, model_id="toy", seed=31, trainable=True)
    padded = ToyBackend(tokenizer=tokenizer, model_id="toy", seed=31, trainable=True)
    sequences = [tokenizer.encode("abc"), tokenizer.encode("a substantially longer row")]
    positions = [[0, 1], [0, 3, len(sequences[1]) - 1]]
    weights = [torch.tensor([1.0, 3.0]), torch.tensor([2.0, 1.0, 4.0])]
    targets = [
        torch.tensor([sequences[0][1], sequences[0][2]]),
        torch.tensor([sequences[1][1], sequences[1][4], sequences[1][-1]]),
    ]
    generator = torch.Generator().manual_seed(919)
    teacher_logits = [
        torch.randn(len(row), tokenizer.vocab_size, generator=generator) for row in positions
    ]

    def providers():  # type: ignore[no-untyped-def]
        if target_shape == "exact":
            return [
                ExactTargetProvider(teacher_logits_fn=lambda a, b, values=values: values[a:b])
                for values in teacher_logits
            ]
        output = []
        for values in teacher_logits:
            indices, log_probs, tail = teacher_topk_targets(values, top_k=8)
            output.append(
                BucketedTargetProvider(
                    topk_indices=indices,
                    topk_log_probs=log_probs,
                    tail_log_prob=tail,
                )
            )
        return output

    reference_providers = providers()
    for sequence, selected, weight, target, provider in zip(
        sequences, positions, weights, targets, reference_providers, strict=True
    ):
        hidden = sequential.hidden_states_at(sequence, selected, with_grad=True)
        chunked_selected_position_loss(
            hidden_states=hidden,
            lm_head=sequential.project,
            weights=weight,
            provider=provider,
            target_token_ids=target,
            chunk_size=2,
            backward=True,
            loss_scale=0.5,
        )

    batch = build_padded_trajectory_batch(
        token_ids=sequences,
        selected_positions=positions,
        pad_token_id=tokenizer.pad_token_id,
        device="cpu",
    )
    hidden = padded.hidden_states_at_batch(batch, with_grad=True)
    chunked_selected_position_loss(
        hidden_states=hidden,
        lm_head=padded.project,
        weights=normalize_trajectory_weights(weights),
        weight_normalizer=float(len(sequences)),
        provider=concatenate_target_providers(providers(), tuple(len(row) for row in positions)),
        target_token_ids=torch.cat(targets),
        chunk_size=2,
        backward=True,
    )

    sequential_grads = {
        name: parameter.grad for name, parameter in sequential.model.named_parameters()
    }
    padded_grads = {name: parameter.grad for name, parameter in padded.model.named_parameters()}
    assert sequential_grads.keys() == padded_grads.keys()
    for name in sequential_grads:
        assert sequential_grads[name] is not None, name
        assert padded_grads[name] is not None, name
        assert torch.allclose(sequential_grads[name], padded_grads[name], atol=2e-6, rtol=2e-5), (
            name
        )


def test_batched_projection_never_receives_batch_or_sequence_dimensions() -> None:
    from miniverl.losses.chunked import chunked_selected_position_loss
    from miniverl.models.tokenizers import ToyTokenizer
    from miniverl.models.toy import ToyBackend
    from miniverl.training.batching import (
        build_padded_trajectory_batch,
        normalize_trajectory_weights,
    )

    tokenizer = ToyTokenizer()
    backend = ToyBackend(tokenizer=tokenizer, model_id="toy", seed=41, trainable=True)
    sequences = [tokenizer.encode("small"), tokenizer.encode("larger selected sequence")]
    positions = [[0, 1], [0, 2, 4, 6]]
    batch = build_padded_trajectory_batch(
        token_ids=sequences,
        selected_positions=positions,
        pad_token_id=tokenizer.pad_token_id,
        device="cpu",
    )
    observed: list[tuple[int, ...]] = []

    def project(hidden):  # type: ignore[no-untyped-def]
        observed.append(tuple(hidden.shape))
        return backend.project(hidden)

    hidden = backend.hidden_states_at_batch(batch, with_grad=True)
    targets = torch.tensor(
        [sequences[i][position + 1] for i, row in enumerate(positions) for position in row]
    )
    chunked_selected_position_loss(
        hidden_states=hidden,
        lm_head=project,
        weights=normalize_trajectory_weights([torch.ones(2), torch.ones(4)]),
        weight_normalizer=2.0,
        target_token_ids=targets,
        ce_weight=1.0,
        chunk_size=3,
        backward=True,
    )
    assert observed == [(3, backend.hidden_size), (3, backend.hidden_size)]
