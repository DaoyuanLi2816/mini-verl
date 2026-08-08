"""Chunked selected-position loss: value and gradient equivalence.

The two-stage backward trick in :mod:`miniverl.losses.chunked` is the single
most load-bearing systems optimization in miniVERL.  If it were subtly wrong
every GPU run would be silently mistrained, so it is checked against a naive
reference that materializes everything at once.
"""

from __future__ import annotations

import pytest

from tests.conftest import requires_torch

pytestmark = [requires_torch, pytest.mark.torch]

torch = pytest.importorskip("torch")


class _Backbone(torch.nn.Module):
    """A trivial stand-in for a transformer backbone."""

    def __init__(self, hidden: int, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.linear = torch.nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and squash."""
        return torch.tanh(self.linear(x))


def _setup(n: int = 37, hidden: int = 16, vocab: int = 48, seed: int = 2026):
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(n, hidden, generator=generator)
    backbone = _Backbone(hidden, seed=seed)
    lm_head = torch.nn.Linear(hidden, vocab, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(torch.randn(vocab, hidden, generator=generator) * 0.1)
    teacher_logits = torch.randn(n, vocab, generator=generator) * 2.0
    weights = torch.rand(n, generator=generator) + 0.1
    targets = torch.randint(0, vocab, (n,), generator=generator)
    return inputs, backbone, lm_head, teacher_logits, weights, targets


def _naive_loss(hidden, lm_head, teacher_logits, weights, divergence="reverse_kl"):
    from miniverl.losses.exact import exact_divergence

    student_logits = lm_head(hidden)
    per_token = exact_divergence(teacher_logits, student_logits, divergence=divergence)
    return (per_token * weights).sum() / weights.sum()


@pytest.mark.parametrize("chunk_size", [1, 3, 8, 37, 1000])
def test_chunked_value_matches_unchunked(chunk_size):
    from miniverl.losses.chunked import ExactTargetProvider, chunked_selected_position_loss

    inputs, backbone, lm_head, teacher_logits, weights, _ = _setup()
    hidden = backbone(inputs)
    provider = ExactTargetProvider(
        teacher_logits_fn=lambda a, b: teacher_logits[a:b], divergence_name="reverse_kl"
    )
    out = chunked_selected_position_loss(
        hidden_states=hidden,
        lm_head=lm_head,
        weights=weights,
        provider=provider,
        chunk_size=chunk_size,
        backward=False,
    )
    reference = _naive_loss(hidden, lm_head, teacher_logits, weights)
    assert out.loss == pytest.approx(float(reference.detach()), abs=1e-5)
    assert out.num_positions == 37
    assert out.num_chunks == (37 + chunk_size - 1) // chunk_size if chunk_size <= 37 else 1


@pytest.mark.parametrize("chunk_size", [1, 5, 37])
@pytest.mark.parametrize("divergence", ["forward_kl", "reverse_kl", "jsd"])
def test_chunked_gradients_match_unchunked(chunk_size, divergence):
    from miniverl.losses.chunked import ExactTargetProvider, chunked_selected_position_loss

    inputs, backbone, lm_head, teacher_logits, weights, _ = _setup()

    # Reference: one big backward.
    ref_backbone = _Backbone(16, seed=2026)
    ref_head = torch.nn.Linear(16, 48, bias=False)
    with torch.no_grad():
        ref_head.weight.copy_(lm_head.weight)
    ref_hidden = ref_backbone(inputs)
    from miniverl.losses.exact import exact_divergence

    per_token = exact_divergence(teacher_logits, ref_head(ref_hidden), divergence=divergence)
    ((per_token * weights).sum() / weights.sum()).backward()

    # Chunked path.
    hidden = backbone(inputs)
    provider = ExactTargetProvider(
        teacher_logits_fn=lambda a, b: teacher_logits[a:b], divergence_name=divergence
    )
    chunked_selected_position_loss(
        hidden_states=hidden,
        lm_head=lm_head,
        weights=weights,
        provider=provider,
        chunk_size=chunk_size,
        backward=True,
    )

    assert backbone.linear.weight.grad is not None
    assert torch.allclose(backbone.linear.weight.grad, ref_backbone.linear.weight.grad, atol=1e-5)
    assert torch.allclose(backbone.linear.bias.grad, ref_backbone.linear.bias.grad, atol=1e-5)
    assert torch.allclose(lm_head.weight.grad, ref_head.weight.grad, atol=1e-5)


def test_zero_weight_positions_contribute_nothing():
    """A masked position must change neither the loss nor any gradient."""
    from miniverl.losses.chunked import ExactTargetProvider, chunked_selected_position_loss

    inputs, backbone, lm_head, teacher_logits, weights, _ = _setup(n=12)
    masked = weights.clone()
    masked[3] = 0.0
    masked[7] = 0.0
    provider = ExactTargetProvider(teacher_logits_fn=lambda a, b: teacher_logits[a:b])

    hidden = backbone(inputs)
    out_masked = chunked_selected_position_loss(
        hidden_states=hidden,
        lm_head=lm_head,
        weights=masked,
        provider=provider,
        chunk_size=4,
        backward=True,
    )
    grad_masked = backbone.linear.weight.grad.clone()

    keep = [i for i in range(12) if i not in (3, 7)]
    backbone2 = _Backbone(16, seed=2026)
    head2 = torch.nn.Linear(16, 48, bias=False)
    with torch.no_grad():
        head2.weight.copy_(lm_head.weight)
    sub_teacher = teacher_logits[keep]
    provider2 = ExactTargetProvider(teacher_logits_fn=lambda a, b: sub_teacher[a:b])
    hidden2 = backbone2(inputs)[keep]
    out_kept = chunked_selected_position_loss(
        hidden_states=hidden2,
        lm_head=head2,
        weights=masked[keep],
        provider=provider2,
        chunk_size=4,
        backward=True,
    )
    assert out_masked.loss == pytest.approx(out_kept.loss, abs=1e-6)
    assert torch.allclose(grad_masked, backbone2.linear.weight.grad, atol=1e-6)


def test_all_zero_weights_give_a_safe_zero_loss():
    from miniverl.losses.chunked import ExactTargetProvider, chunked_selected_position_loss

    inputs, backbone, lm_head, teacher_logits, _, _ = _setup(n=6)
    hidden = backbone(inputs)
    out = chunked_selected_position_loss(
        hidden_states=hidden,
        lm_head=lm_head,
        weights=torch.zeros(6),
        provider=ExactTargetProvider(teacher_logits_fn=lambda a, b: teacher_logits[a:b]),
        chunk_size=2,
        backward=True,
    )
    assert out.loss == pytest.approx(0.0, abs=1e-9)
    assert out.total_weight == pytest.approx(0.0)
    assert backbone.linear.weight.grad is not None
    assert float(backbone.linear.weight.grad.abs().sum()) == pytest.approx(0.0, abs=1e-9)


def test_per_token_components_for_pure_sft_are_explicit() -> None:
    from miniverl.losses.chunked import chunked_selected_position_loss

    inputs, backbone, lm_head, _, weights, targets = _setup(n=7)
    output = chunked_selected_position_loss(
        hidden_states=backbone(inputs),
        lm_head=lm_head,
        weights=weights,
        provider=None,
        target_token_ids=targets,
        ce_weight=1.0,
        chunk_size=3,
    )
    assert output.per_token_divergence is None
    assert output.per_token_ce is not None
    assert torch.equal(output.per_token_objective, output.per_token_ce)
    assert len(output.per_token_objective) == 7
    expected = float(((output.per_token_objective * weights).sum() / weights.sum()).detach())
    assert output.loss == pytest.approx(expected, abs=1e-6)


def test_per_token_components_for_pure_and_mixed_distillation() -> None:
    from miniverl.losses.chunked import ExactTargetProvider, chunked_selected_position_loss

    inputs, backbone, lm_head, teacher_logits, weights, targets = _setup(n=9)
    provider = ExactTargetProvider(teacher_logits_fn=lambda start, end: teacher_logits[start:end])
    pure = chunked_selected_position_loss(
        hidden_states=backbone(inputs),
        lm_head=lm_head,
        weights=weights,
        provider=provider,
        target_token_ids=targets,
        ce_weight=0.0,
        chunk_size=4,
    )
    assert pure.per_token_divergence is not None
    assert pure.per_token_ce is None
    assert torch.equal(pure.per_token_objective, pure.per_token_divergence)

    mixed = chunked_selected_position_loss(
        hidden_states=backbone(inputs),
        lm_head=lm_head,
        weights=weights,
        provider=provider,
        target_token_ids=targets,
        ce_weight=0.25,
        chunk_size=4,
    )
    assert mixed.per_token_divergence is not None
    assert mixed.per_token_ce is not None
    expected_tokens = 0.75 * mixed.per_token_divergence + 0.25 * mixed.per_token_ce
    assert torch.allclose(mixed.per_token_objective, expected_tokens)
    expected_loss = float(((expected_tokens * weights).sum() / weights.sum()).detach())
    assert mixed.loss == pytest.approx(expected_loss, abs=1e-6)


def test_empty_batch_preserves_component_applicability() -> None:
    from miniverl.losses.chunked import ExactTargetProvider, chunked_selected_position_loss

    empty_hidden = torch.zeros(0, 3)
    head = torch.nn.Linear(3, 5)
    provider = ExactTargetProvider(teacher_logits_fn=lambda _start, _end: torch.zeros(0, 5))
    output = chunked_selected_position_loss(
        hidden_states=empty_hidden,
        lm_head=head,
        weights=torch.zeros(0),
        provider=provider,
        chunk_size=2,
    )
    assert output.per_token_objective.numel() == 0
    assert output.per_token_divergence is not None
    assert output.per_token_divergence.numel() == 0
    assert output.per_token_ce is None


def test_span_objective_aggregation_uses_token_weights_and_strict_lengths() -> None:
    from types import SimpleNamespace

    from miniverl.trainer import OPDTrainer

    sample = SimpleNamespace(
        alignment=SimpleNamespace(
            token_weights=[1.0, 9.0],
            span_types=["assistant_final", "assistant_final"],
        )
    )
    totals = OPDTrainer._loss_by_span_type(
        None,
        sample,
        torch.tensor([10.0, 0.0]),
    )
    assert totals["assistant_final"] == [10.0, 10.0]
    assert totals["assistant_final"][0] / totals["assistant_final"][1] == pytest.approx(1.0)

    broken = SimpleNamespace(
        alignment=SimpleNamespace(
            token_weights=[1.0],
            span_types=["assistant_final", "assistant_text"],
        )
    )
    with pytest.raises(ValueError, match="zip"):
        OPDTrainer._loss_by_span_type(None, broken, torch.tensor([1.0, 2.0]))


def test_empty_selection_is_a_documented_no_op():
    from miniverl.losses.chunked import ExactTargetProvider, chunked_selected_position_loss

    out = chunked_selected_position_loss(
        hidden_states=torch.zeros(0, 16),
        lm_head=torch.nn.Linear(16, 8, bias=False),
        weights=torch.zeros(0),
        provider=ExactTargetProvider(teacher_logits_fn=lambda a, b: torch.zeros(0, 8)),
        chunk_size=4,
        backward=True,
    )
    assert out.loss == 0.0
    assert out.num_positions == 0
    assert out.metrics["empty_batch"] == 1.0


def test_bucketed_provider_matches_direct_call():
    from miniverl.losses.bucketed import bucketed_divergence, teacher_topk_targets
    from miniverl.losses.chunked import BucketedTargetProvider, chunked_selected_position_loss

    inputs, backbone, lm_head, teacher_logits, weights, _ = _setup(n=20, vocab=64)
    idx, topk_lp, tail_lp = teacher_topk_targets(teacher_logits, top_k=8)
    hidden = backbone(inputs)
    provider = BucketedTargetProvider(
        topk_indices=idx, topk_log_probs=topk_lp, tail_log_prob=tail_lp
    )
    out = chunked_selected_position_loss(
        hidden_states=hidden,
        lm_head=lm_head,
        weights=weights,
        provider=provider,
        chunk_size=7,
        backward=False,
        collect_teacher_entropy=True,
    )
    direct = bucketed_divergence(
        teacher_topk_log_probs=topk_lp,
        teacher_tail_log_prob=tail_lp,
        topk_indices=idx,
        student_logits=lm_head(hidden),
        divergence="reverse_kl",
    )
    # detach before float(): this is a reference value, not part of any graph,
    # and converting a requires_grad tensor to a scalar warns.
    expected = float(((direct * weights).sum() / weights.sum()).detach())
    assert out.loss == pytest.approx(expected, abs=1e-6)
    assert out.teacher_entropy is not None
    assert out.teacher_entropy.shape == (20,)


def test_cross_entropy_only_mode_matches_torch_reference():
    from miniverl.losses.chunked import chunked_selected_position_loss

    inputs, backbone, lm_head, _, weights, targets = _setup(n=15, vocab=32)
    hidden = backbone(inputs)
    out = chunked_selected_position_loss(
        hidden_states=hidden,
        lm_head=lm_head,
        weights=weights,
        provider=None,
        target_token_ids=targets,
        ce_weight=1.0,
        chunk_size=4,
        backward=False,
    )
    reference_per_token = torch.nn.functional.cross_entropy(
        lm_head(hidden).float(), targets, reduction="none"
    )
    expected = float(((reference_per_token * weights).sum() / weights.sum()).detach())
    assert out.loss == pytest.approx(expected, abs=1e-6)


def test_ce_mixing_is_a_convex_combination():
    from miniverl.losses.chunked import ExactTargetProvider, chunked_selected_position_loss

    inputs, backbone, lm_head, teacher_logits, weights, targets = _setup(n=11, vocab=32)
    hidden = backbone(inputs)
    provider = ExactTargetProvider(teacher_logits_fn=lambda a, b: teacher_logits[a:b])
    kwargs = {
        "hidden_states": hidden,
        "lm_head": lm_head,
        "weights": weights,
        "target_token_ids": targets,
        "chunk_size": 5,
        "backward": False,
    }
    kd_only = chunked_selected_position_loss(provider=provider, ce_weight=0.0, **kwargs)
    ce_only = chunked_selected_position_loss(provider=None, ce_weight=1.0, **kwargs)
    mixed = chunked_selected_position_loss(provider=provider, ce_weight=0.25, **kwargs)
    assert mixed.loss == pytest.approx(0.75 * kd_only.loss + 0.25 * ce_only.loss, abs=1e-6)


def test_loss_scale_only_affects_gradients():
    from miniverl.losses.chunked import ExactTargetProvider, chunked_selected_position_loss

    inputs, backbone, lm_head, teacher_logits, weights, _ = _setup(n=9)
    provider = ExactTargetProvider(teacher_logits_fn=lambda a, b: teacher_logits[a:b])
    hidden = backbone(inputs)
    out = chunked_selected_position_loss(
        hidden_states=hidden,
        lm_head=lm_head,
        weights=weights,
        provider=provider,
        chunk_size=3,
        backward=True,
        loss_scale=0.5,
    )
    scaled_grad = backbone.linear.weight.grad.clone()

    backbone2 = _Backbone(16, seed=2026)
    head2 = torch.nn.Linear(16, 48, bias=False)
    with torch.no_grad():
        head2.weight.copy_(lm_head.weight)
    out2 = chunked_selected_position_loss(
        hidden_states=backbone2(inputs),
        lm_head=head2,
        weights=weights,
        provider=ExactTargetProvider(teacher_logits_fn=lambda a, b: teacher_logits[a:b]),
        chunk_size=3,
        backward=True,
        loss_scale=1.0,
    )
    assert out.loss == pytest.approx(out2.loss, abs=1e-6)
    assert torch.allclose(scaled_grad, 0.5 * backbone2.linear.weight.grad, atol=1e-6)


def test_invalid_arguments_are_rejected():
    from miniverl.losses.chunked import chunked_selected_position_loss

    with pytest.raises(ValueError, match="teacher target provider or ce_weight"):
        chunked_selected_position_loss(
            hidden_states=torch.zeros(2, 4),
            lm_head=torch.nn.Linear(4, 5),
            weights=torch.ones(2),
            provider=None,
            ce_weight=0.0,
        )
    with pytest.raises(ValueError, match="target_token_ids are required"):
        chunked_selected_position_loss(
            hidden_states=torch.zeros(2, 4),
            lm_head=torch.nn.Linear(4, 5),
            weights=torch.ones(2),
            provider=None,
            ce_weight=1.0,
        )
    with pytest.raises(ValueError, match="chunk_size must be"):
        chunked_selected_position_loss(
            hidden_states=torch.zeros(2, 4),
            lm_head=torch.nn.Linear(4, 5),
            weights=torch.ones(2),
            provider=None,
            target_token_ids=torch.zeros(2, dtype=torch.long),
            ce_weight=1.0,
            chunk_size=0,
        )
