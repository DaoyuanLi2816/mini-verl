"""Compressed ``top-k + tail`` divergences."""

from __future__ import annotations

import math

import pytest

from tests.conftest import requires_torch

pytestmark = [requires_torch, pytest.mark.torch]

torch = pytest.importorskip("torch")


@pytest.fixture
def logits() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(4080)
    teacher = torch.randn(9, 64, generator=generator) * 2.5
    student = torch.randn(9, 64, generator=generator) * 2.5
    return teacher, student


def test_log1mexp_matches_reference():
    from miniverl.losses.numerics import log1mexp

    xs = torch.tensor([-1e-6, -1e-3, -0.1, -0.6931471805599453, -1.0, -10.0, -60.0])
    got = log1mexp(xs)
    expected = torch.tensor([math.log1p(-math.exp(float(x))) for x in xs])
    assert torch.allclose(got, expected, atol=1e-6)
    assert bool(torch.isfinite(got).all())


def test_log1mexp_is_finite_and_differentiable_near_zero():
    from miniverl.losses.numerics import log1mexp

    x = torch.tensor([-1e-12, -1e-9, -1e-3], requires_grad=True)
    out = log1mexp(x)
    assert bool(torch.isfinite(out).all())
    out.sum().backward()
    assert x.grad is not None
    assert bool(torch.isfinite(x.grad).all())


def test_topk_targets_have_valid_tail_mass(logits):
    from miniverl.losses.bucketed import teacher_topk_targets

    teacher, _ = logits
    idx, topk_lp, tail_lp = teacher_topk_targets(teacher, top_k=8)
    assert idx.shape == (9, 8)
    covered = torch.logsumexp(topk_lp, dim=-1).exp()
    tail = tail_lp.exp()
    assert bool(((covered + tail - 1.0).abs() < 1e-5).all())
    assert bool((tail >= 0.0).all())
    assert bool((covered <= 1.0 + 1e-6).all())


def test_topk_equal_to_vocab_gives_exactly_empty_tail(logits):
    from miniverl.losses.bucketed import teacher_topk_targets

    teacher, _ = logits
    _, topk_lp, tail_lp = teacher_topk_targets(teacher, top_k=teacher.shape[-1])
    assert bool(torch.isinf(tail_lp).all()) and bool((tail_lp < 0).all())
    assert torch.allclose(torch.logsumexp(topk_lp, dim=-1), torch.zeros(9), atol=1e-6)


@pytest.mark.parametrize("divergence", ["forward_kl", "reverse_kl", "jsd"])
def test_full_k_converges_to_the_exact_loss(logits, divergence):
    """With ``k == V`` the coarse-graining is the identity, so the two agree."""
    from miniverl.losses.bucketed import bucketed_divergence, teacher_topk_targets
    from miniverl.losses.exact import exact_divergence

    teacher, student = logits
    vocab = teacher.shape[-1]
    idx, topk_lp, tail_lp = teacher_topk_targets(teacher, top_k=vocab)
    bucketed = bucketed_divergence(
        teacher_topk_log_probs=topk_lp,
        teacher_tail_log_prob=tail_lp,
        topk_indices=idx,
        student_logits=student,
        divergence=divergence,
    )
    exact = exact_divergence(teacher, student, divergence=divergence)
    assert torch.allclose(bucketed, exact, atol=1e-5)


@pytest.mark.parametrize("divergence", ["forward_kl", "reverse_kl", "jsd"])
@pytest.mark.parametrize("top_k", [1, 2, 8, 32])
def test_bucketed_lower_bounds_exact(logits, divergence, top_k):
    """Coarse-graining can only destroy information (data-processing inequality)."""
    from miniverl.losses.bucketed import bucketed_divergence, teacher_topk_targets
    from miniverl.losses.exact import exact_divergence

    teacher, student = logits
    idx, topk_lp, tail_lp = teacher_topk_targets(teacher, top_k=top_k)
    bucketed = bucketed_divergence(
        teacher_topk_log_probs=topk_lp,
        teacher_tail_log_prob=tail_lp,
        topk_indices=idx,
        student_logits=student,
        divergence=divergence,
    )
    exact = exact_divergence(teacher, student, divergence=divergence)
    assert bool((bucketed <= exact + 1e-5).all())
    assert bool((bucketed >= -1e-6).all())


def test_bucketed_is_monotone_non_decreasing_in_k(logits):
    from miniverl.losses.bucketed import bucketed_divergence, teacher_topk_targets

    teacher, student = logits
    previous = None
    for top_k in (1, 2, 4, 8, 16, 32, 64):
        idx, topk_lp, tail_lp = teacher_topk_targets(teacher, top_k=top_k)
        value = bucketed_divergence(
            teacher_topk_log_probs=topk_lp,
            teacher_tail_log_prob=tail_lp,
            topk_indices=idx,
            student_logits=student,
            divergence="forward_kl",
        )
        if previous is not None:
            assert bool((value >= previous - 1e-5).all())
        previous = value


def test_identical_distributions_give_zero(logits):
    from miniverl.losses.bucketed import bucketed_divergence, teacher_topk_targets

    teacher, _ = logits
    idx, topk_lp, tail_lp = teacher_topk_targets(teacher, top_k=16)
    for divergence in ("forward_kl", "reverse_kl", "jsd"):
        value = bucketed_divergence(
            teacher_topk_log_probs=topk_lp,
            teacher_tail_log_prob=tail_lp,
            topk_indices=idx,
            student_logits=teacher.clone(),
            divergence=divergence,
        )
        assert torch.allclose(value, torch.zeros_like(value), atol=1e-6)


def test_tail_edge_case_when_topk_mass_is_almost_one():
    """A near-deterministic teacher must not produce inf or nan."""
    from miniverl.losses.bucketed import bucketed_divergence, teacher_topk_targets

    teacher = torch.full((3, 128), -60.0)
    teacher[:, 0] = 60.0  # essentially all mass on one token
    student = torch.randn(3, 128, generator=torch.Generator().manual_seed(1))
    idx, topk_lp, tail_lp = teacher_topk_targets(teacher, top_k=4)
    for divergence in ("forward_kl", "reverse_kl", "jsd"):
        value = bucketed_divergence(
            teacher_topk_log_probs=topk_lp,
            teacher_tail_log_prob=tail_lp,
            topk_indices=idx,
            student_logits=student,
            divergence=divergence,
        )
        assert bool(torch.isfinite(value).all()), divergence
        assert bool((value >= -1e-6).all())


def test_reverse_kl_tail_penalty_is_bounded_by_log_one_over_epsilon():
    """Teacher tail == 0 while the student leaks mass must stay bounded."""
    from miniverl.losses.bucketed import bucketed_divergence, teacher_topk_targets

    teacher = torch.full((1, 256), -80.0)
    teacher[0, :2] = 40.0
    student = torch.zeros(1, 256)  # uniform: 254/256 of its mass is in the tail
    idx, topk_lp, tail_lp = teacher_topk_targets(teacher, top_k=2)
    tail_epsilon = 1e-9
    value = bucketed_divergence(
        teacher_topk_log_probs=topk_lp,
        teacher_tail_log_prob=tail_lp,
        topk_indices=idx,
        student_logits=student,
        divergence="reverse_kl",
        tail_epsilon=tail_epsilon,
    )
    assert bool(torch.isfinite(value).all())
    assert float(value.max()) <= math.log(1.0 / tail_epsilon) + 1.0


@pytest.mark.parametrize("divergence", ["forward_kl", "reverse_kl", "jsd"])
def test_gradients_flow_to_the_student_only(logits, divergence):
    from miniverl.losses.bucketed import bucketed_divergence, teacher_topk_targets

    teacher, student = logits
    idx, topk_lp, tail_lp = teacher_topk_targets(teacher, top_k=8)
    student = student.clone().requires_grad_(True)
    value = bucketed_divergence(
        teacher_topk_log_probs=topk_lp,
        teacher_tail_log_prob=tail_lp,
        topk_indices=idx,
        student_logits=student,
        divergence=divergence,
    ).sum()
    value.backward()
    assert student.grad is not None
    assert bool(torch.isfinite(student.grad).all())
    assert float(student.grad.abs().sum()) > 0.0


def test_student_bucket_log_probs_sum_to_one(logits):
    from miniverl.losses.bucketed import student_bucket_log_probs, teacher_topk_targets

    teacher, student = logits
    idx, _, _ = teacher_topk_targets(teacher, top_k=10)
    topk, tail = student_bucket_log_probs(student, idx)
    total = torch.logsumexp(torch.cat([topk, tail.unsqueeze(-1)], dim=-1), dim=-1)
    assert torch.allclose(total, torch.zeros_like(total), atol=1e-5)


def test_bucketed_entropy_lower_bounds_the_exact_entropy(logits):
    from miniverl.losses.bucketed import bucketed_teacher_entropy, teacher_topk_targets
    from miniverl.losses.exact import exact_teacher_entropy

    teacher, _ = logits
    _, topk_lp, tail_lp = teacher_topk_targets(teacher, top_k=8)
    coarse = bucketed_teacher_entropy(topk_lp, tail_lp)
    exact = exact_teacher_entropy(teacher)
    assert bool((coarse <= exact + 1e-5).all())


def test_half_precision_student_logits_are_upcast(logits):
    from miniverl.losses.bucketed import bucketed_divergence, teacher_topk_targets

    teacher, student = logits
    idx, topk_lp, tail_lp = teacher_topk_targets(teacher, top_k=8)
    value = bucketed_divergence(
        teacher_topk_log_probs=topk_lp,
        teacher_tail_log_prob=tail_lp,
        topk_indices=idx,
        student_logits=student.to(torch.bfloat16),
        divergence="reverse_kl",
    )
    assert value.dtype == torch.float32
    assert bool(torch.isfinite(value).all())


def test_invalid_arguments_are_rejected(logits):
    from miniverl.losses.bucketed import (
        bucketed_divergence,
        build_bucket_distributions,
        teacher_topk_targets,
    )

    teacher, student = logits
    with pytest.raises(ValueError, match="top_k must be"):
        teacher_topk_targets(teacher, top_k=0)
    idx, topk_lp, tail_lp = teacher_topk_targets(teacher, top_k=4)
    with pytest.raises(ValueError, match="unknown divergence"):
        bucketed_divergence(
            teacher_topk_log_probs=topk_lp,
            teacher_tail_log_prob=tail_lp,
            topk_indices=idx,
            student_logits=student,
            divergence="chi_squared",
        )
    with pytest.raises(ValueError, match="tail_epsilon"):
        build_bucket_distributions(topk_lp, tail_lp, topk_lp, tail_lp, tail_epsilon=0.0)
