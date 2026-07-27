"""Property-based invariants for the divergences and the reduction.

Hypothesis searches for the logit configurations a hand-written example would
miss: near-degenerate distributions, huge dynamic range, a single dominant
token, exact ties.  The properties asserted here are the ones the whole
objective rests on.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from tests.conftest import requires_torch

pytestmark = [requires_torch, pytest.mark.torch]

torch = pytest.importorskip("torch")

SETTINGS = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)

logits_row = st.lists(
    st.floats(min_value=-30.0, max_value=30.0, allow_nan=False, allow_infinity=False),
    min_size=2,
    max_size=24,
)


@st.composite
def logit_pairs(draw: st.DrawFn, dtype: torch.dtype = torch.float32):
    """Two ``[N, V]`` logit tensors with matching shapes."""
    vocab = draw(st.integers(min_value=2, max_value=24))
    rows = draw(st.integers(min_value=1, max_value=5))
    values = st.floats(min_value=-30.0, max_value=30.0, allow_nan=False, allow_infinity=False)
    teacher = draw(
        st.lists(st.lists(values, min_size=vocab, max_size=vocab), min_size=rows, max_size=rows)
    )
    student = draw(
        st.lists(st.lists(values, min_size=vocab, max_size=vocab), min_size=rows, max_size=rows)
    )
    return torch.tensor(teacher, dtype=dtype), torch.tensor(student, dtype=dtype)


# --------------------------------------------------------------- exact


@SETTINGS
@given(pair=logit_pairs())
def test_exact_divergences_are_finite_and_non_negative(pair):
    from miniverl.losses.exact import exact_forward_kl, exact_jsd, exact_reverse_kl

    teacher, student = pair
    for value in (
        exact_forward_kl(teacher, student),
        exact_reverse_kl(teacher, student),
        exact_jsd(teacher, student),
    ):
        assert bool(torch.isfinite(value).all())
        assert bool((value >= -1e-5).all())


@SETTINGS
@given(pair=logit_pairs())
def test_symmetric_jsd_is_symmetric_and_bounded(pair):
    from miniverl.losses.exact import exact_jsd

    teacher, student = pair
    forward = exact_jsd(teacher, student, beta=0.5)
    backward = exact_jsd(student, teacher, beta=0.5)
    assert torch.allclose(forward, backward, atol=1e-5)
    assert bool((forward <= math.log(2.0) + 1e-5).all())


@SETTINGS
@given(row=logits_row)
def test_self_divergence_is_zero(row):
    from miniverl.losses.exact import exact_forward_kl, exact_jsd, exact_reverse_kl

    logits = torch.tensor([row], dtype=torch.float32)
    for fn in (exact_forward_kl, exact_reverse_kl, exact_jsd):
        assert float(fn(logits, logits.clone()).abs().max()) < 1e-5


@SETTINGS
@given(pair=logit_pairs(), shift=st.floats(min_value=-20.0, max_value=20.0))
def test_divergence_is_invariant_to_a_constant_logit_shift(pair, shift):
    """Softmax is shift-invariant, so the objective must be too."""
    from miniverl.losses.exact import exact_reverse_kl

    teacher, student = pair
    base = exact_reverse_kl(teacher, student)
    shifted = exact_reverse_kl(teacher + shift, student)
    assert torch.allclose(base, shifted, atol=1e-4)


@SETTINGS
@given(pair=logit_pairs())
def test_entropy_is_between_zero_and_log_vocab(pair):
    from miniverl.losses.exact import exact_teacher_entropy

    teacher, _ = pair
    entropy = exact_teacher_entropy(teacher)
    assert bool((entropy >= -1e-6).all())
    assert bool((entropy <= math.log(teacher.shape[-1]) + 1e-5).all())


# ------------------------------------------------------------ bucketed


@SETTINGS
@given(pair=logit_pairs(dtype=torch.float64), k=st.integers(min_value=1, max_value=24))
def test_bucketed_never_exceeds_exact(pair, k):
    """Data-processing inequality: coarse-graining cannot add information.

    Evaluated in float64 so the assertion tests the *mathematics* rather than
    float32 accumulation. The bound is tight (equality) whenever the
    coarse-graining is the identity -- for example ``V == k + 1`` -- so in
    float32 the two sides can differ by a few ulps in either direction; the
    float32 behaviour is covered by tests/unit/test_losses_bucketed.py.

    The allowance carries a cancellation term, which Hypothesis forced into
    existence twice, with mirror-image inputs:

    * ``teacher=[0, 0]``, ``student=[20, 0.03]``, ``k=1`` -- the *student*
      saturates its top bucket;
    * ``teacher=[0, 19.28]``, ``student=[0, 0]``, ``k=1`` -- the *teacher* does.

    Either way, a distribution holding all but ``d`` of its mass in the top
    bucket has a top log-prob of ``-d``, formed as ``logit - logsumexp``: a
    difference of two nearly equal numbers, so it carries about one ulp of
    *absolute* error at any precision. ``log1mexp`` then turns that into the tail
    log-prob and, because its slope there is ``1 / d``, divides the error by the
    tail mass. The divergence multiplies that log by the *other* distribution's
    mass on the same bucket, so the error reaching the result scales with
    ``q_tail / p_tail`` for reverse KL and ``p_tail / q_tail`` for forward KL --
    which is why the bound below is symmetric in the two, and why a term keyed to
    only one side let the mirror case through.

    Concretely, in the second example the teacher's tail mass is 4.2e-9 while the
    uniform student puts 0.5 there, so a 4.3e-15 error becomes 5e-7. That is real
    arithmetic, not a violated inequality. For any non-degenerate input both
    ratios are near 1 and the whole term is about 1e-14, so the test stays sharp
    where sharpness means something. At ``k == V`` both tail masses are exactly
    zero, the tail bucket contributes nothing, and the clamps below leave the
    term at ``2 * eps * (1 + max|logit|)``.
    """
    from miniverl.losses.bucketed import (
        bucketed_divergence,
        student_bucket_log_probs,
        teacher_topk_targets,
    )
    from miniverl.losses.exact import exact_divergence

    teacher, student = pair
    k = min(k, teacher.shape[-1])
    idx, lp, tail = teacher_topk_targets(teacher, top_k=k)

    eps = torch.finfo(student.dtype).eps
    _, student_tail = student_bucket_log_probs(student, idx)
    # One ulp of `logit - logsumexp`, amplified by the ratio of the two tail
    # masses in whichever direction the divergence happens to weight it.
    scale = eps * (1.0 + torch.maximum(teacher.abs().amax(dim=-1), student.abs().amax(dim=-1)))
    p_tail = tail.exp().clamp_min(eps)
    q_tail = student_tail.exp().clamp_min(eps)
    cancellation = scale * (q_tail / p_tail + p_tail / q_tail)

    for divergence in ("forward_kl", "reverse_kl", "jsd"):
        bucketed = bucketed_divergence(
            teacher_topk_log_probs=lp,
            teacher_tail_log_prob=tail,
            topk_indices=idx,
            student_logits=student,
            divergence=divergence,
        )
        exact = exact_divergence(teacher, student, divergence=divergence)
        assert bool(torch.isfinite(bucketed).all())
        assert bool((bucketed >= -1e-5).all())
        allowance = exact.abs() * 1e-9 + 1e-9 + cancellation
        assert bool((bucketed <= exact + allowance).all()), divergence


@SETTINGS
@given(pair=logit_pairs(dtype=torch.float64))
def test_full_k_reproduces_exact(pair):
    from miniverl.losses.bucketed import bucketed_divergence, teacher_topk_targets
    from miniverl.losses.exact import exact_divergence

    teacher, student = pair
    vocab = teacher.shape[-1]
    idx, lp, tail = teacher_topk_targets(teacher, top_k=vocab)
    for divergence in ("forward_kl", "reverse_kl", "jsd"):
        bucketed = bucketed_divergence(
            teacher_topk_log_probs=lp,
            teacher_tail_log_prob=tail,
            topk_indices=idx,
            student_logits=student,
            divergence=divergence,
        )
        exact = exact_divergence(teacher, student, divergence=divergence)
        assert torch.allclose(bucketed, exact, atol=1e-9), divergence


@SETTINGS
@given(pair=logit_pairs(), k=st.integers(min_value=1, max_value=24))
def test_teacher_topk_mass_and_tail_sum_to_one(pair, k):
    from miniverl.losses.bucketed import teacher_topk_targets

    teacher, _ = pair
    k = min(k, teacher.shape[-1])
    _, lp, tail = teacher_topk_targets(teacher, top_k=k)
    covered = torch.logsumexp(lp, dim=-1).exp()
    total = covered + torch.where(torch.isinf(tail), torch.zeros_like(tail), tail.exp())
    assert bool(((total - 1.0).abs() < 1e-4).all())
    assert bool((covered <= 1.0 + 1e-5).all())


@SETTINGS
@given(x=st.floats(min_value=-60.0, max_value=-1e-12, allow_nan=False))
def test_log1mexp_matches_the_reference_outside_the_clamp(x):
    """In float64 the clamp is ~2.2e-16, so the reference holds almost everywhere."""
    from miniverl.losses.numerics import log1mexp, neg_clamp_for

    assume(x <= neg_clamp_for(torch.float64))
    value = float(log1mexp(torch.tensor([x], dtype=torch.float64)))
    assert math.isfinite(value)
    expected = math.log1p(-math.exp(x))
    assert value == pytest.approx(expected, abs=1e-6, rel=1e-6)


@SETTINGS
@given(
    x=st.floats(min_value=-1e-7, max_value=-0.0, allow_nan=False),
    dtype_name=st.sampled_from(["float32", "float64"]),
)
def test_log1mexp_clamps_instead_of_diverging_near_zero(x, dtype_name):
    """``log(1 - exp(x))`` -> -inf as x -> 0; the clamp bounds it, by design.

    Without the bound, a teacher whose top-k captures the entire mass would make
    the tail bucket ``-inf`` and poison the loss. The bound is the dtype's own
    resolution near 1.0, so float32 saturates at about 1e-7 while float64
    resolves nine more decades. This pins both.
    """
    from miniverl.losses.numerics import log1mexp, neg_clamp_for

    dtype = getattr(torch, dtype_name)
    clamp = neg_clamp_for(dtype)
    value = float(log1mexp(torch.tensor([x], dtype=dtype)))
    assert math.isfinite(value)
    if x > clamp:
        floor = float(log1mexp(torch.tensor([clamp], dtype=dtype)))
        assert value == pytest.approx(floor, rel=1e-5)
    else:
        assert value == pytest.approx(math.log1p(-math.exp(x)), rel=1e-4)


# ----------------------------------------------------------- reduction


@SETTINGS
@given(
    values=st.lists(
        st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=32,
    ),
    weights=st.lists(
        st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=32,
    ),
)
def test_weighted_mean_is_a_weighted_mean(values, weights):
    from miniverl.losses.reduction import total_weight, weighted_mean

    n = min(len(values), len(weights))
    per_token = torch.tensor(values[:n], dtype=torch.float32)
    w = torch.tensor(weights[:n], dtype=torch.float32)
    result = float(weighted_mean(per_token, w))
    denominator = float(total_weight(w))
    if denominator == 0.0:
        # Every position masked out: a documented, safe zero.
        assert result == 0.0
        return
    # Any positive weight sum -- however tiny -- must give the true weighted
    # mean, which is always bounded by the range of the values.
    #
    # The reference is computed in float64 on purpose. Computing `(per_token * w)`
    # in float32, as this test used to, reproduces the very error it is meant to
    # detect: at `w = 1.4e-45` (float32's smallest subnormal) the product
    # `1.5 * 1.4e-45` rounds to twice the subnormal minimum, so a float32
    # reference "expects" 2.0 as the mean of a single 1.5. Hypothesis found that
    # input, `weighted_mean` now accumulates in float64 and returns 1.5, and the
    # reference has to be at least as accurate as the code it checks.
    expected = float((per_token.double() * w.double()).sum() / w.double().sum())
    assert result == pytest.approx(expected, rel=1e-4, abs=1e-5)
    assert min(values[:n]) - 1e-4 <= result <= max(values[:n]) + 1e-4


@SETTINGS
@given(
    values=st.lists(
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False), min_size=2, max_size=16
    )
)
def test_zero_weights_mask_positions_exactly(values):
    from miniverl.losses.reduction import weighted_mean

    per_token = torch.tensor(values, dtype=torch.float32)
    weights = torch.ones(len(values))
    weights[0] = 0.0
    masked = float(weighted_mean(per_token, weights))
    dropped = float(weighted_mean(per_token[1:], torch.ones(len(values) - 1)))
    assert masked == pytest.approx(dropped, rel=1e-5, abs=1e-6)


@SETTINGS
@given(
    values=st.lists(
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False), min_size=1, max_size=8
    )
)
def test_weighted_mean_rejects_a_shape_mismatch(values):
    from miniverl.losses.reduction import weighted_mean

    assume(len(values) >= 1)
    with pytest.raises(ValueError, match="shape"):
        weighted_mean(torch.tensor(values), torch.ones(len(values) + 1))


# ------------------------------------------------------------ selection


@SETTINGS
@given(
    ratio=st.floats(min_value=0.01, max_value=1.0),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_selection_is_reproducible_for_any_ratio_and_seed(ratio, seed):
    from miniverl.config.models import SelectionConfig, SelectorName
    from miniverl.schemas.trajectory import Span, SpanType, TerminationReason, Trajectory, Turn
    from miniverl.selection.selectors import select_positions
    from miniverl.trajectory.masks import build_masks

    spans = [
        Span(span_type=SpanType.SYSTEM, start=0, end=5, turn_id=0, text="s"),
        Span(span_type=SpanType.ASSISTANT_TOOL_CALL, start=5, end=15, turn_id=0, text="c"),
        Span(span_type=SpanType.TOOL_RESULT, start=15, end=20, turn_id=0, text="r"),
        Span(span_type=SpanType.ASSISTANT_FINAL, start=20, end=30, turn_id=0, text="f"),
    ]
    model, critical = build_masks(spans, 30)
    traj = Trajectory(
        trajectory_id="prop",
        task_id="t",
        environment="calculator",
        token_ids=list(range(30)),
        attention_mask=[1] * 30,
        model_generated_mask=model,
        critical_mask=critical,
        spans=spans,
        turns=[Turn(turn_id=0)],
        tokenizer_fingerprint="fp",
        model_id="m",
        termination_reason=TerminationReason.FINAL_ANSWER,
    )
    for selector in SelectorName:
        config = SelectionConfig(selector=selector, ratio=ratio)
        first = select_positions(traj, config, run_seed=seed)
        second = select_positions(traj, config, run_seed=seed)
        assert first.positions == second.positions
        assert first.weights == second.weights
        for position in first.positions:
            assert traj.model_generated_mask[position]
        assert first.positions == sorted(set(first.positions))
