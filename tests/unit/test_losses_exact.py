"""Exact divergences checked against brute-force references.

The references are written from the textbook definitions with plain Python
loops so that a bug in the vectorized implementation cannot hide behind the
same expression on both sides.
"""

from __future__ import annotations

import math

import pytest

from tests.conftest import requires_torch

pytestmark = [requires_torch, pytest.mark.torch]

torch = pytest.importorskip("torch")


def _softmax_rows(logits: list[list[float]], temperature: float = 1.0) -> list[list[float]]:
    out = []
    for row in logits:
        scaled = [v / temperature for v in row]
        m = max(scaled)
        exps = [math.exp(v - m) for v in scaled]
        total = sum(exps)
        out.append([e / total for e in exps])
    return out


def _brute_kl(p: list[list[float]], q: list[list[float]]) -> list[float]:
    return [
        sum(pi * (math.log(pi) - math.log(qi)) for pi, qi in zip(prow, qrow) if pi > 0.0)
        for prow, qrow in zip(p, q)
    ]


def _brute_jsd(p: list[list[float]], q: list[list[float]], beta: float) -> list[float]:
    out = []
    for prow, qrow in zip(p, q):
        m = [beta * pi + (1.0 - beta) * qi for pi, qi in zip(prow, qrow)]
        kl_pm = sum(pi * (math.log(pi) - math.log(mi)) for pi, mi in zip(prow, m) if pi > 0.0)
        kl_qm = sum(qi * (math.log(qi) - math.log(mi)) for qi, mi in zip(qrow, m) if qi > 0.0)
        out.append(beta * kl_pm + (1.0 - beta) * kl_qm)
    return out


@pytest.fixture
def logit_pair() -> tuple[list[list[float]], list[list[float]]]:
    generator = torch.Generator().manual_seed(20260727)
    teacher = torch.randn(7, 13, generator=generator) * 2.0
    student = torch.randn(7, 13, generator=generator) * 2.0
    return teacher.tolist(), student.tolist()


def test_forward_kl_matches_brute_force(logit_pair):
    from miniverl.losses.exact import exact_forward_kl

    teacher, student = logit_pair
    expected = _brute_kl(_softmax_rows(teacher), _softmax_rows(student))
    got = exact_forward_kl(torch.tensor(teacher), torch.tensor(student))
    assert got.shape == (7,)
    for a, b in zip(got.tolist(), expected):
        assert a == pytest.approx(b, abs=1e-5)


def test_reverse_kl_matches_brute_force(logit_pair):
    from miniverl.losses.exact import exact_reverse_kl

    teacher, student = logit_pair
    expected = _brute_kl(_softmax_rows(student), _softmax_rows(teacher))
    got = exact_reverse_kl(torch.tensor(teacher), torch.tensor(student))
    for a, b in zip(got.tolist(), expected):
        assert a == pytest.approx(b, abs=1e-5)


@pytest.mark.parametrize("beta", [0.1, 0.5, 0.9])
def test_jsd_matches_brute_force(logit_pair, beta):
    from miniverl.losses.exact import exact_jsd

    teacher, student = logit_pair
    expected = _brute_jsd(_softmax_rows(teacher), _softmax_rows(student), beta)
    got = exact_jsd(torch.tensor(teacher), torch.tensor(student), beta=beta)
    for a, b in zip(got.tolist(), expected):
        assert a == pytest.approx(b, abs=1e-5)


def test_orientation_forward_and_reverse_differ(logit_pair):
    """A swapped-argument bug must be detectable, so the two must not coincide."""
    from miniverl.losses.exact import exact_forward_kl, exact_reverse_kl

    teacher, student = logit_pair
    t = torch.tensor(teacher)
    s = torch.tensor(student)
    fwd = exact_forward_kl(t, s)
    rev = exact_reverse_kl(t, s)
    assert not torch.allclose(fwd, rev, atol=1e-3)
    # Reversing the orientation of the arguments swaps the two objectives.
    assert torch.allclose(exact_forward_kl(s, t), rev, atol=1e-5)
    assert torch.allclose(exact_reverse_kl(s, t), fwd, atol=1e-5)


def test_identical_distributions_are_zero():
    from miniverl.losses.exact import exact_forward_kl, exact_jsd, exact_reverse_kl

    logits = torch.randn(5, 17, generator=torch.Generator().manual_seed(3))
    for fn in (exact_forward_kl, exact_reverse_kl):
        assert torch.allclose(fn(logits, logits.clone()), torch.zeros(5), atol=1e-6)
    assert torch.allclose(exact_jsd(logits, logits.clone()), torch.zeros(5), atol=1e-6)


def test_divergences_are_non_negative(logit_pair):
    from miniverl.losses.exact import exact_forward_kl, exact_jsd, exact_reverse_kl

    teacher, student = logit_pair
    t, s = torch.tensor(teacher), torch.tensor(student)
    for value in (exact_forward_kl(t, s), exact_reverse_kl(t, s), exact_jsd(t, s)):
        assert bool((value >= -1e-6).all())


def test_jsd_is_bounded_by_log_two(logit_pair):
    """Symmetric JSD in nats can never exceed log 2."""
    from miniverl.losses.exact import exact_jsd

    teacher, student = logit_pair
    value = exact_jsd(torch.tensor(teacher), torch.tensor(student), beta=0.5)
    assert bool((value <= math.log(2.0) + 1e-6).all())


def test_jsd_rejects_degenerate_beta(logit_pair):
    from miniverl.losses.exact import exact_jsd

    teacher, student = logit_pair
    for beta in (0.0, 1.0):
        with pytest.raises(ValueError, match="strictly inside"):
            exact_jsd(torch.tensor(teacher), torch.tensor(student), beta=beta)


@pytest.mark.parametrize("scale", [1e3, 1e4])
def test_extreme_logits_stay_finite(scale):
    from miniverl.losses.exact import exact_forward_kl, exact_jsd, exact_reverse_kl

    generator = torch.Generator().manual_seed(11)
    teacher = torch.randn(4, 32, generator=generator) * scale
    student = torch.randn(4, 32, generator=generator) * scale
    for fn in (exact_forward_kl, exact_reverse_kl, exact_jsd):
        value = fn(teacher, student)
        assert bool(torch.isfinite(value).all()), f"{fn.__name__} produced non-finite values"


@pytest.mark.parametrize("divergence", ["forward_kl", "reverse_kl", "jsd"])
def test_gradients_are_finite_and_flow_to_student(divergence):
    from miniverl.losses.exact import exact_divergence

    generator = torch.Generator().manual_seed(5)
    teacher = (torch.randn(6, 23, generator=generator) * 3.0).requires_grad_(False)
    student = (torch.randn(6, 23, generator=generator) * 3.0).requires_grad_(True)
    value = exact_divergence(teacher, student, divergence=divergence).sum()
    value.backward()
    assert student.grad is not None
    assert bool(torch.isfinite(student.grad).all())
    assert float(student.grad.abs().sum()) > 0.0


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_half_precision_inputs_reduce_in_float32(dtype):
    from miniverl.losses.exact import exact_forward_kl

    torch_dtype = getattr(torch, dtype)
    generator = torch.Generator().manual_seed(7)
    teacher = torch.randn(3, 64, generator=generator)
    student = torch.randn(3, 64, generator=generator)
    value = exact_forward_kl(teacher.to(torch_dtype), student.to(torch_dtype))
    assert value.dtype == torch.float32
    assert bool(torch.isfinite(value).all())
    reference = exact_forward_kl(teacher, student)
    assert torch.allclose(value, reference, atol=2e-2)


def test_temperature_squared_scaling_is_applied():
    from miniverl.losses.exact import exact_forward_kl

    generator = torch.Generator().manual_seed(13)
    teacher = torch.randn(4, 11, generator=generator)
    student = torch.randn(4, 11, generator=generator)
    unscaled = exact_forward_kl(
        teacher, student, temperature=2.0, scale_by_temperature_squared=False
    )
    scaled = exact_forward_kl(teacher, student, temperature=2.0, scale_by_temperature_squared=True)
    assert torch.allclose(scaled, unscaled * 4.0, atol=1e-6)


def test_temperature_one_is_unaffected_by_the_scaling_flag():
    from miniverl.losses.exact import exact_reverse_kl

    generator = torch.Generator().manual_seed(17)
    teacher = torch.randn(3, 9, generator=generator)
    student = torch.randn(3, 9, generator=generator)
    a = exact_reverse_kl(teacher, student, temperature=1.0, scale_by_temperature_squared=True)
    b = exact_reverse_kl(teacher, student, temperature=1.0, scale_by_temperature_squared=False)
    assert torch.allclose(a, b)


def test_unknown_divergence_name_is_rejected():
    from miniverl.losses.exact import exact_divergence

    with pytest.raises(ValueError, match="unknown divergence"):
        exact_divergence(torch.zeros(1, 3), torch.zeros(1, 3), divergence="hellinger")


def test_teacher_entropy_matches_brute_force(logit_pair):
    from miniverl.losses.exact import exact_teacher_entropy

    teacher, _ = logit_pair
    probs = _softmax_rows(teacher)
    expected = [-sum(p * math.log(p) for p in row if p > 0) for row in probs]
    got = exact_teacher_entropy(torch.tensor(teacher))
    for a, b in zip(got.tolist(), expected):
        assert a == pytest.approx(b, abs=1e-5)
