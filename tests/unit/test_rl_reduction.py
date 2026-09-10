"""Logical-batch reduction must be invariant to physical microbatching."""

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.torch

from miniverl.losses.reduction import rl_reduction_weights  # noqa: E402


@pytest.mark.parametrize(
    "mode",
    [
        "token-mean",
        "token-sum",
        "seq-mean-token-sum",
        "seq-mean-token-mean",
        "seq-mean-token-sum-norm",
    ],
)
@pytest.mark.parametrize("scale", [None, 7])
def test_upstream_scalar_gradient_and_partition(mode, scale):
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    values = torch.arange(12, dtype=torch.float64).reshape(3, 4).requires_grad_()
    rows, denominator = rl_reduction_weights(
        list(mask), mode, response_length=4, loss_scale_factor=scale
    )
    actual = sum((v * w).sum() / denominator for v, w in zip(values, rows, strict=True))
    sums = (values * mask).sum(-1)
    if mode == "token-mean":
        expected = sums.sum() / mask.sum()
    elif mode == "token-sum":
        expected = sums.sum()
    elif mode == "seq-mean-token-mean":
        expected = (sums / (mask.sum(-1) + 1e-8)).sum() / 2
    else:
        expected = sums.sum() / 2
        if mode.endswith("-norm"):
            expected = expected / (scale or 4)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, values, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, values)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


def test_reject_empty_mask_and_unknown_reduction():
    with pytest.raises(ValueError, match="select"):
        rl_reduction_weights([torch.zeros(2)], "token-mean", response_length=2)
    with pytest.raises(ValueError, match="aggregation"):
        rl_reduction_weights([torch.ones(2)], "jitter", response_length=2)
