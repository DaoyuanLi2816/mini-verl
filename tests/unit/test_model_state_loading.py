"""Trainable checkpoint tensors are validated completely before mutation."""

from __future__ import annotations

import pytest

from miniverl.errors import BackendError
from tests.conftest import requires_torch

pytestmark = [requires_torch, pytest.mark.torch]

torch = pytest.importorskip("torch")


def _backend():
    from miniverl.models.tokenizers import ToyTokenizer
    from miniverl.models.toy import ToyBackend

    return ToyBackend(
        tokenizer=ToyTokenizer(),
        model_id="state-test",
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        intermediate_size=32,
        max_position_embeddings=128,
    )


def test_missing_trainable_key_is_rejected_before_any_parameter_changes() -> None:
    backend = _backend()
    original = backend.trainable_state_dict()
    incomplete = {name: tensor.clone() for name, tensor in original.items()}
    changed = next(iter(incomplete))
    incomplete[changed].add_(1)
    incomplete.pop(next(reversed(incomplete)))

    with pytest.raises(BackendError, match="missing"):
        backend.load_trainable_state_dict(incomplete)

    restored = backend.trainable_state_dict()
    assert all(torch.equal(original[name], restored[name]) for name in original)


def test_shape_mismatch_is_rejected_before_any_parameter_changes() -> None:
    backend = _backend()
    original = backend.trainable_state_dict()
    malformed = {name: tensor.clone() for name, tensor in original.items()}
    changed = next(iter(malformed))
    malformed[changed] = malformed[changed].reshape(-1)[:1]

    with pytest.raises(BackendError, match="shape"):
        backend.load_trainable_state_dict(malformed)

    restored = backend.trainable_state_dict()
    assert all(torch.equal(original[name], restored[name]) for name in original)
