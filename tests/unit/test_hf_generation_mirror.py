from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from miniverl.errors import BackendError  # noqa: E402
from miniverl.models.hf import HFBackend  # noqa: E402
from miniverl.runtime.policy_sync import adapter_tensor_digest  # noqa: E402

pytestmark = pytest.mark.torch


class _MirrorEndpoint:
    def __init__(self, value: float) -> None:
        self.model = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.model.weight.fill_(value)

    def trainable_state_dict(self):  # type: ignore[no-untyped-def]
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }

    def load_trainable_state_dict(self, state):  # type: ignore[no-untyped-def]
        own = dict(self.model.named_parameters())
        with torch.no_grad():
            for name, value in state.items():
                own[name].copy_(value)


def test_generation_mirror_sync_copies_and_verifies_live_tensors() -> None:
    source = _MirrorEndpoint(2.0)
    mirror = _MirrorEndpoint(-1.0)
    expected = adapter_tensor_digest(source)

    HFBackend.synchronize_cached_generation_mirror(  # type: ignore[arg-type]
        source,
        mirror,
        expected_adapter_digest=expected,
    )

    assert adapter_tensor_digest(mirror) == expected
    assert torch.equal(source.model.weight, mirror.model.weight)
    assert mirror.model.training is False


def test_generation_mirror_sync_rejects_actor_change_after_identity() -> None:
    source = _MirrorEndpoint(2.0)
    mirror = _MirrorEndpoint(-1.0)
    stale_digest = adapter_tensor_digest(source)
    with torch.no_grad():
        source.model.weight[0, 0].add_(1.0)
    before = mirror.model.weight.detach().clone()

    with pytest.raises(BackendError, match="changed after rollout policy identity"):
        HFBackend.synchronize_cached_generation_mirror(  # type: ignore[arg-type]
            source,
            mirror,
            expected_adapter_digest=stale_digest,
        )

    assert torch.equal(mirror.model.weight, before)


def test_generation_mirror_sync_rejects_nonidentical_mirror_digest() -> None:
    source = _MirrorEndpoint(2.0)
    mirror = _MirrorEndpoint(-1.0)

    def lossy_load(state):  # type: ignore[no-untyped-def]
        del state
        with torch.no_grad():
            mirror.model.weight.fill_(0.0)

    mirror.load_trainable_state_dict = lossy_load  # type: ignore[method-assign]

    with pytest.raises(BackendError, match="does not exactly match"):
        HFBackend.synchronize_cached_generation_mirror(  # type: ignore[arg-type]
            source,
            mirror,
            expected_adapter_digest=adapter_tensor_digest(source),
        )
