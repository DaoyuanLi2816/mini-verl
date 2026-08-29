from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from miniverl.models.base import GenerationOutput
from miniverl.runtime.backends.hf_cached import HFCachedGenerationBackend
from miniverl.runtime.backends.hf_reference import HFReferenceGenerationBackend
from miniverl.runtime.generation import (
    BackendLifecycleState,
    GenerationRequest,
    PolicySnapshot,
    RolloutBackendKind,
    RolloutGroupIdentity,
    RolloutPolicyIdentity,
    SamplingParameters,
)
from miniverl.runtime.policy_sync import adapter_tensor_digest, build_rollout_policy_identity


class _FakeModelBackend:
    def __init__(self) -> None:
        self.capabilities = SimpleNamespace(device="cpu")
        self.reference_calls = 0
        self.cached_calls = 0
        self.compile_calls = 0

    def enable_cached_generation_compilation(self) -> None:
        self.compile_calls += 1

    @staticmethod
    def _output(seed: int | None, *, record_logprobs: bool) -> GenerationOutput:
        token = int(seed or 0) % 17 + 2
        return GenerationOutput(
            token_ids=[token, token + 1],
            text=f"{token}:{token + 1}",
            stop_reason="max_new_tokens",
            logprobs=[-0.25, -0.5] if record_logprobs else [],
        )

    def generate(self, prefix_token_ids, **kwargs):  # type: ignore[no-untyped-def]
        del prefix_token_ids
        self.reference_calls += 1
        return self._output(kwargs["seed"], record_logprobs=kwargs["record_logprobs"])

    def generate_batch(self, prefix_token_ids, **kwargs):  # type: ignore[no-untyped-def]
        self.reference_calls += 1
        return [
            self._output(seed, record_logprobs=kwargs["record_logprobs"])
            for seed in kwargs["seeds"]
        ]

    def generate_batch_cached(self, prefix_token_ids, **kwargs):  # type: ignore[no-untyped-def]
        self.cached_calls += 1
        return [
            self._output(seed, record_logprobs=kwargs["record_logprobs"])
            for seed in kwargs["seeds"]
        ]


class _FakeNF4MirrorBackend(_FakeModelBackend):
    def __init__(self) -> None:
        super().__init__()
        self.capabilities.quantization = "none"
        self.synced_digests: list[str] = []
        self.discard_calls = 0

    def discard_cached_generation_mirror(self) -> None:
        self.discard_calls += 1


class _FakeNF4SourceBackend(_FakeModelBackend):
    def __init__(self) -> None:
        super().__init__()
        self.capabilities.quantization = "nf4"
        self.mirror = _FakeNF4MirrorBackend()
        self.build_calls = 0

    def build_cached_generation_mirror(self) -> _FakeNF4MirrorBackend:
        self.build_calls += 1
        return self.mirror

    def synchronize_cached_generation_mirror(
        self, mirror: _FakeNF4MirrorBackend, *, expected_adapter_digest: str
    ) -> None:
        assert mirror is self.mirror
        mirror.synced_digests.append(expected_adapter_digest)


def _identity(kind: RolloutBackendKind, *, version: int = 1) -> RolloutPolicyIdentity:
    return RolloutPolicyIdentity(
        parameter_version=version,
        base_model_id="org/model",
        base_model_revision=None,
        tokenizer_structural_identity="1" * 64,
        student_adapter_manifest_digest="2" * 64,
        adapter_tensor_digest="3" * 64,
        quantization="none",
        dtype="float32",
        generation_backend=kind,
        backend_version=f"{kind.value}-v1",
        profile_identity="4" * 64,
        execution_plan_digest="5" * 64,
    )


def _request(identity: RolloutPolicyIdentity, index: int) -> GenerationRequest:
    return GenerationRequest(
        request_id=f"request-{index}",
        group=RolloutGroupIdentity(
            prompt_group_id="group",
            prompt_digest="a" * 64,
            sample_index=index,
            samples_per_prompt=4,
        ),
        deterministic_sample_seed=100 + index,
        prompt_token_ids=(7, 8 + index),
        max_new_tokens=2,
        sampling=SamplingParameters(temperature=0.8, top_p=0.9, top_k=8),
        need_sampled_token_logprobs=True,
        expected_policy_identity=identity,
    )


@pytest.mark.parametrize(
    ("backend_type", "kind", "counter"),
    [
        (HFReferenceGenerationBackend, RolloutBackendKind.HF_REFERENCE, "reference_calls"),
        (HFCachedGenerationBackend, RolloutBackendKind.HF_CACHED, "cached_calls"),
    ],
)
def test_backends_fail_closed_until_exact_policy_is_synchronized(
    backend_type,
    kind,
    counter,  # type: ignore[no-untyped-def]
) -> None:
    model = _FakeModelBackend()
    backend = backend_type(model)
    identity = _identity(kind)

    assert backend.state is BackendLifecycleState.NEW
    with pytest.raises(RuntimeError, match="synchronized"):
        backend.generate([_request(identity, 0)])
    sync = backend.synchronize(PolicySnapshot(identity))
    assert sync.active_policy_digest == identity.digest
    assert backend.state is BackendLifecycleState.SYNCHRONIZED

    batch = backend.generate([_request(identity, 0), _request(identity, 1)])
    assert [result.request_id for result in batch.results] == ["request-0", "request-1"]
    assert all(result.policy_identity == identity for result in batch.results)
    assert all(len(result.sampled_token_logprobs) == 2 for result in batch.results)
    assert getattr(model, counter) == 1

    stale = replace(identity, parameter_version=2)
    with pytest.raises(RuntimeError, match="policy identity"):
        backend.generate([_request(stale, 0)])
    backend.quiesce()
    with pytest.raises(RuntimeError, match="synchronized"):
        backend.generate([_request(identity, 0)])
    backend.close()
    assert backend.state is BackendLifecycleState.CLOSED
    with pytest.raises(RuntimeError, match="closed"):
        backend.synchronize(PolicySnapshot(identity))


def test_cached_backend_is_invariant_to_physical_partition() -> None:
    identity = _identity(RolloutBackendKind.HF_CACHED)
    requests = [_request(identity, index) for index in range(4)]

    whole = HFCachedGenerationBackend(_FakeModelBackend())
    whole.synchronize(PolicySnapshot(identity))
    whole_results = whole.generate(requests).results

    partitioned = HFCachedGenerationBackend(_FakeModelBackend())
    partitioned.synchronize(PolicySnapshot(identity))
    split_results = (
        *partitioned.generate(requests[:2]).results,
        *partitioned.generate(requests[2:]).results,
    )

    assert [result.output_token_ids for result in whole_results] == [
        result.output_token_ids for result in split_results
    ]
    assert [result.sampled_token_logprobs for result in whole_results] == [
        result.sampled_token_logprobs for result in split_results
    ]


def test_cached_compilation_is_explicit_and_changes_backend_identity() -> None:
    model = _FakeModelBackend()
    backend = HFCachedGenerationBackend(model, compile_backend=True)

    assert model.compile_calls == 1
    assert backend.inspect().backend_version == "hf_cached-v1+inductor-no-cudagraph"


def test_compiled_nf4_backend_owns_and_synchronizes_bf16_mirror() -> None:
    source = _FakeNF4SourceBackend()
    backend = HFCachedGenerationBackend(source, compile_backend=True)

    assert source.build_calls == 1
    assert source.compile_calls == 0
    assert source.mirror.compile_calls == 1
    assert backend.model_backend is source.mirror
    assert backend.inspect().backend_version == "hf_cached-v2+nf4-bf16-mirror+inductor"

    first = replace(
        _identity(RolloutBackendKind.HF_CACHED),
        backend_version=backend.backend_version,
        adapter_tensor_digest="a" * 64,
    )
    second = replace(first, parameter_version=2, adapter_tensor_digest="b" * 64)
    backend.synchronize(PolicySnapshot(first))
    backend.synchronize(PolicySnapshot(second))

    assert source.mirror.synced_digests == ["a" * 64, "b" * 64]
    backend.close()
    assert source.mirror.discard_calls == 1


def test_compiled_nf4_mirror_sync_failure_does_not_activate_policy() -> None:
    source = _FakeNF4SourceBackend()

    def fail_sync(mirror, *, expected_adapter_digest):  # type: ignore[no-untyped-def]
        del mirror, expected_adapter_digest
        raise RuntimeError("injected mirror copy failure")

    source.synchronize_cached_generation_mirror = fail_sync  # type: ignore[method-assign]
    backend = HFCachedGenerationBackend(source, compile_backend=True)
    identity = replace(
        _identity(RolloutBackendKind.HF_CACHED),
        backend_version=backend.backend_version,
    )

    with pytest.raises(RuntimeError, match="injected mirror copy failure"):
        backend.synchronize(PolicySnapshot(identity))
    assert backend.state is BackendLifecycleState.NEW
    with pytest.raises(RuntimeError, match="synchronized"):
        backend.generate([_request(identity, 0)])


def test_compiled_nf4_mirror_is_discarded_when_compilation_fails() -> None:
    source = _FakeNF4SourceBackend()

    def fail_compile() -> None:
        raise RuntimeError("injected compilation failure")

    source.mirror.enable_cached_generation_compilation = fail_compile
    with pytest.raises(RuntimeError, match="injected compilation failure"):
        HFCachedGenerationBackend(source, compile_backend=True)

    assert source.mirror.discard_calls == 1


def test_policy_identity_digest_tracks_live_trainable_tensors() -> None:
    torch = pytest.importorskip("torch")
    from miniverl.models.tokenizers import ToyTokenizer
    from miniverl.models.toy import ToyBackend

    backend = ToyBackend(tokenizer=ToyTokenizer(), model_id="toy", seed=7, trainable=True)
    before = adapter_tensor_digest(backend)
    parameter = backend.trainable_parameters()[0]
    with torch.no_grad():
        parameter.view(-1)[0].add_(1.0)
    after = adapter_tensor_digest(backend)

    assert before != after
    identity = build_rollout_policy_identity(
        backend=backend,
        parameter_version=4,
        generation_backend=RolloutBackendKind.HF_CACHED,
        backend_version="hf_cached-v1",
        profile_identity={"profile": "unit", "version": 1},
        execution_plan_digest="9" * 64,
    )
    assert identity.parameter_version == 4
    assert identity.adapter_tensor_digest == after
    assert identity.generation_backend is RolloutBackendKind.HF_CACHED
