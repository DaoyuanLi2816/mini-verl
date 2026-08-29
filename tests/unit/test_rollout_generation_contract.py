from __future__ import annotations

from dataclasses import replace

import pytest

from miniverl.runtime.generation import (
    BackendLifecycleState,
    GenerationRequest,
    PolicySnapshot,
    RolloutBackendKind,
    RolloutGroupIdentity,
    RolloutPolicyIdentity,
    SamplingParameters,
    derive_sample_seed,
)


def _identity(*, version: int = 3, backend: str = "hf_cached") -> RolloutPolicyIdentity:
    return RolloutPolicyIdentity(
        parameter_version=version,
        base_model_id="org/model",
        base_model_revision="a" * 40,
        tokenizer_structural_identity="b" * 64,
        student_adapter_manifest_digest="c" * 64,
        adapter_tensor_digest="d" * 64,
        quantization="none",
        dtype="bfloat16",
        generation_backend=RolloutBackendKind(backend),
        backend_version="hf-cached-v1",
        profile_identity="e" * 64,
        execution_plan_digest="f" * 64,
    )


def test_generation_request_carries_complete_group_and_policy_identity() -> None:
    identity = _identity()
    group = RolloutGroupIdentity(
        prompt_group_id="group-7",
        prompt_digest="1" * 64,
        sample_index=2,
        samples_per_prompt=4,
    )
    request = GenerationRequest(
        request_id="request-7-2",
        group=group,
        deterministic_sample_seed=derive_sample_seed(
            run_seed=11,
            prompt_digest=group.prompt_digest,
            policy_version=identity.parameter_version,
            sample_index=group.sample_index,
        ),
        prompt_token_ids=(10, 20, 30),
        max_new_tokens=16,
        sampling=SamplingParameters(temperature=0.7, top_p=0.9, top_k=20),
        stop_sequences=("</answer>",),
        need_sampled_token_logprobs=True,
        expected_policy_identity=identity,
    )

    assert request.group.samples_per_prompt == 4
    assert request.expected_policy_identity.digest == identity.digest
    assert request.prompt_token_ids == (10, 20, 30)


def test_seed_derivation_is_versioned_and_batch_partition_independent() -> None:
    expected = [
        derive_sample_seed(
            run_seed=91,
            prompt_digest="a" * 64,
            policy_version=5,
            sample_index=index,
        )
        for index in range(4)
    ]

    assert expected == [
        derive_sample_seed(
            run_seed=91,
            prompt_digest="a" * 64,
            policy_version=5,
            sample_index=index,
        )
        for partition in ((0, 1), (2, 3))
        for index in partition
    ]
    assert len(set(expected)) == 4
    assert expected != [
        derive_sample_seed(
            run_seed=91,
            prompt_digest="a" * 64,
            policy_version=6,
            sample_index=index,
        )
        for index in range(4)
    ]


def test_generation_models_fail_closed_on_invalid_identity_and_bounds() -> None:
    with pytest.raises(ValueError, match="sample_index"):
        RolloutGroupIdentity(
            prompt_group_id="g",
            prompt_digest="a" * 64,
            sample_index=4,
            samples_per_prompt=4,
        )
    with pytest.raises(ValueError, match="prompt_token_ids"):
        GenerationRequest(
            request_id="r",
            group=RolloutGroupIdentity(
                prompt_group_id="g",
                prompt_digest="a" * 64,
                sample_index=0,
                samples_per_prompt=1,
            ),
            deterministic_sample_seed=1,
            prompt_token_ids=(),
            max_new_tokens=1,
            sampling=SamplingParameters(),
            expected_policy_identity=_identity(),
        )
    with pytest.raises(ValueError, match="generation backend"):
        PolicySnapshot(identity=replace(_identity(), generation_backend="unknown"))


def test_lifecycle_names_are_manifest_safe() -> None:
    assert [state.value for state in BackendLifecycleState] == [
        "new",
        "synchronized",
        "quiesced",
        "closed",
    ]
