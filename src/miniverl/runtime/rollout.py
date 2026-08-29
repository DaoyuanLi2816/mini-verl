"""Source-agnostic rollout runtimes for tool episodes and prompt datasets."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from miniverl.agent.loop import RolloutRunner
from miniverl.config.models import RolloutConfig, VerlParquetSourceConfig
from miniverl.data.verl_parquet import RenderedPrompt
from miniverl.errors import GpuMemoryError
from miniverl.models.base import GenerationOutput
from miniverl.runtime.backends import HFCachedGenerationBackend, HFReferenceGenerationBackend
from miniverl.runtime.generation import (
    GenerationRequest,
    GenerationResult,
    PolicySnapshot,
    RolloutBackendKind,
    RolloutGroupIdentity,
    SamplingParameters,
    derive_sample_seed,
)
from miniverl.runtime.policy_sync import build_rollout_policy_identity
from miniverl.schemas.trajectory import (
    TRAJECTORY_SCHEMA_VERSION,
    Span,
    SpanType,
    TerminationReason,
    Trajectory,
    derive_grouped_trajectory_id,
    validate_trajectory_groups,
)

__all__ = [
    "GeneratedPromptBatch",
    "PreparedPromptBatch",
    "PromptDatasetRolloutRuntime",
    "RolloutRuntime",
    "ToolEnvironmentRolloutRuntime",
]


@runtime_checkable
class RolloutRuntime(Protocol):
    """Minimal lifecycle used by the source-agnostic trainer path."""

    def prepare_batch(self, inputs: Any) -> Any:
        """Validate and physically batch logical rollout inputs."""
        ...

    def generate(self, batch: Any, *, policy_version: int, seed: int) -> Any:
        """Generate under one explicitly identified actor policy."""
        ...

    def to_trajectories(
        self, batch: Any, generated: Any, *, policy_version: int
    ) -> list[Trajectory]:
        """Freeze exact token provenance for training and teacher scoring."""
        ...

    def close(self) -> None:
        """Release runtime-owned resources."""
        ...


@dataclass(frozen=True)
class PreparedPromptBatch:
    """One logical prompt batch and its deterministic physical partition."""

    prompts: tuple[RenderedPrompt, ...]
    groups: tuple[RolloutGroupIdentity, ...]
    physical_batches: tuple[tuple[int, ...], ...]
    group_cursor: int


@dataclass(frozen=True)
class GeneratedPromptBatch:
    """Outputs restored to logical order plus the actual physical batch sizes."""

    outputs: tuple[GenerationOutput, ...]
    groups: tuple[RolloutGroupIdentity, ...]
    generation_seeds: tuple[int, ...]
    rollout_policy_identity_digest: str
    policy_version: int
    physical_batch_sizes: tuple[int, ...]
    oom_downshifts: int = 0


class PromptDatasetRolloutRuntime:
    """Single-turn padded actor generation over already-rendered prompts."""

    def __init__(
        self,
        *,
        backend: Any,
        source_config: VerlParquetSourceConfig,
        rollout_config: RolloutConfig,
        profile_identity: object | None = None,
        execution_plan_digest: str | None = None,
    ) -> None:
        self.backend = backend
        self.source_config = source_config
        self.config = rollout_config
        self.profile_identity = profile_identity
        self.execution_plan_digest = execution_plan_digest
        self.generation_backend = (
            HFCachedGenerationBackend(backend, compile_backend=rollout_config.compile_backend)
            if rollout_config.backend is RolloutBackendKind.HF_CACHED
            else HFReferenceGenerationBackend(backend)
        )
        self._closed = False

    @property
    def max_new_tokens(self) -> int:
        """Independent response bound for the verl Parquet profile."""
        return min(self.config.max_new_tokens_per_turn, self.source_config.max_response_length)

    def prepare_batch(
        self, inputs: list[RenderedPrompt], *, group_cursor: int = 0
    ) -> PreparedPromptBatch:
        if self._closed:
            raise RuntimeError("prompt rollout runtime is closed")
        if not inputs:
            raise ValueError("prompt rollout batch cannot be empty")
        if group_cursor < 0:
            raise ValueError("group_cursor must be non-negative")
        max_new = self.max_new_tokens
        for prompt in inputs:
            if len(prompt.token_ids) + max_new > self.config.max_total_tokens:
                raise ValueError(
                    f"prompt {prompt.record.row_digest[:12]} plus {max_new} response tokens "
                    f"exceeds rollout.max_total_tokens={self.config.max_total_tokens}"
                )
        groups: list[tuple[int, ...]] = []
        current: list[int] = []
        current_width = 0
        logical_groups: list[RolloutGroupIdentity] = []
        for prompt_index, prompt in enumerate(inputs):
            prompt_group_id = f"g{group_cursor + prompt_index:012d}-{prompt.record.row_digest[:12]}"
            logical_groups.extend(
                RolloutGroupIdentity(
                    prompt_group_id=prompt_group_id,
                    prompt_digest=prompt.rendered_prompt_digest,
                    sample_index=sample_index,
                    samples_per_prompt=self.config.samples_per_prompt,
                )
                for sample_index in range(self.config.samples_per_prompt)
            )
        for index, _logical_group in enumerate(logical_groups):
            prompt = inputs[index // self.config.samples_per_prompt]
            candidate_width = max(current_width, len(prompt.token_ids))
            candidate_size = len(current) + 1
            padded_tokens = candidate_size * (candidate_width + max_new)
            if current and (
                candidate_size > self.config.prompt_batch_size
                or padded_tokens > self.config.max_padded_tokens
            ):
                groups.append(tuple(current))
                current = []
                current_width = 0
            current.append(index)
            current_width = max(current_width, len(prompt.token_ids))
            if current_width + max_new > self.config.max_padded_tokens:
                raise ValueError(
                    f"one prompt needs {current_width + max_new} padded tokens, above "
                    f"rollout.max_padded_tokens={self.config.max_padded_tokens}"
                )
        if current:
            groups.append(tuple(current))
        return PreparedPromptBatch(
            prompts=tuple(inputs),
            groups=tuple(logical_groups),
            physical_batches=tuple(groups),
            group_cursor=group_cursor,
        )

    def _generate_group(
        self,
        batch: PreparedPromptBatch,
        indices: tuple[int, ...],
        *,
        base_seed: int,
        policy_version: int,
        policy_identity: Any,
    ) -> tuple[list[tuple[int, GenerationResult]], list[int], int]:
        try:
            requests = []
            for index in indices:
                logical_group = batch.groups[index]
                prompt_index = index // self.config.samples_per_prompt
                prompt = batch.prompts[prompt_index]
                sample_seed = (
                    base_seed * 1_000_003 + prompt_index
                    if self.config.backend is RolloutBackendKind.HF_REFERENCE
                    and self.config.samples_per_prompt == 1
                    else derive_sample_seed(
                        run_seed=base_seed,
                        prompt_digest=prompt.rendered_prompt_digest,
                        policy_version=policy_version,
                        sample_index=logical_group.sample_index,
                    )
                )
                requests.append(
                    GenerationRequest(
                        request_id=hashlib.sha256(
                            (
                                f"{logical_group.prompt_group_id}:{policy_version}:"
                                f"{logical_group.sample_index}"
                            ).encode("ascii")
                        ).hexdigest(),
                        group=logical_group,
                        deterministic_sample_seed=sample_seed,
                        prompt_token_ids=tuple(prompt.token_ids),
                        max_new_tokens=self.max_new_tokens,
                        sampling=SamplingParameters(
                            temperature=self.config.temperature,
                            top_p=self.config.top_p,
                            top_k=self.config.top_k,
                        ),
                        need_sampled_token_logprobs=self.config.record_logprobs,
                        expected_policy_identity=policy_identity,
                    )
                )
            generated = self.generation_backend.generate(requests)
            output = list(generated.results)
        except BaseException as exc:
            message = str(exc).lower()
            is_oom = type(exc).__name__ in {"OutOfMemoryError", "CudaOutOfMemoryError"} or (
                "out of memory" in message and "cuda" in message
            )
            if not is_oom:
                raise
            if len(indices) == 1:
                raise GpuMemoryError(
                    "prompt generation ran out of GPU memory at physical batch size 1",
                    hint="lower max_prompt_length/max_response_length or choose a smaller model",
                ) from exc
            midpoint = len(indices) // 2
            left, left_sizes, left_down = self._generate_group(
                batch,
                indices[:midpoint],
                base_seed=base_seed,
                policy_version=policy_version,
                policy_identity=policy_identity,
            )
            right, right_sizes, right_down = self._generate_group(
                batch,
                indices[midpoint:],
                base_seed=base_seed,
                policy_version=policy_version,
                policy_identity=policy_identity,
            )
            return left + right, left_sizes + right_sizes, left_down + right_down + 1
        if len(output) != len(indices):
            raise RuntimeError(
                f"backend returned {len(output)} generations for physical batch {len(indices)}"
            )
        return list(zip(indices, output, strict=True)), [len(indices)], 0

    def generate(
        self,
        batch: PreparedPromptBatch,
        *,
        policy_version: int,
        seed: int,
    ) -> GeneratedPromptBatch:
        if self._closed:
            raise RuntimeError("prompt rollout runtime is closed")
        capabilities = self.generation_backend.inspect()
        policy_identity = build_rollout_policy_identity(
            backend=self.backend,
            parameter_version=policy_version,
            generation_backend=self.config.backend,
            backend_version=capabilities.backend_version,
            profile_identity=self.profile_identity,
            execution_plan_digest=self.execution_plan_digest,
        )
        self.generation_backend.synchronize(PolicySnapshot(policy_identity))
        indexed: list[tuple[int, GenerationResult]] = []
        physical_sizes: list[int] = []
        downshifts = 0
        for group in batch.physical_batches:
            rows, sizes, count = self._generate_group(
                batch,
                group,
                base_seed=seed,
                policy_version=policy_version,
                policy_identity=policy_identity,
            )
            indexed.extend(rows)
            physical_sizes.extend(sizes)
            downshifts += count
        indexed.sort(key=lambda item: item[0])
        results = tuple(result for _, result in indexed)
        return GeneratedPromptBatch(
            outputs=tuple(
                GenerationOutput(
                    token_ids=list(result.output_token_ids),
                    text=result.decoded_text,
                    stop_reason=result.stop_reason,
                    matched_stop=result.matched_stop,
                    logprobs=list(result.sampled_token_logprobs),
                )
                for result in results
            ),
            groups=tuple(result.group for result in results),
            generation_seeds=tuple(
                (
                    seed * 1_000_003 + index
                    if self.config.backend is RolloutBackendKind.HF_REFERENCE
                    and self.config.samples_per_prompt == 1
                    else derive_sample_seed(
                        run_seed=seed,
                        prompt_digest=result.group.prompt_digest,
                        policy_version=policy_version,
                        sample_index=result.group.sample_index,
                    )
                )
                for index, result in enumerate(results)
            ),
            rollout_policy_identity_digest=policy_identity.digest,
            policy_version=policy_version,
            physical_batch_sizes=tuple(physical_sizes),
            oom_downshifts=downshifts,
        )

    def to_trajectories(
        self,
        batch: PreparedPromptBatch,
        generated: GeneratedPromptBatch,
        *,
        policy_version: int,
    ) -> list[Trajectory]:
        if generated.policy_version != policy_version:
            raise ValueError(
                f"generated policy version {generated.policy_version} does not match {policy_version}"
            )
        if len(generated.outputs) != len(batch.groups):
            raise ValueError("generated sample count does not match the prepared batch")
        trajectories: list[Trajectory] = []
        model_id = str(getattr(self.backend, "model_id", self.backend.capabilities.name))
        revision = getattr(self.backend, "model_revision", None)
        for logical_index, (output, group, generation_seed) in enumerate(
            zip(
                generated.outputs,
                generated.groups,
                generated.generation_seeds,
                strict=True,
            )
        ):
            prompt_index = logical_index // self.config.samples_per_prompt
            prompt = batch.prompts[prompt_index]
            prompt_ids = list(prompt.token_ids)
            response_ids = list(output.token_ids)
            if not response_ids:
                raise ValueError(
                    f"actor produced an empty response for prompt {prompt.record.row_digest[:12]}"
                )
            boundary = len(prompt_ids)
            token_ids = [*prompt_ids, *response_ids]
            trajectory_id = derive_grouped_trajectory_id(
                prompt_group_id=group.prompt_group_id,
                sample_index=group.sample_index,
                rollout_policy_identity_digest=generated.rollout_policy_identity_digest,
                generation_seed=generation_seed,
            )
            metadata = {
                "source_kind": "verl_parquet",
                "data_source": prompt.record.data_source,
                "ability": prompt.record.ability,
                "reward_model": prompt.record.reward_model,
                "extra_info": prompt.record.extra_info,
                "source_file": prompt.record.source_file,
                "source_row_index": prompt.record.source_row_index,
                "row_digest": prompt.record.row_digest,
                "rendered_prompt_digest": prompt.rendered_prompt_digest,
                "tokenizer_identity": prompt.tokenizer_identity,
                "prompt_token_count": prompt.prompt_token_count,
                "original_prompt_token_count": prompt.original_prompt_token_count,
                "truncation_decision": prompt.truncation_decision,
                "response_token_count": len(response_ids),
                "generation_stop_reason": output.stop_reason,
                "rollout_backend": self.config.backend.value,
                "rollout_backend_version": self.generation_backend.inspect().backend_version,
            }
            if self.config.record_logprobs:
                if len(output.logprobs) != len(response_ids):
                    raise ValueError(
                        "PG rollout requested sampled-token log-probabilities but the backend "
                        "did not return exactly one value per response token"
                    )
                metadata["actor_rollout_log_probs"] = list(output.logprobs)
                metadata["actor_rollout_policy_version"] = policy_version
            trajectories.append(
                Trajectory(
                    schema_version=TRAJECTORY_SCHEMA_VERSION,
                    trajectory_id=trajectory_id,
                    task_id=prompt.record.row_digest,
                    environment="verl_parquet",
                    token_ids=token_ids,
                    attention_mask=[1] * len(token_ids),
                    model_generated_mask=[False] * boundary + [True] * len(response_ids),
                    critical_mask=[False] * len(token_ids),
                    spans=[
                        Span(
                            span_type=SpanType.USER,
                            start=0,
                            end=boundary,
                            turn_id=0,
                            text=prompt.text,
                            metadata={"rendered_prompt_digest": prompt.rendered_prompt_digest},
                        ),
                        Span(
                            span_type=SpanType.ASSISTANT_TEXT,
                            start=boundary,
                            end=len(token_ids),
                            turn_id=0,
                            text=output.text,
                        ),
                    ],
                    policy_version=policy_version,
                    tokenizer_fingerprint=str(self.backend.tokenizer.fingerprint),
                    model_id=model_id,
                    model_revision=revision,
                    prompt_group_id=group.prompt_group_id,
                    prompt_digest=group.prompt_digest,
                    sample_index=group.sample_index,
                    samples_per_prompt=group.samples_per_prompt,
                    generation_seed=generation_seed,
                    rollout_backend=self.config.backend.value,
                    rollout_policy_identity_digest=(generated.rollout_policy_identity_digest),
                    termination_reason=(
                        TerminationReason.EOS_WITHOUT_FINAL
                        if output.stop_reason == "eos"
                        else TerminationReason.MAX_TOKENS
                    ),
                    generated_token_count=len(response_ids),
                    assistant_turns=1,
                    metadata=metadata,
                )
            )
        validate_trajectory_groups(trajectories)
        return trajectories

    def close(self) -> None:
        self.generation_backend.close()
        self._closed = True


class ToolEnvironmentRolloutRuntime:
    """Adapter preserving the existing multi-turn tool runner byte semantics."""

    def __init__(self, runner: RolloutRunner) -> None:
        self.runner = runner

    def prepare_batch(self, inputs: Any) -> tuple[Any, ...]:
        return tuple(inputs)

    def generate(
        self, batch: tuple[Any, ...], *, policy_version: int, seed: int
    ) -> list[Trajectory]:
        return [
            self.runner.rollout(
                task,
                policy_version=policy_version,
                seed=seed * 1_000_003 + index,
            )
            for index, task in enumerate(batch)
        ]

    def to_trajectories(
        self,
        batch: tuple[Any, ...],
        generated: list[Trajectory],
        *,
        policy_version: int,
    ) -> list[Trajectory]:
        del batch
        if any(item.policy_version != policy_version for item in generated):
            raise ValueError("tool rollout policy version changed inside one batch")
        return generated

    def close(self) -> None:
        """The trainer owns and closes the backward-compatible environment."""
