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
from miniverl.schemas.trajectory import Span, SpanType, TerminationReason, Trajectory

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
    physical_batches: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class GeneratedPromptBatch:
    """Outputs restored to logical order plus the actual physical batch sizes."""

    outputs: tuple[GenerationOutput, ...]
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
    ) -> None:
        self.backend = backend
        self.source_config = source_config
        self.config = rollout_config
        self._closed = False

    def prepare_batch(self, inputs: list[RenderedPrompt]) -> PreparedPromptBatch:
        if self._closed:
            raise RuntimeError("prompt rollout runtime is closed")
        if not inputs:
            raise ValueError("prompt rollout batch cannot be empty")
        max_new = self.config.max_new_tokens_per_turn
        for prompt in inputs:
            if len(prompt.token_ids) + max_new > self.config.max_total_tokens:
                raise ValueError(
                    f"prompt {prompt.record.row_digest[:12]} plus {max_new} response tokens "
                    f"exceeds rollout.max_total_tokens={self.config.max_total_tokens}"
                )
        groups: list[tuple[int, ...]] = []
        current: list[int] = []
        current_width = 0
        for index, prompt in enumerate(inputs):
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
        return PreparedPromptBatch(prompts=tuple(inputs), physical_batches=tuple(groups))

    def _generate_group(
        self,
        batch: PreparedPromptBatch,
        indices: tuple[int, ...],
        *,
        base_seed: int,
    ) -> tuple[list[tuple[int, GenerationOutput]], list[int], int]:
        try:
            output = self.backend.generate_batch(
                [batch.prompts[index].token_ids for index in indices],
                max_new_tokens=self.config.max_new_tokens_per_turn,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                seeds=[base_seed * 1_000_003 + index for index in indices],
            )
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
                batch, indices[:midpoint], base_seed=base_seed
            )
            right, right_sizes, right_down = self._generate_group(
                batch, indices[midpoint:], base_seed=base_seed
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
        indexed: list[tuple[int, GenerationOutput]] = []
        physical_sizes: list[int] = []
        downshifts = 0
        for group in batch.physical_batches:
            rows, sizes, count = self._generate_group(batch, group, base_seed=seed)
            indexed.extend(rows)
            physical_sizes.extend(sizes)
            downshifts += count
        indexed.sort(key=lambda item: item[0])
        return GeneratedPromptBatch(
            outputs=tuple(output for _, output in indexed),
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
        if len(generated.outputs) != len(batch.prompts):
            raise ValueError("generated prompt count does not match the prepared batch")
        trajectories: list[Trajectory] = []
        model_id = str(getattr(self.backend, "model_id", self.backend.capabilities.name))
        revision = getattr(self.backend, "model_revision", None)
        for prompt, output in zip(batch.prompts, generated.outputs, strict=True):
            prompt_ids = list(prompt.token_ids)
            response_ids = list(output.token_ids)
            if not response_ids:
                raise ValueError(
                    f"actor produced an empty response for prompt {prompt.record.row_digest[:12]}"
                )
            boundary = len(prompt_ids)
            token_ids = [*prompt_ids, *response_ids]
            trajectory_id = hashlib.sha256(
                f"{prompt.record.row_digest}:{policy_version}".encode("ascii")
            ).hexdigest()
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
            }
            trajectories.append(
                Trajectory(
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
        return trajectories

    def close(self) -> None:
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
