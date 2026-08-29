# ruff: noqa: E402

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytestmark = pytest.mark.torch

from miniverl.config.models import RolloutConfig, VerlParquetSourceConfig
from miniverl.data.verl_parquet import PromptRecord, RenderedPrompt
from miniverl.models.tokenizers import ToyTokenizer
from miniverl.models.toy import ToyBackend
from miniverl.runtime.rollout import PromptDatasetRolloutRuntime
from miniverl.schemas.trajectory import TRAJECTORY_SCHEMA_VERSION, validate_trajectory_groups


def _record(index: int) -> PromptRecord:
    return PromptRecord(
        prompt=f"prompt {index}",
        data_source="unit",
        ability=None,
        reward_model=None,
        extra_info={"index": index},
        source_file="train.parquet",
        source_row_index=index,
        row_digest=f"{index:064x}",
        canonical_payload="{}",
    )


def test_real_padded_greedy_generation_matches_sequential() -> None:
    tokenizer = ToyTokenizer()
    backend = ToyBackend(tokenizer=tokenizer, model_id="toy", seed=19, trainable=False)
    prompts = [tokenizer.encode("short"), tokenizer.encode("a longer prompt")]

    expected = [
        backend.generate(row, max_new_tokens=7, temperature=0.0, seed=100 + index)
        for index, row in enumerate(prompts)
    ]
    actual = backend.generate_batch(
        prompts,
        max_new_tokens=7,
        temperature=0.0,
        seeds=[100, 101],
    )

    assert [row.token_ids for row in actual] == [row.token_ids for row in expected]
    assert [row.text for row in actual] == [row.text for row in expected]
    assert [len(row.token_ids) for row in actual] == [len(row.token_ids) for row in expected]

    single = backend.generate_batch(prompts[:1], max_new_tokens=7, temperature=0.0, seeds=[100])
    assert single[0].token_ids == expected[0].token_ids


def test_prompt_runtime_records_exact_rollout_logprobs_for_pg_targets() -> None:
    tokenizer = ToyTokenizer()
    backend = ToyBackend(tokenizer=tokenizer, model_id="toy", seed=23, trainable=False)
    runtime = PromptDatasetRolloutRuntime(
        backend=backend,
        source_config=VerlParquetSourceConfig(
            train_files=["unused.parquet"], allow_plain_string_prompts=True
        ),
        rollout_config=RolloutConfig(
            backend="hf_cached",
            max_new_tokens_per_turn=4,
            max_total_tokens=32,
            temperature=0.0,
            prompt_batch_size=2,
            max_padded_tokens=64,
            record_logprobs=True,
        ),
    )
    rendered = [
        RenderedPrompt(
            record=_record(index),
            text=text,
            token_ids=tuple(tokenizer.encode(text)),
            tokenizer_identity=tokenizer.identity,
            rendered_prompt_digest=f"{index + 40:064x}",
            prompt_token_count=len(tokenizer.encode(text)),
            truncation_decision="not_needed",
            original_prompt_token_count=len(tokenizer.encode(text)),
        )
        for index, text in enumerate(("short", "longer prompt"))
    ]

    batch = runtime.prepare_batch(rendered)
    generated = runtime.generate(batch, policy_version=7, seed=11)
    trajectories = runtime.to_trajectories(batch, generated, policy_version=7)

    for output, trajectory in zip(generated.outputs, trajectories, strict=True):
        assert len(output.logprobs) == len(output.token_ids)
        assert all(value <= 0.0 for value in output.logprobs)
        assert trajectory.metadata["actor_rollout_log_probs"] == output.logprobs
        assert trajectory.metadata["actor_rollout_policy_version"] == 7
        assert trajectory.metadata["rollout_backend"] == "hf_cached"


def test_prompt_trajectories_never_select_prompt_or_padding_tokens() -> None:
    tokenizer = ToyTokenizer()
    backend = ToyBackend(tokenizer=tokenizer, model_id="toy", seed=7, trainable=False)
    runtime = PromptDatasetRolloutRuntime(
        backend=backend,
        source_config=VerlParquetSourceConfig(
            train_files=["unused.parquet"], allow_plain_string_prompts=True
        ),
        rollout_config=RolloutConfig(
            max_new_tokens_per_turn=5,
            max_total_tokens=32,
            temperature=0.0,
            prompt_batch_size=2,
            max_padded_tokens=64,
        ),
    )
    rendered = [
        RenderedPrompt(
            record=_record(index),
            text=text,
            token_ids=tuple(tokenizer.encode(text)),
            tokenizer_identity=tokenizer.identity,
            rendered_prompt_digest=f"{index + 10:064x}",
            prompt_token_count=len(tokenizer.encode(text)),
            truncation_decision="not_needed",
            original_prompt_token_count=len(tokenizer.encode(text)),
        )
        for index, text in enumerate(("short", "a longer prompt"))
    ]

    batch = runtime.prepare_batch(rendered)
    generated = runtime.generate(batch, policy_version=4, seed=90)
    trajectories = runtime.to_trajectories(batch, generated, policy_version=4)

    assert [trajectory.task_id for trajectory in trajectories] == [
        row.record.row_digest for row in rendered
    ]
    for prompt, output, trajectory in zip(rendered, generated.outputs, trajectories, strict=True):
        assert trajectory.policy_version == 4
        assert trajectory.token_ids == [*prompt.token_ids, *output.token_ids]
        assert not any(trajectory.model_generated_mask[: prompt.prompt_token_count])
        assert all(trajectory.model_generated_mask[prompt.prompt_token_count :])
        assert trajectory.model_token_positions() == list(
            range(prompt.prompt_token_count, len(trajectory.token_ids))
        )
        assert trajectory.metadata["response_token_count"] == len(output.token_ids)


def test_prompt_runtime_enforces_independent_response_bound() -> None:
    tokenizer = ToyTokenizer()
    backend = ToyBackend(tokenizer=tokenizer, model_id="toy", seed=7, trainable=False)
    runtime = PromptDatasetRolloutRuntime(
        backend=backend,
        source_config=VerlParquetSourceConfig(
            train_files=["unused.parquet"],
            allow_plain_string_prompts=True,
            max_response_length=3,
        ),
        rollout_config=RolloutConfig(
            max_new_tokens_per_turn=20,
            max_total_tokens=64,
            temperature=0.0,
            max_padded_tokens=64,
        ),
    )
    prompt_ids = tuple(tokenizer.encode("short"))
    rendered = RenderedPrompt(
        record=_record(0),
        text="short",
        token_ids=prompt_ids,
        tokenizer_identity=tokenizer.identity,
        rendered_prompt_digest="f" * 64,
        prompt_token_count=len(prompt_ids),
        truncation_decision="not_needed",
        original_prompt_token_count=len(prompt_ids),
    )
    generated = runtime.generate(runtime.prepare_batch([rendered]), policy_version=0, seed=1)
    assert len(generated.outputs[0].token_ids) == 3


def test_physical_batches_respect_padded_token_budget_without_reordering() -> None:
    tokenizer = ToyTokenizer()
    backend = ToyBackend(tokenizer=tokenizer, model_id="toy", seed=7, trainable=False)
    runtime = PromptDatasetRolloutRuntime(
        backend=backend,
        source_config=VerlParquetSourceConfig(
            train_files=["unused.parquet"], allow_plain_string_prompts=True
        ),
        rollout_config=RolloutConfig(
            max_new_tokens_per_turn=4,
            max_total_tokens=32,
            temperature=0.0,
            prompt_batch_size=8,
            max_padded_tokens=24,
        ),
    )
    rendered = [
        RenderedPrompt(
            record=_record(index),
            text="x" * length,
            token_ids=tuple(tokenizer.encode("x" * length)),
            tokenizer_identity=tokenizer.identity,
            rendered_prompt_digest=f"{index + 20:064x}",
            prompt_token_count=length,
            truncation_decision="not_needed",
            original_prompt_token_count=length,
        )
        for index, length in enumerate((2, 5, 3, 6))
    ]

    batch = runtime.prepare_batch(rendered)
    assert [item.record.source_row_index for item in batch.prompts] == [0, 1, 2, 3]
    assert all(
        len(group) * (max(len(batch.prompts[index].token_ids) for index in group) + 4) <= 24
        for group in batch.physical_batches
    )
    generated = runtime.generate(batch, policy_version=0, seed=1)
    assert generated.physical_batch_sizes == tuple(len(group) for group in batch.physical_batches)
    assert len(generated.outputs) == 4


def test_oom_downshift_changes_only_physical_batching(monkeypatch) -> None:
    tokenizer = ToyTokenizer()
    backend = ToyBackend(tokenizer=tokenizer, model_id="toy", seed=7, trainable=False)
    original = backend.generate_batch

    def fail_multi(prefixes, **kwargs):  # type: ignore[no-untyped-def]
        if len(prefixes) > 1:
            raise RuntimeError("CUDA out of memory: injected")
        return original(prefixes, **kwargs)

    monkeypatch.setattr(backend, "generate_batch", fail_multi)
    rollout_config = RolloutConfig(
        max_new_tokens_per_turn=2,
        max_total_tokens=32,
        temperature=0.0,
        prompt_batch_size=2,
        max_padded_tokens=64,
    )
    runtime = PromptDatasetRolloutRuntime(
        backend=backend,
        source_config=VerlParquetSourceConfig(
            train_files=["unused.parquet"], allow_plain_string_prompts=True
        ),
        rollout_config=rollout_config,
    )
    rendered = [
        RenderedPrompt(
            record=_record(index),
            text="x",
            token_ids=tuple(tokenizer.encode("x")),
            tokenizer_identity=tokenizer.identity,
            rendered_prompt_digest=f"{index + 30:064x}",
            prompt_token_count=1,
            truncation_decision="not_needed",
            original_prompt_token_count=1,
        )
        for index in range(2)
    ]

    generated = runtime.generate(runtime.prepare_batch(rendered), policy_version=3, seed=4)

    assert generated.physical_batch_sizes == (1, 1)
    assert generated.oom_downshifts == 1
    assert rollout_config.prompt_batch_size == 2
    assert generated.policy_version == 3


def test_n4_group_identity_and_outputs_are_partition_invariant() -> None:
    tokenizer = ToyTokenizer()
    rendered = [
        RenderedPrompt(
            record=_record(index),
            text=f"prompt {index}",
            token_ids=tuple(tokenizer.encode(f"prompt {index}")),
            tokenizer_identity=tokenizer.identity,
            rendered_prompt_digest=f"{index + 50:064x}",
            prompt_token_count=len(tokenizer.encode(f"prompt {index}")),
            truncation_decision="not_needed",
            original_prompt_token_count=len(tokenizer.encode(f"prompt {index}")),
        )
        for index in range(2)
    ]

    def run(batch_size: int):  # type: ignore[no-untyped-def]
        backend = ToyBackend(tokenizer=tokenizer, model_id="toy", seed=31, trainable=False)
        runtime = PromptDatasetRolloutRuntime(
            backend=backend,
            source_config=VerlParquetSourceConfig(
                train_files=["unused.parquet"], allow_plain_string_prompts=True
            ),
            rollout_config=RolloutConfig(
                backend="hf_cached",
                samples_per_prompt=4,
                max_new_tokens_per_turn=4,
                max_total_tokens=32,
                temperature=0.8,
                prompt_batch_size=batch_size,
                max_padded_tokens=256,
                record_logprobs=True,
            ),
        )
        batch = runtime.prepare_batch(rendered, group_cursor=9)
        generated = runtime.generate(batch, policy_version=2, seed=123)
        return generated, runtime.to_trajectories(batch, generated, policy_version=2)

    whole, trajectories = run(8)
    partitioned, partitioned_trajectories = run(2)

    assert len(trajectories) == 8
    assert [row.token_ids for row in whole.outputs] == [
        row.token_ids for row in partitioned.outputs
    ]
    assert [row.logprobs for row in whole.outputs] == [row.logprobs for row in partitioned.outputs]
    assert [row.trajectory_id for row in trajectories] == [
        row.trajectory_id for row in partitioned_trajectories
    ]
    assert {row.schema_version for row in trajectories} == {TRAJECTORY_SCHEMA_VERSION}
    assert {row.prompt_group_id for row in trajectories} == {
        f"g{index:012d}-{rendered[index - 9].record.row_digest[:12]}" for index in (9, 10)
    }
    assert [row.sample_index for row in trajectories] == [0, 1, 2, 3, 0, 1, 2, 3]
    assert {row.samples_per_prompt for row in trajectories} == {4}
    assert len({row.generation_seed for row in trajectories}) == 8
    assert len({row.trajectory_id for row in trajectories}) == 8
    validate_trajectory_groups(trajectories)
