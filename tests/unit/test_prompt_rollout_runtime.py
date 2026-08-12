from __future__ import annotations

from miniverl.config.models import RolloutConfig, VerlParquetSourceConfig
from miniverl.data.verl_parquet import PromptRecord, RenderedPrompt
from miniverl.models.tokenizers import ToyTokenizer
from miniverl.models.toy import ToyBackend
from miniverl.runtime.rollout import PromptDatasetRolloutRuntime


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
