from __future__ import annotations

import json
from pathlib import Path

import pytest

from miniverl.schemas.trajectory import (
    TRAJECTORY_SCHEMA_VERSION,
    Span,
    SpanType,
    TerminationReason,
    Trajectory,
    derive_grouped_trajectory_id,
    validate_trajectory_groups,
)
from miniverl.training.checkpoint import CheckpointState
from miniverl.trajectory.io import (
    append_trajectory_groups,
    read_trajectories,
    write_trajectories,
)
from miniverl.utils.runs import write_json_atomic


def _group(group_index: int, *, n: int = 2) -> list[Trajectory]:
    group_id = f"group-{group_index}"
    prompt_digest = f"{group_index + 1:064x}"
    policy_digest = "a" * 64
    rows = []
    for sample_index in range(n):
        seed = 1000 + group_index * 10 + sample_index
        rows.append(
            Trajectory(
                schema_version=TRAJECTORY_SCHEMA_VERSION,
                trajectory_id=derive_grouped_trajectory_id(
                    prompt_group_id=group_id,
                    sample_index=sample_index,
                    rollout_policy_identity_digest=policy_digest,
                    generation_seed=seed,
                ),
                task_id=f"task-{group_index}",
                environment="verl_parquet",
                token_ids=[1],
                attention_mask=[1],
                model_generated_mask=[False],
                critical_mask=[False],
                spans=[Span(span_type=SpanType.USER, start=0, end=1, turn_id=0)],
                tokenizer_fingerprint="tokenizer",
                model_id="model",
                termination_reason=TerminationReason.MAX_TOKENS,
                prompt_group_id=group_id,
                prompt_digest=prompt_digest,
                sample_index=sample_index,
                samples_per_prompt=n,
                generation_seed=seed,
                rollout_backend="hf_cached",
                rollout_policy_identity_digest=policy_digest,
            )
        )
    return rows


def test_group_validator_rejects_a_partial_group() -> None:
    with pytest.raises(ValueError, match="incomplete"):
        validate_trajectory_groups(_group(0)[:1])


def test_checkpoint_round_trips_group_and_prompt_cursors() -> None:
    state = CheckpointState(
        task_cursor=12,
        prompt_cursor=12,
        rollout_group_cursor=6,
        trajectory_count=24,
        samples_per_prompt=4,
        pending_group_identity=[],
        committed_group_identity=["group-5"],
        backend_sync_identity="b" * 64,
    )

    restored = CheckpointState.from_dict(state.to_dict())

    assert restored.prompt_cursor == 12
    assert restored.rollout_group_cursor == 6
    assert restored.trajectory_count == 24
    assert restored.samples_per_prompt == 4
    assert restored.committed_group_identity == ["group-5"]
    assert restored.backend_sync_identity == "b" * 64


def test_transactional_group_append_never_publishes_half_a_group(tmp_path: Path) -> None:
    target = tmp_path / "trajectories.jsonl"
    append_trajectory_groups(target, _group(0), transaction_id="first")

    rows = read_trajectories(target)
    assert len(rows) == 2
    assert {row.sample_index for row in rows} == {0, 1}
    assert not (tmp_path / ".trajectories.jsonl.group-transaction.json").exists()


def test_exact_committed_group_replay_is_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "trajectories.jsonl"
    group = _group(0)
    append_trajectory_groups(target, group, transaction_id="first")
    before = target.read_bytes()

    written = append_trajectory_groups(target, group, transaction_id="replayed")

    assert written == 0
    assert target.read_bytes() == before
    assert len(read_trajectories(target)) == 2


def test_partial_committed_group_replay_fails_closed(tmp_path: Path) -> None:
    target = tmp_path / "trajectories.jsonl"
    target.write_text(
        json.dumps(_group(0)[0].model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="partially present"):
        append_trajectory_groups(target, _group(0), transaction_id="partial-replay")


def test_generic_writer_upgrades_n1_legacy_object_to_current_schema(tmp_path: Path) -> None:
    target = tmp_path / "trajectories.jsonl"
    legacy = _group(0, n=1)[0].model_copy(
        update={
            "schema_version": 2,
            "trajectory_id": "legacy-id",
            "prompt_group_id": None,
            "prompt_digest": None,
            "sample_index": None,
            "samples_per_prompt": None,
            "generation_seed": None,
            "rollout_backend": None,
            "rollout_policy_identity_digest": None,
        }
    )

    write_trajectories(target, [legacy])
    loaded = read_trajectories(target)[0]

    assert loaded.schema_version == TRAJECTORY_SCHEMA_VERSION
    assert loaded.samples_per_prompt == 1
    assert loaded.sample_index == 0
    assert loaded.metadata["legacy_trajectory_id"] == "legacy-id"
    assert loaded.trajectory_id == derive_grouped_trajectory_id(
        prompt_group_id=str(loaded.prompt_group_id),
        sample_index=0,
        rollout_policy_identity_digest=str(loaded.rollout_policy_identity_digest),
        generation_seed=int(loaded.generation_seed or 0),
    )
    assert legacy.trajectory_id == loaded.trajectory_id


def test_failed_commit_publication_restores_old_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import miniverl.trajectory.io as trajectory_io

    target = tmp_path / "trajectories.jsonl"
    append_trajectory_groups(target, _group(0), transaction_id="first")
    before = target.read_bytes()
    calls = 0
    original = trajectory_io.write_json_atomic

    def fail_commit(path, payload):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected committed-journal failure")
        return original(path, payload)

    monkeypatch.setattr(trajectory_io, "write_json_atomic", fail_commit)
    with pytest.raises(OSError, match="committed-journal"):
        append_trajectory_groups(target, _group(1), transaction_id="second")

    assert target.read_bytes() == before
    assert [row.prompt_group_id for row in read_trajectories(target)] == ["group-0"] * 2
    assert not (tmp_path / ".trajectories.jsonl.group-transaction.json").exists()


def test_pending_crash_journal_is_rolled_back_before_the_next_group(
    tmp_path: Path,
) -> None:
    target = tmp_path / "trajectories.jsonl"
    append_trajectory_groups(target, _group(0), transaction_id="first")
    committed = target.read_bytes()
    with target.open("ab") as handle:
        handle.write(b'{"partial":')
    journal = tmp_path / ".trajectories.jsonl.group-transaction.json"
    write_json_atomic(
        journal,
        {
            "schema_version": 1,
            "status": "pending",
            "transaction_id": "crashed",
            "start_offset": len(committed),
            "groups": ["group-1"],
            "trajectories": 2,
        },
    )

    append_trajectory_groups(target, _group(2), transaction_id="after-recovery")

    rows = read_trajectories(target)
    assert [row.prompt_group_id for row in rows] == ["group-0"] * 2 + ["group-2"] * 2
    assert json.loads(target.read_text(encoding="utf-8").splitlines()[-1])["schema_version"] == 3
    assert not journal.exists()
