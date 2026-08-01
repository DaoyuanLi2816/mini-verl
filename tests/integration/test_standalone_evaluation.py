"""Standalone evaluation must restore an exact checkpoint without training state."""

from __future__ import annotations

import multiprocessing
import threading
from pathlib import Path

import pytest

from miniverl.errors import CheckpointError
from tests.conftest import requires_torch
from tests.integration.test_resume_and_swap import _config

pytestmark = [requires_torch, pytest.mark.torch]

pytest.importorskip("torch")


def _attempt_checkpoint_mutation(
    output_root: str,
    run_id: str,
    checkpoint_file: str,
    result_queue,
) -> None:  # type: ignore[no-untyped-def]
    from miniverl.errors import RunLockedError
    from miniverl.utils.locking import RunLock

    try:
        with RunLock(Path(output_root), run_id):
            Path(checkpoint_file).write_bytes(b"concurrent-writer-corruption")
            result_queue.put("mutated")
    except RunLockedError:
        result_queue.put("locked")


def test_standalone_eval_loads_only_weights_and_records_checkpoint_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import safetensors.torch

    import miniverl.training.trainer as trainer_module
    from miniverl.evaluation.evaluator import evaluate_run
    from miniverl.trainer import OPDTrainer
    from miniverl.training.checkpoint import validate_checkpoint

    trainer = OPDTrainer.from_config(
        _config(tmp_path, train={"cycles": 1}),
        run_id="standalone-eval",
    )
    trainer.train()
    run_dir = trainer.paths.root
    checkpoint = trainer.paths.checkpoints / "final"
    expected = validate_checkpoint(checkpoint)
    trainer.close()

    def refuse_optimizer(*_args, **_kwargs):
        raise AssertionError("standalone evaluation allocated an optimizer")

    monkeypatch.setattr(trainer_module, "build_optimizer", refuse_optimizer)
    original_load_file = safetensors.torch.load_file
    loaded: list[str] = []

    def record_load(path, *args, **kwargs):
        loaded.append(Path(path).name)
        return original_load_file(path, *args, **kwargs)

    monkeypatch.setattr(safetensors.torch, "load_file", record_load)
    payload = evaluate_run(run_dir, tasks=1)

    assert loaded == ["adapter.safetensors"]
    assert payload["checkpoint"] == str(checkpoint)
    assert payload["checkpoint_digest"] == expected.content_digest
    assert payload["checkpoint_integrity"] == "checksummed_v1"
    assert payload["checkpoint_global_step"] == expected.state.global_step
    expected_parameter_version = (
        expected.state.policy_version
        if expected.state.parameter_version is None
        else expected.state.parameter_version
    )
    expected_rollout_iteration = (
        max(expected.state.cycle + 1, 0)
        if expected.state.rollout_iteration is None
        else expected.state.rollout_iteration
    )
    expected_rollout_policy_version = (
        expected.state.policy_version
        if expected.state.rollout_policy_version is None
        else expected.state.rollout_policy_version
    )
    assert payload["global_step"] == expected.state.global_step
    assert payload["global_optimizer_step"] == expected.state.global_step
    assert payload["parameter_version"] == expected_parameter_version
    assert payload["policy_version"] == expected_parameter_version
    assert payload["rollout_iteration"] == expected_rollout_iteration
    assert payload["rollout_policy_version"] == expected_rollout_policy_version


def test_standalone_eval_refuses_a_run_without_a_checkpoint(tmp_path: Path) -> None:
    from miniverl.evaluation.evaluator import evaluate_run
    from miniverl.trainer import OPDTrainer

    trainer = OPDTrainer.from_config(
        _config(tmp_path, train={"cycles": 0}),
        run_id="no-checkpoint",
    )
    run_dir = trainer.paths.root
    trainer.close()

    with pytest.raises(CheckpointError, match="no checkpoint"):
        evaluate_run(run_dir)


def test_standalone_eval_acquires_lock_before_opening_mutable_run_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import miniverl.evaluation.evaluator as evaluator_module
    from miniverl.errors import RunLockedError
    from miniverl.evaluation.evaluator import evaluate_run
    from miniverl.utils.locking import RunLock

    run_dir = tmp_path / "locked-before-open"
    opened = False

    def forbidden_open(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        nonlocal opened
        opened = True
        raise AssertionError("run state was opened before lock acquisition")

    monkeypatch.setattr(evaluator_module.RunPaths, "open", forbidden_open)
    with (
        RunLock(run_dir.parent, run_dir.name),
        pytest.raises(RunLockedError, match="locked-before-open"),
    ):
        evaluate_run(run_dir)

    assert opened is False


def test_concurrent_writer_cannot_replace_validated_checkpoint_during_eval_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import miniverl.training.checkpoint as checkpoint_module
    from miniverl.evaluation.evaluator import evaluate_run
    from miniverl.trainer import OPDTrainer

    trainer = OPDTrainer.from_config(
        _config(tmp_path, train={"cycles": 1}),
        run_id="eval-checkpoint-owner",
    )
    trainer.train()
    run_dir = trainer.paths.root
    checkpoint_file = trainer.paths.checkpoints / "final" / "adapter.safetensors"
    trainer.close()

    validated = threading.Event()
    continue_eval = threading.Event()
    real_validate = checkpoint_module.validate_checkpoint

    def pause_after_validation(*args, **kwargs):  # type: ignore[no-untyped-def]
        result = real_validate(*args, **kwargs)
        validated.set()
        assert continue_eval.wait(timeout=30)
        return result

    monkeypatch.setattr(checkpoint_module, "validate_checkpoint", pause_after_validation)
    payloads: list[dict] = []
    failures: list[BaseException] = []

    def run_eval() -> None:
        try:
            payloads.append(evaluate_run(run_dir, tasks=1))
        except BaseException as exc:  # pragma: no cover - diagnostic capture
            failures.append(exc)

    eval_thread = threading.Thread(target=run_eval)
    eval_thread.start()
    assert validated.wait(timeout=30)

    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    writer = context.Process(
        target=_attempt_checkpoint_mutation,
        args=(str(run_dir.parent), run_dir.name, str(checkpoint_file), result_queue),
    )
    writer.start()
    writer.join(timeout=30)
    assert not writer.is_alive()
    assert result_queue.get(timeout=5) == "locked"

    continue_eval.set()
    eval_thread.join(timeout=60)
    assert not eval_thread.is_alive()
    assert failures == []
    assert len(payloads) == 1
    assert (
        payloads[0]["checkpoint_digest"]
        == real_validate(trainer.paths.checkpoints / "final").content_digest
    )


def test_eval_write_failure_preserves_manifest_and_releases_transferred_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import miniverl.evaluation.evaluator as evaluator_module
    from miniverl.evaluation.evaluator import evaluate_run
    from miniverl.trainer import OPDTrainer
    from miniverl.utils.locking import RunLock

    trainer = OPDTrainer.from_config(
        _config(tmp_path, train={"cycles": 1}),
        run_id="eval-write-failure",
    )
    trainer.train()
    run_dir = trainer.paths.root
    trainer.close()
    before = (run_dir / "manifest.json").read_bytes()
    monkeypatch.setattr(
        evaluator_module,
        "write_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("injected eval write failure")),
    )

    with pytest.raises(OSError, match="injected eval write failure"):
        evaluate_run(run_dir, tasks=1)

    assert (run_dir / "manifest.json").read_bytes() == before
    with RunLock(run_dir.parent, run_dir.name):
        pass


@pytest.mark.parametrize("was_training", [False, True])
def test_evaluation_attachment_restores_mode_and_manifest_after_rollout_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    was_training: bool,
) -> None:
    from miniverl.trainer import OPDTrainer, TrainerState

    config = _config(tmp_path, train={"cycles": 1})
    owner = OPDTrainer.from_config(config, run_id="attachment-mode-failure")
    owner.train()
    run_dir = owner.paths.root
    owner.close()
    before = (run_dir / "manifest.json").read_bytes()

    attachment = OPDTrainer.from_config(
        config,
        output_dir=run_dir.parent,
        run_id=run_dir.name,
        write_artifacts=False,
        for_evaluation=True,
    )
    attachment.student.set_train(was_training)
    injected = RuntimeError("injected attachment rollout failure")
    monkeypatch.setattr(
        attachment.runner, "rollout", lambda *_a, **_k: (_ for _ in ()).throw(injected)
    )

    with pytest.raises(RuntimeError, match="injected attachment rollout failure") as caught:
        attachment.evaluate(write=False)

    assert caught.value is injected
    assert attachment.student.model.training is was_training
    assert attachment.state is TrainerState.READY
    assert attachment._operation_guard.acquire(blocking=False)
    attachment._operation_guard.release()
    attachment.close()
    assert (run_dir / "manifest.json").read_bytes() == before
