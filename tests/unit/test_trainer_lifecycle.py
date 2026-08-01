"""Trainer ownership and teardown are explicit, destructive, and idempotent."""

from __future__ import annotations

import gc
import json
import threading
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.conftest import requires_torch

pytestmark = [requires_torch, pytest.mark.torch]


def _config(tmp_path, *, eval_tasks: int = 1):  # type: ignore[no-untyped-def]
    from tests.integration.test_toy_pipeline import _config as toy_config

    return toy_config(
        tmp_path,
        train={"cycles": 1, "rollouts_per_cycle": 1, "gradient_accumulation_steps": 1},
        eval={"enabled": False, "tasks": eval_tasks},
        report={"enabled": False},
    )


def test_close_releases_owned_resources_once_and_is_idempotent(tmp_path) -> None:
    from miniverl.trainer import OPDTrainer

    trainer = OPDTrainer.from_config(_config(tmp_path), run_id="release-once")
    student_release = Mock(wraps=trainer.student.release)
    teacher_release = Mock(wraps=trainer.teacher.release)
    environment_close = Mock()
    cache_flush = Mock()
    trainer.student.release = student_release
    trainer.teacher.release = teacher_release
    trainer.environment.close = environment_close
    trainer._cache = SimpleNamespace(flush=cache_flush)

    trainer.close()
    trainer.close()

    student_release.assert_called_once_with()
    teacher_release.assert_called_once_with()
    environment_close.assert_called_once_with()
    cache_flush.assert_called_once_with()
    assert trainer.student is None
    assert trainer.teacher is None
    assert trainer.runner is None
    assert trainer.scorer is None
    assert trainer.optimizer is None
    assert trainer._cache is None
    assert trainer._offline_samples is None


def test_model_construction_failure_never_leaves_an_orphan_run_directory(
    tmp_path, monkeypatch
) -> None:
    import miniverl.training.trainer as trainer_module
    from miniverl.trainer import OPDTrainer

    config = _config(tmp_path)
    run_id = "model-construction-failure"
    run_directory = Path(config.run.output_dir) / run_id
    monkeypatch.setattr(
        trainer_module,
        "build_student",
        Mock(side_effect=RuntimeError("injected model construction failure")),
    )

    with pytest.raises(RuntimeError, match="injected model construction failure"):
        OPDTrainer.from_config(config, run_id=run_id)

    if run_directory.exists():
        manifest_path = run_directory / "manifest.json"
        assert manifest_path.is_file()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["status"] == "failed_construction"
        assert manifest["failure"]["type"] == "RuntimeError"


def test_close_drops_model_optimizer_runner_scorer_and_target_provider_references(
    tmp_path,
) -> None:
    from miniverl.trainer import OPDTrainer

    trainer = OPDTrainer.from_config(_config(tmp_path), run_id="drop-strong-references")
    student_ref = weakref.ref(trainer.student)
    teacher_ref = weakref.ref(trainer.teacher)
    runner_ref = weakref.ref(trainer.runner)
    scorer_ref = weakref.ref(trainer.scorer)
    optimizer_ref = weakref.ref(trainer.optimizer)

    class ProviderOwner:
        pass

    owner = ProviderOwner()
    owner_ref = weakref.ref(owner)
    provider = lambda owner=owner: owner  # noqa: E731 - closure is the retention probe
    trainer._offline_samples = [SimpleNamespace(teacher=SimpleNamespace(provider=provider))]
    del owner, provider

    trainer.close()
    gc.collect()

    assert student_ref() is None
    assert teacher_ref() is None
    assert runner_ref() is None
    assert scorer_ref() is None
    assert optimizer_ref() is None
    assert owner_ref() is None


@pytest.mark.parametrize(
    ("method_name", "args"),
    [
        ("train", ()),
        ("evaluate", ()),
        ("save_checkpoint", ()),
        ("load_from_checkpoint", ("unused",)),
    ],
)
def test_public_operations_fail_actionably_after_close(
    tmp_path, method_name: str, args: tuple[object, ...]
) -> None:
    from miniverl.errors import LifecycleError
    from miniverl.trainer import OPDTrainer

    trainer = OPDTrainer.from_config(_config(tmp_path), run_id=f"closed-{method_name}")
    trainer.close()

    with pytest.raises(LifecycleError, match=rf"cannot {method_name}.*closed"):
        getattr(trainer, method_name)(*args)


def test_context_manager_preserves_the_original_error_when_cleanup_also_fails(tmp_path) -> None:
    from miniverl.trainer import OPDTrainer

    trainer = OPDTrainer.from_config(_config(tmp_path), run_id="preserve-original-error")
    student_release = Mock(side_effect=RuntimeError("cleanup exploded"))
    teacher_release = Mock(wraps=trainer.teacher.release)
    trainer.student.release = student_release
    trainer.teacher.release = teacher_release
    failed_train = Mock(side_effect=RuntimeError("training exploded"))
    trainer.train = failed_train

    with pytest.raises(RuntimeError, match="training exploded"), trainer:
        trainer.train()

    failed_train.assert_called_once_with()
    student_release.assert_called_once_with()
    teacher_release.assert_called_once_with()
    assert trainer.student is None
    assert trainer.teacher is None


def test_from_config_releases_partial_construction_when_startup_artifacts_fail(
    tmp_path, monkeypatch
) -> None:
    import miniverl.training.trainer as trainer_module
    from miniverl.trainer import OPDTrainer

    student_release = Mock()
    teacher_release = Mock()
    environment_close = Mock()
    real_build_student = trainer_module.build_student
    real_build_teacher = trainer_module.build_teacher
    real_make_environment = trainer_module.make_environment

    def build_student(*args, **kwargs):  # type: ignore[no-untyped-def]
        backend = real_build_student(*args, **kwargs)
        backend.release = student_release
        return backend

    def build_teacher(*args, **kwargs):  # type: ignore[no-untyped-def]
        backend = real_build_teacher(*args, **kwargs)
        backend.release = teacher_release
        return backend

    def make_environment(*args, **kwargs):  # type: ignore[no-untyped-def]
        environment = real_make_environment(*args, **kwargs)
        environment.close = environment_close
        return environment

    monkeypatch.setattr(trainer_module, "build_student", build_student)
    monkeypatch.setattr(trainer_module, "build_teacher", build_teacher)
    monkeypatch.setattr(trainer_module, "make_environment", make_environment)
    monkeypatch.setattr(
        OPDTrainer,
        "_write_startup_artifacts",
        Mock(side_effect=RuntimeError("artifact write failed")),
    )

    with pytest.raises(RuntimeError, match="artifact write failed"):
        OPDTrainer.from_config(_config(tmp_path), run_id="partial-construction")

    student_release.assert_called_once_with()
    teacher_release.assert_called_once_with()
    environment_close.assert_called_once_with()


def _artifact_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _manifest_status(trainer) -> str:  # type: ignore[no-untyped-def]
    return json.loads(trainer.paths.manifest.read_text(encoding="utf-8"))["status"]


def _parameter_snapshot(trainer) -> dict[str, object]:  # type: ignore[no-untyped-def]
    return {
        name: tensor.detach().clone()
        for name, tensor in trainer.student.trainable_state_dict().items()
    }


def _assert_parameters_equal(
    before: dict[str, object],
    trainer,  # type: ignore[no-untyped-def]
) -> None:
    import torch

    after = trainer.student.trainable_state_dict()
    assert set(after) == set(before)
    assert all(torch.equal(before[name], after[name]) for name in before)  # type: ignore[arg-type]


def _progress_snapshot(trainer) -> tuple[object, ...]:  # type: ignore[no-untyped-def]
    return (
        trainer.global_step,
        trainer.policy_version,
        trainer.parameter_version,
        trainer.cycle,
        trainer.task_cursor,
        trainer._cycles_completed,
        trainer._start_cycle,
        trainer._resumed,
        trainer._resumed_from,
    )


def _resource_snapshot(trainer) -> tuple[object, ...]:  # type: ignore[no-untyped-def]
    return (
        trainer.student,
        trainer.teacher,
        trainer.optimizer,
        trainer.runner,
        trainer.environment,
        trainer.metrics_log,
        trainer.events,
    )


def _trained_checkpoint(tmp_path, run_id: str):  # type: ignore[no-untyped-def]
    from miniverl.trainer import OPDTrainer

    config = _config(tmp_path)
    source = OPDTrainer.from_config(config, run_id=f"{run_id}-source")
    source.train()
    checkpoint = source.paths.checkpoints / "final"
    source.close()
    return config, checkpoint


def _thread_call(call):  # type: ignore[no-untyped-def]
    results: list[object] = []
    failures: list[BaseException] = []

    def run() -> None:
        try:
            results.append(call())
        except BaseException as exc:  # pragma: no cover - diagnostic capture
            failures.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    return thread, results, failures


def test_new_trainer_manifest_is_ready_and_close_is_terminal(tmp_path) -> None:
    from miniverl.trainer import OPDTrainer

    trainer = OPDTrainer.from_config(_config(tmp_path), run_id="ready-close")

    assert _manifest_status(trainer) == "ready"
    trainer.close()

    assert _manifest_status(trainer) == "closed_before_training"


def test_context_exit_without_training_closes_ready_manifest(tmp_path) -> None:
    from miniverl.trainer import OPDTrainer

    with OPDTrainer.from_config(_config(tmp_path), run_id="ready-context") as trainer:
        assert _manifest_status(trainer) == "ready"

    assert _manifest_status(trainer) == "closed_before_training"


def test_evaluation_only_attachment_close_does_not_change_training_manifest(tmp_path) -> None:
    from miniverl.trainer import OPDTrainer

    config = _config(tmp_path)
    with OPDTrainer.from_config(config, run_id="evaluation-attachment") as trainer:
        trainer.train()
        run_root = trainer.paths.root
    before = (run_root / "manifest.json").read_bytes()

    attachment = OPDTrainer.from_config(
        config,
        output_dir=run_root.parent,
        run_id=run_root.name,
        write_artifacts=False,
        for_evaluation=True,
    )
    attachment.close()

    assert (run_root / "manifest.json").read_bytes() == before


def test_manifest_transition_failure_emits_no_run_start_and_is_not_stale_running(
    tmp_path,
    monkeypatch,
) -> None:
    from miniverl.trainer import OPDTrainer, TrainerState

    trainer = OPDTrainer.from_config(_config(tmp_path), run_id="transition-failure")
    monkeypatch.setattr(
        trainer,
        "_transition_manifest_to_running",
        Mock(side_effect=OSError("injected transition failure")),
        raising=False,
    )

    with pytest.raises(OSError, match="injected transition failure"):
        trainer.train()

    assert trainer.state is TrainerState.FAILED
    assert _manifest_status(trainer) != "running"
    events = (
        trainer.paths.events.read_text(encoding="utf-8") if trainer.paths.events.exists() else ""
    )
    assert '"event":"run_start"' not in events.replace(" ", "")
    trainer.close()


@pytest.mark.parametrize("operation", ["evaluate", "save_checkpoint"])
def test_external_operations_are_rejected_while_training(
    tmp_path,
    monkeypatch,
    operation: str,
) -> None:
    from miniverl.errors import LifecycleError
    from miniverl.trainer import OPDTrainer

    trainer = OPDTrainer.from_config(_config(tmp_path), run_id=f"external-{operation}")
    entered = threading.Event()
    release = threading.Event()
    real_train_impl = trainer._train_impl

    def blocked_train_impl():  # type: ignore[no-untyped-def]
        entered.set()
        assert release.wait(timeout=10)
        return real_train_impl()

    monkeypatch.setattr(trainer, "_train_impl", blocked_train_impl)
    failures: list[BaseException] = []

    def run_training() -> None:
        try:
            trainer.train()
        except BaseException as exc:  # pragma: no cover - diagnostic capture
            failures.append(exc)

    thread = threading.Thread(target=run_training)
    thread.start()
    assert entered.wait(timeout=10)
    before = _artifact_bytes(trainer.paths.root)

    with pytest.raises(LifecycleError, match=rf"cannot {operation}.*running"):
        getattr(trainer, operation)()

    assert _artifact_bytes(trainer.paths.root) == before
    release.set()
    thread.join(timeout=30)
    assert not thread.is_alive()
    assert failures == []
    trainer.close()


def test_public_evaluation_works_in_ready_and_completed_states(tmp_path) -> None:
    from miniverl.trainer import OPDTrainer

    trainer = OPDTrainer.from_config(_config(tmp_path), run_id="public-evaluation")

    assert trainer.evaluate(write=False)["tasks"] == 1
    trainer.train()
    assert trainer.evaluate(write=False)["tasks"] == 1
    trainer.close()


@pytest.mark.parametrize("was_training", [False, True])
def test_successful_evaluation_restores_exact_previous_model_mode(
    tmp_path,
    was_training: bool,
) -> None:
    from miniverl.trainer import OPDTrainer

    trainer = OPDTrainer.from_config(
        _config(tmp_path),
        run_id=f"successful-eval-mode-{was_training}",
    )
    trainer.student.set_train(was_training)

    assert trainer.evaluate(write=False)["tasks"] == 1
    assert trainer.student.model.training is was_training
    trainer.close()


def test_no_public_operation_can_observe_evaluations_temporary_eval_mode(
    tmp_path,
    monkeypatch,
) -> None:
    from miniverl.errors import LifecycleError
    from miniverl.trainer import OPDTrainer

    trainer = OPDTrainer.from_config(_config(tmp_path), run_id="private-eval-mode")
    trainer.student.set_train(True)
    entered = threading.Event()
    release = threading.Event()
    real_rollout = trainer.runner.rollout

    def blocked_rollout(*args, **kwargs):  # type: ignore[no-untyped-def]
        entered.set()
        assert release.wait(timeout=10)
        return real_rollout(*args, **kwargs)

    monkeypatch.setattr(trainer.runner, "rollout", blocked_rollout)
    thread, _results, failures = _thread_call(lambda: trainer.evaluate(write=False))
    assert entered.wait(timeout=10)
    assert trainer.student.model.training is False

    try:
        with pytest.raises(LifecycleError, match="cannot save_checkpoint"):
            trainer.save_checkpoint()
    finally:
        release.set()
        thread.join(timeout=30)

    assert not thread.is_alive()
    assert failures == []
    assert trainer.student.model.training is True
    trainer.close()


def test_train_is_one_shot_and_rejection_changes_no_artifact(tmp_path) -> None:
    from miniverl.errors import LifecycleError
    from miniverl.trainer import OPDTrainer, TrainerState

    trainer = OPDTrainer.from_config(_config(tmp_path), run_id="one-shot")
    trainer.train()
    assert trainer.state is TrainerState.COMPLETED
    before = _artifact_bytes(trainer.paths.root)

    with pytest.raises(LifecycleError, match="completed"):
        trainer.train()

    assert _artifact_bytes(trainer.paths.root) == before
    trainer.close()
    assert trainer.state is TrainerState.CLOSED


def test_two_threads_cannot_both_enter_train(tmp_path, monkeypatch) -> None:
    from miniverl.errors import LifecycleError
    from miniverl.trainer import OPDTrainer, TrainerState

    trainer = OPDTrainer.from_config(_config(tmp_path), run_id="thread-one-shot")
    entered = threading.Event()
    release = threading.Event()
    real_train_impl = trainer._train_impl

    def blocked_train_impl():  # type: ignore[no-untyped-def]
        entered.set()
        assert release.wait(timeout=10)
        return real_train_impl()

    monkeypatch.setattr(trainer, "_train_impl", blocked_train_impl)
    failures: list[BaseException] = []

    def run_training() -> None:
        try:
            trainer.train()
        except BaseException as exc:  # pragma: no cover - diagnostic capture
            failures.append(exc)

    thread = threading.Thread(target=run_training)
    thread.start()
    assert entered.wait(timeout=10)
    assert trainer.state is TrainerState.RUNNING

    with pytest.raises(LifecycleError, match="running"):
        trainer.train()

    release.set()
    thread.join(timeout=30)
    assert not thread.is_alive()
    assert failures == []
    assert trainer.state is TrainerState.COMPLETED
    trainer.close()


@pytest.mark.parametrize(
    ("exception", "expected_state"),
    [
        (RuntimeError("boom"), "failed"),
        (KeyboardInterrupt(), "interrupted"),
    ],
)
def test_train_failure_sets_an_explicit_terminal_state(
    tmp_path,
    monkeypatch,
    exception: BaseException,
    expected_state: str,
) -> None:
    from miniverl.trainer import OPDTrainer

    trainer = OPDTrainer.from_config(_config(tmp_path), run_id=f"terminal-{expected_state}")
    monkeypatch.setattr(trainer, "_train_impl", Mock(side_effect=exception))
    with pytest.raises(
        type(exception),
        match="boom" if expected_state == "failed" else None,
    ):
        trainer.train()
    assert trainer.state.value == expected_state
    trainer.close()
    assert trainer.state.value == "closed"


def test_close_refuses_to_tear_down_a_running_trainer(tmp_path, monkeypatch) -> None:
    from miniverl.errors import LifecycleError
    from miniverl.trainer import OPDTrainer

    trainer = OPDTrainer.from_config(_config(tmp_path), run_id="close-running")
    entered = threading.Event()
    release = threading.Event()
    real_train_impl = trainer._train_impl

    def blocked_train_impl():  # type: ignore[no-untyped-def]
        entered.set()
        assert release.wait(timeout=10)
        return real_train_impl()

    monkeypatch.setattr(trainer, "_train_impl", blocked_train_impl)
    thread = threading.Thread(target=trainer.train)
    thread.start()
    assert entered.wait(timeout=10)
    with pytest.raises(LifecycleError, match="cannot close while another trainer operation"):
        trainer.close()
    release.set()
    thread.join(timeout=30)
    assert not thread.is_alive()
    trainer.close()


@pytest.mark.parametrize("operation", ["evaluate", "save_checkpoint"])
def test_close_cannot_race_with_an_active_public_operation(
    tmp_path,
    monkeypatch,
    operation: str,
) -> None:
    from miniverl.errors import LifecycleError
    from miniverl.trainer import OPDTrainer, TrainerState

    trainer = OPDTrainer.from_config(_config(tmp_path), run_id=f"close-vs-{operation}")
    entered = threading.Event()
    release = threading.Event()
    impl_name = "_evaluate_impl" if operation == "evaluate" else "_save_checkpoint_impl"
    real_impl = getattr(trainer, impl_name)

    def blocked(*args, **kwargs):  # type: ignore[no-untyped-def]
        entered.set()
        assert release.wait(timeout=10)
        return real_impl(*args, **kwargs)

    monkeypatch.setattr(trainer, impl_name, blocked)
    call = trainer.evaluate if operation == "evaluate" else trainer.save_checkpoint
    thread, _results, failures = _thread_call(call)
    assert entered.wait(timeout=10)
    artifacts = _artifact_bytes(trainer.paths.root)
    resources = _resource_snapshot(trainer)
    state = trainer.state

    try:
        with pytest.raises(LifecycleError, match="cannot close while another trainer operation"):
            trainer.close()
        assert trainer.state is state is TrainerState.READY
        assert _resource_snapshot(trainer) == resources
        assert _artifact_bytes(trainer.paths.root) == artifacts
    finally:
        release.set()
        thread.join(timeout=30)

    assert not thread.is_alive()
    assert failures == []
    trainer.close()
    trainer.close()
    assert trainer.state is TrainerState.CLOSED


def test_close_cannot_race_with_checkpoint_load(tmp_path, monkeypatch) -> None:
    import miniverl.training.trainer as trainer_module
    from miniverl.errors import LifecycleError
    from miniverl.trainer import OPDTrainer, TrainerState

    config, checkpoint = _trained_checkpoint(tmp_path, "close-vs-load")
    trainer = OPDTrainer.from_config(config, run_id="close-vs-load-target")
    entered = threading.Event()
    release = threading.Event()
    real_load = trainer_module.load_checkpoint

    def blocked(*args, **kwargs):  # type: ignore[no-untyped-def]
        entered.set()
        assert release.wait(timeout=10)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(trainer_module, "load_checkpoint", blocked)
    thread, _results, failures = _thread_call(lambda: trainer.load_from_checkpoint(checkpoint))
    assert entered.wait(timeout=10)
    artifacts = _artifact_bytes(trainer.paths.root)
    resources = _resource_snapshot(trainer)

    try:
        with pytest.raises(LifecycleError, match="cannot close while another trainer operation"):
            trainer.close()
        assert trainer.state is TrainerState.READY
        assert _resource_snapshot(trainer) == resources
        assert _artifact_bytes(trainer.paths.root) == artifacts
    finally:
        release.set()
        thread.join(timeout=30)

    assert not thread.is_alive()
    assert failures == []
    trainer.close()
    trainer.close()


def test_two_simultaneous_close_calls_have_one_owner(tmp_path, monkeypatch) -> None:
    from miniverl.errors import LifecycleError
    from miniverl.trainer import OPDTrainer, TrainerState

    trainer = OPDTrainer.from_config(_config(tmp_path), run_id="two-close-owners")
    entered = threading.Event()
    release = threading.Event()
    real_release = trainer.student.release

    def blocked_release() -> None:
        entered.set()
        assert release.wait(timeout=10)
        real_release()

    monkeypatch.setattr(trainer.student, "release", blocked_release)
    thread, _results, failures = _thread_call(trainer.close)
    assert entered.wait(timeout=10)
    try:
        with pytest.raises(LifecycleError, match="cannot close while another trainer operation"):
            trainer.close()
    finally:
        release.set()
        thread.join(timeout=30)

    assert not thread.is_alive()
    assert failures == []
    assert trainer.state is TrainerState.CLOSED
    trainer.close()


def test_cleanup_failure_releases_operation_ownership_and_remains_idempotent(
    tmp_path,
    monkeypatch,
) -> None:
    from miniverl.errors import LifecycleError
    from miniverl.trainer import OPDTrainer, TrainerState

    trainer = OPDTrainer.from_config(_config(tmp_path), run_id="failed-cleanup-ownership")
    monkeypatch.setattr(
        trainer.student,
        "release",
        Mock(side_effect=RuntimeError("injected release failure")),
    )

    with pytest.raises(LifecycleError, match="injected release failure"):
        trainer.close()

    assert trainer.state is TrainerState.CLOSED
    assert trainer._operation_guard.acquire(blocking=False)
    trainer._operation_guard.release()
    trainer.close()


@pytest.mark.parametrize(
    ("active", "loser"),
    [
        ("evaluate", "evaluate"),
        ("evaluate", "save_checkpoint"),
        ("save_checkpoint", "evaluate"),
        ("save_checkpoint", "save_checkpoint"),
    ],
)
def test_evaluate_and_save_checkpoint_share_one_operation_owner(
    tmp_path,
    monkeypatch,
    active: str,
    loser: str,
) -> None:
    from miniverl.errors import LifecycleError
    from miniverl.trainer import OPDTrainer, TrainerState

    trainer = OPDTrainer.from_config(_config(tmp_path), run_id=f"{active}-vs-{loser}")
    entered = threading.Event()
    release = threading.Event()
    impl_name = "_evaluate_impl" if active == "evaluate" else "_save_checkpoint_impl"
    real_impl = getattr(trainer, impl_name)

    def blocked(*args, **kwargs):  # type: ignore[no-untyped-def]
        entered.set()
        assert release.wait(timeout=10)
        return real_impl(*args, **kwargs)

    monkeypatch.setattr(trainer, impl_name, blocked)
    active_call = trainer.evaluate if active == "evaluate" else trainer.save_checkpoint
    losing_call = trainer.evaluate if loser == "evaluate" else trainer.save_checkpoint
    thread, _results, failures = _thread_call(active_call)
    assert entered.wait(timeout=10)
    artifacts = _artifact_bytes(trainer.paths.root)
    resources = _resource_snapshot(trainer)

    try:
        with pytest.raises(LifecycleError, match=rf"cannot {loser}"):
            losing_call()
        assert trainer.state is TrainerState.READY
        assert _resource_snapshot(trainer) == resources
        assert _artifact_bytes(trainer.paths.root) == artifacts
    finally:
        release.set()
        thread.join(timeout=30)

    assert not thread.is_alive()
    assert failures == []
    trainer.close()


@pytest.mark.parametrize("active", ["evaluate", "save_checkpoint"])
def test_checkpoint_load_loses_to_an_active_evaluate_or_save_without_mutation(
    tmp_path,
    monkeypatch,
    active: str,
) -> None:
    from miniverl.errors import LifecycleError
    from miniverl.trainer import OPDTrainer, TrainerState

    config, checkpoint = _trained_checkpoint(tmp_path, f"{active}-vs-load")
    trainer = OPDTrainer.from_config(config, run_id=f"{active}-vs-load-target")
    entered = threading.Event()
    release = threading.Event()
    impl_name = "_evaluate_impl" if active == "evaluate" else "_save_checkpoint_impl"
    real_impl = getattr(trainer, impl_name)

    def blocked(*args, **kwargs):  # type: ignore[no-untyped-def]
        entered.set()
        assert release.wait(timeout=10)
        return real_impl(*args, **kwargs)

    monkeypatch.setattr(trainer, impl_name, blocked)
    call = trainer.evaluate if active == "evaluate" else trainer.save_checkpoint
    thread, _results, failures = _thread_call(call)
    assert entered.wait(timeout=10)
    artifacts = _artifact_bytes(trainer.paths.root)
    parameters = _parameter_snapshot(trainer)
    progress = _progress_snapshot(trainer)

    try:
        with pytest.raises(LifecycleError, match="cannot load_from_checkpoint"):
            trainer.load_from_checkpoint(checkpoint)
        assert trainer.state is TrainerState.READY
        _assert_parameters_equal(parameters, trainer)
        assert _progress_snapshot(trainer) == progress
        assert _artifact_bytes(trainer.paths.root) == artifacts
    finally:
        release.set()
        thread.join(timeout=30)

    assert not thread.is_alive()
    assert failures == []
    trainer.close()


@pytest.mark.parametrize("loser", ["evaluate", "save_checkpoint", "load_from_checkpoint", "close"])
def test_active_checkpoint_load_excludes_every_other_public_resource_operation(
    tmp_path,
    monkeypatch,
    loser: str,
) -> None:
    import miniverl.training.trainer as trainer_module
    from miniverl.errors import LifecycleError
    from miniverl.trainer import OPDTrainer, TrainerState

    config, checkpoint = _trained_checkpoint(tmp_path, f"load-vs-{loser}")
    trainer = OPDTrainer.from_config(config, run_id=f"load-vs-{loser}-target")
    entered = threading.Event()
    release = threading.Event()
    real_load = trainer_module.load_checkpoint

    def blocked(*args, **kwargs):  # type: ignore[no-untyped-def]
        entered.set()
        assert release.wait(timeout=10)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(trainer_module, "load_checkpoint", blocked)
    thread, _results, failures = _thread_call(lambda: trainer.load_from_checkpoint(checkpoint))
    assert entered.wait(timeout=10)
    artifacts = _artifact_bytes(trainer.paths.root)
    parameters = _parameter_snapshot(trainer)
    progress = _progress_snapshot(trainer)
    resources = _resource_snapshot(trainer)
    args = (checkpoint,) if loser == "load_from_checkpoint" else ()

    try:
        with pytest.raises(
            LifecycleError, match=rf"cannot {loser if loser != 'close' else 'close'}"
        ):
            getattr(trainer, loser)(*args)
        assert trainer.state is TrainerState.READY
        _assert_parameters_equal(parameters, trainer)
        assert _progress_snapshot(trainer) == progress
        assert _resource_snapshot(trainer) == resources
        assert _artifact_bytes(trainer.paths.root) == artifacts
    finally:
        release.set()
        thread.join(timeout=30)

    assert not thread.is_alive()
    assert failures == []
    trainer.close()


def test_checkpoint_load_owns_operation_before_train_can_transition(tmp_path, monkeypatch) -> None:
    import miniverl.training.checkpoint as checkpoint_module
    from miniverl.errors import LifecycleError
    from miniverl.trainer import OPDTrainer, TrainerState

    config, checkpoint = _trained_checkpoint(tmp_path, "load-vs-train-transition")
    trainer = OPDTrainer.from_config(config, run_id="load-vs-train-transition-target")
    entered = threading.Event()
    release = threading.Event()
    real_validate = checkpoint_module.validate_checkpoint

    def blocked(*args, **kwargs):  # type: ignore[no-untyped-def]
        result = real_validate(*args, **kwargs)
        entered.set()
        assert release.wait(timeout=10)
        return result

    monkeypatch.setattr(checkpoint_module, "validate_checkpoint", blocked)
    thread, _results, failures = _thread_call(lambda: trainer.load_from_checkpoint(checkpoint))
    assert entered.wait(timeout=10)
    artifacts = _artifact_bytes(trainer.paths.root)
    parameters = _parameter_snapshot(trainer)
    progress = _progress_snapshot(trainer)

    try:
        with pytest.raises(LifecycleError, match="cannot train"):
            trainer.train()
        assert trainer.state is TrainerState.READY
        _assert_parameters_equal(parameters, trainer)
        assert _progress_snapshot(trainer) == progress
        assert _artifact_bytes(trainer.paths.root) == artifacts
    finally:
        release.set()
        thread.join(timeout=30)

    assert not thread.is_alive()
    assert failures == []
    trainer.close()


@pytest.mark.parametrize(
    "failure_stage",
    ["first_rollout", "later_rollout", "trajectory_append", "diagnostics", "metrics", "event"],
)
@pytest.mark.parametrize("was_training", [False, True])
def test_evaluation_restores_exact_previous_model_mode_after_every_failure(
    tmp_path,
    monkeypatch,
    failure_stage: str,
    was_training: bool,
) -> None:
    import miniverl.evaluation.diagnostics as diagnostics_module
    import miniverl.training.trainer as trainer_module
    from miniverl.trainer import OPDTrainer, TrainerState

    trainer = OPDTrainer.from_config(
        _config(tmp_path, eval_tasks=2),
        run_id=f"eval-mode-{failure_stage}-{was_training}",
    )
    trainer.student.set_train(was_training)
    injected = RuntimeError(f"injected {failure_stage} failure")

    if failure_stage in {"first_rollout", "later_rollout"}:
        real_rollout = trainer.runner.rollout
        calls = 0

        def rollout(*args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal calls
            calls += 1
            if failure_stage == "first_rollout" or calls == 2:
                raise injected
            return real_rollout(*args, **kwargs)

        monkeypatch.setattr(trainer.runner, "rollout", rollout)
    elif failure_stage == "trajectory_append":
        monkeypatch.setattr(trainer_module, "append_trajectories", Mock(side_effect=injected))
    elif failure_stage == "diagnostics":
        monkeypatch.setattr(
            diagnostics_module,
            "lenient_diagnostic_success_rate",
            Mock(side_effect=injected),
        )
    elif failure_stage == "metrics":
        monkeypatch.setattr(trainer.metrics_log, "write", Mock(side_effect=injected))
    else:
        monkeypatch.setattr(trainer.events, "emit", Mock(side_effect=injected))

    with pytest.raises(RuntimeError, match=rf"injected {failure_stage} failure") as caught:
        trainer.evaluate()

    assert caught.value is injected
    assert trainer.student.model.training is was_training
    assert trainer.state is TrainerState.READY
    assert trainer._operation_guard.acquire(blocking=False)
    trainer._operation_guard.release()
    trainer.close()


def test_second_resume_loses_the_run_lock_before_model_loading(
    tmp_path,
    monkeypatch,
) -> None:
    import miniverl.training.trainer as trainer_module
    from miniverl.errors import RunLockedError
    from miniverl.trainer import OPDTrainer

    config = _config(tmp_path)
    owner = OPDTrainer.from_config(config, run_id="locked-resume")
    before = _artifact_bytes(owner.paths.root)
    tokenizer_probe = Mock(side_effect=AssertionError("model loading must not start"))
    monkeypatch.setattr(trainer_module, "build_tokenizer", tokenizer_probe)

    with pytest.raises(RunLockedError, match="locked-resume"):
        OPDTrainer.from_config(config, resume=owner.paths.root)

    tokenizer_probe.assert_not_called()
    assert _artifact_bytes(owner.paths.root) == before
    owner.close()


def test_overwrite_cannot_replace_a_run_while_its_owner_holds_the_lock(
    tmp_path,
    monkeypatch,
) -> None:
    import miniverl.training.trainer as trainer_module
    from miniverl.errors import RunLockedError
    from miniverl.trainer import OPDTrainer

    config = _config(tmp_path)
    owner = OPDTrainer.from_config(config, run_id="locked-overwrite")
    before = _artifact_bytes(owner.paths.root)
    tokenizer_probe = Mock(side_effect=AssertionError("model loading must not start"))
    monkeypatch.setattr(trainer_module, "build_tokenizer", tokenizer_probe)

    with pytest.raises(RunLockedError, match="locked-overwrite"):
        OPDTrainer.from_config(config, run_id="locked-overwrite", overwrite=True)

    tokenizer_probe.assert_not_called()
    assert _artifact_bytes(owner.paths.root) == before
    owner.close()


def test_file_backed_recipe_writes_submitted_validated_legacy_and_resolved_layers(
    tmp_path,
) -> None:
    from miniverl.config import RunConfig
    from miniverl.trainer import OPDTrainer

    programmatic = _config(tmp_path)
    submitted = ("# submitted bytes remain exact\n" + programmatic.to_yaml()).encode()
    recipe = tmp_path / "submitted.yaml"
    recipe.write_bytes(submitted)
    config = RunConfig.from_yaml(recipe)

    trainer = OPDTrainer.from_config(config, run_id="config-layers")
    paths = trainer.paths
    assert paths.config_submitted.read_bytes() == submitted
    assert paths.config_validated.read_text(encoding="utf-8") == config.to_yaml()
    assert paths.config_original.read_bytes() == paths.config_validated.read_bytes()
    assert paths.config_resolved.read_bytes() != paths.config_validated.read_bytes()
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    assert manifest["manifest_schema_version"] == 3
    assert manifest["config_provenance"]["submitted"] == "verbatim_file_bytes"
    assert set(manifest["config_digests"]) == {
        "submitted",
        "validated",
        "legacy_original",
        "resolved",
    }
    trainer.close()


def test_programmatic_recipe_marks_submitted_layer_as_generated(tmp_path) -> None:
    from miniverl.trainer import OPDTrainer

    trainer = OPDTrainer.from_config(_config(tmp_path), run_id="generated-config")
    assert not trainer.paths.config_submitted.exists()
    assert trainer.paths.config_validated.is_file()
    manifest = json.loads(trainer.paths.manifest.read_text(encoding="utf-8"))
    assert manifest["config_provenance"]["submitted"] == "generated_no_source_bytes"
    trainer.close()
