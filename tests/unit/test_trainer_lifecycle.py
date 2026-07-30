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


def _config(tmp_path):  # type: ignore[no-untyped-def]
    from tests.integration.test_toy_pipeline import _config as toy_config

    return toy_config(
        tmp_path,
        train={"cycles": 1, "rollouts_per_cycle": 1, "gradient_accumulation_steps": 1},
        eval={"enabled": False, "tasks": 1},
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
    with pytest.raises(LifecycleError, match="running"):
        trainer.close()
    release.set()
    thread.join(timeout=30)
    assert not thread.is_alive()
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
