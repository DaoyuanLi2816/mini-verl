"""Trainer ownership and teardown are explicit, destructive, and idempotent."""

from __future__ import annotations

import gc
import json
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
