"""Standalone evaluation must restore an exact checkpoint without training state."""

from __future__ import annotations

from pathlib import Path

import pytest

from miniverl.errors import CheckpointError
from tests.conftest import requires_torch
from tests.integration.test_resume_and_swap import _config

pytestmark = [requires_torch, pytest.mark.torch]

pytest.importorskip("torch")


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
