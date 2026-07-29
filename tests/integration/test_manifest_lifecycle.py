"""Run manifests distinguish immutable startup provenance from final state."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conftest import requires_torch
from tests.integration.test_resume_and_swap import _config

pytestmark = [requires_torch, pytest.mark.torch]

pytest.importorskip("torch")


def test_successful_training_finalizes_manifest_without_changing_startup(
    tmp_path: Path,
) -> None:
    from miniverl.trainer import OPDTrainer

    trainer = OPDTrainer.from_config(
        _config(tmp_path, train={"cycles": 1}),
        run_id="manifest-complete",
    )
    startup_bytes = trainer.paths.manifest_start.read_bytes()
    result = trainer.train()
    manifest = json.loads(trainer.paths.manifest.read_text(encoding="utf-8"))

    assert trainer.paths.manifest_start.read_bytes() == startup_bytes
    assert manifest["status"] == "completed"
    assert manifest["started_at"]
    assert manifest["completed_at"]
    assert manifest["global_step"] == result.global_step
    assert manifest["actual_optimizer_updates"] == result.global_step
    assert manifest["final_memory"]["projection_chunk_size"] == trainer.plan.chunk_size
    assert manifest["final_checkpoint"]["integrity"] == "checksummed_v1"
    assert len(manifest["final_checkpoint"]["digest"]) == 64
    assert manifest["all_expected_artifacts_complete"] is True
    trainer.close()


def test_training_failure_is_recorded_without_hiding_the_original_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from miniverl.trainer import OPDTrainer

    trainer = OPDTrainer.from_config(
        _config(tmp_path, train={"cycles": 1}),
        run_id="manifest-failed",
    )
    startup_bytes = trainer.paths.manifest_start.read_bytes()

    def fail_cycle():
        raise RuntimeError("injected training failure")

    monkeypatch.setattr(trainer, "_run_cycle", fail_cycle)
    with pytest.raises(RuntimeError, match="injected training failure"):
        trainer.train()

    manifest = json.loads(trainer.paths.manifest.read_text(encoding="utf-8"))
    assert trainer.paths.manifest_start.read_bytes() == startup_bytes
    assert manifest["status"] == "failed"
    assert manifest["failure"]["type"] == "RuntimeError"
    assert manifest["failure"]["message"] == "injected training failure"
    assert manifest["failed_at"]
    trainer.close()


def test_keyboard_interrupt_is_recorded_as_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from miniverl.trainer import OPDTrainer

    trainer = OPDTrainer.from_config(
        _config(tmp_path, train={"cycles": 1}),
        run_id="manifest-interrupted",
    )

    def interrupt():
        raise KeyboardInterrupt

    monkeypatch.setattr(trainer, "_run_cycle", interrupt)
    with pytest.raises(KeyboardInterrupt):
        trainer.train()

    manifest = json.loads(trainer.paths.manifest.read_text(encoding="utf-8"))
    assert manifest["status"] == "interrupted"
    assert manifest["failure"]["type"] == "KeyboardInterrupt"
    trainer.close()
