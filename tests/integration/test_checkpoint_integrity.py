"""Checkpoint selection, integrity, atomicity and legacy compatibility."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from miniverl.errors import CheckpointError
from tests.conftest import requires_torch

pytestmark = [requires_torch, pytest.mark.torch]

torch = pytest.importorskip("torch")


class _Backend:
    def __init__(self) -> None:
        self.loaded: dict[str, Any] | None = None

    def load_trainable_state_dict(self, state: dict[str, Any]) -> None:
        if set(state) != {"weight"}:
            raise CheckpointError(f"unexpected trainable keys: {sorted(state)}")
        self.loaded = state


def _write_checkpoint(
    path: Path,
    *,
    step: int,
    weight: float = 1.0,
) -> Path:
    from miniverl.training.checkpoint import CheckpointState, save_checkpoint
    from miniverl.utils.seeding import capture_rng

    parameter = torch.nn.Parameter(torch.tensor([weight]))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    return save_checkpoint(
        path,
        trainable_state={"weight": parameter.detach().clone()},
        optimizer=optimizer,
        state=CheckpointState(
            miniverl_version="0.2.1.dev0",
            global_step=step,
            policy_version=step,
            cycle=max(step - 1, 0),
            config_digest="config-a",
        ),
        rng=capture_rng(),
        identity={"student_model_id": "toy-student", "tokenizer_identity": "toy-v2"},
    )


def test_latest_checkpoint_prefers_final_when_it_is_the_highest_step(tmp_path: Path) -> None:
    from miniverl.training.checkpoint import latest_checkpoint

    root = tmp_path / "checkpoints"
    _write_checkpoint(root / "step-000008", step=8)
    _write_checkpoint(root / "step-000016", step=16)
    _write_checkpoint(root / "final", step=20)

    assert latest_checkpoint(root) == root / "final"


def test_empty_execution_plan_identity_preserves_legacy_checkpoint_shape(tmp_path: Path) -> None:
    from miniverl.training.checkpoint import validate_checkpoint

    checkpoint = _write_checkpoint(tmp_path / "legacy-compatible", step=2)
    state = (checkpoint / "state.json").read_text(encoding="utf-8")
    manifest = (checkpoint / "checkpoint.json").read_text(encoding="utf-8")
    assert "execution_plan_digest" not in state
    assert "execution_plan_digest" not in manifest
    assert validate_checkpoint(checkpoint).state.execution_plan_digest == ""


def test_latest_checkpoint_uses_state_step_instead_of_lexicographic_name(
    tmp_path: Path,
) -> None:
    from miniverl.training.checkpoint import latest_checkpoint

    root = tmp_path / "checkpoints"
    _write_checkpoint(root / "final", step=10)
    _write_checkpoint(root / "step-000016", step=16)

    assert latest_checkpoint(root) == root / "step-000016"


def test_latest_checkpoint_rejects_ambiguous_same_step_content(tmp_path: Path) -> None:
    from miniverl.training.checkpoint import latest_checkpoint

    root = tmp_path / "checkpoints"
    _write_checkpoint(root / "candidate-a", step=12, weight=1.0)
    _write_checkpoint(root / "candidate-b", step=12, weight=2.0)

    with pytest.raises(CheckpointError, match="ambiguous"):
        latest_checkpoint(root)


def test_incomplete_temporary_checkpoint_is_ignored(tmp_path: Path) -> None:
    from miniverl.training.checkpoint import latest_checkpoint

    root = tmp_path / "checkpoints"
    expected = _write_checkpoint(root / "step-000004", step=4)
    temporary = root / ".step-000999.tmp-deadbeef"
    temporary.mkdir()
    (temporary / "state.json").write_text('{"global_step": 999}', encoding="utf-8")

    assert latest_checkpoint(root) == expected


def test_checkpoint_checksum_corruption_is_detected_before_loading(tmp_path: Path) -> None:
    from miniverl.training.checkpoint import load_checkpoint

    checkpoint = _write_checkpoint(tmp_path / "checkpoint", step=3)
    with (checkpoint / "adapter.safetensors").open("ab") as handle:
        handle.write(b"corruption")

    with pytest.raises(CheckpointError, match="checksum"):
        load_checkpoint(
            checkpoint,
            backend=_Backend(),
            optimizer=None,
            include_optimizer=False,
            include_rng=False,
        )


def test_missing_trainable_weights_are_rejected(tmp_path: Path) -> None:
    from miniverl.training.checkpoint import load_checkpoint

    checkpoint = _write_checkpoint(tmp_path / "checkpoint", step=3)
    (checkpoint / "adapter.safetensors").unlink()

    with pytest.raises(CheckpointError, match=r"adapter\.safetensors"):
        load_checkpoint(
            checkpoint,
            backend=_Backend(),
            optimizer=None,
            include_optimizer=False,
            include_rng=False,
        )


def test_failed_atomic_save_preserves_the_previous_complete_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import miniverl.training.checkpoint as checkpoint_module

    target = _write_checkpoint(tmp_path / "checkpoint", step=2, weight=1.0)
    before = {path.name: path.read_bytes() for path in target.iterdir() if path.is_file()}

    def fail_write(*_args: Any, **_kwargs: Any) -> Path:
        raise OSError("injected state write failure")

    monkeypatch.setattr(checkpoint_module, "write_text", fail_write)
    with pytest.raises(OSError, match="injected"):
        _write_checkpoint(target, step=3, weight=2.0)

    after = {path.name: path.read_bytes() for path in target.iterdir() if path.is_file()}
    assert after == before
    assert not list(tmp_path.glob(".checkpoint.tmp-*"))


def test_manifestless_v02_checkpoint_is_labeled_legacy_unchecksummed(
    tmp_path: Path,
) -> None:
    from miniverl.training.checkpoint import validate_checkpoint

    checkpoint = _write_checkpoint(tmp_path / "checkpoint", step=2)
    (checkpoint / "checkpoint.json").unlink()

    validated = validate_checkpoint(checkpoint)

    assert validated.integrity == "legacy_unchecksummed"
    assert validated.state.global_step == 2
