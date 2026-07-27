"""Checkpoint/resume equivalence and the resident-vs-swap contract.

Two claims are checked here, both of which are easy to get subtly wrong:

1. **Resume is exact.** Training A->C in one process must produce the same
   parameters as training A->B, stopping, restoring, and training B->C. If the
   optimizer moments, the LR schedule position, the task-sampler cursor or any
   RNG stream were not restored, the two runs diverge and this fails.

2. **``swap`` changes only *where* tensors live.** Moving the student to host
   memory, scoring with the teacher, and bringing the student back must produce
   the same update as keeping both resident. If ``swap`` silently dropped the
   optimizer state or re-scored under a different policy version, the two runs
   would differ.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.conftest import requires_torch

pytestmark = [requires_torch, pytest.mark.torch]

torch = pytest.importorskip("torch")


def _config(tmp_path: Path, **overrides: Any):
    from miniverl.config import RunConfig

    payload: dict[str, Any] = {
        "schema_version": 1,
        "run": {
            "name": "resume-test",
            "mode": "opd",
            "seed": 11,
            "output_dir": str(tmp_path),
            "deterministic": True,
        },
        "models": {
            "backend": "toy",
            "device": "cpu",
            "student": {
                "model_id": "toy-student",
                "lora": {"enabled": False},
                "toy": {
                    "hidden_size": 32,
                    "num_layers": 2,
                    "num_heads": 4,
                    "intermediate_size": 64,
                    "max_position_embeddings": 512,
                },
            },
            "teacher": {
                "model_id": "toy-teacher",
                "toy_pretrain_steps": 5,
                "toy": {
                    "hidden_size": 32,
                    "num_layers": 2,
                    "num_heads": 4,
                    "intermediate_size": 64,
                    "max_position_embeddings": 512,
                },
            },
        },
        "environment": {
            "name": "calculator",
            "difficulty": "easy",
            "params": {"prompt_style": "compact"},
            "train_tasks": 12,
            "eval_tasks": 2,
            "test_tasks": 2,
            "split_seed": 5,
        },
        "rollout": {"max_turns": 2, "max_new_tokens_per_turn": 10, "max_total_tokens": 400},
        "selection": {"selector": "all_model_tokens"},
        "loss": {
            "mode": "bucketed_topk_tail",
            "divergence": "reverse_kl",
            "top_k": 8,
            "chunk_size": 32,
        },
        "train": {
            "cycles": 4,
            "rollouts_per_cycle": 2,
            "gradient_accumulation_steps": 2,
            "learning_rate": 0.005,
            "sft_warmup_cycles": 0,
            "lr_schedule": "cosine",
        },
        "memory": {"strategy": "resident"},
        "cache": {"entries_per_shard": 2, "keep_cycles": 10},
        "eval": {"enabled": False},
        "report": {"enabled": False},
    }
    for section, values in overrides.items():
        _deep_update(payload.setdefault(section, {}), values)
    return RunConfig.model_validate(payload)


def _deep_update(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _flat_state(backend: Any) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in backend.trainable_state_dict().items()}


def _assert_states_equal(
    a: dict[str, torch.Tensor], b: dict[str, torch.Tensor], *, atol: float
) -> None:
    assert set(a) == set(b)
    worst = 0.0
    for key in sorted(a):
        delta = float((a[key] - b[key]).abs().max())
        worst = max(worst, delta)
    assert worst <= atol, f"largest parameter difference {worst:.3e} exceeds {atol:.1e}"


# ------------------------------------------------------------------ resume


def test_uninterrupted_and_resumed_training_agree_exactly(tmp_path: Path):
    from miniverl.trainer import OPDTrainer
    from miniverl.utils.seeding import seed_everything

    # Reference: four cycles in one process.
    seed_everything(11, deterministic=True)
    reference = OPDTrainer.from_config(_config(tmp_path), run_id="reference")
    reference.train()
    reference_state = _flat_state(reference.student)
    reference_step = reference.global_step
    reference_version = reference.policy_version
    reference_cursor = reference.task_cursor
    reference.close()

    # Interrupted: two cycles, checkpoint, then a fresh trainer resumes.
    seed_everything(11, deterministic=True)
    # Simulate an interruption after cycle 1 by running the full config but
    # stopping early: two cycles, then checkpoint.
    first = OPDTrainer.from_config(_config(tmp_path), run_id="first-half")
    first._prepare_toy_teacher()
    for cycle in range(2):
        first.cycle = cycle
        first._run_cycle()
    checkpoint = first.save_checkpoint(name="interrupt")
    first.close()

    # Resume with the FULL schedule: the config digest is checked, and the LR
    # schedule's total_steps must match the original budget or the remaining
    # steps would use a different learning rate.
    seed_everything(11, deterministic=True)
    second = OPDTrainer.from_config(_config(tmp_path), run_id="second-half")
    state = second.load_from_checkpoint(checkpoint)
    assert state.global_step == 2
    assert state.policy_version == 2
    # train() must continue at cycle 2 rather than replaying cycles 0 and 1.
    second.train()
    resumed_state = _flat_state(second.student)
    resumed_step = second.global_step
    resumed_version = second.policy_version
    resumed_cursor = second.task_cursor
    second.close()

    assert resumed_step == reference_step
    assert resumed_version == reference_version
    assert resumed_cursor == reference_cursor
    _assert_states_equal(reference_state, resumed_state, atol=1e-6)


def test_checkpoint_round_trip_restores_optimizer_and_schedule(tmp_path: Path):
    from miniverl.trainer import OPDTrainer
    from miniverl.training.checkpoint import load_checkpoint

    trainer = OPDTrainer.from_config(_config(tmp_path, train={"cycles": 2}), run_id="ckpt")
    trainer.train()
    before = {
        "step": trainer.global_step,
        "version": trainer.policy_version,
        "cursor": trainer.task_cursor,
        "lr": trainer.schedule.lr_at(trainer.global_step),
    }
    moments_before = {
        id(p): {k: v.clone() for k, v in s.items() if isinstance(v, torch.Tensor)}
        for p, s in trainer.optimizer.state.items()
    }
    assert moments_before, "the optimizer must have accumulated state to make this meaningful"
    path = trainer.save_checkpoint(name="rt")
    trainer.close()

    fresh = OPDTrainer.from_config(_config(tmp_path, train={"cycles": 2}), run_id="ckpt-restore")
    state = load_checkpoint(path, backend=fresh.student, optimizer=fresh.optimizer, device="cpu")
    assert state.global_step == before["step"]
    assert state.policy_version == before["version"]
    assert state.task_cursor == before["cursor"]
    from miniverl.training.optim import LearningRateSchedule

    restored_schedule = LearningRateSchedule.from_state_dict(state.scheduler)
    assert restored_schedule.lr_at(before["step"]) == pytest.approx(before["lr"])
    restored_moments = [
        v for s in fresh.optimizer.state.values() for v in s.values() if isinstance(v, torch.Tensor)
    ]
    assert restored_moments, "optimizer moments must be restored, not silently dropped"
    fresh.close()


def test_checkpoint_files_are_pickle_free(tmp_path: Path):
    from miniverl.trainer import OPDTrainer

    trainer = OPDTrainer.from_config(_config(tmp_path, train={"cycles": 1}), run_id="pickle-free")
    trainer.train()
    path = trainer.save_checkpoint(name="inspect")
    trainer.close()
    names = sorted(p.name for p in path.iterdir())
    assert names == ["adapter.safetensors", "optimizer.safetensors", "state.json"]
    # A pickle stream starts with a protocol opcode; safetensors starts with a
    # little-endian header length. Assert the files are not pickles.
    for name in ("adapter.safetensors", "optimizer.safetensors"):
        head = (path / name).read_bytes()[:2]
        assert head[:1] != b"\x80", f"{name} looks like a pickle stream"
    import json

    json.loads((path / "state.json").read_text(encoding="utf-8"))


def test_resuming_a_checkpoint_from_a_different_config_is_refused(tmp_path: Path):
    from miniverl.errors import ConfigError
    from miniverl.trainer import OPDTrainer

    trainer = OPDTrainer.from_config(_config(tmp_path, train={"cycles": 1}), run_id="digest-a")
    trainer.train()
    path = trainer.save_checkpoint(name="c")
    trainer.close()

    other = OPDTrainer.from_config(
        _config(tmp_path, train={"cycles": 1, "learning_rate": 0.001}), run_id="digest-b"
    )
    with pytest.raises(ConfigError, match="different configuration"):
        other.load_from_checkpoint(path)
    other.close()


# -------------------------------------------------------------------- swap


def test_swap_and_resident_produce_the_same_update(tmp_path: Path):
    """`swap` must change only where tensors live, never the objective."""
    from miniverl.trainer import OPDTrainer
    from miniverl.utils.seeding import seed_everything

    seed_everything(11, deterministic=True)
    resident = OPDTrainer.from_config(
        _config(tmp_path, memory={"strategy": "resident"}), run_id="resident"
    )
    resident.train()
    resident_state = _flat_state(resident.student)
    resident.close()

    seed_everything(11, deterministic=True)
    swapped = OPDTrainer.from_config(
        _config(tmp_path, memory={"strategy": "swap"}), run_id="swapped"
    )
    swapped.train()
    swapped_state = _flat_state(swapped.student)
    swapped.close()

    _assert_states_equal(resident_state, swapped_state, atol=1e-6)


def test_swap_reads_its_targets_back_through_the_cache(tmp_path: Path):
    """Under swap the teacher is gone by update time, so the cache is the source."""
    from miniverl.cache.store import TeacherCache
    from miniverl.trainer import OPDTrainer

    trainer = OPDTrainer.from_config(
        _config(tmp_path, memory={"strategy": "swap"}, train={"cycles": 2}), run_id="swap-cache"
    )
    trainer.train()
    cache = TeacherCache.open(trainer.paths.teacher_cache)
    assert len(cache) > 0
    assert cache.validate() == []
    trainer.close()


def test_swap_is_refused_for_a_quantized_model(tmp_path: Path):
    """bitsandbytes parameters are device-pinned, so swap cannot work."""
    from miniverl.config import RunConfig
    from miniverl.errors import ConfigError
    from miniverl.trainer import OPDTrainer

    payload = _config(tmp_path).model_dump(mode="json")
    payload["models"]["backend"] = "hf"
    payload["models"]["student"]["model_id"] = "unused/model"
    payload["models"]["student"]["quantization"] = "nf4"
    payload["models"]["student"]["lora"]["enabled"] = True
    payload["models"]["teacher"]["model_id"] = "unused/teacher"
    payload["memory"]["strategy"] = "swap"
    config = RunConfig.model_validate(payload)
    with pytest.raises(ConfigError, match="cannot be used with a quantized model"):
        OPDTrainer.from_config(config, run_id="swap-quantized")


def test_auto_resolves_to_resident_on_cpu_and_records_the_reason(tmp_path: Path):
    from miniverl.trainer import OPDTrainer

    trainer = OPDTrainer.from_config(
        _config(tmp_path, memory={"strategy": "auto"}, train={"cycles": 1}), run_id="auto"
    )
    assert trainer.plan.strategy.value == "resident"
    assert "no CUDA device" in trainer.plan.reason
    resolved = trainer.paths.config_resolved.read_text(encoding="utf-8")
    assert "strategy: resident" in resolved
    manifest = trainer.build_manifest()
    assert manifest["memory"]["strategy"] == "resident"
    assert manifest["memory"]["reason"]
    trainer.close()


# ------------------------------------------------------- OOM retry contract


def test_oom_retry_only_shrinks_the_chunk_and_gives_up_with_advice():
    from miniverl.config.models import MemoryConfig, MemoryStrategy
    from miniverl.errors import GpuMemoryError
    from miniverl.training.memory import MemoryPlan, run_with_oom_retry

    plan = MemoryPlan(
        strategy=MemoryStrategy.RESIDENT, chunk_size=256, device="cuda", reason="test"
    )
    memory = MemoryConfig(oom_retries=3, min_chunk_size=32)
    attempts: list[int] = []

    def flaky(chunk: int) -> str:
        attempts.append(chunk)
        if chunk > 64:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return "ok"

    assert run_with_oom_retry(flaky, plan=plan, memory=memory) == "ok"
    assert attempts == [256, 128, 64]
    assert plan.chunk_size == 64
    assert plan.oom_retries_used == 2
    assert plan.chunk_size_history == [256]

    always = MemoryPlan(
        strategy=MemoryStrategy.RESIDENT, chunk_size=64, device="cuda", reason="test"
    )
    with pytest.raises(GpuMemoryError, match="retries were exhausted") as excinfo:

        def never(chunk: int) -> str:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")

        run_with_oom_retry(
            never, plan=always, memory=MemoryConfig(oom_retries=2, min_chunk_size=32)
        )
    assert "rollout.max_total_tokens" in (excinfo.value.hint or "")


def test_a_non_oom_error_is_not_retried():
    from miniverl.config.models import MemoryConfig, MemoryStrategy
    from miniverl.training.memory import MemoryPlan, run_with_oom_retry

    plan = MemoryPlan(strategy=MemoryStrategy.RESIDENT, chunk_size=64, device="cuda", reason="t")
    calls = 0

    def boom(chunk: int) -> str:
        nonlocal calls
        calls += 1
        raise RuntimeError("shape mismatch")

    with pytest.raises(RuntimeError, match="shape mismatch"):
        run_with_oom_retry(boom, plan=plan, memory=MemoryConfig(oom_retries=3))
    assert calls == 1, "a real bug must surface immediately, not be retried away"
