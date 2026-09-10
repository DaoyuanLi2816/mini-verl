"""Direct execution checks hardware separately and never trusts config code."""

import hashlib
from pathlib import Path

import pytest
import yaml

from miniverl.bridge.direct import apply_bindings, build_native, compile_direct
from miniverl.bridge.direct_runtime import hardware_plan
from miniverl.errors import ConfigError
from miniverl.rewards.binding import BoundVerlReward

ROOT = Path(__file__).resolve().parents[2]


def test_hardware_block_does_not_change_experiment():
    source = yaml.safe_load(
        (ROOT / "benchmarks/compatibility/verl-v0.9.0/ppo-complete.yaml").read_text()
    )
    native = build_native(source, cycles=2)
    plan = hardware_plan(native, parameter_counts={"actor": 8_000_000_000}, available_gpu_gib=16)
    assert plan["status"] == "hardware_blocked"
    assert "actor" in plan["blocking_roles"]
    assert native["train"]["rollouts_per_cycle"] == 1024
    assert native["train"]["gradient_accumulation_steps"] == 256
    assert native["train"]["adam_beta2"] == 0.999
    assert native["train"]["lr_step_unit"] == "rollout_iteration"


def test_bindings_are_typed_and_do_not_mutate_source():
    source = {
        "actor_rollout_ref": {"model": {"path": "large"}},
        "critic": {"model": {"path": "large"}},
    }
    effective, ledger = apply_bindings(
        source, ["actor.model=small", "data.train_files=[a.parquet,b.parquet]"]
    )
    assert source["actor_rollout_ref"]["model"]["path"] == "large"
    assert effective["critic"]["model"]["path"] == "small"
    assert len(ledger) == 3
    with pytest.raises(ConfigError, match="unknown"):
        apply_bindings(source, ["hf_token=secret"])
    with pytest.raises(ConfigError, match="only data"):
        apply_bindings(source, ["actor.model=[a,b]"])


def test_reward_code_requires_matching_explicit_approval(tmp_path):
    path = tmp_path / "reward.py"
    path.write_text("raise RuntimeError('must not execute')")
    with pytest.raises(ConfigError, match="checksum"):
        BoundVerlReward(f"{path}:compute_score", approved_sha256="0" * 64)
    path.write_text("def compute_score(**kwargs):\n    return 1.0\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    provider = BoundVerlReward(f"{path}:compute_score", approved_sha256=digest)
    assert provider.identity.config_digest == digest


def test_cli_direct_auto_selects_new_profile():
    from typer.testing import CliRunner

    from miniverl.cli import app

    source = ROOT / "benchmarks/compatibility/verl-v0.9.0/grpo-complete.yaml"
    result = CliRunner().invoke(app, ["run", str(source), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "semantic: accepted" in result.output
    assert "requires_bindings" in result.output


def test_bad_sampling_cannot_be_marked_accepted(tmp_path):
    source = ROOT / "benchmarks/compatibility/verl-v0.9.0/grpo-complete.yaml"
    payload = yaml.safe_load(source.read_text())
    payload["actor_rollout_ref"]["rollout"]["do_sample"] = False
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload))
    result = compile_direct(path)
    assert result["semantic_status"] == "rejected"


def test_competing_reward_bindings_reject():
    with pytest.raises(ConfigError, match="one reward"):
        apply_bindings({}, ["reward.provider=target_length", "reward.function=reward.py:score"])


def test_direct_export_preserves_new_semantics():
    from miniverl.bridge.export import _rl_v09_overrides

    source = yaml.safe_load((ROOT / "src/miniverl/resources/verl_direct_ppo.yaml").read_text())
    source["actor_rollout_ref"]["actor"]["loss_agg_mode"] = "seq-mean-token-sum-norm"
    source["actor_rollout_ref"]["actor"]["loss_scale_factor"] = 37
    source["critic"]["ppo_mini_batch_size"] = 1
    source["critic"]["model"]["path"] = "independent/critic"
    native = build_native(source, cycles=2)
    result = _rl_v09_overrides(
        native,
        {"rank": 16, "alpha": 32, "target_modules": ["q_proj"]},
        {"train": ["data/train.parquet"], "val": ["data/val.parquet"]},
    )
    actor = result["actor_rollout_ref"]["actor"]
    assert actor["loss_agg_mode"] == "seq-mean-token-sum-norm"
    assert actor["loss_scale_factor"] == 37
    assert actor["ppo_mini_batch_size"] == 2
    assert actor["optim"]["betas"] == [0.9, 0.999]
    assert result["critic"]["ppo_mini_batch_size"] == 1
    assert result["critic"]["model"]["path"] == "critic/base"
    assert result["data"]["filter_overlong_prompts"] is True
    assert result["trainer"]["val_before_train"] is True


def test_independent_frozen_base_reference_is_not_a_teacher():
    source = yaml.safe_load((ROOT / "src/miniverl/resources/verl_direct_grpo.yaml").read_text())
    source["actor_rollout_ref"]["actor"]["use_kl_loss"] = True
    source["actor_rollout_ref"]["actor"]["kl_loss_coef"] = 0.02
    native = build_native(source, cycles=2)
    assert native["models"]["reference"]["frozen_base"] is True
    assert native["models"]["reference"]["adapter"] is None
    assert native["models"]["runtime"] == "dual_model"
    assert native["algorithm"]["actor_kl_coef"] == 0.02
