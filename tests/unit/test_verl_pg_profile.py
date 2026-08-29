from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from miniverl.errors import ConfigError

PG_PROFILE = "verl-opd-v0.8-single-gpu-pg-k1-v1"
REWARDED_PROFILE = "verl-opd-v0.8-single-gpu-pg-k1-rewarded-v1"


def _payload() -> dict[str, object]:
    source = Path(__file__).resolve().parents[2] / "examples/verl-opd-v0.8-single-gpu.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    loss = payload["distillation"]["distillation_loss"]
    loss.update(
        {
            "loss_mode": "k1",
            "topk": None,
            "use_policy_gradient": True,
            "policy_loss_mode": "vanilla",
            "clip_ratio": 0.2,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.2,
            "loss_max_clamp": 10.0,
            "log_prob_min_clamp": None,
        }
    )
    payload["algorithm"]["adv_estimator"] = None
    return payload


def _set(payload: dict[str, object], path: str, value: object) -> None:
    current = payload
    parts = path.split(".")
    for part in parts[:-1]:
        current = current[part]  # type: ignore[assignment]
    current[parts[-1]] = value


def _rewarded_payload() -> dict[str, object]:
    payload = _payload()
    loss = payload["distillation"]["distillation_loss"]  # type: ignore[index]
    loss.update(  # type: ignore[union-attr]
        {
            "use_task_rewards": True,
            "task_reward_coef": 1.0,
            "task_advantage_mode": "group_center",
            "reward_provider": "exact_answer",
        }
    )
    payload["actor_rollout_ref"]["rollout"]["n"] = 4  # type: ignore[index]
    return payload


def test_pg_profile_compiles_as_a_separate_executable_contract() -> None:
    from miniverl.bridge.opd_pg_v08 import compile_verl_pg_k1_v08
    from miniverl.bridge.opd_runtime import build_system_plan, compile_native_run_config

    plan = compile_verl_pg_k1_v08(_payload())
    assert plan.profile == PG_PROFILE
    assert plan.executable is True
    assert plan.local_execution["loss_mode"] == "k1"
    assert plan.local_execution["policy_gradient"] is True
    assert plan.local_execution["policy_loss_mode"] == "vanilla"
    assert plan.local_execution["task_rewards"] is False
    assert plan.source.distillation.distillation_loss.topk is None
    native = compile_native_run_config(plan, system_plan=build_system_plan(plan))
    assert native.loss.mode.value == "verl_pg_k1"
    assert native.loss.policy_loss_mode == "vanilla"
    assert native.loss.clip_ratio == pytest.approx(0.2)
    assert native.rollout.record_logprobs is True
    assert native.train.opd_freshness.value == "strict"
    assert native.train.gradient_accumulation_steps == native.train.rollouts_per_cycle


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("distillation.distillation_loss.loss_mode", "k3"),
        ("distillation.distillation_loss.use_policy_gradient", False),
        ("distillation.distillation_loss.policy_loss_mode", "dppo_tv"),
        ("distillation.distillation_loss.use_task_rewards", True),
        ("distillation.distillation_loss.topk", 32),
        ("distillation.distillation_loss.clip_ratio", 0.1),
        ("actor_rollout_ref.actor.use_kl_loss", True),
        ("algorithm.use_kl_in_reward", True),
        ("algorithm.adv_estimator", "grpo"),
        ("actor_rollout_ref.rollout.n", 2),
        ("trainer.n_gpus_per_node", 2),
    ],
)
def test_pg_profile_rejects_out_of_contract_semantics(path: str, value: object) -> None:
    from miniverl.bridge.opd_pg_v08 import compile_verl_pg_k1_v08

    payload = _payload()
    _set(payload, path, value)
    with pytest.raises(ConfigError, match="not executable") as exc_info:
        compile_verl_pg_k1_v08(payload)
    assert path in str(exc_info.value)


def test_registry_lists_pg_profile_with_independent_identity() -> None:
    from miniverl.bridge.profiles import get_profile, list_profiles

    names = [profile.name for profile in list_profiles()]
    assert names == [
        "verl-opd-v0.8-single-gpu-v1",
        PG_PROFILE,
        "verl-opd-v0.8-single-gpu-grouped-v1",
        "verl-opd-v0.8-single-gpu-pg-k1-grouped-v1",
        REWARDED_PROFILE,
    ]
    direct = get_profile(names[0])
    pg = get_profile(PG_PROFILE)
    assert pg.identity.digest != direct.identity.digest
    assert pg.identity.loss_conformance_version == "pg-k1-verl-v0.8-v1"
    assert pg.summary.objective == "k1 + vanilla policy loss"
    assert pg.summary.teacher_target == "sampled-token teacher log-probability"


def test_rewarded_pg_profile_compiles_to_explicit_reward_contract() -> None:
    from miniverl.bridge.opd_pg_v08 import compile_verl_pg_k1_v08
    from miniverl.bridge.opd_runtime import build_system_plan, compile_native_run_config

    plan = compile_verl_pg_k1_v08(
        _rewarded_payload(),
        profile_name=REWARDED_PROFILE,
        allow_grouped_samples=True,
        rewarded=True,
    )
    native = compile_native_run_config(plan, system_plan=build_system_plan(plan))
    assert native.loss.mode.value == "verl_pg_k1_rewarded"
    assert native.loss.advantage_mode.value == "group_center"
    assert native.loss.task_reward_coef == pytest.approx(1.0)
    assert native.source.use_task_rewards is True
    assert native.reward.enabled is True
    assert native.reward.provider.value == "exact_answer"


def test_rewarded_profile_rejects_n1_group_normalization() -> None:
    from miniverl.bridge.opd_pg_v08 import compile_verl_pg_k1_v08

    payload = _rewarded_payload()
    payload["actor_rollout_ref"]["rollout"]["n"] = 1  # type: ignore[index]
    with pytest.raises(ConfigError, match="task_advantage_mode"):
        compile_verl_pg_k1_v08(
            payload,
            profile_name=REWARDED_PROFILE,
            allow_grouped_samples=True,
            rewarded=True,
        )
