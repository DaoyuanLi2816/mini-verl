from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from miniverl.algorithms.contract import UPSTREAM_VERL_COMMIT
from miniverl.bridge.rl_v09 import VERL_RL_V09_PROFILE, publish_imported_verl_rl_v09
from miniverl.config import RunConfig
from miniverl.errors import ConfigError


def source_payload() -> dict:  # type: ignore[type-arg]
    return {
        "data": {
            "train_files": ["train.parquet"],
            "val_files": ["val.parquet"],
            "prompt_key": "prompt",
            "train_batch_size": 2,
            "max_prompt_length": 128,
            "max_response_length": 32,
            "truncation": "error",
            "shuffle": True,
            "seed": 17,
        },
        "actor_rollout_ref": {
            "model": {
                "path": "Qwen/Qwen3-0.6B",
                "enable_gradient_checkpointing": True,
                "lora_rank": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj", "v_proj"],
            },
            "actor": {
                "optim": {"lr": "1e-5", "weight_decay": 0.01, "lr_warmup_steps": 1},
                "ppo_mini_batch_size": 8,
                "ppo_epochs": 1,
                "loss_agg_mode": "token-mean",
                "clip_ratio": 0.2,
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.2,
                "clip_ratio_c": 3.0,
                "use_kl_loss": False,
                "entropy_coeff": 0.0,
            },
            "rollout": {
                "name": "vllm",
                "n": 4,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 0,
                "tensor_model_parallel_size": 8,
                "gpu_memory_utilization": 0.6,
            },
        },
        "algorithm": {
            "adv_estimator": "grpo",
            "norm_adv_by_std_in_grpo": True,
            "gamma": 1.0,
            "lam": 1.0,
            "use_kl_in_reward": False,
        },
        "trainer": {
            "total_training_steps": 3,
            "total_epochs": 30,
            "project_name": "verl-home",
            "experiment_name": "local-grpo",
            "save_freq": -1,
            "test_freq": -1,
            "logger": ["console"],
            "n_gpus_per_node": 8,
            "nnodes": 2,
        },
        "miniverl": {
            "reward": {"provider": "exact_answer"},
            "actor": {"lora_enabled": True, "dtype": "auto", "quantization": "none"},
            "rollout": {"backend": "hf_cached"},
            "memory": {"strategy": "auto"},
            "update": {"trajectory_batch_size": 1},
        },
    }


def _write(tmp_path: Path, payload: dict) -> Path:  # type: ignore[type-arg]
    path = tmp_path / "verl.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_v09_grpo_import_preserves_semantics_and_lowers_placement(tmp_path: Path) -> None:
    out = tmp_path / "local.yaml"
    report = publish_imported_verl_rl_v09(
        _write(tmp_path, source_payload()),
        out=out,
        target_verl=UPSTREAM_VERL_COMMIT,
    )
    config = RunConfig.from_yaml(out)

    assert report["status"] == "accepted"
    assert report["profile"] == VERL_RL_V09_PROFILE
    assert config.algorithm.name.value == "grpo"
    assert config.source.train_files == ["train.parquet"]
    assert config.resolved_for_runtime().source.train_files == [
        str((tmp_path / "train.parquet").resolve())
    ]
    assert config.rollout.samples_per_prompt == 4
    assert config.train.cycles == 3
    assert config.train.rollouts_per_cycle == 2
    assert config.train.gradient_accumulation_steps == 8
    assert config.train.opd_freshness.value == "strict"
    assert config.train.learning_rate == pytest.approx(1e-5)
    distributed = [
        row for row in report["field_classification"] if row["classification"] == "distributed_only"
    ]
    assert {row["upstream_field"] for row in distributed} == {
        "actor_rollout_ref.rollout.tensor_model_parallel_size",
        "trainer.n_gpus_per_node",
        "trainer.nnodes",
    }
    assert report["local_lowering"][2]["local_value"] == {"devices": 1, "processes": 1}
    assert not [
        row
        for row in report["field_classification"]
        if row["classification"] in {"not_implemented", "unsupported"}
    ]
    lam = next(
        row for row in report["field_classification"] if row["upstream_field"] == "algorithm.lam"
    )
    assert lam["classification"] == "informational_only"


def test_v09_grpo_false_std_maps_to_dr_grpo(tmp_path: Path) -> None:
    payload = source_payload()
    payload["algorithm"]["norm_adv_by_std_in_grpo"] = False
    out = tmp_path / "local.yaml"
    publish_imported_verl_rl_v09(_write(tmp_path, payload), out=out, target_verl="v0.9.0")
    assert RunConfig.from_yaml(out).algorithm.name.value == "dr_grpo"


def test_v09_import_accepts_explicit_builtin_target_length_reward(tmp_path: Path) -> None:
    payload = source_payload()
    payload["miniverl"]["reward"]["provider"] = "target_length"
    out = tmp_path / "local.yaml"

    report = publish_imported_verl_rl_v09(_write(tmp_path, payload), out=out, target_verl="v0.9.0")

    assert report["status"] == "accepted"
    assert RunConfig.from_yaml(out).reward.provider.value == "target_length"


def test_v09_actor_minibatch_preserves_multiple_updates_as_explicit_replay(
    tmp_path: Path,
) -> None:
    payload = source_payload()
    payload["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] = 4
    out = tmp_path / "local.yaml"

    publish_imported_verl_rl_v09(_write(tmp_path, payload), out=out, target_verl="v0.9.0")
    config = RunConfig.from_yaml(out)

    assert config.train.gradient_accumulation_steps == 4
    assert config.train.opd_freshness.value == "replay"
    assert config.is_on_policy is False


def test_v09_fixed_reference_kl_lowers_to_shared_temporal_role(tmp_path: Path) -> None:
    payload = source_payload()
    payload["algorithm"].update(
        {
            "use_kl_in_reward": True,
            "kl_penalty": "low_var_kl",
            "kl_ctrl": {"type": "fixed", "kl_coef": 0.02},
        }
    )
    payload["miniverl"]["reference"] = {
        "adapter_path": "reference-adapter",
        "adapter_source": "local",
    }
    out = tmp_path / "local.yaml"

    report = publish_imported_verl_rl_v09(_write(tmp_path, payload), out=out, target_verl="v0.9.0")
    config = RunConfig.from_yaml(out)

    assert report["status"] == "accepted"
    assert config.models.runtime.value == "shared_backbone"
    assert config.models.reference is not None
    assert config.models.reference.adapter.path == "reference-adapter"
    assert config.resolved_for_runtime().models.reference.adapter.path == str(
        (tmp_path / "reference-adapter").resolve()
    )
    assert config.algorithm.kl_coef == pytest.approx(0.02)
    assert config.algorithm.kl_penalty == "low_var_kl"


def test_v09_import_emits_non_executable_template_when_choices_are_missing(tmp_path: Path) -> None:
    payload = source_payload()
    del payload["trainer"]["total_training_steps"]
    del payload["miniverl"]["reward"]
    del payload["miniverl"]["actor"]["lora_enabled"]
    out = tmp_path / "local.yaml"
    report = publish_imported_verl_rl_v09(_write(tmp_path, payload), out=out, target_verl="v0.9.0")

    assert report["status"] == "needs_user_input"
    assert not out.exists()
    template = tmp_path / "local.template.yaml"
    assert template.exists()
    with pytest.raises(ValidationError):
        RunConfig.from_yaml(template)
    assert {row["field"] for row in report["required_user_input"]} == {
        "trainer.total_training_steps",
        "miniverl.reward.provider",
        "miniverl.actor.lora_enabled",
    }


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (("algorithm", "adv_estimator", "gae"), "critic lifecycle"),
        (("actor_rollout_ref.actor", "use_kl_loss", True), "not implemented"),
        (("actor_rollout_ref.actor", "ppo_epochs", 2), "must be 1"),
    ],
)
def test_v09_import_fails_closed_for_unimplemented_semantics(
    tmp_path: Path, mutate: tuple[str, str, object], match: str
) -> None:
    payload = source_payload()
    parent, key, value = mutate
    current = payload
    for part in parent.split("."):
        current = current[part]
    current[key] = value
    with pytest.raises(ConfigError, match=match):
        publish_imported_verl_rl_v09(
            _write(tmp_path, payload), out=tmp_path / "local.yaml", target_verl="v0.9.0"
        )


def test_v09_import_rejects_unknown_fields_and_non_finite_numbers(tmp_path: Path) -> None:
    payload = source_payload()
    payload["trainer"]["mystery"] = 1
    out = tmp_path / "unknown.yaml"
    report = publish_imported_verl_rl_v09(_write(tmp_path, payload), out=out, target_verl="v0.9.0")
    assert report["status"] == "rejected"
    assert not out.exists()
    assert "trainer.mystery" in report["unsupported_fields"]
    on_disk = json.loads((tmp_path / "unknown.import-report.json").read_text(encoding="utf-8"))
    assert on_disk == report

    payload = deepcopy(source_payload())
    payload["actor_rollout_ref"]["actor"]["optim"]["lr"] = ".nan"
    with pytest.raises(ConfigError, match="must be finite"):
        publish_imported_verl_rl_v09(
            _write(tmp_path, payload), out=tmp_path / "nan.yaml", target_verl="v0.9.0"
        )
