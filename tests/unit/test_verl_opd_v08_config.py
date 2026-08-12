"""Typed, fail-closed compilation of the pinned verl v0.8 OPD subset."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from miniverl.errors import ConfigError


def _profile() -> dict[str, object]:
    return {
        "data": {
            "train_files": ["data/train.parquet"],
            "val_files": ["data/val.parquet"],
            "prompt_key": "prompt",
            "train_batch_size": 8,
            "max_prompt_length": 256,
            "max_response_length": 64,
            "filter_overlong_prompts": False,
            "truncation": "error",
            "shuffle": True,
            "seed": 17,
        },
        "actor_rollout_ref": {
            "model": {
                "path": "Qwen/Qwen3-0.6B",
                "enable_gradient_checkpointing": True,
                "lora_rank": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "v_proj"],
                "lora_adapter_path": None,
            },
            "actor": {
                "optim": {
                    "lr": "1e-5",
                    "weight_decay": 0.01,
                    "lr_warmup_steps": 0,
                },
                "loss_agg_mode": "token-mean",
                "use_kl_loss": False,
                "ppo_mini_batch_size": 8,
                "ppo_max_token_len_per_gpu": 2048,
                "use_dynamic_bsz": True,
            },
            "rollout": {
                "name": "vllm",
                "n": 1,
                "temperature": 1.0,
                "top_p": 0.95,
                "tensor_model_parallel_size": 1,
                "gpu_memory_utilization": 0.5,
                "max_model_len": 320,
                "max_num_batched_tokens": 2048,
                "max_num_seqs": 8,
            },
        },
        "algorithm": {"use_kl_in_reward": False},
        "distillation": {
            "enabled": True,
            "teacher_key": "data_source",
            "n_gpus_per_node": 1,
            "nnodes": 1,
            "teacher_models": {
                "teacher_model": {
                    "model_path": "Qwen/Qwen3-1.7B",
                    "num_replicas": 1,
                    "inference": {
                        "name": "vllm",
                        "dtype": "bfloat16",
                        "tensor_model_parallel_size": 1,
                        "data_parallel_size": 1,
                        "pipeline_model_parallel_size": 1,
                        "gpu_memory_utilization": 0.5,
                        "max_model_len": 321,
                    },
                }
            },
            "distillation_loss": {
                "loss_mode": "forward_kl_topk",
                "topk": 32,
                "use_task_rewards": False,
                "distillation_loss_coef": 1.0,
                "loss_max_clamp": None,
                "log_prob_min_clamp": -10.0,
                "use_policy_gradient": False,
            },
        },
        "trainer": {
            "project_name": "mini-verl",
            "experiment_name": "opd-smoke",
            "save_freq": 10,
            "test_freq": 10,
            "total_epochs": 1,
            "total_training_steps": 2,
            "n_gpus_per_node": 1,
            "nnodes": 1,
        },
        "miniverl": {
            "runtime": {"mode": "auto"},
            "memory": {"vram_limit_gib": 16, "headroom_gib": 1.5},
            "batching": {
                "rollout_batch_size": 2,
                "teacher_score_batch_size": 2,
                "update_trajectory_batch_size": 2,
            },
            "teacher_adapter": {"path": None, "revision": None},
        },
    }


def test_compiler_is_typed_deterministic_and_classifies_every_source_leaf() -> None:
    from miniverl.bridge.opd_v08 import VerlOPDV08Profile, compile_verl_opd_v08

    first = compile_verl_opd_v08(_profile())
    second = compile_verl_opd_v08(copy.deepcopy(_profile()))

    assert isinstance(first.source, VerlOPDV08Profile)
    assert first.executable is True
    assert first.compiled_digest == second.compiled_digest
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.source.actor_rollout_ref.actor.optim.lr == pytest.approx(1e-5)
    paths = {item.upstream_field for item in first.compatibility}
    assert paths == set(first.source_leaf_fields)
    classes = {item.upstream_field: item.classification for item in first.compatibility}
    assert classes["data.train_files"] == "exact"
    assert classes["actor_rollout_ref.rollout.name"] == "locally_reinterpreted"
    assert classes["trainer.total_epochs"] == "locally_reinterpreted"
    assert classes["miniverl.runtime.mode"] == "informational_only"


def test_override_precedence_and_scientific_notation_are_safe() -> None:
    from miniverl.bridge.opd_v08 import compile_verl_opd_v08

    plan = compile_verl_opd_v08(
        _profile(),
        overrides=(
            "actor_rollout_ref.actor.optim.lr=2e-5",
            'data.train_files=["new.parquet"]',
            "distillation.distillation_loss.topk=64",
        ),
    )

    assert plan.source.actor_rollout_ref.actor.optim.lr == pytest.approx(2e-5)
    assert plan.source.data.train_files == ["new.parquet"]
    assert plan.source.distillation.distillation_loss.topk == 64
    assert [item.expression for item in plan.overrides] == [
        "actor_rollout_ref.actor.optim.lr=2e-5",
        'data.train_files=["new.parquet"]',
        "distillation.distillation_loss.topk=64",
    ]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("distillation.enabled", False),
        ("distillation.distillation_loss.loss_mode", "k3"),
        ("distillation.distillation_loss.use_policy_gradient", True),
        ("distillation.distillation_loss.use_task_rewards", True),
        ("actor_rollout_ref.actor.use_kl_loss", True),
        ("algorithm.use_kl_in_reward", True),
        ("actor_rollout_ref.rollout.n", 2),
        ("actor_rollout_ref.rollout.tensor_model_parallel_size", 2),
        (
            "distillation.teacher_models.teacher_model.inference.data_parallel_size",
            2,
        ),
        ("trainer.nnodes", 2),
    ],
)
def test_unsupported_algorithm_and_distributed_semantics_fail_closed(
    path: str, value: object
) -> None:
    from miniverl.bridge.opd_v08 import compile_verl_opd_v08

    payload = _profile()
    current: dict[str, object] = payload
    parts = path.split(".")
    for part in parts[:-1]:
        current = current[part]  # type: ignore[assignment]
    current[parts[-1]] = value

    with pytest.raises(ConfigError, match="not executable") as exc_info:
        compile_verl_opd_v08(payload)
    assert path in str(exc_info.value)


def test_unknown_fields_and_multiple_teachers_fail_closed() -> None:
    from miniverl.bridge.opd_v08 import compile_verl_opd_v08

    unknown = _profile()
    unknown["critic"] = {"strategy": "fsdp"}
    with pytest.raises(ConfigError, match=r"critic\.strategy"):
        compile_verl_opd_v08(unknown)

    multiple = _profile()
    teachers = multiple["distillation"]["teacher_models"]  # type: ignore[index]
    teachers["teacher_model_2"] = copy.deepcopy(teachers["teacher_model"])  # type: ignore[index]
    with pytest.raises(ConfigError, match="multi-teacher"):
        compile_verl_opd_v08(multiple)


@pytest.mark.parametrize("value", ["${oc.env:MODEL}", "${model.path", "${HOME}/model"])
def test_unresolved_interpolation_is_rejected(value: str) -> None:
    from miniverl.bridge.opd_v08 import compile_verl_opd_v08

    payload = _profile()
    payload["actor_rollout_ref"]["model"]["path"] = value  # type: ignore[index]
    with pytest.raises(ConfigError, match="unresolved interpolation"):
        compile_verl_opd_v08(payload)


@pytest.mark.parametrize("value", ["nan", ".inf", "-Infinity", float("nan")])
def test_non_finite_numbers_are_rejected(value: object) -> None:
    from miniverl.bridge.opd_v08 import compile_verl_opd_v08

    payload = _profile()
    payload["actor_rollout_ref"]["actor"]["optim"]["lr"] = value  # type: ignore[index]
    with pytest.raises(ConfigError, match="finite"):
        compile_verl_opd_v08(payload)


def test_report_can_be_inspected_without_authorizing_unsupported_input() -> None:
    from miniverl.bridge.opd_v08 import compile_verl_opd_v08

    payload = _profile()
    payload["actor_rollout_ref"]["rollout"]["n"] = 4  # type: ignore[index]
    report = compile_verl_opd_v08(payload, require_executable=False)

    assert report.executable is False
    item = next(x for x in report.compatibility if x.upstream_field.endswith("rollout.n"))
    assert item.classification == "unsupported"
    assert item.executable is False
    assert "one generation" in item.reason
    json.dumps(report.model_dump(mode="json"), sort_keys=True)


def test_packaged_resolved_fixture_compiles_and_shell_inputs_are_refused(tmp_path: Path) -> None:
    from miniverl.bridge.opd_v08 import load_verl_opd_v08

    root = Path(__file__).resolve().parents[2]
    plan = load_verl_opd_v08(root / "examples" / "verl-opd-v0.8-single-gpu.yaml")
    assert plan.executable is True
    assert plan.upstream["commit"] == "7aed6b230776f963fa09509c10d9c3a767d1102c"

    script = tmp_path / "profile.sh"
    script.write_text("echo never-executed", encoding="utf-8")
    with pytest.raises(ConfigError, match="not a shell script"):
        load_verl_opd_v08(script)
