from __future__ import annotations

import hashlib
import importlib.metadata
import json
import struct
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml


def _safetensors_bytes() -> bytes:
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        separators=(",", ":"),
    ).encode()
    header += b" " * ((8 - len(header) % 8) % 8)
    return struct.pack("<Q", len(header)) + header + struct.pack("<f", 0.0)


def _opd_run(tmp_path: Path, *, teacher_adapter: bool = False) -> tuple[Path, Path, Path]:
    run = tmp_path / "run"
    model = run / "final-peft-adapter"
    model.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    (model / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "base_model_name_or_path": "Qwen/Qwen3-0.6B",
                "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
                "target_modules": ["q_proj", "v_proj"],
                "r": 8,
                "lora_alpha": 16,
            }
        ),
        encoding="utf-8",
    )
    (model / "adapter_model.safetensors").write_bytes(_safetensors_bytes())
    (model / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "Qwen2Tokenizer"}), encoding="utf-8"
    )
    rows = [
        {
            "data_source": "unit",
            "prompt": [{"role": "user", "content": f"prompt-{index}"}],
            "ability": "chat",
            "reward_model": None,
            "extra_info": {"row_identity": index},
        }
        for index in range(3)
    ]
    train = tmp_path / "source-train.parquet"
    val = tmp_path / "source-val.parquet"
    pq.write_table(pa.Table.from_pylist(rows), train)
    pq.write_table(pa.Table.from_pylist(rows[:1]), val)
    source = {
        "data": {
            "train_files": [str(train)],
            "val_files": [str(val)],
            "prompt_key": "prompt",
            "train_batch_size": 2,
            "max_prompt_length": 128,
            "max_response_length": 16,
            "filter_overlong_prompts": True,
            "truncation": "error",
            "shuffle": False,
            "seed": 7,
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
                "optim": {"lr": 1e-5, "weight_decay": 0.0, "lr_warmup_steps": 0},
                "loss_agg_mode": "token-mean",
                "use_kl_loss": False,
                "ppo_mini_batch_size": 2,
            },
            "rollout": {
                "name": "hf",
                "n": 1,
                "temperature": 0.0,
                "top_p": 1.0,
                "tensor_model_parallel_size": 1,
            },
        },
        "algorithm": {"use_kl_in_reward": False},
        "distillation": {
            "enabled": True,
            "n_gpus_per_node": 1,
            "nnodes": 1,
            "teacher_models": {
                "teacher_model": {
                    "model_path": "Qwen/Qwen3-1.7B",
                    "num_replicas": 1,
                    "inference": {
                        "name": "hf",
                        "dtype": "bfloat16",
                        "tensor_model_parallel_size": 1,
                        "data_parallel_size": 1,
                        "pipeline_model_parallel_size": 1,
                    },
                }
            },
            "distillation_loss": {
                "loss_mode": "forward_kl_topk",
                "topk": 32,
                "use_task_rewards": False,
                "distillation_loss_coef": 1.0,
                "log_prob_min_clamp": -10.0,
                "use_policy_gradient": False,
            },
        },
        "trainer": {
            "project_name": "mini-verl",
            "experiment_name": "opd-export",
            "save_freq": 1,
            "test_freq": 0,
            "total_training_steps": 1,
            "n_gpus_per_node": 1,
            "nnodes": 1,
        },
        "miniverl": {
            "student_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
            "teacher_revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
            "runtime": {"mode": "dual_model"},
            "teacher_adapter": {
                "path": "teacher-adapter" if teacher_adapter else None,
                "revision": "a" * 40 if teacher_adapter else None,
            },
        },
    }
    (run / "verl-source-config.json").write_text(json.dumps(source), encoding="utf-8")
    (run / "verl-compatibility-report.json").write_text(
        json.dumps(
            {
                "profile": "verl-opd-v0.8-single-gpu-v1",
                "executable": True,
                "compiled_digest": "b" * 64,
                "source": source,
            }
        ),
        encoding="utf-8",
    )
    (run / "local-execution-plan.json").write_text(
        json.dumps({"profile": "verl-opd-v0.8-single-gpu-v1", "compiled_digest": "b" * 64}),
        encoding="utf-8",
    )
    return run, train, val


def test_pure_opd_export_is_reward_free_and_preserves_data_bytes(tmp_path: Path) -> None:
    from miniverl.bridge.contract import VERL_TAG
    from miniverl.bridge.doctor import inspect_bridge_bundle
    from miniverl.bridge.export import export_verl_bundle

    run, train, val = _opd_run(tmp_path)
    out = tmp_path / "bundle"
    report = export_verl_bundle(run, target_verl=VERL_TAG, out=out)

    assert (out / "recipe/verl-opd-overrides.yaml").is_file()
    assert not (out / "recipe/verl-overrides.yaml").exists()
    assert not (out / "reward").exists()
    assert (out / "teacher/teacher-model.json").is_file()
    assert (out / "provenance/source-config.json").is_file()
    assert (out / "provenance/compiled-plan.json").is_file()
    assert (out / "data/train.parquet").read_bytes() == train.read_bytes()
    assert (out / "data/val.parquet").read_bytes() == val.read_bytes()

    overrides = yaml.safe_load((out / "recipe/verl-opd-overrides.yaml").read_text())
    assert overrides["distillation"]["enabled"] is True
    loss = overrides["distillation"]["distillation_loss"]
    assert loss == {
        "loss_mode": "forward_kl_topk",
        "topk": 32,
        "use_task_rewards": False,
        "distillation_loss_coef": 1.0,
        "log_prob_min_clamp": -10.0,
        "use_policy_gradient": False,
    }
    assert overrides["actor_rollout_ref"]["actor"]["use_kl_loss"] is False
    assert overrides["algorithm"]["use_kl_in_reward"] is False
    assert report["artifact_complete"] is True
    assert report["config_semantics_supported"] is True
    assert report["student_artifact_loadable"] is True
    assert report["teacher_artifact_loadable"] is False
    assert report["dataset_loadable"] is True
    assert report["upstream_parse_passed"] is False
    assert report["upstream_tiny_smoke_passed"] is False
    assert report["launchable"] is False
    assert report["distributed_execution_tested"] is False
    assert report["target_semantics"] == "pure GKD forward_kl_topk OPD"
    assert (
        report["data_round_trip"]["train"][0]["sha256"]
        == hashlib.sha256(train.read_bytes()).hexdigest()
    )

    diagnosis = inspect_bridge_bundle(out)
    assert diagnosis["verdict"] == "ok"
    assert diagnosis["config_profile"]["profile"] == "verl-opd-v0.8-single-gpu-v1"
    assert diagnosis["reward_verification_level"] == "not_applicable_pure_opd"
    assert diagnosis["launchable"] is False


def test_teacher_adapter_requires_upstream_materialization(tmp_path: Path) -> None:
    from miniverl.bridge.contract import VERL_TAG
    from miniverl.bridge.export import export_verl_bundle

    run, _, _ = _opd_run(tmp_path, teacher_adapter=True)
    report = export_verl_bundle(run, target_verl=VERL_TAG, out=tmp_path / "bundle")

    assert report["teacher_artifact_loadable"] is False
    assert report["launchable"] is False
    assert "teacher adapter" in " ".join(report["launch_blockers"]).lower()
    identity = json.loads(
        (tmp_path / "bundle/teacher/teacher-model.json").read_text(encoding="utf-8")
    )
    assert identity["adapter"]["path"] == "teacher-adapter"
    assert identity["upstream_materialization_required"] is True


@pytest.mark.verl_conformance
def test_pure_opd_overrides_parse_under_the_pinned_upstream_config(tmp_path: Path) -> None:
    try:
        importlib.metadata.distribution("verl")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("official verl v0.8.0 is not installed")
    from miniverl.bridge.contract import VERL_TAG
    from miniverl.bridge.doctor import inspect_bridge_bundle
    from miniverl.bridge.export import export_verl_bundle

    run, _, _ = _opd_run(tmp_path)
    out = tmp_path / "bundle"
    export_verl_bundle(run, target_verl=VERL_TAG, out=out)
    diagnosis = inspect_bridge_bundle(out, require_verl=True)

    assert diagnosis["upstream_parse_passed"] is True, diagnosis["upstream_config_parse_recheck"]
    assert diagnosis["config_semantics_supported"] is True
    assert diagnosis["model_data_load_smoke_passed"] is False
    assert diagnosis["distributed_execution_tested"] is False
