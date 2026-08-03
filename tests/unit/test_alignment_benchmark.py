"""Frozen Alignment Lab benchmark construction tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from miniverl.alignment.benchmark import (
    ALIGNMENT_BENCHMARK_METHODS,
    build_alignment_benchmark_config,
    load_alignment_preregistration,
)
from miniverl.config import RunConfig
from miniverl.errors import ConfigError
from miniverl.utils.runs import canonical_json


def _preregistration(path: Path) -> tuple[dict[str, object], str]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "name": "alignment-lab-v1",
        "status": "preregistered_before_final_test",
        "execution": {
            "methods": list(ALIGNMENT_BENCHMARK_METHODS),
            "student_seeds": [3, 5, 7],
            "optimizer_updates": 2,
            "rollouts_per_update": 2,
            "trajectory_batch_size": 2,
            "learning_rate": 0.001,
        },
        "final_test": {"read_count": 1, "tasks": 6},
        "teacher_selection": {"eval_tasks": 4},
        "task_set": {"split_seed": 11},
        "starting_checkpoint": {"content_sha256": "a" * 64},
        "policy_artifact": {
            "id": "policy",
            "revision": "v1",
            "sha256": "b" * 64,
            "license": "Apache-2.0",
        },
        "verifier_gate": {
            "version": "gate-v1",
            "signal": "policy_critical_span",
            "decision_scope": "span",
            "threshold": 1.0,
            "calibrated_on": "eval",
            "frozen_before_test": True,
        },
        "dpo": {
            "trl_version": "1.8.0",
            "exact_config_sha256_by_seed": {
                "3": "c" * 64,
                "5": "1" * 64,
                "7": "2" * 64,
            },
            "dataset_sha256": "d" * 64,
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return payload, hashlib.sha256(path.read_bytes()).hexdigest()


def _base(tmp_path: Path) -> RunConfig:
    return RunConfig.from_mapping(
        {
            "models": {
                "backend": "toy",
                "device": "cpu",
                "student": {
                    "model_id": "student",
                    "quantization": "none",
                    "lora": {"enabled": False},
                },
                "teacher": {
                    "model_id": "teacher",
                    "quantization": "none",
                    "mode": "privileged_context",
                    "toy_pretrain_steps": 1,
                },
            },
            "environment": {
                "name": "tool_policy",
                "train_tasks": 12,
                "eval_tasks": 6,
                "test_tasks": 6,
                "params": {"protocol_version": "v2"},
            },
            "train": {"sft_warmup_cycles": 1},
            "cache": {"dtype": "float32"},
            "alignment": {
                "method": "standard_opd",
                "teacher_mode": "policy_conditioned",
                "policy": {"id": "policy", "revision": "v1", "sha256": "b" * 64},
            },
            "run": {"output_dir": str(tmp_path)},
        }
    )


def test_preregistration_digest_and_six_arm_config_contract(tmp_path: Path) -> None:
    path = tmp_path / "prereg.yaml"
    payload, digest = _preregistration(path)
    loaded = load_alignment_preregistration(path, expected_sha256=digest)
    checkpoint = tmp_path / "checkpoint"
    for method in ALIGNMENT_BENCHMARK_METHODS:
        if method == "dpo":
            adapter = tmp_path / "dpo-seed-3"
            adapter.mkdir()
            manifest = {
                "method": "dpo",
                "trl_version": "1.8.0",
                "exact_config_sha256": "c" * 64,
                "dataset": {"id": "preferences", "revision": "v1", "sha256": "d" * 64},
                "reference": {"adapter_weights_sha256": "e" * 64},
                "adapter": {"weights_sha256": "f" * 64},
                "seed": 3,
            }
            (adapter / "dpo_manifest.json").write_text(canonical_json(manifest), encoding="utf-8")
            dpo_manifest = adapter / "dpo_manifest.json"
        else:
            dpo_manifest = None
        config = build_alignment_benchmark_config(
            _base(tmp_path),
            loaded,
            method=method,
            seed=3,
            split="test",
            starting_checkpoint=checkpoint,
            dpo_manifest=dpo_manifest,
        )
        assert config.alignment is not None
        assert config.alignment.method.value == method
        assert config.eval.tasks == 6
        assert config.eval.baseline_enabled is False
        assert config.train.sft_warmup_cycles == 0

    assert payload["status"] == "preregistered_before_final_test"


def test_preregistration_rejects_digest_drift_and_unregistered_seed(tmp_path: Path) -> None:
    path = tmp_path / "prereg.yaml"
    loaded, _ = _preregistration(path)
    with pytest.raises(ConfigError, match="digest mismatch"):
        load_alignment_preregistration(path, expected_sha256="0" * 64)
    with pytest.raises(ConfigError, match="not preregistered"):
        build_alignment_benchmark_config(
            _base(tmp_path),
            loaded,
            method="standard_opd",
            seed=99,
            split="eval",
            starting_checkpoint=tmp_path / "checkpoint",
        )


def test_dpo_config_digest_is_path_independent_and_seed_specific() -> None:
    from miniverl.alignment.dpo import build_dpo_training_config, dpo_config_digest

    first = build_dpo_training_config(seed=3, max_steps=4, learning_rate=5e-5, beta=0.1)
    repeated = build_dpo_training_config(seed=3, max_steps=4, learning_rate=5e-5, beta=0.1)
    other_seed = build_dpo_training_config(seed=5, max_steps=4, learning_rate=5e-5, beta=0.1)
    assert first["output_dir"] == "<OUTPUT>"
    assert dpo_config_digest(first) == dpo_config_digest(repeated)
    assert dpo_config_digest(first) != dpo_config_digest(other_seed)
