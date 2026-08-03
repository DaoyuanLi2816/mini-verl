"""Preregistered one-GPU Alignment Lab benchmark configuration builder."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

import yaml

from miniverl.alignment.schema import AlignmentMethod
from miniverl.config import RunConfig
from miniverl.errors import ConfigError
from miniverl.utils.runs import read_json

__all__ = [
    "ALIGNMENT_BENCHMARK_METHODS",
    "build_alignment_benchmark_config",
    "load_alignment_preregistration",
]

ALIGNMENT_BENCHMARK_METHODS = tuple(method.value for method in AlignmentMethod)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_alignment_preregistration(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Load the frozen benchmark contract and optionally enforce its byte digest."""
    source = Path(path)
    if not source.is_file():
        raise ConfigError(f"alignment preregistration not found: {source}")
    actual = _sha256(source)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ConfigError(
            "alignment preregistration digest mismatch",
            hint=f"declared {expected_sha256}, measured {actual}",
        )
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigError("alignment preregistration must be a YAML mapping")
    if payload.get("schema_version") != 1 or payload.get("name") != "alignment-lab-v1":
        raise ConfigError("unsupported alignment preregistration identity")
    if payload.get("status") != "preregistered_before_final_test":
        raise ConfigError("alignment preregistration is not frozen for final testing")
    execution = payload.get("execution")
    if not isinstance(execution, dict):
        raise ConfigError("alignment preregistration has no execution contract")
    methods = execution.get("methods")
    seeds = execution.get("student_seeds")
    if methods != list(ALIGNMENT_BENCHMARK_METHODS):
        raise ConfigError("alignment preregistration must contain all six ordered baselines")
    if not isinstance(seeds, list) or len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ConfigError("alignment preregistration requires at least three unique seeds")
    return payload


def _artifact(
    *,
    identity: str,
    revision: str,
    sha256: str,
    license_name: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"id": identity, "revision": revision, "sha256": sha256}
    if license_name is not None:
        payload["license"] = license_name
    return payload


def _dpo_provenance(manifest_path: Path, preregistration: dict[str, Any]) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("method") != "dpo":
        raise ConfigError(f"invalid DPO manifest: {manifest_path}")
    dpo = preregistration["dpo"]
    if manifest.get("trl_version") != dpo["trl_version"]:
        raise ConfigError("DPO manifest uses a different TRL version than preregistered")
    expected_configs = dpo["exact_config_sha256_by_seed"]
    expected_config = expected_configs.get(str(manifest.get("seed")))
    if manifest.get("exact_config_sha256") != expected_config:
        raise ConfigError("DPO manifest config digest differs from preregistration")
    if (manifest.get("dataset") or {}).get("sha256") != dpo["dataset_sha256"]:
        raise ConfigError("DPO preference dataset digest differs from preregistration")
    adapter = manifest.get("adapter") or {}
    reference = manifest.get("reference") or {}
    return {
        "trl_version": manifest["trl_version"],
        "exact_config_sha256": manifest["exact_config_sha256"],
        "reference_model": _artifact(
            identity="common-sft-adapter-reference",
            revision="alignment-lab-v1",
            sha256=reference["adapter_weights_sha256"],
        ),
        "dataset": _artifact(
            identity=manifest["dataset"]["id"],
            revision=manifest["dataset"]["revision"],
            sha256=manifest["dataset"]["sha256"],
            license_name="Apache-2.0",
        ),
        "checkpoint": _artifact(
            identity="common-sft-checkpoint",
            revision="alignment-lab-v1",
            sha256=preregistration["starting_checkpoint"]["content_sha256"],
        ),
        "adapter": _artifact(
            identity=manifest_path.parent.name,
            revision=f"seed-{manifest['seed']}",
            sha256=adapter["weights_sha256"],
        ),
    }


def build_alignment_benchmark_config(
    base: RunConfig,
    preregistration: dict[str, Any],
    *,
    method: str,
    seed: int,
    split: Literal["eval", "test"],
    starting_checkpoint: str | Path,
    dpo_manifest: str | Path | None = None,
) -> RunConfig:
    """Build one matched arm without exposing post-freeze budget knobs."""
    execution = preregistration["execution"]
    if method not in execution["methods"]:
        raise ConfigError(f"method {method!r} is not preregistered")
    if seed not in execution["student_seeds"]:
        raise ConfigError(f"student seed {seed} is not preregistered")
    if split == "test" and preregistration["final_test"]["read_count"] != 1:
        raise ConfigError("final-test contract must declare exactly one read")

    config = base.model_dump(mode="json")
    config["run"].update(
        {
            "name": f"alignment-lab-{method}-seed-{seed}",
            "mode": "opd",
            "seed": seed,
            "run_id": None,
            "tags": ["alignment-lab-v1", method, split],
        }
    )
    config["train"].update(
        {
            "cycles": execution["optimizer_updates"],
            "rollouts_per_cycle": execution["rollouts_per_update"],
            "gradient_accumulation_steps": execution["rollouts_per_update"],
            "trajectory_batch_size": execution["trajectory_batch_size"],
            "sft_warmup_cycles": 0,
            "learning_rate": execution["learning_rate"],
            "save_every_cycles": 0,
            "eval_every_cycles": 0,
            "max_selected_training_tokens": None,
            "max_wall_seconds": None,
        }
    )
    config["eval"].update(
        {
            "enabled": True,
            "baseline_enabled": False,
            "split": split,
            "tasks": preregistration["final_test"]["tasks"]
            if split == "test"
            else preregistration["teacher_selection"]["eval_tasks"],
            "temperature": 0.0,
            "seed": preregistration["task_set"]["split_seed"],
        }
    )
    config["alignment"] = {
        "method": method,
        "teacher_mode": "policy_conditioned",
        "starting_sft_checkpoint": str(Path(starting_checkpoint).resolve()),
        "starting_sft_checkpoint_sha256": preregistration["starting_checkpoint"]["content_sha256"],
        "policy": preregistration["policy_artifact"],
        "evaluation_adapters": base.alignment.evaluation_adapters if base.alignment else [],
        "limitations": [
            "One model family, one deterministic sandbox policy suite and one measured GPU.",
            "Three seeds describe observed variation; they do not establish a population claim.",
        ],
    }
    config["selection"] = {"selector": "all_model_tokens"}
    config["offline_kd"] = {}
    config["cache"].update(
        {
            "strict_policy_version": True,
            "reuse_across_policy_versions": False,
        }
    )

    if method in {"sft_checkpoint", "continued_sft", "dpo"}:
        config["run"]["mode"] = "sft"
        config["alignment"]["teacher_mode"] = None
    if method == "sft_checkpoint":
        config["train"]["cycles"] = 0
    elif method == "continued_sft":
        config["train"]["cycles"] = execution["optimizer_updates"]
    elif method == "dpo":
        if dpo_manifest is None:
            raise ConfigError("the DPO arm requires its immutable dpo_manifest.json")
        manifest_path = Path(dpo_manifest).resolve()
        config["train"]["cycles"] = 0
        config["alignment"]["dpo"] = _dpo_provenance(manifest_path, preregistration)
        config["alignment"]["dpo_adapter_path"] = str(manifest_path.parent)
    elif method == "offline_distillation":
        config["run"]["mode"] = "offline_kd"
        config["cache"]["reuse_across_policy_versions"] = True
        config["offline_kd"] = {
            "trajectory_source": "frozen_student",
            "collection_seed": seed,
            "collection_tasks": execution["rollouts_per_update"],
        }
    elif method == "verifier_gated_opd":
        gate = preregistration["verifier_gate"]
        config["selection"] = {"selector": "verifier_gated", "gate": gate}
        config["alignment"]["gate"] = gate

    return RunConfig.from_mapping(config)
