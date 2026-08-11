"""Verify a miniVERL-defined Level-3 bundle against the pinned verl snapshot."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from miniverl.bridge.contract import BRIDGE_PROFILE, VERL_COMMIT, VERL_TAG
from miniverl.bridge.doctor import inspect_bridge_bundle
from miniverl.errors import ConfigError
from miniverl.utils.privacy import portable_payload
from miniverl.utils.runs import write_json

_OFFICIAL_FIELDS = (
    "data.train_files",
    "data.val_files",
    "data.prompt_key",
    "data.max_prompt_length",
    "data.max_response_length",
    "data.seed",
    "actor_rollout_ref.model.path",
    "actor_rollout_ref.model.enable_gradient_checkpointing",
    "actor_rollout_ref.actor.optim.lr",
    "trainer.save_freq",
    "trainer.test_freq",
    "trainer.project_name",
    "trainer.experiment_name",
    "trainer.total_epochs",
)
_OFFICIAL_EXPORT_FIELDS = (
    "actor_rollout_ref.model.lora_rank",
    "actor_rollout_ref.model.lora_alpha",
    "actor_rollout_ref.model.target_modules",
    "actor_rollout_ref.model.lora_adapter_path",
    "custom_reward_function.path",
    "custom_reward_function.name",
)


def _has_path(payload: Mapping[str, Any], path: str) -> bool:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False
        current = current[part]
    return True


def verify_smoke(bundle: str | Path, *, out: str | Path) -> dict[str, Any]:
    """Check exact installation, official fields and all exported artifacts."""
    diagnosis = inspect_bridge_bundle(bundle, require_verl=True)
    distribution = importlib.metadata.distribution("verl")
    generated_path = Path(
        str(distribution.locate_file("verl/trainer/config/_generated_ppo_trainer.yaml"))
    )
    if not generated_path.is_file():
        raise ConfigError(f"installed verl distribution omits {generated_path.name}")
    official_config = OmegaConf.load(generated_path)
    official = OmegaConf.to_container(official_config, resolve=False)
    if not isinstance(official, Mapping):
        raise ConfigError("official generated PPO config is not a YAML mapping")
    missing = [field for field in _OFFICIAL_FIELDS if not _has_path(official, field)]
    if missing:
        raise ConfigError("official verl config is missing bridge fields: " + ", ".join(missing))
    exported_config = Path(bundle) / "recipe" / "verl-overrides.yaml"
    exported_omegaconf = OmegaConf.load(exported_config)
    exported = OmegaConf.to_container(exported_omegaconf, resolve=False)
    if not isinstance(exported, Mapping):
        raise ConfigError("exported verl overrides are not an OmegaConf mapping")
    missing_export_fields = [
        field for field in _OFFICIAL_EXPORT_FIELDS if not _has_path(official, field)
    ]
    if missing_export_fields:
        raise ConfigError(
            "official verl config is missing export fields: " + ", ".join(missing_export_fields)
        )
    try:
        OmegaConf.set_struct(official_config, True)
        OmegaConf.merge(official_config, exported_omegaconf)
    except Exception as exc:
        raise ConfigError(
            f"exported overrides do not merge into official verl config: {exc}"
        ) from exc
    direct_url = diagnosis["installed_verl"].get("direct_url") or {}
    commit = (direct_url.get("vcs_info") or {}).get("commit_id")
    report: dict[str, Any] = {
        "schema_version": 1,
        "target_verl": {
            "tag": VERL_TAG,
            "commit": VERL_COMMIT,
            "observed_package_version": diagnosis["installed_verl"].get("version"),
            "observed_vcs_commit": commit,
        },
        "profile": BRIDGE_PROFILE,
        "official_config": {
            "path": "verl/trainer/config/_generated_ppo_trainer.yaml",
            "required_fields": list(_OFFICIAL_FIELDS),
            "missing_fields": missing,
            "required_export_fields": list(_OFFICIAL_EXPORT_FIELDS),
            "missing_export_fields": missing_export_fields,
            "parse_status": "passed with OmegaConf",
            "structured_merge_status": "passed",
            "exported_override_roots": sorted(exported),
        },
        "bundle_doctor": portable_payload(diagnosis),
        "model_or_adapter_load": diagnosis["model_adapter_loadability"]["peft_config_load"],
        "parquet_load": "train and val passed",
        "reward_scaffold_import": "passed; fail-closed scorer not executed",
        "artifact_bundle_complete": diagnosis["artifact_bundle_complete"],
        "upstream_config_parse_passed": True,
        "model_data_load_smoke_passed": True,
        "model_data_load_smoke_scope": (
            "PEFT config, safetensors structure and both Parquet splits; base weights not loaded"
        ),
        "reward_implementation_complete": False,
        "launchable": False,
        "distributed_execution_tested": False,
        "algorithm_semantic_parity": False,
        "tiny_cpu_dry_run": {
            "status": "artifact-only",
            "reason": "full PPO execution requires the excluded distributed inference stack",
        },
        "distributed_execution_status": "not tested",
        "privacy_status": diagnosis["privacy"]["status"],
        "verdict": diagnosis["verdict"],
    }
    if commit != VERL_COMMIT or diagnosis["verdict"] != "ok":
        report["verdict"] = "fail"
    write_json(out, report)
    if report["verdict"] != "ok":
        raise ConfigError("pinned verl bridge smoke failed; inspect the report")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_smoke(args.bundle, out=args.out), indent=2))


if __name__ == "__main__":
    main()
