"""Transactional miniVERL-defined Level-3 artifact-bundle export."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shlex
import shutil
import uuid
from pathlib import Path
from typing import Any

import yaml

from miniverl import __version__
from miniverl.bridge.contract import (
    BRIDGE_PROFILE,
    COMPATIBILITY_LEVELS,
    VERL_COMMIT,
    VERL_REPOSITORY,
    VERL_TAG,
    required_verl_text,
    validate_target_verl,
)
from miniverl.errors import ConfigError
from miniverl.utils.privacy import portable_payload
from miniverl.utils.runs import read_json, write_json, write_text

__all__ = ["export_verl_bundle"]

_MODEL_REQUIRED = ("adapter_config.json", "adapter_model.safetensors")
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
)
_UNSUPPORTED = (
    "optimizer state",
    "distributed RNG state",
    "FSDP native checkpoint",
    "Megatron native checkpoint",
    "Ray runtime state",
    "teacher cache as PPO reference cache",
    "PPO advantage or clipping semantics",
    "GRPO group semantics",
)
_REVISION = re.compile(r"^[0-9a-f]{40}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_source(run: Path) -> Path:
    candidates = (run / "model", run / "exported-adapter", run / "adapter")
    for candidate in candidates:
        if all((candidate / name).is_file() for name in _MODEL_REQUIRED):
            return candidate
    raise ConfigError(
        "run has no standard PEFT adapter",
        hint=(
            "place adapter_config.json and adapter_model.safetensors under <run>/model; "
            "export a miniVERL checkpoint with `miniverl export-adapter` first"
        ),
    )


def _copy_model(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    for name in _MODEL_REQUIRED:
        shutil.copy2(source / name, destination / name)
    copied_tokenizer = False
    for name in _TOKENIZER_FILES:
        path = source / name
        if path.is_file():
            shutil.copy2(path, destination / name)
            copied_tokenizer = True
    if not copied_tokenizer:
        raise ConfigError(
            f"standard adapter directory {source} has no tokenizer metadata",
            hint="copy tokenizer_config.json and the tokenizer vocabulary/snapshot metadata",
        )


def _adapter_contract(source: Path) -> dict[str, Any]:
    try:
        payload = json.loads((source / "adapter_config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read standard PEFT adapter config: {exc}") from exc
    if not isinstance(payload, dict) or str(payload.get("peft_type", "")).upper() != "LORA":
        raise ConfigError("verl bridge export requires a standard LoRA PEFT adapter")
    base_model = payload.get("base_model_name_or_path")
    revision = payload.get("revision")
    rank = payload.get("r")
    alpha = payload.get("lora_alpha")
    target_modules = payload.get("target_modules")
    if not isinstance(base_model, str) or not base_model.strip():
        raise ConfigError("adapter_config.json has no portable base_model_name_or_path")
    if not isinstance(revision, str) or not _REVISION.fullmatch(revision):
        raise ConfigError(
            "adapter_config.json has no immutable 40-character base revision",
            hint="re-export the adapter from a run whose base model revision is pinned",
        )
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ConfigError("adapter_config.json has no positive LoRA rank")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or alpha <= 0:
        raise ConfigError("adapter_config.json has no positive LoRA alpha")
    if isinstance(target_modules, str):
        targets = [target_modules]
    elif (
        isinstance(target_modules, list)
        and target_modules
        and all(isinstance(item, str) and item for item in target_modules)
    ):
        targets = target_modules
    else:
        raise ConfigError("adapter_config.json has no explicit LoRA target_modules")
    return {
        "base_model": base_model,
        "revision": revision,
        "rank": rank,
        "alpha": alpha,
        "target_modules": targets,
    }


def _get(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _source_config(run: Path) -> tuple[dict[str, Any], str | None]:
    for name in ("config.resolved.yaml", "config.original.yaml"):
        path = run / name
        if not path.is_file():
            continue
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigError(f"cannot read source-run {name}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ConfigError(f"source-run {name} must contain a YAML mapping")
        return payload, name
    return {}, None


def _source_run_values(config: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "run.name",
        "run.seed",
        "environment.name",
        "rollout.max_total_tokens",
        "rollout.max_new_tokens_per_turn",
        "train.learning_rate",
        "train.cycles",
        "train.save_every_cycles",
        "train.eval_every_cycles",
    )
    return {
        field: portable_payload(value)
        for field in fields
        if (value := _get(config, field)) is not None
    }


def _positive_source_number(config: dict[str, Any], path: str) -> float | None:
    value = _get(config, path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number > 0 else None


def _positive_source_integer(config: dict[str, Any], path: str) -> int | None:
    value = _get(config, path)
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _verl_overrides(
    adapter: dict[str, Any], source_config: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    response_length = _positive_source_integer(source_config, "rollout.max_new_tokens_per_turn")
    learning_rate = _positive_source_number(source_config, "train.learning_rate")
    seed = _get(source_config, "run.seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        seed = None
    run_name = _get(source_config, "run.name")
    if not isinstance(run_name, str) or not run_name.strip():
        run_name = None
    placeholders: list[dict[str, Any]] = [
        {
            "field": "data.max_prompt_length",
            "value": 512,
            "reason": (
                "miniVERL records max_total_tokens, not an equivalent standalone prompt limit"
            ),
            "source_run_intent": False,
        },
        {
            "field": "trainer.total_epochs",
            "value": 1,
            "reason": "miniVERL continuation cycles are not proven equivalent to verl epochs",
            "source_run_intent": False,
        },
        {
            "field": "trainer.save_freq",
            "value": 1,
            "reason": "miniVERL cycle frequency units are not proven equivalent",
            "source_run_intent": False,
        },
        {
            "field": "trainer.test_freq",
            "value": 1,
            "reason": "miniVERL cycle frequency units are not proven equivalent",
            "source_run_intent": False,
        },
    ]
    if response_length is None:
        response_length = 128
        placeholders.append(
            {
                "field": "data.max_response_length",
                "value": response_length,
                "reason": "no validated source-run response bound was available",
                "source_run_intent": False,
            }
        )
    if learning_rate is None:
        learning_rate = 1e-5
        placeholders.append(
            {
                "field": "actor_rollout_ref.actor.optim.lr",
                "value": learning_rate,
                "reason": "no validated source-run learning rate was available",
                "source_run_intent": False,
            }
        )
    if seed is None:
        seed = 1234
        placeholders.append(
            {
                "field": "data.seed",
                "value": seed,
                "reason": "no validated source-run seed was available",
                "source_run_intent": False,
            }
        )
    if run_name is None:
        run_name = "exported-profile"
        placeholders.append(
            {
                "field": "trainer.experiment_name",
                "value": run_name,
                "reason": "no validated source-run name was available",
                "source_run_intent": False,
            }
        )

    overrides = {
        "data": {
            "train_files": ["data/train.parquet"],
            "val_files": ["data/val.parquet"],
            "prompt_key": "prompt",
            "max_prompt_length": 512,
            "max_response_length": response_length,
            "seed": seed,
        },
        "actor_rollout_ref": {
            "model": {
                "path": "model/base",
                "enable_gradient_checkpointing": True,
                "lora_rank": adapter["rank"],
                "lora_alpha": adapter["alpha"],
                "target_modules": adapter["target_modules"],
                "lora_adapter_path": "model",
            },
            "actor": {"optim": {"lr": learning_rate}},
        },
        "custom_reward_function": {
            "path": "reward/reward_or_verifier_scaffold.py",
            "name": "compute_score",
        },
        "trainer": {
            "save_freq": 1,
            "test_freq": 1,
            "project_name": "miniverl-bridge",
            "experiment_name": run_name,
            "total_epochs": 1,
        },
    }
    return overrides, placeholders


def _launch_script(adapter: dict[str, Any], overrides: dict[str, Any]) -> str:
    base_model = shlex.quote(adapter["base_model"])
    revision = shlex.quote(adapter["revision"])
    targets = shlex.quote(json.dumps(adapter["target_modules"], separators=(",", ":")))
    data = overrides["data"]
    trainer = overrides["trainer"]
    learning_rate = overrides["actor_rollout_ref"]["actor"]["optim"]["lr"]
    return f"""#!/usr/bin/env bash
set -euo pipefail

# Generated for {VERL_REPOSITORY}@{VERL_COMMIT} ({VERL_TAG}).
# Review the reward scaffold before launching domain-specific work.
BUNDLE_ROOT="$(cd "$(dirname "${{BASH_SOURCE[0]}}")/.." && pwd)"
if [[ ! -f "$BUNDLE_ROOT/model/base/config.json" ]]; then
  echo "materialize the exact base snapshot before launch:" >&2
  echo "  hf download {base_model} --revision {revision} --local-dir \"$BUNDLE_ROOT/model/base\"" >&2
  exit 2
fi
if grep -q "complete and test reward_or_verifier_scaffold" "$BUNDLE_ROOT/reward/reward_or_verifier_scaffold.py"; then
  echo "complete and test the fail-closed reward scaffold before launch" >&2
  exit 2
fi
python -m verl.trainer.main_ppo \\
  data.train_files="['$BUNDLE_ROOT/data/train.parquet']" \\
  data.val_files="['$BUNDLE_ROOT/data/val.parquet']" \\
  data.prompt_key=prompt \\
  data.max_prompt_length={data["max_prompt_length"]} \\
  data.max_response_length={data["max_response_length"]} \\
  actor_rollout_ref.model.path="$BUNDLE_ROOT/model/base" \\
  actor_rollout_ref.model.enable_gradient_checkpointing=true \\
  actor_rollout_ref.model.lora_rank={adapter["rank"]} \\
  actor_rollout_ref.model.lora_alpha={adapter["alpha"]} \\
  actor_rollout_ref.model.target_modules={targets} \\
  actor_rollout_ref.model.lora_adapter_path="$BUNDLE_ROOT/model" \\
  actor_rollout_ref.actor.optim.lr={learning_rate} \\
  custom_reward_function.path="$BUNDLE_ROOT/reward/reward_or_verifier_scaffold.py" \\
  custom_reward_function.name=compute_score \\
  trainer.total_epochs={trainer["total_epochs"]} \\
  trainer.save_freq={trainer["save_freq"]} trainer.test_freq={trainer["test_freq"]}
"""


def _reward_scaffold() -> str:
    return '''"""Fail-closed reward scaffold generated by miniVERL.

Replace the body only after implementing and testing domain-specific scoring.
Importing this module is side-effect free.
"""

from __future__ import annotations

from typing import Any


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> float:
    """Return a domain reward after the maintainer supplies verified logic."""
    del data_source, solution_str, ground_truth, extra_info
    raise RuntimeError(
        "complete and test reward_or_verifier_scaffold.compute_score before a verl launch"
    )
'''


def _bundle_readme() -> str:
    return f"""# miniVERL → verl scale-out bundle

This bundle exchanges standard Hugging Face, PEFT, safetensors and Parquet
artifacts with the documented `{BRIDGE_PROFILE}` subset of
[`verl {VERL_TAG}`]({VERL_REPOSITORY}/tree/{VERL_TAG}), pinned to
`{VERL_COMMIT}`.

Run `miniverl bridge doctor . --require-verl` after installing the exact pin.
By default the doctor inspects portable metadata only: it reports
`dataset_content_privacy: not_inspected` and `model_weight_privacy:
not_inspected`, neither of which means `passed`. Add `--scan-dataset-text` for
a bounded heuristic scan of string-like Parquet fields, and
`--require-tokenizer-load` to demand a real local tokenizer load and identity
check instead of accepting `metadata_only`.

The generated reward scaffold fails closed until domain logic is supplied and
tested. Materialize `model/base` from the exact identity in
`model/base-model.json` before launch; the adapter directory alone is not a
base-model checkpoint. `recipe/launch.template.sh` is not launch-ready. This is
a PPO/reward scaffold, not an executable continuation of miniVERL OPD
semantics. The bundle has **not** executed a distributed job. It does not convert
optimizer state, distributed RNG, FSDP/Megatron checkpoints, Ray state, or a
miniVERL teacher cache into PPO reference log-probabilities.
"""


def _write_hashes(root: Path) -> None:
    lines = []
    checksum = root / "provenance" / "SHA256SUMS"
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item != checksum):
        lines.append(f"{_sha256(path)}  {path.relative_to(root).as_posix()}")
    write_text(checksum, "\n".join(lines) + "\n")


def export_verl_bundle(
    run: str | Path,
    *,
    target_verl: str,
    out: str | Path,
) -> dict[str, Any]:
    """Export one immutable, self-checking artifact bundle."""
    validate_target_verl(target_verl)
    run_path = Path(run)
    manifest_path = run_path / "manifest.json"
    if not run_path.is_dir() or not manifest_path.is_file():
        raise ConfigError(f"miniVERL run is missing manifest.json: {run_path}")
    model_source = _model_source(run_path)
    adapter = _adapter_contract(model_source)
    source_config, source_config_file = _source_config(run_path)
    source_run_values = _source_run_values(source_config)
    overrides, placeholder_defaults = _verl_overrides(adapter, source_config)
    data_source = run_path / "data"
    for split in ("train.parquet", "val.parquet"):
        if not (data_source / split).is_file():
            raise ConfigError(f"run data is missing {split}: {data_source / split}")

    destination = Path(out)
    if destination.exists():
        raise ConfigError(
            f"export destination already exists: {destination}",
            hint="choose a new directory so an earlier verified bundle cannot be mixed in",
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    report: dict[str, Any] = {
        "schema_version": 1,
        "compatibility_level": 3,
        "compatibility_level_name": "miniVERL-defined compatibility Level 3",
        "compatibility_levels": COMPATIBILITY_LEVELS,
        "profile": BRIDGE_PROFILE,
        "target_verl": {
            "repository": VERL_REPOSITORY,
            "tag": VERL_TAG,
            "commit": VERL_COMMIT,
        },
        "miniverl_version": __version__,
        "base_model": {
            "model_id": adapter["base_model"],
            "revision": adapter["revision"],
            "materialized_path": "model/base",
            "status": "not bundled; materialize the exact snapshot before launch",
        },
        "supported_artifacts": [
            "Hugging Face model/config/tokenizer",
            "standard PEFT adapter",
            "safetensors",
            "Parquet prompt data",
            "result/provenance manifests",
        ],
        "unsupported_semantics": list(_UNSUPPORTED),
        "source_config_file": source_config_file,
        "source_run_values": source_run_values,
        "placeholder_defaults": placeholder_defaults,
        "artifact_bundle_complete": True,
        "upstream_config_parse_passed": False,
        "model_data_load_smoke_passed": False,
        "reward_implementation_complete": False,
        "launchable": False,
        "distributed_execution_tested": False,
        "algorithm_semantic_parity": False,
        "target_semantics": "PPO/reward scaffold",
        "local_smoke_status": "generated; run bridge doctor",
        "distributed_execution_status": "not tested",
        "claim": (
            "A single-GPU runtime for a documented subset of verl-style post-training workflows."
        ),
    }
    try:
        _copy_model(model_source, temporary / "model")
        write_json(
            temporary / "model" / "base-model.json",
            {
                "materialized_path": "model/base",
                "model_id": adapter["base_model"],
                "revision": adapter["revision"],
                "status": "not bundled; materialize the exact snapshot before launch",
            },
        )
        data_destination = temporary / "data"
        data_destination.mkdir()
        shutil.copy2(data_source / "train.parquet", data_destination / "train.parquet")
        shutil.copy2(data_source / "val.parquet", data_destination / "val.parquet")

        recipe = temporary / "recipe"
        recipe.mkdir()
        write_text(
            recipe / "verl-overrides.yaml",
            yaml.safe_dump(overrides, sort_keys=False, allow_unicode=True, width=100),
        )
        write_text(recipe / "launch.template.sh", _launch_script(adapter, overrides))
        write_text(recipe / "REQUIRED_VERL.txt", required_verl_text())

        reward = temporary / "reward"
        reward.mkdir()
        write_text(reward / "reward_or_verifier_scaffold.py", _reward_scaffold())

        provenance = temporary / "provenance"
        provenance.mkdir()
        write_json(
            provenance / "miniverl-manifest.json", portable_payload(read_json(manifest_path))
        )
        result_path = run_path / "result.json"
        source_result = (
            portable_payload(read_json(result_path))
            if result_path.is_file()
            else {"status": "not present in source run"}
        )
        write_json(provenance / "source-result.json", source_result)
        write_json(provenance / "compatibility-report.json", report)
        write_text(temporary / "README.md", _bundle_readme())
        _write_hashes(temporary)
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report
