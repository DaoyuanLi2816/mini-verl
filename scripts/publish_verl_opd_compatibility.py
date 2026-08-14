"""Publish compiler-bound compatibility reports for the pinned OPD profile."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from miniverl.bridge.contract import VERL_COMMIT, VERL_REPOSITORY, VERL_TAG
from miniverl.bridge.opd_runtime import build_system_plan, compile_native_run_config
from miniverl.bridge.opd_v08 import compile_verl_opd_v08, load_verl_opd_v08
from miniverl.errors import ConfigError
from miniverl.utils.runs import canonical_json


def build_matrix(source: Path) -> dict[str, Any]:
    compiled = load_verl_opd_v08(source)
    fields = [item.model_dump(mode="json") for item in compiled.compatibility]
    return {
        "schema_version": 2,
        "profile": compiled.profile,
        "upstream": compiled.upstream,
        "source_fixture": source.as_posix(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "compiled_digest": compiled.compiled_digest,
        "executable": compiled.executable,
        "field_count": len(fields),
        "classification_counts": dict(
            sorted(Counter(item["classification"] for item in fields).items())
        ),
        "reinterpretation_acceptance": compiled.reinterpretation_acceptance,
        "fields": fields,
        "scope": (
            "config conformance for one documented local pure-OPD profile; "
            "not full verl compatibility or distributed execution"
        ),
    }


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(item, path))
        return result
    return {prefix: value}


def _set_path(payload: dict[str, Any], path: str, value: Any) -> None:
    current = payload
    parts = path.split(".")
    for part in parts[:-1]:
        current = current[part]
    current[parts[-1]] = value


_FIXED_CONSTRAINTS = {
    "actor_rollout_ref.actor.loss_agg_mode",
    "actor_rollout_ref.actor.use_kl_loss",
    "actor_rollout_ref.rollout.n",
    "actor_rollout_ref.rollout.tensor_model_parallel_size",
    "algorithm.use_kl_in_reward",
    "distillation.enabled",
    "distillation.n_gpus_per_node",
    "distillation.nnodes",
    "distillation.teacher_models.teacher_model.num_replicas",
    "distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size",
    "distillation.teacher_models.teacher_model.inference.data_parallel_size",
    "distillation.teacher_models.teacher_model.inference.pipeline_model_parallel_size",
    "distillation.distillation_loss.loss_mode",
    "distillation.distillation_loss.use_task_rewards",
    "distillation.distillation_loss.use_policy_gradient",
    "trainer.n_gpus_per_node",
    "trainer.nnodes",
}


def _alternate(path: str, value: Any) -> Any:
    special: dict[str, Any] = {
        "data.train_files": ["data/train-alternate.parquet"],
        "data.val_files": ["data/val-alternate.parquet"],
        "data.prompt_key": "messages",
        "data.max_prompt_length": 255,
        "data.max_response_length": 63,
        "data.truncation": "left",
        "actor_rollout_ref.model.path": "Qwen/Qwen3-0.7B",
        "actor_rollout_ref.model.target_modules": ["q_proj", "k_proj", "v_proj"],
        "actor_rollout_ref.rollout.name": "sglang",
        "actor_rollout_ref.rollout.max_num_seqs": 1,
        "distillation.teacher_models.teacher_model.model_path": "Qwen/Qwen3-1.8B",
        "distillation.teacher_models.teacher_model.inference.name": "sglang",
        "distillation.teacher_models.teacher_model.inference.dtype": "float16",
        "distillation.distillation_loss.loss_max_clamp": 2.0,
        "miniverl.runtime.mode": "dual_model_resident",
        "miniverl.actor_runtime.dtype": "float16",
        "miniverl.actor_runtime.quantization": "int8",
        "miniverl.actor_runtime.attn_implementation": "eager",
        "miniverl.teacher_runtime.quantization": "int8",
        "miniverl.teacher_runtime.attn_implementation": "eager",
    }
    if path in special:
        return special[path]
    if path == "actor_rollout_ref.actor.optim.lr":
        return float(value) * 2
    if path in _FIXED_CONSTRAINTS:
        if isinstance(value, bool):
            return not value
        if isinstance(value, int):
            return value + 1
        return f"unsupported-{value}"
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        if path.endswith("top_p"):
            return 0.9 if value != 0.9 else 0.8
        return value + (0.1 if value == 0 else abs(value) * 0.1)
    if isinstance(value, str):
        return value + "-alternate"
    if isinstance(value, list):
        return [*value, "alternate"]
    if value is None:
        return 1.0
    raise TypeError(f"no alternate value for {path}: {value!r}")


def _effect_projection(compiled: Any) -> dict[str, Any]:
    system = build_system_plan(compiled)
    native = compile_native_run_config(compiled, system_plan=system)
    return {
        "system_plan": system.model_dump(mode="json", exclude={"compiled_digest", "overrides"}),
        "native_run_config": native.model_dump(mode="json"),
    }


def build_field_effects(source: Path) -> dict[str, Any]:
    """Prove every executable non-informational compatibility claim has an effect."""
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a mapping in {source}")
    base = load_verl_opd_v08(source)
    base_projection = _effect_projection(base)
    base_flat = _flatten(base_projection)
    critical = {
        "system_plan.loss.mode",
        "system_plan.loss.top_k",
        "system_plan.student.model_id",
        "system_plan.teacher.model_id",
        "native_run_config.run.mode",
        "native_run_config.train.cycles",
        "native_run_config.train.rollouts_per_cycle",
    }
    records: list[dict[str, Any]] = []
    for entry in base.compatibility:
        if not entry.executable or entry.classification in {"informational_only", "unsupported"}:
            continue
        mutated = copy.deepcopy(payload)
        alternate = _alternate(entry.upstream_field, entry.source_value)
        _set_path(mutated, entry.upstream_field, alternate)
        try:
            changed_compiled = compile_verl_opd_v08(mutated)
            changed_projection = _effect_projection(changed_compiled)
        except ConfigError as exc:
            if entry.upstream_field not in _FIXED_CONSTRAINTS:
                raise AssertionError(
                    f"{entry.upstream_field} has no valid field-effect mutation: {exc}"
                ) from exc
            records.append(
                {
                    "upstream_field": entry.upstream_field,
                    "source_values": [entry.source_value, alternate],
                    "classification": entry.classification,
                    "declared_local_target": entry.local_target,
                    "observed_changed_paths": ["validation.executable_constraint"],
                    "observed_unchanged_critical_paths": sorted(critical),
                    "test_status": "fixed_constraint_rejected_alternate",
                }
            )
            continue
        changed_flat = _flatten(changed_projection)
        changed_paths = sorted(
            path
            for path in set(base_flat) | set(changed_flat)
            if base_flat.get(path) != changed_flat.get(path)
        )
        if not changed_paths:
            raise AssertionError(
                f"{entry.upstream_field} declares {entry.local_target} but has no effect"
            )
        unchanged = sorted(
            path for path in critical if base_flat.get(path) == changed_flat.get(path)
        )
        records.append(
            {
                "upstream_field": entry.upstream_field,
                "source_values": [entry.source_value, alternate],
                "classification": entry.classification,
                "declared_local_target": entry.local_target,
                "observed_changed_paths": changed_paths,
                "observed_unchanged_critical_paths": unchanged,
                "test_status": "effect_observed",
            }
        )
    expected = sum(
        item.executable
        and item.classification not in {"informational_only", "unsupported"}
        and item.local_target is not None
        for item in base.compatibility
    )
    if len(records) != expected:
        raise AssertionError(f"field-effect coverage is {len(records)}/{expected}, expected 100%")
    return {
        "schema_version": 1,
        "profile": base.profile,
        "source_fixture": source.as_posix(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "coverage": {"covered": len(records), "eligible": expected, "percent": 100.0},
        "fields": records,
    }


def build_official_report(root: Path) -> dict[str, object]:
    fixture = root / "tests/fixtures/verl/opd-v0.8-official-example-fields.yaml"
    plan = load_verl_opd_v08(fixture, require_executable=False)
    counts = Counter(item.classification for item in plan.compatibility)
    return {
        "schema_version": 1,
        "profile": plan.profile,
        "provenance": {
            "repository": VERL_REPOSITORY,
            "tag": VERL_TAG,
            "commit": VERL_COMMIT,
            "source_path": "examples/on_policy_distillation_trainer/run_qwen3_8b_fsdp.sh",
            "license": "Apache-2.0",
            "fixture": fixture.relative_to(root).as_posix(),
        },
        "official_example_fields_total": len(plan.compatibility),
        "classifications": {
            name: counts.get(name, 0)
            for name in (
                "exact",
                "semantically_conformant",
                "locally_reinterpreted",
                "derived",
                "informational_only",
                "unsupported",
            )
        },
        "executable": plan.executable,
        "unsupported_fields": [
            item.upstream_field
            for item in plan.compatibility
            if item.classification == "unsupported"
        ],
        "claim": (
            "Every fixture leaf is classified; coverage does not imply that "
            "policy-gradient, FSDP, or distributed execution is supported."
        ),
    }


def render_official_report(root: Path) -> str:
    return json.dumps(build_official_report(root), indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("examples/verl-opd-v0.8-single-gpu.yaml")
    )
    parser.add_argument(
        "--field-effects-out",
        type=Path,
        default=Path("docs/generated/verl-opd-v0.8-field-effects.json"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("docs/generated/verl-opd-v0.8-compatibility.json")
    )
    parser.add_argument(
        "--official-out",
        type=Path,
        default=Path("docs/generated/verl-opd-v08-official-fields.json"),
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    rendered = canonical_json(build_matrix(args.source))
    official = render_official_report(root)
    field_effects = canonical_json(build_field_effects(args.source))
    if args.check:
        current = args.out.is_file() and args.out.read_text(encoding="utf-8") == rendered
        official_current = (
            args.official_out.is_file()
            and args.official_out.read_text(encoding="utf-8") == official
        )
        effects_current = (
            args.field_effects_out.is_file()
            and args.field_effects_out.read_text(encoding="utf-8") == field_effects
        )
        return 0 if current and official_current and effects_current else 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered, encoding="utf-8", newline="\n")
    args.official_out.parent.mkdir(parents=True, exist_ok=True)
    args.official_out.write_text(official, encoding="utf-8", newline="\n")
    args.field_effects_out.parent.mkdir(parents=True, exist_ok=True)
    args.field_effects_out.write_text(field_effects, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
