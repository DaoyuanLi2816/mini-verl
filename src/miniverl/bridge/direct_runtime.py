"""Runtime inputs and hardware preflight for the immutable direct-config compiler."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from miniverl.bridge.direct import _digest, _get, _put, build_native
from miniverl.config import RunConfig
from miniverl.config.models import VerlParquetSourceConfig
from miniverl.errors import ConfigError


def hardware_plan(
    native: dict[str, Any],
    *,
    parameter_counts: dict[str, int],
    available_gpu_gib: float,
) -> dict[str, Any]:
    """A conservative preflight, not a fitted benchmark or a guarantee of fit.

    Physical microbatch one and temporal critic/reward placement preserve the
    logical batch. Counts come from model metadata, never a model-name guess.
    """
    actor = parameter_counts["actor"]
    lora = native["models"]["student"]["lora"]["enabled"]
    # Weights plus gradients/moments for dense AdamW; LoRA keeps the base frozen.
    actor_bytes = 4 if native["models"]["student"].get("dtype") == "float32" else 2
    state = actor * actor_bytes * (1 if lora else 4) / 2**30
    peak = state + 2.0
    roles = {"actor": {"parameters": actor, "minimum_state_gib": state, "estimated_peak_gib": peak}}
    if native["critic"]["enabled"]:
        critic_count = parameter_counts.get("critic", actor)
        critic_lora = (native["critic"].get("model") or native["models"]["student"])["lora"][
            "enabled"
        ]
        critic_spec = native["critic"].get("model") or native["models"]["student"]
        # Critic construction is on CPU before temporal GPU placement; auto
        # therefore selects float32, unlike the CUDA-resident actor.
        critic_bytes = 2 if critic_spec.get("dtype") in {"bfloat16", "float16"} else 4
        critic_state = critic_count * critic_bytes * (1 if critic_lora else 4) / 2**30
        roles["critic"] = {
            "parameters": critic_count,
            "minimum_state_gib": critic_state,
            "estimated_peak_gib": critic_state + 2.0,
        }
    if "reward" in parameter_counts:
        reward_state = parameter_counts["reward"] * 2 / 2**30
        roles["reward"] = {
            "parameters": parameter_counts["reward"],
            "minimum_state_gib": reward_state,
            "estimated_peak_gib": reward_state + 1.0,
        }
    blockers = [
        name for name, role in roles.items() if role["estimated_peak_gib"] > available_gpu_gib
    ]
    return {
        "status": "hardware_blocked"
        if blockers or available_gpu_gib <= 0
        else "executable_with_local_lowering",
        "available_gpu_gib": available_gpu_gib,
        "roles": roles,
        "blocking_roles": blockers,
        "estimate_only": True,
        "physical_microbatch": 1,
        "placement": "temporal_cpu_offload",
        "logical_prompt_batch": native["train"]["rollouts_per_cycle"],
        "logical_trajectory_minibatch": native["train"]["gradient_accumulation_steps"],
        "reference_placement": "cpu" if native["models"].get("reference") else None,
        "note": "state-size estimate plus activation reserve; OOM remains possible; explicit smaller model bindings are supported",
    }


def _snapshot(model: str, revision: str | None, *, offline: bool) -> tuple[str | None, int]:
    """Resolve identity and parameter metadata without downloading weights."""
    from huggingface_hub import HfApi

    if Path(model).is_dir():
        # Local weight indexes contain total_size, not trainable parameter counts.
        # Avoid pretending bytes are parameters without the tensor dtypes.
        from safetensors import safe_open

        count = 0
        for path in sorted(Path(model).glob("*.safetensors")):
            with safe_open(path, framework="pt", device="cpu") as handle:
                for key in handle.keys():  # noqa: SIM118 - safetensors is not an iterable mapping
                    shape = handle.get_slice(key).get_shape()
                    size = 1
                    for dimension in shape:
                        size *= dimension
                    count += size
        if not count:
            raise ConfigError(
                "local model binding requires safetensors weights for hardware preflight"
            )
        return revision, count  # a content digest is not a fabricated Hub commit
    if offline:
        from huggingface_hub import snapshot_download

        snapshot = Path(
            snapshot_download(
                model, revision=revision, local_files_only=True, allow_patterns=["*.safetensors"]
            )
        )
        _, count = _snapshot(str(snapshot), revision, offline=True)
        return snapshot.name, count
    info = HfApi().model_info(model, revision=revision, files_metadata=False)
    metadata = info.safetensors
    total = getattr(metadata, "total", None)
    if not info.sha or not isinstance(total, int) or total <= 0:
        raise ConfigError(f"model {model} lacks verified safetensors parameter metadata")
    return info.sha, total


def _local_snapshot_digest(model: str) -> str | None:
    root = Path(model)
    if not root.is_dir():
        return None
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in {".json", ".safetensors", ".model", ".txt", ".py"}:
            digest.update(path.relative_to(root).as_posix().encode() + b"\0")
            digest.update(str(path.stat().st_size).encode() + b"\0")
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


def prepare_direct(
    report: dict[str, Any], *, offline: bool = False
) -> tuple[RunConfig | None, dict[str, Any]]:
    """Resolve local rows, immutable snapshots and hardware before reserving a run."""
    import copy

    import torch

    from miniverl.data.verl_parquet import VerlParquetDataset
    from miniverl.models.factory import build_tokenizer

    native_builder = build_native
    if report.get("profile") == "verl-rl-v0.9-single-gpu-v5":
        from miniverl.bridge.direct_v5 import build_native as native_builder

    if report["semantic_status"] != "accepted":
        return None, report
    effective = copy.deepcopy(report["effective_source"])
    required = [row for row in report["required_bindings"] if row["field"] != "dataset"]
    if required:
        return None, {**report, "required_bindings": required}
    counts = {}
    actor_revision, counts["actor"] = _snapshot(
        _get(effective, "actor_rollout_ref.model.path"),
        _get(effective, "miniverl.actor.revision", None),
        offline=offline,
    )
    _put(effective, "miniverl.actor.revision", actor_revision)
    if _get(effective, "algorithm.adv_estimator") == "gae":
        critic_revision, counts["critic"] = _snapshot(
            _get(effective, "critic.model.path", _get(effective, "actor_rollout_ref.model.path")),
            _get(effective, "miniverl.critic.revision", None),
            offline=offline,
        )
        _put(effective, "miniverl.critic.revision", critic_revision)
    if _get(effective, "reward.reward_model.enable", False):
        revision, counts["reward"] = _snapshot(
            _get(effective, "reward.reward_model.model_path"),
            _get(effective, "miniverl.reward.revision", None),
            offline=offline,
        )
        _put(effective, "miniverl.reward.revision", revision)
    native = native_builder(
        effective, cycles=_get(effective, "trainer.total_training_steps", None) or 1
    )
    available = torch.cuda.mem_get_info()[0] / 2**30 if torch.cuda.is_available() else 0.0
    plan = hardware_plan(native, parameter_counts=counts, available_gpu_gib=available)
    updated = {
        **report,
        "hardware_plan": plan,
        "execution_status": plan["status"],
        "effective_source": effective,
    }
    if plan["status"] == "hardware_blocked":
        return None, updated
    provisional = RunConfig.model_validate(native)
    assert isinstance(provisional.source, VerlParquetSourceConfig)
    tokenizer = build_tokenizer(provisional, local_files_only=offline)
    dataset = VerlParquetDataset(provisional.source, tokenizer=tokenizer)
    manifest = dataset.inspect()
    rows = manifest.rows["train"]
    batch = native["train"]["rollouts_per_cycle"]
    batches = rows // batch
    if batches < 1:
        raise ConfigError(
            f"retained dataset has {rows} prompts, smaller than logical batch {batch}; no drop-last batches"
        )
    epochs = _get(effective, "trainer.total_epochs", 1)
    limit = _get(effective, "trainer.total_training_steps", None)
    cycles = min(batches * epochs, limit) if limit is not None else batches * epochs
    native = native_builder(effective, cycles=cycles)
    native["memory"]["strategy"] = "swap"  # actor/critic/RM temporal placement
    native["rollout"]["prompt_batch_size"] = 1
    native["train"]["trajectory_batch_size"] = 1
    native["run"]["profile_identity"].update(
        {
            "source_config_sha256": report["source_config_sha256"],
            "runtime_bindings": report["runtime_bindings"],
            **({"composition": report["composition"]} if "composition" in report else {}),
            "local_placement": {"physical_microbatch": 1, "strategy": "swap", "reference": "cpu"},
            "dataset_content_digest": manifest.content_digest,
            "local_snapshot_digests": {
                role: _local_snapshot_digest(_get(effective, field))
                for role, field in {
                    "actor": "actor_rollout_ref.model.path",
                    "critic": "critic.model.path",
                    "reward": "reward.reward_model.model_path",
                }.items()
                if role in counts
            },
        }
    )
    native["run"]["execution_plan_digest"] = _digest(native)
    config = RunConfig.model_validate(native)
    return config, {
        **updated,
        "required_bindings": [],
        "resolved_native_config": config.model_dump(mode="json"),
        "ir_sha256": _digest(config.model_dump(mode="json")),
        "filtered_dataset_rows": manifest.rows,
        "dataset_content_digest": manifest.content_digest,
        "schedule": {"epochs": epochs, "batches_per_epoch": batches, "cycles": cycles},
    }


def execute_direct(
    source: Path,
    *,
    bindings: list[str],
    dry_run: bool,
    output: Path | None,
    run_id: str | None,
    resume: Path | None,
    offline: bool,
    approved_reward_sha256: str | None,
    resume_from: Path | None = None,
    profile: str = "verl-rl-v0.9-single-gpu-v4",
    composition: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from miniverl.bridge.direct import compile_direct

    if profile == "verl-rl-v0.9-single-gpu-v5":
        from miniverl.bridge.direct_v5 import compile_direct
    elif profile != "verl-rl-v0.9-single-gpu-v4":
        raise ConfigError("unsupported direct runtime profile")
    report = compile_direct(source, bindings=bindings)
    if composition is not None:
        report.update({"composition_status": "resolved", "composition": composition})
    if dry_run or report["semantic_status"] != "accepted":
        return report
    binding = next(
        (row["bound"] for row in report["runtime_bindings"] if row["binding"] == "reward.function"),
        None,
    )
    if binding is not None and not approved_reward_sha256:
        raise ConfigError("local reward code requires --trust-reward-code SHA256")
    resume_root = resume or (resume_from.parent.parent if resume_from is not None else None)
    if resume_root is not None:
        from miniverl.reporting.inspection import read_object

        previous = read_object(resume_root / "verl-direct-report.json")
        if (
            previous.get("source_config_sha256") != report["source_config_sha256"]
            or previous.get("runtime_bindings") != report["runtime_bindings"]
            or previous.get("composition") != report.get("composition")
            or previous.get("profile") != report.get("profile")
        ):
            raise ConfigError("resume requires the original upstream bytes and runtime bindings")
        previous_native = previous.get("resolved_native_config", {})
        previous_actor = _get(previous_native, "models.student", {})
        _put(report["effective_source"], "miniverl.actor.revision", previous_actor.get("revision"))
        if _get(previous_native, "critic.model", None):
            _put(
                report["effective_source"],
                "miniverl.critic.revision",
                _get(previous_native, "critic.model.revision"),
            )
        if _get(previous_native, "reward.model", None):
            _put(
                report["effective_source"],
                "miniverl.reward.revision",
                _get(previous_native, "reward.model.revision"),
            )
    config, report = prepare_direct(report, offline=offline)
    if config is None:
        return report
    provider = None
    if binding is not None:
        from miniverl.rewards.binding import BoundVerlReward

        if not approved_reward_sha256:
            raise ConfigError("local reward code requires --trust-reward-code SHA256")
        provider = BoundVerlReward(
            binding,
            approved_sha256=approved_reward_sha256,
            preserve_metrics=profile == "verl-rl-v0.9-single-gpu-v5",
        )
        config.run.profile_identity["reward_code_sha256"] = approved_reward_sha256
    config.run.execution_plan_digest = None
    config.run.execution_plan_digest = _digest(config.model_dump(mode="json"))
    report["resolved_native_config"] = config.model_dump(mode="json")
    report["ir_sha256"] = _digest(report["resolved_native_config"])
    from miniverl.trainer import OPDTrainer
    from miniverl.utils.runs import write_json_atomic

    with OPDTrainer.from_config(
        config,
        output_dir=output,
        run_id=run_id,
        resume=resume,
        resume_from=resume_from,
        local_files_only=offline,
        reward_provider=provider,
    ) as trainer:
        original = source.read_bytes()
        if hashlib.sha256(original).hexdigest() != report["source_config_sha256"]:
            raise ConfigError("source config changed after semantic compilation")
        (trainer.paths.root / "verl-source.yaml").write_bytes(original)
        write_json_atomic(trainer.paths.root / "verl-direct-report.json", report)
        if composition is not None:
            write_json_atomic(trainer.paths.root / "verl-composition.json", composition)
        result = trainer.train()
        return {
            **report,
            "execution_status": "completed",
            "run_dir": str(result.run_dir),
            "actor_updates": result.global_step,
        }
