"""Local planning and native-runtime compilation for the pinned verl OPD profile."""

from __future__ import annotations

import math
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from miniverl.bridge.opd_capabilities import decide_placement
from miniverl.bridge.opd_v08 import CompiledLocalExecutionPlan
from miniverl.config.models import RunConfig
from miniverl.errors import ConfigError

__all__ = ["OPDSystemPlan", "build_system_plan", "compile_native_run_config"]


class OPDSystemPlan(BaseModel):
    """A weight-free plan whose estimates are never labelled as measurements."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    profile: str
    upstream: dict[str, str]
    compiled_digest: str
    executable: bool
    overrides: list[dict[str, Any]]
    reinterpretation_acceptance: dict[str, Any]
    acknowledgement_required_mappings: list[dict[str, Any]]
    field_classification_counts: dict[str, int]
    unsupported_fields: list[str]
    student: dict[str, Any]
    teacher: dict[str, Any]
    data: dict[str, Any]
    loss: dict[str, Any]
    local_execution: dict[str, Any]
    memory: dict[str, Any]
    batching: dict[str, Any]
    disk: dict[str, Any]
    time_to_first_update: dict[str, Any]


def _parameter_estimate(model_id: str) -> tuple[int | None, str]:
    match = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*[bB](?![A-Za-z])", model_id)
    if match is None:
        return None, "unknown"
    return int(float(match.group(1)) * 1_000_000_000), "estimated_from_model_identity"


def _gib(value: float) -> float:
    return round(value / (1024**3), 3)


def _weight_bytes_per_parameter(*, quantization: str, dtype: str) -> float:
    if quantization == "nf4":
        return 0.65
    if quantization == "int8":
        return 1.15
    if dtype == "float32":
        return 4.2
    return 2.2


def build_system_plan(compiled: CompiledLocalExecutionPlan) -> OPDSystemPlan:
    """Build a deterministic plan without importing torch or loading weights."""
    source = compiled.source
    student_params, student_basis = _parameter_estimate(source.actor_rollout_ref.model.path)
    teacher_params, teacher_basis = _parameter_estimate(
        source.distillation.teacher_models.teacher_model.model_path
    )
    rank = source.actor_rollout_ref.model.lora_rank
    actor_runtime = source.miniverl.actor_runtime
    teacher_runtime = source.miniverl.teacher_runtime
    teacher_dtype = source.distillation.teacher_models.teacher_model.inference.dtype
    student_static = (
        None
        if student_params is None
        else student_params
        * _weight_bytes_per_parameter(
            quantization=actor_runtime.quantization, dtype=actor_runtime.dtype
        )
    )
    teacher_static = (
        None
        if teacher_params is None
        else teacher_params
        * _weight_bytes_per_parameter(
            quantization=teacher_runtime.quantization, dtype=teacher_dtype
        )
    )
    trainable_optimizer = (
        None if student_params is None else max(96 * rank * math.sqrt(student_params), 64 * 1024**2)
    )
    max_tokens = source.data.max_prompt_length + source.data.max_response_length
    rollout_batch = source.miniverl.batching.rollout_batch_size
    token_buffers = max_tokens * rollout_batch * 64 * 1024
    target_bytes = (
        source.data.train_batch_size
        * source.data.max_response_length
        * source.distillation.distillation_loss.topk
        * 12
    )
    known_static = sum(
        value
        for value in (student_static, teacher_static, trainable_optimizer)
        if value is not None
    )
    limit = source.miniverl.memory.vram_limit_gib
    headroom = source.miniverl.memory.headroom_gib
    resident_feasible = (
        None
        if student_static is None or teacher_static is None
        else _gib(known_static + token_buffers + target_bytes) <= limit - headroom
    )
    decision = decide_placement(
        requested=source.miniverl.runtime.mode,
        student_quantization=actor_runtime.quantization,
        teacher_quantization=teacher_runtime.quantization,
        resident_feasible=resident_feasible,
        shared_backbone_feasible=(
            source.actor_rollout_ref.model.path
            == source.distillation.teacher_models.teacher_model.model_path
        ),
    )
    classifications: dict[str, int] = {}
    for item in compiled.compatibility:
        classifications[item.classification] = classifications.get(item.classification, 0) + 1
    return OPDSystemPlan(
        profile=compiled.profile,
        upstream=compiled.upstream,
        compiled_digest=compiled.compiled_digest,
        executable=compiled.executable and decision.executable_without_probe,
        overrides=[item.model_dump(mode="json") for item in compiled.overrides],
        reinterpretation_acceptance=compiled.reinterpretation_acceptance,
        acknowledgement_required_mappings=[
            item.model_dump(mode="json")
            for item in compiled.compatibility
            if item.user_confirmation_required
        ],
        field_classification_counts=classifications,
        unsupported_fields=[
            item.upstream_field
            for item in compiled.compatibility
            if item.classification == "unsupported"
        ],
        student={
            "model_id": source.actor_rollout_ref.model.path,
            "revision": source.miniverl.student_revision,
            "parameter_count": student_params,
            "parameter_count_status": student_basis,
            "runtime": actor_runtime.model_dump(mode="json"),
            "adapter": {
                "path": source.actor_rollout_ref.model.lora_adapter_path,
                **source.miniverl.student_adapter.model_dump(mode="json"),
            },
        },
        teacher={
            "model_id": source.distillation.teacher_models.teacher_model.model_path,
            "revision": source.miniverl.teacher_revision,
            "parameter_count": teacher_params,
            "parameter_count_status": teacher_basis,
            "adapter": source.miniverl.teacher_adapter.model_dump(mode="json"),
            "runtime": {
                "dtype": teacher_dtype,
                **teacher_runtime.model_dump(mode="json"),
            },
        },
        data={
            "train_files": source.data.train_files,
            "val_files": source.data.val_files,
            "row_counts": "unknown_until_scan",
            "prompt_tokens_max": source.data.max_prompt_length,
            "response_tokens_max": source.data.max_response_length,
        },
        loss={
            "mode": source.distillation.distillation_loss.loss_mode,
            "aggregation": source.actor_rollout_ref.actor.loss_agg_mode,
            "top_k": source.distillation.distillation_loss.topk,
            "task_rewards": False,
        },
        local_execution={
            "strategy": decision.strategy,
            "requested_strategy": source.miniverl.runtime.mode,
            "reason": decision.reason,
            "placement_not_proven": decision.placement_not_proven,
            "executable_without_probe": decision.executable_without_probe,
            "swap_feasible": decision.swap_feasible,
            "resident_feasible": decision.resident_feasible,
            "shared_backbone_feasible": decision.shared_backbone_feasible,
            "roles": [
                "ActorRuntime",
                "TeacherRuntime",
                "ReferenceRuntime:not_used",
                "RolloutRuntime",
                "UpdateRuntime",
                "ArtifactRuntime",
            ],
            "phases": [
                "actor_rollout",
                "teacher_scoring",
                "actor_forward_backward_update",
                "checkpoint_evaluation",
            ],
            "distributed_execution": False,
            "declared_rollout_engine": source.actor_rollout_ref.rollout.name,
            "declared_teacher_engine": (
                source.distillation.teacher_models.teacher_model.inference.name
            ),
            "dynamic_physical_batching": source.actor_rollout_ref.actor.use_dynamic_bsz,
            "context_limits": {
                "actor": source.actor_rollout_ref.rollout.max_model_len,
                "teacher": (
                    source.distillation.teacher_models.teacher_model.inference.max_model_len
                ),
            },
        },
        memory={
            "status": "estimated",
            "static_weight_gib": {
                "student": None if student_static is None else _gib(student_static),
                "teacher": None if teacher_static is None else _gib(teacher_static),
            },
            "trainable_and_optimizer_gib": (
                None if trainable_optimizer is None else _gib(trainable_optimizer)
            ),
            "kv_and_token_buffer_gib": _gib(token_buffers),
            "teacher_target_gib": _gib(target_bytes),
            "configured_vram_limit_gib": limit,
            "configured_headroom_gib": headroom,
            "declared_rollout_memory_fraction": (
                source.actor_rollout_ref.rollout.gpu_memory_utilization
            ),
            "declared_teacher_memory_fraction": (
                source.distillation.teacher_models.teacher_model.inference.gpu_memory_utilization
            ),
            "measured_peak_reserved_gib": None,
        },
        batching={
            "rollout": source.miniverl.batching.rollout_batch_size,
            "declared_rollout_sequence_cap": source.actor_rollout_ref.rollout.max_num_seqs,
            "declared_rollout_token_cap": (source.actor_rollout_ref.rollout.max_num_batched_tokens),
            "teacher_score": source.miniverl.batching.teacher_score_batch_size,
            "update_trajectories": source.miniverl.batching.update_trajectory_batch_size,
            "update_token_cap": source.actor_rollout_ref.actor.ppo_max_token_len_per_gpu,
        },
        disk={
            "status": "estimated",
            "model_download_gib": _gib(known_static * 2),
            "run_artifacts_gib": round(max(0.25, _gib(target_bytes) * 4), 3),
        },
        time_to_first_update={"status": "unknown", "seconds": None},
    )


def compile_native_run_config(
    compiled: CompiledLocalExecutionPlan,
    *,
    system_plan: OPDSystemPlan | None = None,
) -> RunConfig:
    """Translate the supported profile into an executable, validated RunConfig."""
    source = compiled.source
    plan = system_plan or build_system_plan(compiled)
    strategy = plan.local_execution["strategy"]
    if not plan.executable:
        if strategy == "requires_probe":
            raise ConfigError(
                "OPD placement requires a resident feasibility probe before execution",
                hint="use model identities with known sizes or an explicitly proven legal placement",
            )
        raise ConfigError("OPD system plan is not executable under the configured placement")
    if strategy == "shared_backbone":
        raise ConfigError(
            "shared_backbone execution needs a materialized frozen teacher adapter",
            hint="the packaged Qwen3 pair uses auto/dual_model_resident/swap",
        )
    if not source.miniverl.student_revision or not source.miniverl.teacher_revision:
        raise ConfigError(
            "runnable OPD plans require immutable student and teacher revisions",
            hint="set miniverl.student_revision and miniverl.teacher_revision",
        )
    total_steps = source.trainer.total_training_steps
    if total_steps is None:
        total_steps = source.trainer.total_epochs
    logical_batch = source.data.train_batch_size
    memory_strategy = "resident" if strategy == "dual_model_resident" else strategy
    student_adapter = None
    if source.actor_rollout_ref.model.lora_adapter_path is not None:
        student_adapter = {
            "path": source.actor_rollout_ref.model.lora_adapter_path,
            **source.miniverl.student_adapter.model_dump(mode="json"),
        }
    from miniverl.bridge.profiles import get_profile

    profile_identity = get_profile(compiled.profile).identity.model_dump(mode="json")
    payload = {
        "schema_version": 1,
        "run": {
            "name": source.trainer.experiment_name,
            "mode": "opd",
            "seed": source.data.seed or 0,
            "output_dir": "runs",
            "deterministic": True,
            "tags": [compiled.profile, "verl-v0.8.0", "pure-opd"],
            "profile_identity": profile_identity,
        },
        "models": {
            "backend": "hf",
            "runtime": "dual_model",
            "device": "cuda",
            "student": {
                "model_id": source.actor_rollout_ref.model.path,
                "revision": source.miniverl.student_revision,
                "dtype": source.miniverl.actor_runtime.dtype,
                "quantization": source.miniverl.actor_runtime.quantization,
                "attn_implementation": source.miniverl.actor_runtime.attn_implementation,
                "gradient_checkpointing": source.actor_rollout_ref.model.enable_gradient_checkpointing,
                "lora": {
                    "enabled": True,
                    "r": source.actor_rollout_ref.model.lora_rank,
                    "alpha": source.actor_rollout_ref.model.lora_alpha,
                    "target_modules": source.actor_rollout_ref.model.target_modules,
                },
                "adapter": student_adapter,
            },
            "teacher": {
                "model_id": source.distillation.teacher_models.teacher_model.model_path,
                "revision": source.miniverl.teacher_revision,
                "dtype": source.distillation.teacher_models.teacher_model.inference.dtype,
                "quantization": source.miniverl.teacher_runtime.quantization,
                "attn_implementation": source.miniverl.teacher_runtime.attn_implementation,
                "mode": "standard",
                "toy_pretrain_steps": 0,
                "adapter": (
                    None
                    if source.miniverl.teacher_adapter.path is None
                    else source.miniverl.teacher_adapter.model_dump(mode="json")
                ),
            },
        },
        "source": {
            "kind": "verl_parquet",
            "train_files": source.data.train_files,
            "val_files": source.data.val_files,
            "prompt_key": source.data.prompt_key,
            "allow_plain_string_prompts": False,
            "use_task_rewards": False,
            "max_prompt_length": source.data.max_prompt_length,
            "max_response_length": source.data.max_response_length,
            "truncation": source.data.truncation,
            "shuffle": source.data.shuffle,
            "seed": source.data.seed or 0,
        },
        "rollout": {
            "max_turns": 1,
            "max_total_tokens": source.data.max_prompt_length + source.data.max_response_length,
            "temperature": source.actor_rollout_ref.rollout.temperature,
            "top_p": source.actor_rollout_ref.rollout.top_p,
            "prompt_batch_size": min(
                source.miniverl.batching.rollout_batch_size,
                source.actor_rollout_ref.rollout.max_num_seqs
                or source.miniverl.batching.rollout_batch_size,
            ),
            "max_padded_tokens": source.actor_rollout_ref.rollout.max_num_batched_tokens
            or (source.data.max_prompt_length + source.data.max_response_length)
            * source.miniverl.batching.rollout_batch_size,
        },
        "selection": {"selector": "all_model_tokens"},
        "loss": {
            "mode": "forward_kl_topk",
            "aggregation": "token-mean",
            "divergence": "forward_kl",
            "temperature": 1.0,
            "scale_by_temperature_squared": False,
            "top_k": source.distillation.distillation_loss.topk,
            "log_prob_min_clamp": source.distillation.distillation_loss.log_prob_min_clamp,
            "loss_max_clamp": source.distillation.distillation_loss.loss_max_clamp,
            "sampled_token_nll_weight": 0.0,
        },
        "train": {
            "cycles": total_steps,
            "rollouts_per_cycle": logical_batch,
            "gradient_accumulation_steps": logical_batch,
            "trajectory_batch_size": source.miniverl.batching.update_trajectory_batch_size,
            "length_bucketing": source.actor_rollout_ref.actor.use_dynamic_bsz,
            "max_update_padded_tokens": (source.actor_rollout_ref.actor.ppo_max_token_len_per_gpu),
            "learning_rate": source.actor_rollout_ref.actor.optim.lr,
            "weight_decay": source.actor_rollout_ref.actor.optim.weight_decay,
            "warmup_steps": source.actor_rollout_ref.actor.optim.lr_warmup_steps,
            "save_every_cycles": max(source.trainer.save_freq, 0),
            "eval_every_cycles": max(source.trainer.test_freq, 0),
        },
        "memory": {
            "strategy": memory_strategy,
            "auto_swap_vram_headroom_gb": source.miniverl.memory.headroom_gib,
        },
        "cache": {
            "dtype": "float32",
            "strict_policy_version": True,
            "reuse_across_policy_versions": False,
            "keep_cycles": 1,
        },
        "eval": {"enabled": bool(source.data.val_files), "baseline_enabled": False},
        "report": {"enabled": True},
    }
    return RunConfig.model_validate(payload)
